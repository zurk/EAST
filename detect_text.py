#!/usr/bin/env python3

import argparse
import collections
import datetime
import functools
import json
import logging
import os
import time
import uuid

import chainer
import cv2
import numpy as np
import tqdm
from sqlalchemy.testing.plugin.plugin_base import pre

from see.chainer.text_recognition_demo import create_network, preprocess_image, process
from see.chainer.utils.datatypes import Size

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@functools.lru_cache(maxsize=1)
def get_host_info():
    ret = {}
    with open('/proc/cpuinfo') as f:
        ret['cpuinfo'] = f.read()

    with open('/proc/meminfo') as f:
        ret['meminfo'] = f.read()

    with open('/proc/loadavg') as f:
        ret['loadavg'] = f.read()

    return ret


@functools.lru_cache(maxsize=100)
def get_predictor(checkpoint_path):
    logger.info('loading model')
    import tensorflow as tf
    from EAST import model
    from EAST.eval import resize_image, sort_poly, detect

    input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
    model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
    logger.info('Restore from {}'.format(model_path))
    saver.restore(sess, model_path)

    def predictor(img):
        """
        :return: {
            'text_lines': [
                {
                    'score': ,
                    'x0': ,
                    'y0': ,
                    'x1': ,
                    ...
                    'y3': ,
                }
            ],
            'rtparams': {  # runtime parameters
                'image_size': ,
                'working_size': ,
            },
            'timing': {
                'net': ,
                'restore': ,
                'nms': ,
                'cpuinfo': ,
                'meminfo': ,
                'uptime': ,
            }
        }
        """
        start_time = time.time()
        rtparams = collections.OrderedDict()
        rtparams['start_time'] = datetime.datetime.now().isoformat()
        rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
        timer = collections.OrderedDict([
            ('net', 0),
            ('restore', 0),
            ('nms', 0)
        ])

        im_resized, (ratio_h, ratio_w) = resize_image(img)
        rtparams['working_size'] = '{}x{}'.format(
            im_resized.shape[1], im_resized.shape[0])
        start = time.time()
        score, geometry = sess.run(
            [f_score, f_geometry],
            feed_dict={input_images: [im_resized[:, :, ::-1]]})
        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
        logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

        if boxes is not None:
            scores = boxes[:, 8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        timer['overall'] = duration
        logger.info('[timing] {}'.format(duration))

        text_lines = []
        if boxes is not None:
            text_lines = []
            for box, score in zip(boxes, scores):
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                tl = collections.OrderedDict(zip(
                    ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                    map(float, box.flatten())))
                tl['score'] = float(score)
                text_lines.append(tl)
        ret = {
            'text_lines': text_lines,
            'rtparams': rtparams,
            'timing': timer,
        }
        ret.update(get_host_info())
        return ret

    return predictor


def draw_illu(illu, rst):
    mult = 1.2
    cropped_text = []
    # img_box = cv2.cvtColor(illu.copy(), cv2.COLOR_RGB2BGR)
    img_box = illu.copy()
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 255, 0))

        rect = cv2.minAreaRect(d)
        box = np.int0(cv2.boxPoints(rect))

        x1, y1 = np.min(box, axis=0)
        x2, y2 = np.max(box, axis=0)

        rotated = False
        angle = rect[2]

        if angle < -45:
            angle += 90
            rotated = True

        center = int((x1 + x2) / 2), int((y1 + y2) / 2)
        size = int(mult * (x2 - x1)), int(mult * (y2 - y1))
        M = cv2.getRotationMatrix2D((size[0] / 2, size[1] / 2), angle, 1.0)
        cropped = cv2.getRectSubPix(img_box, size, center)
        cropped = cv2.warpAffine(cropped, M, size)
        croppedW = rect[1][0] if not rotated else rect[1][1]
        croppedH = rect[1][1] if not rotated else rect[1][0]
        croppedRotated = cv2.getRectSubPix(cropped, (int(croppedW * mult), int(croppedH * mult)),
                                           (size[0] / 2, size[1] / 2))
        cropped_text.append(croppedRotated)

    return illu, cropped_text


class TextRecognizer:
    def __init__(self, model_dir, snapshot_name, char_map, gpu):
        self.model_dir = model_dir
        self.snapshot_name = snapshot_name
        self.char_map = char_map
        self.gpu = gpu
        self.log_name = 'log'
        self.dropout_ratio = 0.5
        self.blank_symbol = 0
        # max number of text regions in the image
        self.timesteps = 23
        # max number of characters per word
        self.num_labels = 1

        # open log and extract meta information
        with open(os.path.join(self.model_dir, self.log_name)) as the_log:
            log_data = json.load(the_log)[0]

        self.target_shape = Size._make(log_data['target_size'])
        self.image_size = Size._make(log_data['image_size'])

        self.xp = chainer.cuda.cupy if gpu >= 0 else np
        self.network = create_network(self, log_data)

        # load weights
        with np.load(os.path.join(self.model_dir, self.snapshot_name)) as f:
            chainer.serializers.NpzDeserializer(f).load(self.network)

        # load char map
        with open(self.char_map) as the_map:
            self.char_map = json.load(the_map)

    def recognize_text(self, image):
        image = preprocess_image(image, self.xp, self.image_size)
        return process(image, self.network, self.char_map, self.xp, self)


def save_result(img, rst, save_path, recognizer: TextRecognizer):
    illu, cropped_text = draw_illu(img.copy(), rst)
    cv2.imwrite(save_path, illu)

    cropped_save_path = os.path.splitext(save_path)[0]
    os.makedirs(cropped_save_path, exist_ok=True)
    for i, ct in enumerate(cropped_text):
        word = recognizer.recognize_text(ct)
        cv2.imwrite(os.path.join(cropped_save_path, f"{i}-{word}.jpg"), ct)

    # save json data
    # output_name = os.path.join(config.SAVE_DIR, 'result_{}.json'.format(session_id))
    # with open(output_name, 'w') as f:
    #     json.dump(rst, f)

    # rst['session_id'] = session_id
    # return rst


def detect_text(input_path, output_path, checkpoint_path, recognizer):
    predictor = get_predictor(checkpoint_path)
    if os.path.isdir(input_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.isdir(output_path):
            raise ValueError("Output path should be directory if input path is directory.")
        for img_name in tqdm.tqdm(os.listdir(input_path)):
            img = cv2.imread(os.path.join(input_path, img_name), 1)
            rst = predictor(img)
            save_result(img, rst, os.path.join(output_path, img_name), recognizer)
    else:
        img = cv2.imread(input_path, 1)
        rst = predictor(img)
        save_result(img, rst, output_path, recognizer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_path', required=True)
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument("--text_recognition_model_dir", required=True, help="path to directory where model is saved")
    parser.add_argument("--text_recognition_snapshot_name", required=True, help="name of the snapshot to load")
    parser.add_argument("--text_recognition_char_map", required=True, help="path to char map, that maps class id to character")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint_path):
        raise RuntimeError(
            'Checkpoint `{}` not found'.format(args.checkpoint_path))

    recognizer = TextRecognizer(
        args.text_recognition_model_dir, args.text_recognition_snapshot_name, args.text_recognition_char_map, -1)
    detect_text(args.input, args.output, args.checkpoint_path, recognizer)


if __name__ == '__main__':
    main()
