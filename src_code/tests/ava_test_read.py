#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/8/14
"""

# !/usr/bin/env python
# -- coding: utf-8 --
import sys
import importlib

from src_code.handlers.model_builder import Nima

"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/8/12
"""

import os
import keras
import tensorflow as tf

import cv2
import urllib.request
import numpy as np

from keras.models import Model
from keras.layers import Dropout, Dense
from keras.optimizers import Adam

from root_dir import ROOT_DIR, MODELS_DIR


def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.request.urlopen(url)
    img_np = np.asarray(bytearray(resp.read()), dtype="uint8")
    img_np = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    print('[Info] max: {}, min: {}, avg: {}'.format(np.min(img_np), np.max(img_np), np.mean(img_np)))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    print('[Info] max: {}, min: {}, avg: {}'.format(np.min(img_np), np.max(img_np), np.mean(img_np)))

    return img_np


def norm_img(x):
    x = x.astype('float')
    x /= 127.5
    # x = np.true_divide(x, 127.5)
    x -= 1.
    return x


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist * np.arange(1, 11)).sum()


def main():
    img_url = "https://sm-transfer.oss-cn-hangzhou.aliyuncs.com/zhengsheng.wcl/42042.jpg"
    img_np = url_to_image(img_url)
    print(img_url)
    print(img_np.shape)

    print('[Info] img max: {}, min: {}'.format(np.max(img_np), np.min(img_np)))
    img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_NEAREST)
    print('[Info] max: {}, min: {}, avg: {}'.format(np.min(img_np), np.max(img_np), np.mean(img_np)))

    img_file = os.path.join(ROOT_DIR, 'data', 'tmp.jpg')
    cv2.imwrite(img_file, img_np)

    print('[Info] img max: {}, min: {}'.format(np.max(img_np), np.min(img_np)))
    print(img_np.shape)
    # print(img_np[:10])

    img_np = norm_img(img_np)
    print(img_np[0, 0, :10])
    print('[Info] max: {}, min: {}, avg: {}'.format(np.min(img_np), np.max(img_np), np.mean(img_np)))

    base_model_name = 'MobileNet'
    weights_file = os.path.join(MODELS_DIR, 'MobileNet/weights_mobilenet_technical_0.11.hdf5')

    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)  # 加载参数

    img_np = np.expand_dims(img_np, axis=0)
    score_list = nima.nima_model.predict(img_np)
    print('数据分布: {}'.format(score_list))  # 可以通过


def iqa_model():
    base_model_name = "MobileNet"

    base_module = importlib.import_module('keras.applications.' + base_model_name.lower())

    BaseCnn = getattr(base_module, base_model_name)

    # load pre-trained model，weights不为空, 则加载默认参数
    base_model = BaseCnn(input_shape=(224, 224, 3), weights=None, include_top=False, pooling='avg')

    # add dropout and dense layer，增加1个dropout和dense全连接
    x = Dropout(0)(base_model.output)
    x = Dense(units=10, activation='softmax')(x)

    nima_model = Model(base_model.inputs, x)

    input_data = nima_model.inputs[0]
    inference_op = nima_model.outputs[0]  # 输出

    print("tensors:", input_data.get_shape())
    print("tensors:", inference_op.get_shape())

    # self._loss_op = self.loss(gt_labels, self._inference_op)  # loss


if __name__ == "__main__":
    print('-' * 50)
    print('Hello World!')

    # main()
    iqa_model()
    print('keras: {}'.format(keras.__version__))
    print('tensorflow: {}'.format(tf.__version__))
    print('-' * 50)
