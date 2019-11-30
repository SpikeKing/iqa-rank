#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2019. All rights reserved.
Created by C. L. Wang on 2019/11/30
"""

import os
import base64
import cv2
import tensorflow as tf
from PIL import Image as pil_image
import numpy as np

from root_dir import ROOT_DIR, MODELS_DIR, DATA_DIR
from src_code.handlers.model_builder import Nima


def norm_img(x):
    x = x.astype('float')
    x /= 127.5
    x -= 1.
    return x


def normalize_labels(labels):
    labels_np = np.array(labels)
    return labels_np / labels_np.sum()


def calc_mean_score(score_dist):
    score_dist = normalize_labels(score_dist)
    return (score_dist * np.arange(1, 11)).sum()


def read_img_pil(img_path):
    """
    pillow读取图像，与标准相同
    """
    img = pil_image.open(img_path)
    img = img.resize((224, 224), pil_image.NEAREST)
    img_np = np.asarray(img)
    img_np = norm_img(img_np)
    img_np = img_np.astype(np.float32)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np


def read_img_opencv(img_path):
    """
    opencv读取图像
    """
    img_np = cv2.imread(img_path)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    img_np = cv2.resize(img_np, (224, 224), interpolation=cv2.INTER_NEAREST)
    img_np = norm_img(img_np)
    img_np = np.expand_dims(img_np, axis=0)
    return img_np


def use_serve_mode(img_np):
    # export_path = "data/model-tf"  # 模型文件
    export_path = os.path.join(DATA_DIR, 'model-tf')
    input_name = "input_1:0"
    output_name = "dense_1/Softmax:0"

    with tf.Session(graph=tf.Graph()) as sess:
        tf.saved_model.loader.load(sess, ["serve"], export_path)
        graph = tf.get_default_graph()
        print(graph.get_operations())
        res = sess.run(output_name,
                       feed_dict={input_name: img_np})
        # print('[Info] 最终结果: {}'.format(res))
    return res


def image_decode(image_raw):
    """
    图像解码，base64输入
    """
    image = tf.decode_base64(image_raw)
    image = tf.decode_raw(image, tf.float32)  # 图像需要float32格式，根据不同的数据处理
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, [-1, 224, 224, 3])
    return image


def predict_img(img_path):
    print('[Info] 图像路径: {}'.format(img_path))
    base_model_name = 'MobileNet'
    weights_file = os.path.join(MODELS_DIR, 'MobileNet/weights_mobilenet_aesthetic_0.07.hdf5')
    print('[Info] 模型路径: {}'.format(weights_file))

    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)  # 加载参数

    img_np = read_img_pil(img_path)  # 与标准相同

    print('\n[Info] 默认版本')
    score_list = nima.nima_model.predict(img_np)
    print('[Info] 数据分布: {}'.format(score_list))  # 可以通过
    score = calc_mean_score(score_list)
    print('[Info] 评分: {}'.format(score))

    print('\n[Info] 导出版本')
    img_b64 = base64.urlsafe_b64encode(img_np)  # 转换base64
    img_tf = image_decode(img_b64)
    img_np = tf.Session().run(img_tf)

    score_list_v2 = use_serve_mode(img_np)
    print('[Info] 数据分布: {}'.format(score_list_v2))  # 可以通过
    score_v2 = calc_mean_score(score_list_v2)
    print('[Info] 评分: {}'.format(score_v2))


def predict_img_test():
    img_path = os.path.join(ROOT_DIR, 'src_code/tests/test_images/test1.jpg')
    predict_img(img_path)  # 预测图像


def main():
    predict_img_test()


if __name__ == '__main__':
    main()
