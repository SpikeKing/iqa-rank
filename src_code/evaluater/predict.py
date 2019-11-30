import argparse
import glob
import json
import os
import sys

import cv2
import tensorflow as tf
from keras import backend as K
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resources
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

from root_dir import MODELS_DIR, ROOT_DIR, DATA_DIR

p = os.path.dirname(os.path.dirname(os.path.dirname((os.path.abspath(__file__)))))
if p not in sys.path:
    sys.path.append(p)

from src_code.utils.utils import calc_mean_score, save_json
from src_code.handlers.model_builder import Nima
from src_code.handlers.data_generator import TestDataGenerator


def image_file_to_json(img_path):
    img_dir = os.path.dirname(img_path)
    img_id = os.path.basename(img_path).split('.')[0]

    return img_dir, [{'image_id': img_id}]


def image_dir_to_json(img_dir, img_type='jpg'):
    img_paths = glob.glob(os.path.join(img_dir, '*.' + img_type))

    samples = []
    for img_path in img_paths:
        img_id = os.path.basename(img_path).split('.')[0]
        samples.append({'image_id': img_id})

    return samples


def predict(model, data_generator):
    return model.predict_generator(data_generator, workers=8, use_multiprocessing=True, verbose=1)


def save_model(model, saved_dir):
    """
    存储模型
    """
    input_node_names = [node.op.name for node in model.inputs]
    output_node_names = [node.op.name for node in model.outputs]

    print('[Info] input node names: {}'.format(input_node_names))
    print('[Info] pred node names: {}'.format(output_node_names))

    input_tensors = {}
    for node_name in input_node_names:
        i_tensor = tf.get_default_graph().get_tensor_by_name('%s:0' % node_name)
        input_tensors[node_name] = i_tensor

    output_tensors = {}
    for node_name in output_node_names:
        o_tensor = tf.get_default_graph().get_tensor_by_name('%s:0' % node_name)
        output_tensors[node_name] = o_tensor

    print('[Info] input tensor: {}'.format(input_tensors))
    print('[Info] output tensors: {}'.format(output_tensors))

    sess = K.get_session()

    prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(input_tensors, output_tensors)
    signature_map = {signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature}

    legacy_op = control_flow_ops.group(
        tf.local_variables_initializer(),
        resources.initialize_resources(resources.shared_resources()),
        tf.tables_initializer())

    builder = saved_model_builder.SavedModelBuilder(saved_dir)

    builder.add_meta_graph_and_variables(
        sess, [tag_constants.SERVING],
        signature_def_map=signature_map,
        legacy_init_op=legacy_op)

    builder.save()


def main(base_model_name, weights_file, image_source, predictions_file, img_format='jpg'):
    # load samples
    if os.path.isfile(image_source):
        image_dir, samples = image_file_to_json(image_source)  # 图片文件夹和样本
    else:
        image_dir = image_source
        samples = image_dir_to_json(image_dir, img_type='jpg')

    # build model and load weights
    nima = Nima(base_model_name, weights=None)
    nima.build()
    nima.nima_model.load_weights(weights_file)  # 加载参数

    # 存储模型的逻辑
    model = nima.nima_model
    saved_dir = os.path.join(DATA_DIR, 'model-tf')
    save_model(model, saved_dir)  # 存储模型

    # initialize data generator，生成测试样本
    data_generator = TestDataGenerator(samples, image_dir, 64, 10, nima.preprocessing_function(),
                                       img_format=img_format)

    # get predictions
    predictions = predict(nima.nima_model, data_generator)

    # calc mean scores and add to samples
    for i, sample in enumerate(samples):
        # print(predictions[i])
        sample['mean_score_prediction'] = calc_mean_score(predictions[i])

    print('[Info] 模型输出')
    print(json.dumps(samples, indent=2))

    if predictions_file is not None:
        save_json(samples, predictions_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('-b', '--base-model-name', help='CNN base model name', required=True)
    # parser.add_argument('-w', '--weights-file', help='path of weights file', required=True)
    # parser.add_argument('-is', '--image-source', help='image directory or file', required=True)
    # parser.add_argument('-pf', '--predictions-file', help='file with predictions', required=False, default=None)

    args = parser.parse_args()
    args.base_model_name = 'MobileNet'

    args.weights_file = os.path.join(MODELS_DIR, 'MobileNet/weights_mobilenet_aesthetic_0.07.hdf5')
    print('[Info] 模型路径: {}'.format(args.weights_file))
    args.image_source = os.path.join(ROOT_DIR, 'src_code/tests/test_images/test1.jpg')
    args.predictions_file = None  # 不存储结果

    img_np = cv2.imread(args.image_source)
    print('[Info] 图像尺寸: {}'.format(img_np.shape))

    main(**args.__dict__)
