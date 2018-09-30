#!/usr/bin/env python
# encoding: utf-8

from __future__ import print_function
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
import numpy as np
import grpc
from tensorflow_serving.apis import prediction_service_pb2_grpc

tf.app.flags.DEFINE_string('server', 'localhost:8500',
                           'PredictionService host:port')
FLAGS = tf.app.flags.FLAGS


def main(_):
    # 发送请求
    channel = grpc.insecure_channel(FLAGS.server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    data = np.zeros((1, 9, 9, 7))
    data = data.astype(np.float32)

    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'tf-model'  # name与tensorflow_model_server --model_name="username"对应
    request.model_spec.signature_name = 'predict'  # signature_name与signature_def_map对应
    request.inputs['inputs'].CopyFrom(
        tf.contrib.util.make_tensor_proto(data, shape=data.shape))  # shape与keras的model.input类型对应
    result = stub.Predict(request)
    probs = np.array(result.outputs['output'].float_val)  # 解析时与pb中定义的输出对应
    print(probs[np.argmax(probs)])


if __name__ == '__main__':
    tf.app.run()



