#!/usr/bin/env python
# encoding: utf-8
import os
import tensorflow as tf
import sys

tf.logging.set_verbosity(tf.logging.INFO)


def save_model_for_production(model, version, path='prod_models'):
    tf.keras.backend.set_learning_phase(1)
    if not os.path.exists(path):
        os.mkdir(path)
    export_path = os.path.join(
        tf.compat.as_bytes(path),
        tf.compat.as_bytes(version))
    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    model_input = tf.saved_model.utils.build_tensor_info(model.input)
    model_output = tf.saved_model.utils.build_tensor_info(model.output)

    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'inputs': model_input},
            outputs={'output': model_output},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

    classification_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                    model_input
            },
            outputs={
                tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                    model_output
            },
            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))

    with tf.keras.backend.get_session() as sess:
        builder.add_meta_graph_and_variables(
            sess=sess, tags=[tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict':
                    prediction_signature,
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                    classification_signature,
            },
            main_op=tf.tables_initializer(),
            strip_default_attrs=True)

        builder.save()


if __name__ == '__main__':
    model_file = sys.argv(1)
    if os.path.isfile(model_file):
        print('model file detected. Loading.')
        model = tf.keras.models.load_model(model_file)
        model.summary()

        export_path = "tf-model"
        save_model_for_production(model, "1", export_path)
    else:
        print('No model file detected.')



