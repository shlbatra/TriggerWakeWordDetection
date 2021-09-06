import onnx

from onnx_tf.backend import prepare
import tensorflow as tf


onnx_model = onnx.load("onnx_model.onnx")  # load onnx model

tf_rep = prepare(onnx_model)  # prepare tf representation

# Input nodes to the model
print("inputs:", tf_rep.inputs)

# Output nodes from the model
print("outputs:", tf_rep.outputs)

# All nodes in the model
print("tensor_dict:")
print(tf_rep.tensor_dict)

tf_rep.export_graph("hey_fourth_brain")  # export the model

# Converting a SavedModel.
converter = tf.lite.TFLiteConverter.from_saved_model("hey_fourth_brain")
tflite_model = converter.convert()
open("hey_fourth_brain.tflite", "wb").write(tflite_model)
