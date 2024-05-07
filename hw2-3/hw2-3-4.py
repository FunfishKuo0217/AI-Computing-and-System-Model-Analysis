# 使用 onnx 進行 inference --> 2-4 要用的
import onnx
import onnxruntime
import numpy as np

session = onnxruntime.InferenceSession("./alexnet.onnx")
print('Starting original alexnet inference...')
input_ = np.random.randn(1, 3, 224, 224).astype(np.float32)
ans_origin = session.run(None,{"actual_input_1": input_})
# print(ans_origin)

print('Starting customized alexnet inference...')
session = onnxruntime.InferenceSession("./hw2-3-3.onnx")
ans_test = session.run(None,{"actual_input_1": input_})[0]
# print(ans_test)

print('MSE between two models:', np.square(np.subtract(ans_origin[0], ans_test[0])).mean())
