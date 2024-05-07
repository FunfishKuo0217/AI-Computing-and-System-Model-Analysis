# reference: https://onnx.ai/onnx/intro/python.html (onnx linear regression ex)
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# inputs
A = make_tensor_value_info('A', TensorProto.FLOAT, [128, 128])
B = make_tensor_value_info('B', TensorProto.FLOAT, [128, 128])
# B = make_tensor_value_info('B', TensorProto.FLOAT, [None, None])

# outputs, the shape is left undefined
Y = make_tensor_value_info('Y', TensorProto.FLOAT, [128, 128])

# nodes
# It creates a node defined by the operator type MatMul,
node1 = make_node('MatMul', ['A', 'B'], ['Y'])

# from nodes to graph
# the graph is built from the list of nodes, the list of inputs,
# the list of outputs and a name.
graph = make_graph([node1],  # nodes
                    'lr',  # a name
                    [A, B],  # inputs
                    [Y])  # outputs

# onnx graph
# there is no metadata in this case.

onnx_model = make_model(graph)
check_model(onnx_model)

# Display the model
print(onnx_model)

# The serialization
with open("Subgraph1.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
