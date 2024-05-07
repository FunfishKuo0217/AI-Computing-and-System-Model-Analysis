### 2-3-2 訂正: 參數版
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# input
A = make_tensor_value_info('A', TensorProto.FLOAT, [128, 128])
B = make_tensor_value_info('B', TensorProto.FLOAT, [128, 128])
# output
C = make_tensor_value_info('C', TensorProto.FLOAT, [128, 128])

# segment_size
segment_size = 64
num_seg = int(A.type.tensor_type.shape.dim[0].dim_value/segment_size)

# A splitted submatrix
input_var = ['A', 'B']
output_layer_dict = {}
output_layer_dict['A'] = []
output_layer_dict['B'] = []

for input in input_var:
    for x_dim in range(num_seg):
        for y_dim in range(num_seg):
            output_layer_dict[input].append(f'{input}_{x_dim}{y_dim}')

# split node list
split_node1_1 = make_node("Split", inputs=["A"], outputs=["A_0", "A_1"], axis=0)
split_node1_2 = make_node("Split", inputs=["A_0", "A_1"], outputs=output_layer_dict["A"], axis=1)
split_node2_1 = make_node("Split", inputs=["B"], outputs=["B_0", "B_1"], axis=0)
split_node2_2 = make_node("Split", inputs=["B_0", "B_1"], outputs=output_layer_dict["B"], axis=1) 

# matmal node list (seg^3)
h_A = 2
v_A = 2
h_B = 2
v_B = 2

output_layer_dict['A'][:h_B]*h_B + output_layer_dict['A'][h_B:]*h_B

# construct A
A_list = []
for i in range(0, h_B*v_B, h_B):
    A_list += output_layer_dict['A'][i:h_B*i+h_B]*h_B

B_list = []
for i in range(0, h_B):
    for j in range(0, h_B*h_B, h_B):
        B_list += [output_layer_dict['B'][i+j]]

B_list *= 2

matmul_list = []
matmul_node = []
for a, b in zip(A_list, B_list):
    # print(f'{a}*{b}')
    matmul_list.append(f'{a}*{b}')
    matmul_node.append(make_node('MatMul', [a, b], [f'{a}*{b}'], name="2DMM 64*64*64")) # -> C00)


# operatpr - sum
sum_node = []
matmul_idx = 0
for i in range(v_A):
    for j in range(h_B):   
        sum_node.append(make_node('Sum', matmul_list[matmul_idx*h_A:matmul_idx*h_A+h_A], [f"C_{i}{j}"], name = "Sum"))
        matmul_idx += 1

# opeator = concat
node_concat = make_node('Concat', ['C_00', 'C_01', 'C_10', 'C_11'], ['C'], axis = 0)
concat_node = []
for i in range(v_A):
    concat_node.append(make_node('Concat', [f'C_{i}{x}' for x in range(v_B)], [f'C_{i}'], axis = 1))
concat_node.append(make_node('Concat', [f'C_{x}' for x in range(v_A)], ['C'], axis = 0))

graph = make_graph([split_node1_1, split_node1_2, split_node2_1, split_node2_2] + matmul_node + sum_node + concat_node,  # nodes
                    'subgraph-2',  # a name
                    [A, B],  # inputs
                    [C])  # outputs

onnx_model = make_model(graph)
check_model(onnx_model)

# # The serialization
with open("Subgraph2.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
