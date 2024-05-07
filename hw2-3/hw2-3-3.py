import onnx
from onnx import TensorProto
from onnx.helper import (
    make_model, make_node, make_graph,
    make_tensor_value_info)
from onnx.checker import check_model

# input_segment
def CustomizeGemm(input_name, weight_name, bias_name, output_name, inSeg, outSeg, prefix):
    inputSeg = inSeg
    outputSeg = outSeg
    # output_segment
    splitted_input = [f"{prefix}/Input_{x}" for x in range(inputSeg)]
    splitted_weight_horizon = [f"{prefix}/W_{x}" for x in range(outputSeg)]
    Splitted_list = {}
    for item in splitted_weight_horizon:
        Splitted_list[item] = []
        for i in range(inputSeg):
            Splitted_list[item].append(f'{prefix}/{item}_{i}')
    # Construct the hidden
    Split_in = make_node("Split", [input_name], splitted_input, axis = 1, num_outputs = inputSeg, name = f'{prefix}/Split/input')
    Split_weight_h = make_node("Split", [weight_name], splitted_weight_horizon, axis = 0, num_outputs = outputSeg, name = f'{prefix}/Split/weight_h')
    Split_node = [Split_in, Split_weight_h]
    for i in range(len(splitted_weight_horizon)):
        weight_blk = splitted_weight_horizon[i]
        Split_node.append(make_node("Split", [weight_blk], Splitted_list[splitted_weight_horizon[i]], num_outputs = inputSeg, axis = 1, name = f'{prefix}/Split/{splitted_weight_horizon[i]}'))
    Gemm_node = []
    Sum_node = []
    for i in range(outputSeg):
        for j in range(inputSeg):
            Gemm_node.append(make_node("Gemm", [splitted_input[j], Splitted_list[splitted_weight_horizon[i]][j]], [f'{prefix}/matmal_{i}_{j}'], alpha = 1.0, beta = 1.0, transA = False, transB = True, name=f"{prefix}/2DMM 64*64*64/matmal_{i}_{j}"))
        Sum_node.append(make_node("Sum", [f'{prefix}/matmal_{i}_{t}' for t in range(inputSeg)], [f'{prefix}/C_{i}'], name = f"{prefix}/Sum/{i}"))
    Concat_node = make_node('Concat', [f'{prefix}/C_{i}' for i in range(outputSeg)], [f'{prefix}/before_bias'], axis = 1, name = f"{prefix}/Concat/C")
    Bias = make_node('Sum', [f'{prefix}/before_bias', bias_name], [output_name], name = f"{prefix}/Sum/bias")
    # combine all node
    insert_node_list = Split_node + Gemm_node + Sum_node + [Concat_node] + [Bias]
    return insert_node_list

insert_node_list = CustomizeGemm('/Flatten_output_0', 'learned_10', 'learned_11', '/classifier/classifier.1/Gemm_output_0', 144, 64, 'CG1')
insert_node_list2 = CustomizeGemm('/classifier/classifier.2/Relu_output_0', 'learned_12', 'learned_13', '/classifier/classifier.4/Gemm_output_0', 64, 64, 'CG2')
insert_node_list3 = CustomizeGemm('/classifier/classifier.5/Relu_output_0', 'learned_14', 'learned_15', 'output1', 64, 50, 'CG3')
# node insertion
onnx_model = onnx.load('./alexnet.onnx')
idx = 15
idx2 = 17
idx3 = 19
graph = onnx_model.graph
original_nodes = onnx_model.graph.node
new_nodes = original_nodes[:idx] + insert_node_list + original_nodes[idx+1:idx2] + insert_node_list2 + original_nodes[idx2+1:idx3] + insert_node_list3
onnx_model.graph.ClearField("node")
onnx_model.graph.node.extend(new_nodes)

onnx_model = make_model(graph)
check_model(onnx_model)

with open("hw2-3-3.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())# 使用 onnx 進行 inference --> 2-4 要用的
