import onnx 
import json
onnx_model = onnx.load('./mobilenetv2-10.onnx')
onnx_model = onnx.shape_inference.infer_shapes(onnx_model)

## List all tensor names in the graph
input_nlist = [k.name for k in onnx_model.graph.input]
initializer_nlist = [k.name for k in onnx_model.graph.initializer]
value_info_nlist = [k.name for k in onnx_model.graph.value_info]

def get_size(shape):
    dims = []
    ndim = len(shape.dim)
    size = 1;
    for i in range(ndim):
        size = size * shape.dim[i].dim_value
        dims.append(shape.dim[i].dim_value)
    return dims, size

##### Collect Attribute ####
def OperatorAttr(op_type):
    JSON_list = []
    for i in onnx_model.graph.node:
        if i.op_type == op_type:
            JSON = {}
            JSON[i.name] = {}
            if i.op_type == 'Conv':
                for j in i.input:
                    if j in input_nlist:
                        idx = input_nlist.index(j)
                        (dims, size) = get_size(onnx_model.graph.input[idx].type.tensor_type.shape)
                        c = dims[1]
                        h = dims[2]
                        w = dims[3]
                        
                    elif j in initializer_nlist:
                        idx = initializer_nlist.index(j)
                        dims = onnx_model.graph.initializer[idx].dims
                    elif j in value_info_nlist:
                        idx = value_info_nlist.index(j)
                        (dims, size) = get_size(onnx_model.graph.value_info[idx].type.tensor_type.shape)
                        c = dims[1]
                        h = dims[2]
                        w = dims[3]
                    JSON[i.name]['channel'] = c
                    JSON[i.name]['height'] = h
                    JSON[i.name]['width'] = w
            for attr in i.attribute:
                JSON[i.name][attr.name] = attr.i if len(attr.ints) == 0 else attr.ints
                # str.replace(old, new[, max])
            JSON_list.append(JSON)
    return JSON_list

def Print_JSON(JSON):
    key = list(JSON.keys())[0]
    print(key)
    for ckey in JSON[key]:
        print(f'|- {ckey} : {JSON[key][ckey]}')
    

op_dict = {}
op_operator = set()

# Compute the each op_type
for i in onnx_model.graph.node:
    if i.op_type in op_dict:
        op_dict[i.op_type] += 1
    else:
        op_dict[i.op_type] = 1
    op_operator.add(i.op_type)

print(f'Total operators: {sum(op_dict.values())}')
print(f'Unique Operator: {op_operator}')
print('=====')
for op in op_dict:
    print(f'Operator[{op}]: {op_dict[op]}')
    JSON_list = OperatorAttr(op)
    if len(JSON_list) == 0:
        print('No attribute')
    for item in JSON_list:
        Print_JSON(item)
    print('====')
