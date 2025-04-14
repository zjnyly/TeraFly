import numpy as np



LAYER_NUM = 24
CU = 2
PROCESSOR = 8
INP_NUM = 64
EMBEDDING_SIZE = 2048
PROJECT = "./OPT-1.3b/"
WEIGHT = "../opt-1.3b/"
FILE_PATH = PROJECT + "const_data/"


# lm_weight = np.fromfile("tokenizer/model_weights/lm_head.weight.bin", dtype=np.float32, count=50272 * EMBEDDING_SIZE).reshape([CU, PROCESSOR, -1])


# input_data = np.load(WEIGHT + 'input.npy')
# def gen_float_data_bin(fileName, input_data):
#     binary_data = input_data.flatten().astype(np.float32).tobytes()
#     with open(fileName, "wb") as file:
#         file.write(binary_data)
# gen_float_data_bin(FILE_PATH + "inp_addr.bin", input_data)

# 加载保存的 .npz 文件
data = np.load(WEIGHT + 'opt-1.3b.npz')

# 加载各个权重和偏置
q_weights = data['q_weight']
k_weights = data['k_weight']
v_weights = data['v_weight']
qkv_weights = np.concatenate([q_weights, k_weights, v_weights], axis=1)
out_weights = data['out_weight']
fc1_weights = data['fc1_weight']
fc2_weights = data['fc2_weight']
weights = []

def cyclic_slice(weight_matrix):
    rows, cols = weight_matrix.shape
    rows = rows // CU
    sliced_weight = np.zeros([CU, PROCESSOR, rows // PROCESSOR, cols])
    for cu in range(CU):
        for i in range(PROCESSOR):
            for j in range(rows // PROCESSOR):
                sliced_weight[cu, i, j] = weight_matrix[cu * rows + i + j * PROCESSOR]
    return sliced_weight.reshape(CU, PROCESSOR, -1)

def pack_weight_for_each_layer(qkv_weights, out_weights, fc1_weights, fc2_weights):
    qkv_weights = cyclic_slice(qkv_weights)
    out_weights = cyclic_slice(out_weights)
    fc1_weights = cyclic_slice(fc1_weights)
    fc2_weights = cyclic_slice(fc2_weights)
    packed_weights = np.concatenate([qkv_weights, out_weights, fc1_weights, fc2_weights], axis=2)
    return packed_weights

layer_weights = np.zeros([LAYER_NUM, CU, PROCESSOR, EMBEDDING_SIZE * EMBEDDING_SIZE * 12 // CU // PROCESSOR])
for layer in range(LAYER_NUM):
    packed_weights = pack_weight_for_each_layer(qkv_weights[layer], out_weights[layer], fc1_weights[layer], fc2_weights[layer])
    layer_weights[layer] = packed_weights

def gen_int_data_pack_bin(fileName, input_data):
    int_8_array = input_data.flatten().astype(np.int8)
    binary_data = int_8_array.tobytes()
    with open(fileName, "wb") as file:
        file.write(binary_data)

def gen_int8_half_data_pack_bin(fileName, int_data, half_data):
    int_8_array = int_data.flatten().astype(np.int8)
    # half_array = half_data.flatten().astype(np.float16)
    half_array = np.round(half_data.flatten() * (2**6)).astype(np.uint16)
    binary_data_int = int_8_array.tobytes()
    binary_data_half = half_array.tobytes()
    # print(half_data)
    # exit()

    binary_data = binary_data_int + binary_data_half
    # print(len(binary_data_int))
    # print(len(binary_data_half))
    print(len(binary_data))
    with open(fileName, "wb") as file:
        file.write(binary_data)
        
for cu in range(CU):
    for processor in range(PROCESSOR):
        processor_data = layer_weights[:, cu, processor, :].flatten()
        print(FILE_PATH + f"w{cu}_addr_{processor}.bin")
        packed_weights = np.concatenate([processor_data], axis=-1)
        gen_int_data_pack_bin(FILE_PATH + f"w{cu}_addr_{processor}.bin", packed_weights)
# bias [layers, packed_bias]

qkv_bias = np.concatenate([data['q_bias'], data['k_bias'], data['v_bias']], axis=-1)
qkv_bias = qkv_bias.reshape([LAYER_NUM, CU, -1])
out_bias = data['out_bias'].reshape([LAYER_NUM, CU, -1])
fc1_bias = data['fc1_bias'].reshape([LAYER_NUM, CU, -1])
fc2_bias = data['fc2_bias'].reshape([LAYER_NUM, CU, -1])
bias = np.concatenate([qkv_bias, out_bias, fc1_bias, fc2_bias], axis=-1)

attn_ln_weight = data['attn_ln_weight'].reshape([LAYER_NUM, CU, -1])
attn_ln_bias = data['attn_ln_bias'].reshape([LAYER_NUM, CU, -1])
fc_ln_weight = data['fc_ln_weight'].reshape([LAYER_NUM, CU, -1])
fc_ln_bias = data['fc_ln_bias'].reshape([LAYER_NUM, CU, -1])
ln_weight = np.concatenate([attn_ln_weight, fc_ln_weight], axis=-1)
ln_bias = np.concatenate([attn_ln_bias, fc_ln_bias], axis=-1)

    
packed_bias_ln = np.concatenate([ln_weight, ln_bias, bias], axis=-1)



def gen_float_data_bin(fileName, input_data):
    binary_data = input_data.flatten().astype(np.float32).tobytes()
    with open(fileName, "wb") as file:
        file.write(binary_data)    
        

# print(packed_bias_ln.shape)
# exit()

def gen_scaling_factor_file(fileName, var_name, data):
    with open(fileName, "w") as file:
        num_elements = len(data)
        file.write(f"const float {var_name}[{num_elements}] = {{\n")
        for i, value in enumerate(data):
            if  i < len(data) - 1:
                file.write(f"    {value},\n")
            else:
                file.write(f"    {value}\n")
        file.write("};\n")

# for cu in range(CU):
#     print(FILE_PATH + f"bias_{cu}.txt")
#     gen_scaling_factor_file(FILE_PATH + f"bias_{cu}.txt", "bias", bias[:, cu, :].flatten())


# alpha [layers, packed_alpha]
sample_granularity = CU
q_a = data['q_a'].reshape(-1, 1)
k_a = data['k_a'].reshape(-1, 1)
v_a = data['v_a'].reshape(-1, 1)
qkv_alpha = np.concatenate([q_a, k_a, v_a], axis = -1).repeat(sample_granularity, axis=1).reshape([LAYER_NUM, CU, -1])
out_a = data['out_a'].reshape(-1, 1).repeat(sample_granularity, axis=-1).reshape([LAYER_NUM, CU, -1])
fc1_a = data['fc1_a'].reshape(-1, 1).repeat(sample_granularity, axis=-1).repeat(4, axis=-1).reshape([LAYER_NUM, CU, -1])
fc2_a = data['fc2_a'].reshape(-1, 1).repeat(sample_granularity, axis=-1).reshape([LAYER_NUM, CU, -1])
qk_a = data['qk_a'].reshape(-1, 1)
pv_a = data['pv_a'].reshape(-1, 1)

linear_alpha = np.concatenate([qkv_alpha, out_a, fc1_a, fc2_a], axis = -1)
# for cu in range(CU):
#     print(FILE_PATH + f"linear_alpha_{cu}.txt")
#     gen_scaling_factor_file(FILE_PATH + f"linear_alpha_{cu}.txt", "linear_alpha", linear_alpha[:, cu, :].flatten())

for cu in range(CU):
    cu_data = packed_bias_ln[:, cu, :]
    cu_data = np.concatenate([cu_data.flatten(), linear_alpha[:, cu, :].flatten()], axis=-1).flatten()
    gen_float_data_bin(FILE_PATH + f"const_data_{cu}.bin", cu_data)  


attn_alpha = np.concatenate([qk_a], axis = -1)
ctx_alpha = np.concatenate([pv_a], axis = -1)
gen_scaling_factor_file(FILE_PATH + "attn_alpha.txt", "attn_alpha", attn_alpha.flatten())
gen_scaling_factor_file(FILE_PATH + "ctx_alpha.txt", "ctx_alpha", ctx_alpha.flatten())


# attn_ln_weight = data['attn_ln_weight'].reshape([LAYER_NUM, CU, -1])
# attn_ln_bias = data['attn_ln_bias'].reshape([LAYER_NUM, CU, -1])
# fc_ln_weight = data['fc_ln_weight'].reshape([LAYER_NUM, CU, -1])
# fc_ln_bias = data['fc_ln_bias'].reshape([LAYER_NUM, CU, -1])
# ln_weight = np.concatenate([attn_ln_weight, fc_ln_weight], axis=-1)
# ln_bias = np.concatenate([attn_ln_bias, fc_ln_bias], axis=-1)


# for cu in range(CU):
#     cu_weight = np.concatenate([ln_weight[:, cu, :].flatten()], axis=-1)
#     cu_bias = np.concatenate([ln_bias[:, cu, :].flatten()], axis=-1)
#     gen_scaling_factor_file(FILE_PATH + f"ln_weight_{cu}.txt", "ln_weight", cu_weight)
#     gen_scaling_factor_file(FILE_PATH + f"ln_bias_{cu}.txt", "ln_bias", cu_bias)
    