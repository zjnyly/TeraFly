import math
import json
import os

# PROJECT_NAME = "OPT-125m"
PROJECT_NAME = "OPT-1.3b"
CONST_DATA_DIR = "const_data"
PLATFORM = "xilinx_u50lv_gen3x4_xdma_2_202010_1"
FREQ = 200

with open(PROJECT_NAME + ".json", 'r') as file:
    config = json.load(file)

if not os.path.exists(PROJECT_NAME):
    os.makedirs(PROJECT_NAME)
    
################################################################################################################################################################
# 1. generate params.h
################################################################################################################################################################
def lcm(a, b):
    return abs(a * b) // math.gcd(a, b)

GCD_PE_RT = lcm(config['PROCESSOR'], config['ROUTE_NUM'])

calculated_values = {
    "ROUNDS_FROM": GCD_PE_RT // config["PROCESSOR"],
    "ROUNDS_TO": GCD_PE_RT // config["ROUTE_NUM"],
    "ATTENTION_CHANNELS": config["HEAD_PARALLEL"] * config["HEAD_LEN"] // config["INP_NUM"],
    "INV_FULL_INP_LEN": 1 / float(config["FULL_INP_LEN"]),
    "INP_LEN": config["FULL_INP_LEN"] // config["CU"],
    "DECODE_ITERS": config["FULL_INP_LEN"] // (config["INP_NUM"] // 2) * (50272 // config["CU"]) // config["PROCESSOR"],
    "WEIGHT_SIZE": config["FULL_INP_LEN"] ** 2 // (config["PROCESSOR"] * config["CU"] * config["INP_NUM"]),
    "WEIGHT_SIZE_LAYER": config["FULL_INP_LEN"] ** 2 // (config["PROCESSOR"] * config["CU"] * config["INP_NUM"]) * 12,
    "WEIGHT_SIZE_TOTAL": config["FULL_INP_LEN"] ** 2 // (config["PROCESSOR"] * config["CU"] * config["INP_NUM"]) * 12 * config["NUM_LAYER"],
    "SCALING_FACTOR_SIZE_LAYER": 9,
    "SCALING_FACTOR_SIZE_TOTAL": 9 * config["NUM_LAYER"],
    "BIAS_SIZE_LAYER": (config["FULL_INP_LEN"] // config["CU"]) * 9,
    "BIAS_SIZE_TOTAL": (config["FULL_INP_LEN"] // config["CU"]) * 9 * config["NUM_LAYER"],
    "LN_BIAS_HALF": (config["FULL_INP_LEN"] // config["CU"]) // config["INP_PARALLEL"],
    "LN_BIAS_FULL": (config["FULL_INP_LEN"] // config["CU"]) * 2,
    "LN_SIZE_TOTAL": ((config["FULL_INP_LEN"] // config["CU"]) // config["INP_PARALLEL"]) * 2 * config["NUM_LAYER"],
    "LN_SIZE_LAYER": ((config["FULL_INP_LEN"] // config["CU"]) // config["INP_PARALLEL"]) * 2,
    "MEMORY_SIZE": (config["FULL_INP_LEN"] ** 2 // (config["PROCESSOR"] * config["CU"] * config["INP_NUM"])) * config["NUM_LAYER"] * 12,
    "KV_CACHE_SIZE": config["FULL_SEQ_NUM"] // 2 * config["HEAD_NUM"] // config["CU"] * config["HEAD_LEN"] // config["INP_NUM"],
    "QKV_ROWS": config["FULL_INP_LEN"] // config["PROCESSOR"] // config["CU"] * 3,
    "QKV_COLS": config["FULL_INP_LEN"] // config["INP_NUM"],
    "O_ROWS": config["FULL_INP_LEN"] // config["PROCESSOR"] // config["CU"],
    "O_COLS": config["FULL_INP_LEN"] // config["INP_NUM"],
    "MLP1_ROWS": config["FULL_INP_LEN"] // config["PROCESSOR"] // config["CU"] * 4,
    "MLP1_COLS": config["FULL_INP_LEN"] // config["INP_NUM"],
    "MLP2_ROWS": config["FULL_INP_LEN"] // config["PROCESSOR"] // config["CU"],
    "MLP2_COLS": config["FULL_INP_LEN"] * 4 // config["INP_NUM"],
    "WEIGHT_O_BIAS": (config["FULL_INP_LEN"] ** 2 // (config["PROCESSOR"] * config["CU"] * config["INP_NUM"])) * 3,
    "WEIGHT_MLP1_BIAS": (config["FULL_INP_LEN"] ** 2 // (config["PROCESSOR"] * config["CU"] * config["INP_NUM"])) * 4,
    "WEIGHT_MLP2_BIAS": (config["FULL_INP_LEN"] ** 2 // (config["PROCESSOR"] * config["CU"] * config["INP_NUM"])) * 8,
    "SCALE_O_BIAS" : 3,
    "SCALE_MLP1_BIAS" : 4,
    "SCALE_MLP2_BIAS" : 8,
    "BIAS_O" : (config["FULL_INP_LEN"] // config["CU"]) * 3,
    "BIAS_MLP1" : (config["FULL_INP_LEN"] // config["CU"]) * 4,
    "BIAS_MLP2" : (config["FULL_INP_LEN"] // config["CU"]) * 8,
}

kv_local_bias = []
for cu in range(config["CU"]):
    kv_local_bias.append([])
    cu_bias = cu * calculated_values["INP_LEN"]
    for kv in range(2):
        kv_base_bias = config["FULL_INP_LEN"] * (kv + 1)
        for head in range(config["HEAD_PARALLEL"]):
            head_bias = head * config['HEAD_LEN']
            for pe in range(config['HEAD_LEN'] // config['INP_NUM']):
                pe_bias = pe * config['INP_NUM']
                final_bias = cu_bias + kv_base_bias + head_bias + pe_bias
                kv_local_bias[cu].append(final_bias)



# 将所有值写入文件
with open(PROJECT_NAME + "/params.h", "w") as f:
    f.write("#pragma once\n\n")
    f.write("#include <algorithm>\n#include <ap_axi_sdata.h>\n#include <ap_fixed.h>\n")
    f.write("#include <ap_int.h>\n#include <hls_math.h>\n#include <hls_stream.h>\n")
    f.write("#include <math.h>\n#include <stdint.h>\n#include <hls_half.h>\n\n")
    
    for k, v in {**config, **calculated_values}.items():
        if k == 'INV_FULL_INP_LEN':
            f.write(f"const float {k} = {v};\n")
        else:
            f.write(f"const int {k} = {v};\n")

    for cu in range(config["CU"]):
        f.write(f"const int KV_LOCAL_BIAS_{cu}[{len(kv_local_bias[cu])}] = {{")
        f.write(", ".join(map(str, kv_local_bias[cu])))
        f.write("};\n")
        
    if config['INP_NUM'] == 32:
        f.write("typedef ap_uint<32 * PROCESSOR> proc_pack_float;\ntypedef ap_uint<32 * PROCESSOR> proc_pack_int;\ntypedef ap_uint<32 * INP_PARALLEL> parallel_pack_float;\ntypedef ap_uint<8 * INP_PARALLEL> parallel_pack_int8;\ntypedef ap_uint<32 * ROUTE_NUM> router_pack_float;\ntypedef ap_uint<32 * ROUTE_NUM> router_pack_int;\ntypedef ap_uint<32 * INP_NUM> io_pack_float;\ntypedef ap_uint<8 * INP_NUM> io_pack_int8;\ntypedef ap_uint<32 * INP_NUM> io_pack_int32;\ntypedef ap_uint<16 * 32> datapack_32;\ntypedef ap_uint<17 * 16> datapack_16;\ntypedef ap_uint<18 * 8> datapack_8;\ntypedef ap_uint<19 * 4> datapack_4;\ntypedef ap_uint<20 * 2> datapack_2;\ntypedef union {\n\tfloat f;\n\tuint32_t i;\n} converter_t;\n")
    if config['INP_NUM'] == 64:
        f.write("typedef ap_uint<32 * PROCESSOR> proc_pack_float;\ntypedef ap_uint<32 * PROCESSOR> proc_pack_int;\ntypedef ap_uint<32 * INP_PARALLEL> parallel_pack_float;\ntypedef ap_uint<8 * INP_PARALLEL> parallel_pack_int8;\ntypedef ap_uint<32 * ROUTE_NUM> router_pack_float;\ntypedef ap_uint<32 * ROUTE_NUM> router_pack_int;\ntypedef ap_uint<32 * INP_NUM> io_pack_float;\ntypedef ap_uint<8 * INP_NUM> io_pack_int8;\ntypedef ap_uint<32 * INP_NUM> io_pack_int32;\ntypedef ap_uint<16 * 64> datapack_64;\ntypedef ap_uint<17 * 32> datapack_32;\ntypedef ap_uint<18 * 16> datapack_16;\ntypedef ap_uint<19 * 8> datapack_8;\ntypedef ap_uint<20 * 4> datapack_4;\ntypedef ap_uint<21 * 2> datapack_2;\ntypedef union {\n\tfloat f;\n\tuint32_t i;\n} converter_t;\n")
    
    
################################################################################################################################################################
# 2. generate Makefile
################################################################################################################################################################

with open('template/Makefile', 'r') as file:
    template_content = file.read()
local_params = {
    'PLATFORM': PLATFORM,
}
for key, value in local_params.items():
    value_str = str(value)
    template_content = template_content.replace(f'{{{key}}}', value_str)

with open(PROJECT_NAME + '/Makefile', 'w') as file:
    file.write(template_content)
  
with open('template/makefile_us_alveo.mk', 'r') as file:
    template_content = file.read()
      
kernel_base = '''$(TEMP_DIR)/loopLynx_{cu_id}.xo: loopLynx.cpp loopLynx.h params.h
	mkdir -p $(TEMP_DIR)
	v++ -c $(VPP_FLAGS) -t $(TARGET) --platform $(PLATFORM) -k loopLynx_{cu_id} --temp_dir $(TEMP_DIR)  -I'$(<D)' -o'$@' $^\n\n'''
kernel_str = ""
for cu_id in range(config['CU']):
    formatted_str = kernel_base.format(cu_id=cu_id)
    kernel_str += formatted_str

call_base = "$(TEMP_DIR)/loopLynx_{cu_id}.xo "
call_str = ""
for cu_id in range(config['CU']):
    formatted_str = call_base.format(cu_id=cu_id)
    call_str += formatted_str

local_params = {
    'FREQ': FREQ,
    'XO_REGION' : kernel_str,
    'XO_CALL' : call_str,
}

for key, value in local_params.items():
    value_str = str(value)
    template_content = template_content.replace(f'{{{key}}}', value_str)

with open(PROJECT_NAME + '/makefile_us_alveo.mk', 'w') as file:
    file.write(template_content)

################################################################################################################################################################
# 3. generate layerNorm
################################################################################################################################################################

with open('template/layerNorm.cpp', 'r') as file:
    template_content = file.read()

router_str = ''
if config['CU'] != 1:
    # router_str = '\trouter_ln(for_router, stream_previous, stream_next, from_router);\n\twrite_buffer_ln(from_router, output, output_float, device);'
    router_str = '\trouter_ln(for_router, stream_previous, stream_next, from_router);\n\twrite_buffer_ln(from_router, output, device);'
else:
    # router_str = '\twrite_buffer_ln(for_router, output, output_float, device);'   
    router_str = '\twrite_buffer_ln(for_router, output, device);'   
    
ln_str = ""
current_content = template_content
local_params = {
    # 'LN_INNER_BIAS' : (cu * (params['FULL_INP_LEN'] // params['CU'])),
    'WEATHER_USE_ROUTER' : router_str
}

for key, value in local_params.items():
    value_str = str(value)
    current_content = current_content.replace(f'{{{key}}}', value_str)
ln_str += current_content + '\n\n'

################################################################################################################################################################
# 4. generate gemm
################################################################################################################################################################

with open('template/gemm_quant.cpp', 'r') as file:
    template_content = file.read()

mem_port_base = "\tio_pack_int8 *w_addr_{processor_id},"
mem_port_str = ""
for processor_id in range(config['PROCESSOR']):
    formatted_str = mem_port_base.format(processor_id=processor_id)
    mem_port_str += formatted_str + "\n"

weight_loader_base = "\tweight_loader(w_addr_{processor_id}, block_w_loader[{processor_id}], WEIGHT_BIAS, ROWS * COLS);"
weight_loader_str = ""
for processor_id in range(config['PROCESSOR']):
    formatted_str = weight_loader_base.format(processor_id=processor_id)
    weight_loader_str += formatted_str + "\n"

adapter_str = ''
if config['PROCESSOR'] != config['ROUTE_NUM']:
    adapter_str = '\tacc_result_merger(output, packed_output, ROWS);\n\tadapter(packed_output, repacked_output, ROWS);\n'
else:
    adapter_str = '\tacc_result_merger(output, repacked_output, ROWS);\n'

router_str = ''
if config['CU'] != 1:
    router_str = '\trouter(for_router, stream_previous, stream_next, from_router, ROWS);\n\twrite_buffer(from_router, output_buffer_float, device, ROWS);'
else:
    router_str = '\twrite_buffer(for_router, output_buffer_float, device, ROWS);'
    
gemm_str = ""

current_content = template_content
local_params = {
    'MEM_PORT_REGION': mem_port_str,
    'WEIGHT_LOADER_REGION': weight_loader_str,
    'WEATHER_USE_ADAPTER' : adapter_str,
    'WEATHER_USE_ROUTER' : router_str
}

for key, value in local_params.items():
    value_str = str(value)
    current_content = current_content.replace(f'{{{key}}}', value_str)
gemm_str += current_content + '\n\n'

################################################################################################################################################################
# 5. generate attention
################################################################################################################################################################

with open('template/attention.cpp', 'r') as file:
    template_content = file.read()
    
ring_connection_str = ""
if config['CU'] != 1:
    ring_connection_str = '\trouter_attention(ctx_requant_merged, stream_previous, stream_next, router_out);\n\twrite_buffer_int(router_out, output_buffer, device_id);\n'
else:
    ring_connection_str ='\twrite_buffer_int(ctx_requant_merged, output_buffer, device_id);\n'


k_loader = "\tloader_new(w_addr_{processor_id}, k_stream[{stream_id}], layer_id, seq_id);\n"
k_loader_str = ""
for processor_id in range(calculated_values['ATTENTION_CHANNELS']):
    formatted_str = k_loader.format(processor_id=processor_id, stream_id = processor_id)
    k_loader_str += formatted_str
v_loader = "\tloader_new(w_addr_{processor_id}, v_stream[{stream_id}], layer_id, seq_id);\n"
v_loader_str = ""
for processor_id in range(calculated_values['ATTENTION_CHANNELS']):
    formatted_str = v_loader.format(processor_id=processor_id + calculated_values['ATTENTION_CHANNELS'], stream_id = processor_id)
    v_loader_str += formatted_str

loader_str = k_loader_str + v_loader_str

mem_port_base = "\tio_pack_int8 *w_addr_{processor_id},"
mem_port_str = ""
for processor_id in range(calculated_values['ATTENTION_CHANNELS'] * 2):
    formatted_str = mem_port_base.format(processor_id=processor_id)
    mem_port_str += formatted_str + "\n"

acc_result_str = ""
if config['HEAD_LEN'] // config['INP_NUM'] == 4:
    acc_result_str = "int acc_result = sub_result[parallel * 4 + 1] + sub_result[parallel * 4 + 2] + sub_result[parallel * 4 + 3] + sub_result[parallel * 4 + 3];\n";
elif config['HEAD_LEN'] // config['INP_NUM'] == 2:
    acc_result_str = "int acc_result = sub_result[parallel * 2 + 1] + sub_result[parallel * 2];\n";
elif config['HEAD_LEN'] // config['INP_NUM'] == 1:
    acc_result_str = "int acc_result = sub_result[parallel];\n";
    
    
current_content = template_content
local_params = {
    'CONST_DATA_DIR': CONST_DATA_DIR,
    'MEM_PORT_REGION': mem_port_str,
    'WEATHER_USE_ROUTER_ATTENTION': ring_connection_str,
    'KV_LOADER_REAGION' : loader_str,
    'ACC_RESULT_SELECT' : acc_result_str,
    'LOG_FULL_SEQ_LEN' : int(math.log2(config['FULL_SEQ_NUM']))
}

attention_str = ""
for key, value in local_params.items():
    value_str = str(value)
    current_content = current_content.replace(f'{{{key}}}', value_str)
    current_content = current_content + '\n\n'
attention_str = current_content

################################################################################################################################################################
# 6. generate top
################################################################################################################################################################

with open('template/top.cpp', 'r') as file:
    template_content = file.read()

mem_port_base = "\tio_pack_int8 *w_addr_{processor_id},"
mem_port_str = ""
for processor_id in range(config['PROCESSOR']):
    formatted_str = mem_port_base.format(processor_id=processor_id)
    mem_port_str += formatted_str + "\n"

mem_port_call = "w_addr_{processor_id}, "
mem_port_call_str = ""
for processor_id in range(config['PROCESSOR']):
    formatted_str = mem_port_call.format(processor_id=processor_id)
    mem_port_call_str += formatted_str

mem_pragma = "#pragma HLS interface m_axi port = w_addr_{processor_id} offset = slave bundle = gmem{processor_id}\n"
mem_pragma_str = ""
for processor_id in range(config['PROCESSOR']):
    formatted_str = mem_pragma.format(processor_id=processor_id)
    mem_pragma_str += formatted_str

kv_buffer_onchip = "\t\twrite_kv_buffer_onchip(float_buffer, KV_buffer[{processor_id}], KV_LOCAL_BIAS_0[{processor_id}]);\n";
kv_buffer_onchip_str = ""
for processor_id in range(calculated_values['ATTENTION_CHANNELS'] * 2):
    formatted_str = kv_buffer_onchip.format(processor_id=processor_id)
    kv_buffer_onchip_str += formatted_str

kv_buffer_offchip = "\t\tkv_writer_parallel(KV_buffer[{processor_id}], w_addr_{processor_id}, memory_bias, i_seq);\n";
kv_buffer_offchip_str = ""
for processor_id in range(calculated_values['ATTENTION_CHANNELS'] * 2):
    formatted_str = kv_buffer_offchip.format(processor_id=processor_id)
    kv_buffer_offchip_str += formatted_str


mem_port_call_attn_str = ""
for processor_id in range(calculated_values['ATTENTION_CHANNELS'] * 2):
    formatted_str = mem_port_call.format(processor_id=processor_id)
    mem_port_call_attn_str += formatted_str
        
if config['CU'] != 1:
    ring_connection_str = ',\thls::stream<router_pack_float> &stream_previous,\n\thls::stream<router_pack_float> &stream_next'
else:
    ring_connection_str = ''

ring_connection_str = '\thls::stream<router_pack_float> &stream_previous,\n\thls::stream<router_pack_float> &stream_next'

if len(ring_connection_str) == 0:
    mem_port_str = mem_port_str[:-2]



top_str = ""
for cu in range(config['CU']):
    current_content = template_content
    local_params = {
        'CU_ID': cu,
        'MEM_PORT_CALL': mem_port_call_str,
        'MEM_PORT_CALL_ATTN': mem_port_call_attn_str,
        'MEM_PORT_REGION': mem_port_str,
        'MEM_PRAGMA_REGION': mem_pragma_str,
        'RING_CONNECTION' : ring_connection_str,
        'WRITE_KV_ONCHIP' : kv_buffer_onchip_str,
        'WRITE_KV_OFFCHIP' : kv_buffer_offchip_str
    }

    for key, value in local_params.items():
        value_str = str(value)
        current_content = current_content.replace(f'{{{key}}}', value_str)
    top_str += current_content + '\n\n'

with open('template/top.h', 'r') as file:
    template_content = file.read()

top_str_header = ""
for cu in range(config['CU']):
    current_content = template_content
    local_params = {
        'CU_ID': cu,
        'MEM_PORT_CALL': mem_port_call_str,
        'MEM_PORT_REGION': mem_port_str,
        'MEM_PRAGMA_REGION': mem_pragma_str,
        'RING_CONNECTION' : ring_connection_str
    }

    for key, value in local_params.items():
        value_str = str(value)
        current_content = current_content.replace(f'{{{key}}}', value_str)
    top_str_header += current_content

################################################################################################################################################################
# 7. generate loopLynx.cpp
################################################################################################################################################################

with open('template/kernels.cpp', 'r') as file:
    template_content = file.read()

with open('template/router.cpp', 'r') as file:
    router_str = file.read()

if config['INP_NUM'] == 32:
    with open('template/adder_tree_32.cpp', 'r') as file:
        adder_tree_str = file.read()
if config['INP_NUM'] == 64:
    with open('template/adder_tree_64.cpp', 'r') as file:
        adder_tree_str = file.read()
    
# template_content = template_content  +  "\n\n" + ln_str + "\n\n" + fused_str + "\n\n" + requant_str + "\n\n" + gemm_str + "\n\n" + gemm_half_str + "\n\n" + top_str
template_content = template_content  +  "\n\n"  + adder_tree_str + "\n\n"  + router_str + "\n\n"  + attention_str +  "\n\n" + ln_str  + "\n\n" + gemm_str + "\n\n"  + "\n\n" + top_str

adder_tree_str = ''
if config['INP_NUM'] == 32:
    adder_tree_str = 'Mul_Adder_Tree_32'
if config['INP_NUM'] == 64:
    adder_tree_str = 'Mul_Adder_Tree_64'
if config['INP_NUM'] == 128:
    adder_tree_str = 'Mul_Adder_Tree_128'

local_params = {
    'Mul_Adder_Tree': adder_tree_str
}

for key, value in local_params.items():
    value_str = str(value)
    template_content = template_content.replace(f'{{{key}}}', value_str)

with open(PROJECT_NAME + '/loopLynx.cpp', 'w') as file:
    file.write(template_content)
    
template_content =  '#include "params.h"\n' + top_str_header

with open(PROJECT_NAME + '/loopLynx.h', 'w') as file:
    file.write(template_content)
    
    
