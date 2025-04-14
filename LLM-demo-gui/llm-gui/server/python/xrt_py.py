import time
import os
import sys
import uuid
import re
import json
import numpy as np
from transformers import GPT2Tokenizer
from multiprocessing import Process, Queue


tokenizer = GPT2Tokenizer.from_pretrained("tokenizer-1.3b")

# Add XRT Python bindings path to PYTHONPATH
xrt_python_path = "/opt/xilinx/xrt/python"
if xrt_python_path not in sys.path:
    print("append")
    sys.path.append(xrt_python_path)
import pyxrt 
from utils_binding import *

BASE = 'OPT-1.3b-x2-512bit/OPT-1.3b_optimize/'
BIN = BASE + 'build_dir.hw.xilinx_u50lv_gen3x4_xdma_2_202010_1/loopLynx.xclbin'
DATA = BASE + 'const_data/'
WEIGHT = BASE + 'tokenizer/model_weights/'

CU = 2
FULL_INP_LEN = 2048
INP_LEN = FULL_INP_LEN // CU
NUM_LAYER = 24
PROCESSOR = 8
INP_NUM = 64
FULL_SEQ_NUM = 1024
HEAD_NUM = 32
HEAD_LEN = 64
WEIGHT_SIZE = NUM_LAYER * FULL_INP_LEN * FULL_INP_LEN // CU // PROCESSOR * 12 
KV_CACHE_SIZE = (NUM_LAYER * FULL_SEQ_NUM * HEAD_NUM // CU * HEAD_LEN // 2)


GREEN = "\033[1;32m"  # 亮绿色
YELLOW = "\033[1;33m" # 亮黄色
RESET = "\033[0m"     # 重置颜色

BOLD = "\033[1m"      # 加粗
RED = "\033[1;31m"    # 亮红色
    
    
d = pyxrt.device(0)
xbin = pyxrt.xclbin(BIN)
uuid = d.load_xclbin(xbin)
max_power_consumption = 0

def clear_screen():
    # 检查操作系统类型并执行相应的命令
    os.system('cls' if os.name == 'nt' else 'clear')
    # 打印带有颜色和样式的文本
    print(f"{RED}{BOLD}============================={RESET}")
    print(f"{RED}{BOLD}   RUNNING MODEL OPT-1.3B   {RESET}")
    print(f"{RED}{BOLD}============================={RESET}")
    
    
def getinfo():
    global max_power_consumption
    while True:
        time.sleep(0.1)
        data = json.loads(d.get_info(pyxrt.xrt_info_device(0).electrical))
        max_power_consumption = max(max_power_consumption, float(data["power_consumption_watts"]))
        
    

def runKernel():
    global max_power_consumption
    kernel_names = ["loopLynx_0", "loopLynx_1"]
    kernels = []
    for cu in range(CU):
        kernels.append(pyxrt.kernel(d, uuid, kernel_names[cu], pyxrt.kernel.shared))
    inp_addr_bo = []
    w_addr_bo = []
   
    for cu in range(CU):
        inp_addr_bo.append(pyxrt.bo(d, INP_LEN * 13 * 4, pyxrt.bo.normal, kernels[cu].group_id(1)))
        w_addr_bo.append([])
        for processor in range(PROCESSOR):
            w_addr_bo[cu].append(pyxrt.bo(d, (WEIGHT_SIZE + KV_CACHE_SIZE), pyxrt.bo.normal, kernels[cu].group_id(3 + processor)))
         
        
    for cu in range(CU):
        with open(DATA + 'const_data_' + str(cu) + '.bin', "rb") as f:
            binary_data = bytearray(f.read())
            for layer in range(NUM_LAYER):
                layer_data = binary_data[layer * 13 * INP_LEN * 4 : (layer + 1) * 13 * INP_LEN * 4 ]
                inp_addr_bo[cu].write(layer_data, 0)
                inp_addr_bo[cu].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 13 * INP_LEN * 4, 0)
                run = kernels[cu](0, inp_addr_bo[cu], layer, w_addr_bo[cu][0], w_addr_bo[cu][1], w_addr_bo[cu][2], w_addr_bo[cu][3], w_addr_bo[cu][4], w_addr_bo[cu][5], w_addr_bo[cu][6], w_addr_bo[cu][7], 0, 0)
                run.wait()
            layer_data = binary_data[NUM_LAYER * 13 * INP_LEN * 4: ]
            inp_addr_bo[cu].write(layer_data, 0)
            inp_addr_bo[cu].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 13 * INP_LEN * 4, 0)
            run = kernels[cu](0, inp_addr_bo[cu], NUM_LAYER, w_addr_bo[cu][0], w_addr_bo[cu][1], w_addr_bo[cu][2], w_addr_bo[cu][3], w_addr_bo[cu][4], w_addr_bo[cu][5], w_addr_bo[cu][6], w_addr_bo[cu][7], 0, 0)
            run.wait()
    for cu in range(CU):
        for processor in range(PROCESSOR):
            with open(DATA + 'w' + str(cu) + '_addr_' + str(processor) + '.bin', "rb") as f:
                binary_data = bytearray(f.read())
                w_addr_bo[cu][processor].write(binary_data, 0)
                w_addr_bo[cu][processor].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, WEIGHT_SIZE, 0)
    
    lm_head = np.fromfile(WEIGHT + 'embed_tokens.weight.bin', dtype=np.float32).reshape(-1, FULL_INP_LEN)
    pos_embedding_weight = np.fromfile(WEIGHT + 'pos_embeds.bin', dtype=np.float32).reshape(-1, FULL_INP_LEN)
    layer_norm_weight = np.fromfile(WEIGHT + 'final_layer_norm.weight.bin', dtype=np.float32)
    layer_norm_bias = np.fromfile(WEIGHT + 'final_layer_norm.bias.bin', dtype=np.float32)
    
    
  

    
    
    # info_process = Process(target=getinfo)
    # info_process.start()            
    while True:
        clear_screen()
        print("---------------------------------------------------------")
        print("User input: ", end="")
        input_text = input()
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        
        print()
        print(f"{GREEN}{input_text}{RESET}", end=" ")
        
        # 设置温度
        temperature = 0.9
        last_generated_ids = list(input_ids[0]) 
        
        last_id = 0
        total_time = 0
        count = 0
        for token in range(512):    
            if token < input_ids.shape[-1]:
                input_id = input_ids[0, token]
            else:
                input_id = last_id

            embedding = lm_head[input_id] + pos_embedding_weight[token]
            
            for cu in range(CU):
                embedding_bytearray = bytearray(embedding.tobytes())
                inp_addr_bo[cu].write(embedding_bytearray, 0)
                inp_addr_bo[cu].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, FULL_INP_LEN * 4, 0)
            
            start_time = time.time() * 1000
            for cu in range(CU):
                run = kernels[cu](token, inp_addr_bo[cu], -1, w_addr_bo[cu][0], w_addr_bo[cu][1], w_addr_bo[cu][2], w_addr_bo[cu][3], w_addr_bo[cu][4], w_addr_bo[cu][5], w_addr_bo[cu][6], w_addr_bo[cu][7], 0, 0)
            for cu in range(CU):
                run.wait()
            end_time = time.time() * 1000  # 记录结束时间（毫秒）
            total_time += end_time - start_time  # 计算延迟时间
                
            inp_addr_bo[0].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, FULL_INP_LEN * 4, 0)
            out_hidden_state = np.frombuffer(inp_addr_bo[0].map(), dtype=np.float32)[:2048]
            mean = np.mean(out_hidden_state, axis=-1, keepdims=True)
            std = np.std(out_hidden_state, axis=-1, keepdims=True)
            normalized = (out_hidden_state - mean) / (std + 1e-5) 
            out_layer_norm = layer_norm_weight * normalized + layer_norm_bias
            output = np.matmul(out_layer_norm, lm_head.T)
            # maxIndex = np.argmax(output)
            
            # 对模型输出进行温度缩放
            output_scaled = output / temperature

            # 如果使用重复惩罚，可以在此应用
            repetition_penalty = 1.1
            penalized_output = output_scaled.copy()
            for i in range(len(last_generated_ids)):
                penalized_output[last_generated_ids[i]] /= repetition_penalty

            # 通过 top-k 或 top-p 采样增加多样性
            def top_k_sampling(logits, k=50):
                top_k_values, top_k_indices = np.partition(logits, -k)[-k:], np.argpartition(logits, -k)[-k:]
                probabilities = np.exp(top_k_values) / np.sum(np.exp(top_k_values))
                return np.random.choice(top_k_indices, p=probabilities)

            # 使用 top-k 或 top-p 采样
            maxIndex = top_k_sampling(penalized_output, k=5)
            
            
            last_generated_ids.append(maxIndex)

            # 更新 last_id
            last_id = maxIndex
            
            if maxIndex == 2:
                break
            
            if token >= input_ids.shape[-1] - 1:
                decoded_text = tokenizer.decode(maxIndex, skip_special_tokens=True)
                if maxIndex == 17 or maxIndex == 44 or maxIndex == 48:
                    continue
                if maxIndex == 27:
                    print(f"{YELLOW}'{RESET}", end="", flush=True)
                    continue
                print(f"{YELLOW}{decoded_text}{RESET}", end="", flush=True)
            count += 1
        print()
        print("---------------------------------------------------------")
        print(f"Token {token} latency: {total_time/count:.2f} ms (with python wrapper)", flush=True)  # 打印延迟时间
        
        print()
        print()
        input("Press Enter to continue...")
        
    

def main(args):
    # opt = Options()
    # Options.getOptions(opt, args)
    runKernel()
 

if __name__ == "__main__":
    os.environ["Runtime.xrt_bo"] = "false"
    result = main(sys.argv)
