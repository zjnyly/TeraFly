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

class LoopLynx:
    BASE = '../../OPT-series/OPT-1.3b/No.15-OPT-1.3b-x2-512bit/OPT-1.3b_optimize/'
    BIN = BASE + 'build_dir.hw.xilinx_u50lv_gen3x4_xdma_2_202010_1/loopLynx.xclbin'
    DATA = '../../OPT-series/OPT-1.3b/const_data_x2/'
    WEIGHT = '../../OPT-series/OPT-1.3b/model_weights/'

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
    MAX_SEQ_LEN = 512
    
    kernel_names = ["loopLynx_0", "loopLynx_1"]

    GREEN = "\033[1;32m"  # 亮绿色
    YELLOW = "\033[1;33m" # 亮黄色
    RESET = "\033[0m"     # 重置颜色
    BOLD = "\033[1m"      # 加粗
    RED = "\033[1;31m"    # 亮红色
    
    kernels = []
    inp_addr_bo = []
    w_addr_bo = []
    
    total_time = 0.0
    word = ""
    valid = False

    def __init__(self):
        self.device = pyxrt.device(0)
        self.xbin = pyxrt.xclbin(self.BIN)
        self.uuid = self.device.load_xclbin(self.xbin)
        self.max_power_consumption = 0
        self.initialize()
        


    def clear_screen(self):
        # 检查操作系统类型并执行相应的命令
        os.system('cls' if os.name == 'nt' else 'clear')
        # 打印带有颜色和样式的文本
        print(f"{self.RED}{self.BOLD}============================={self.RESET}")
        print(f"{self.RED}{self.BOLD}   RUNNING MODEL OPT-1.3B   {self.RESET}")
        print(f"{self.RED}{self.BOLD}============================={self.RESET}")
    

    def initialize(self):
        self.d = pyxrt.device(0)
        self.xbin = pyxrt.xclbin(self.BIN)
        self.uuid = self.d.load_xclbin(self.xbin)
        
        for cu in range(self.CU):
            self.kernels.append(pyxrt.kernel(self.d, self.uuid, self.kernel_names[cu], pyxrt.kernel.shared))
        
    
        for cu in range(self.CU):
            self.inp_addr_bo.append(pyxrt.bo(self.d, self.INP_LEN * 13 * 4, pyxrt.bo.normal, self.kernels[cu].group_id(1)))
            self.w_addr_bo.append([])
            for processor in range(self.PROCESSOR):
                self.w_addr_bo[cu].append(pyxrt.bo(self.d, (self.WEIGHT_SIZE + self.KV_CACHE_SIZE), pyxrt.bo.normal, self.kernels[cu].group_id(3 + processor)))
         
        
        for cu in range(self.CU):
            with open(self.DATA + 'const_data_' + str(cu) + '.bin', "rb") as f:
                binary_data = bytearray(f.read())
                for layer in range(self.NUM_LAYER):
                    layer_data = binary_data[layer * 13 * self.INP_LEN * 4 : (layer + 1) * 13 * self.INP_LEN * 4 ]
                    self.inp_addr_bo[cu].write(layer_data, 0)
                    self.inp_addr_bo[cu].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 13 * self.INP_LEN * 4, 0)
                    run = self.kernels[cu](0, self.inp_addr_bo[cu], layer, self.w_addr_bo[cu][0], self.w_addr_bo[cu][1], self.w_addr_bo[cu][2], self.w_addr_bo[cu][3], self.w_addr_bo[cu][4], self.w_addr_bo[cu][5], self.w_addr_bo[cu][6], self.w_addr_bo[cu][7], 0, 0)
                    run.wait()
                layer_data = binary_data[self.NUM_LAYER * 13 * self.INP_LEN * 4: ]
                self.inp_addr_bo[cu].write(layer_data, 0)
                self.inp_addr_bo[cu].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, 13 * self.INP_LEN * 4, 0)
                run = self.kernels[cu](0, self.inp_addr_bo[cu], self.NUM_LAYER, self.w_addr_bo[cu][0], self.w_addr_bo[cu][1], self.w_addr_bo[cu][2], self.w_addr_bo[cu][3], self.w_addr_bo[cu][4], self.w_addr_bo[cu][5], self.w_addr_bo[cu][6], self.w_addr_bo[cu][7], 0, 0)
                run.wait()
        for cu in range(self.CU):
            for processor in range(self.PROCESSOR):
                with open(self.DATA + 'w' + str(cu) + '_addr_' + str(processor) + '.bin', "rb") as f:
                    binary_data = bytearray(f.read())
                    self.w_addr_bo[cu][processor].write(binary_data, 0)
                    self.w_addr_bo[cu][processor].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, self.WEIGHT_SIZE, 0)
    
        self.lm_head = np.fromfile(self.WEIGHT + 'embed_tokens.weight.bin', dtype=np.float32).reshape(-1, self.FULL_INP_LEN)
        self.pos_embedding_weight = np.fromfile(self.WEIGHT + 'pos_embeds.bin', dtype=np.float32).reshape(-1, self.FULL_INP_LEN)
        self.layer_norm_weight = np.fromfile(self.WEIGHT + 'final_layer_norm.weight.bin', dtype=np.float32)
        self.layer_norm_bias = np.fromfile(self.WEIGHT + 'final_layer_norm.bias.bin', dtype=np.float32)
    
    def inference(self):
        embedding = self.lm_head[self.input_id] + self.pos_embedding_weight[self.token]
            
        for cu in range(self.CU):
            embedding_bytearray = bytearray(embedding.tobytes())
            self.inp_addr_bo[cu].write(embedding_bytearray, 0)
            self.inp_addr_bo[cu].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, self.FULL_INP_LEN * 4, 0)
        
        start_time = time.time() * 1000
        for cu in range(self.CU):
            run = self.kernels[cu](self.token, self.inp_addr_bo[cu], -1, self.w_addr_bo[cu][0], self.w_addr_bo[cu][1], self.w_addr_bo[cu][2], self.w_addr_bo[cu][3], self.w_addr_bo[cu][4], self.w_addr_bo[cu][5], self.w_addr_bo[cu][6], self.w_addr_bo[cu][7], 0, 0)
        for cu in range(self.CU):
            run.wait()
        end_time = time.time() * 1000  # 记录结束时间（毫秒）
        self.total_time += end_time - start_time  # 计算延迟时间
            
        self.inp_addr_bo[0].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, self.FULL_INP_LEN * 4, 0)
        out_hidden_state = np.frombuffer(self.inp_addr_bo[0].map(), dtype=np.float32)[:2048]
        mean = np.mean(out_hidden_state, axis=-1, keepdims=True)
        std = np.std(out_hidden_state, axis=-1, keepdims=True)
        normalized = (out_hidden_state - mean) / (std + 1e-5) 
        out_layer_norm = self.layer_norm_weight * normalized + self.layer_norm_bias
        output = np.matmul(out_layer_norm, self.lm_head.T)
        # maxIndex = np.argmax(output)
        
        # 对模型输出进行温度缩放
        output_scaled = output / self.temperature

        # 如果使用重复惩罚，可以在此应用
        repetition_penalty = 1.1
        penalized_output = output_scaled.copy()
        for i in range(len(self.last_generated_ids)):
            penalized_output[self.last_generated_ids[i]] /= repetition_penalty

        # 通过 top-k 或 top-p 采样增加多样性
        def top_k_sampling(logits, k=50):
            top_k_values, top_k_indices = np.partition(logits, -k)[-k:], np.argpartition(logits, -k)[-k:]
            probabilities = np.exp(top_k_values) / np.sum(np.exp(top_k_values))
            return np.random.choice(top_k_indices, p=probabilities)

        # 使用 top-k 或 top-p 采样
        maxIndex = top_k_sampling(penalized_output, k=5)
        
        
        self.last_generated_ids.append(maxIndex)

        # 更新 last_id
        self.last_id = maxIndex
        
        if maxIndex == 2:
            self.finish = True
            return
        
        if self.token >= self.input_ids.shape[-1] - 1:
            decoded_text = tokenizer.decode(maxIndex, skip_special_tokens=True)
            if maxIndex == 17 or maxIndex == 44 or maxIndex == 48:
                self.valid = False
                return
            if maxIndex == 27:
                print(f"{self.YELLOW}'{self.RESET}", end="", flush=True)
                self.word = "'"
                self.valid = True
                return
            print(f"{self.YELLOW}{decoded_text}{self.RESET}", end="", flush=True)
            self.word = str(decoded_text)
            self.valid = True
        self.count += 1
    
    def print_latency(self):
        print()
        print("---------------------------------------------------------")
        print(f"Token {self.token} latency: {self.total_time/self.count:.2f} ms (with python wrapper)", flush=True)  # 打印延迟时间
        print()
        print()
        input("Press Enter to continue...")
    
    def getLatency(self):
        return self.total_time/self.count
        
    def process_prompt(self, input_text):
        
        print("---------------------------------------------------------")
        print("User input: ", end="")
        self.input_ids = tokenizer(input_text, return_tensors="pt").input_ids
        
        print()
        print(f"{self.GREEN}{input_text}{self.RESET}", end=" ")
        
        # 设置温度
        self.temperature = 0.9
        self.last_generated_ids = list(self.input_ids[0]) 
        
        self.last_id = 0
        self.total_time = 0
        self.count = 0
        self.token = 0
        self.finish = False
        self.valid = False
        
        while(self.token < self.input_ids.shape[-1] and self.token < self.MAX_SEQ_LEN):
            self.input_id = self.input_ids[0, self.token]
            self.inference()
            self.token+=1
    
    def process_token(self):
        self.input_id = self.last_id
        self.inference()
        self.token+=1

    def process(self):
        while True:
            self.clear_screen()
            self.process_prompt(input())
            while(self.token < self.MAX_SEQ_LEN and not self.finish):
                self.process_token()
            self.print_latency()
        

        
        
 

if __name__ == "__main__":
    os.environ["Runtime.xrt_bo"] = "false"
    loopLynx = LoopLynx();
    loopLynx.process()
