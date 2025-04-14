import asyncio
import websockets
import json
import time
import os
import sys
import uuid
import re
import numpy as np
from transformers import GPT2Tokenizer
import pyxrt
from utils_binding import *

# WebSocket配置
WS_HOST = "localhost"
WS_PORT = 10088

# ------------------ 硬件初始化部分 ------------------
tokenizer = GPT2Tokenizer.from_pretrained("tokenizer-1.3b")

# XRT配置
xrt_python_path = "/opt/xilinx/xrt/python"
if xrt_python_path not in sys.path:
    sys.path.append(xrt_python_path)

# 硬件参数
BASE = '/home/zjnyly/codegen-uram/OPT-1.3b-x2-512bit/OPT-1.3b_optimize/'
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

# 初始化设备
d = pyxrt.device(0)
xbin = pyxrt.xclbin(BIN)
uuid = d.load_xclbin(xbin)
max_power_consumption = 0

# ------------------ WebSocket处理部分 ------------------
async def wait_ack(websocket, expected_type):
    message = await websocket.recv()
    message = json.loads(message)
    if message["originalType"] != expected_type:
        raise RuntimeError(f"Expected {expected_type}, got {message['type']}")

async def send_response(websocket, response_type, content=None):
    response = {"type": response_type}
    if content:
        response["content"] = content
    await websocket.send(json.dumps(response))
    await wait_ack(websocket, response_type)

async def generate_response(text, websocket):
    # 加载模型组件
    lm_head = np.fromfile(WEIGHT + 'embed_tokens.weight.bin', dtype=np.float32).reshape(-1, FULL_INP_LEN)
    pos_embedding_weight = np.fromfile(WEIGHT + 'pos_embeds.bin', dtype=np.float32).reshape(-1, FULL_INP_LEN)
    layer_norm_weight = np.fromfile(WEIGHT + 'final_layer_norm.weight.bin', dtype=np.float32)
    layer_norm_bias = np.fromfile(WEIGHT + 'final_layer_norm.bias.bin', dtype=np.float32)

    # 初始化硬件内核
    kernel_names = ["loopLynx_0", "loopLynx_1"]
    kernels = [pyxrt.kernel(d, uuid, name, pyxrt.kernel.shared) for name in kernel_names]
    
    # 准备缓冲区对象
    inp_addr_bo = [pyxrt.bo(d, INP_LEN * 13 * 4, pyxrt.bo.normal, kernel.group_id(1)) 
                  for kernel in kernels]
    
    # 模型推理逻辑
    input_ids = tokenizer(text, return_tensors="pt").input_ids
    last_generated_ids = list(input_ids[0])
    total_time = 0
    count = 0
    last_id = 0

    for token in range(512):
        input_id = input_ids[0, token] if token < input_ids.shape[-1] else last_id
        embedding = lm_head[input_id] + pos_embedding_weight[token]

        # 写入硬件缓冲区
        for cu in range(CU):
            embedding_bytearray = bytearray(embedding.tobytes())
            inp_addr_bo[cu].write(embedding_bytearray, 0)
            inp_addr_bo[cu].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, FULL_INP_LEN * 4, 0)

        # 执行硬件推理
        start_time = time.time() * 1000
        runs = [kernel(token, inp_addr_bo[i], -1, *[None]*8, 0, 0) for i, kernel in enumerate(kernels)]
        [run.wait() for run in runs]
        latency = time.time() * 1000 - start_time
        total_time += latency
        count += 1

        # 处理推理结果
        inp_addr_bo[0].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, FULL_INP_LEN * 4, 0)
        out_hidden_state = np.frombuffer(inp_addr_bo[0].map(), dtype=np.float32)[:2048]
        
        # 生成文本
        mean = np.mean(out_hidden_state, axis=-1, keepdims=True)
        std = np.std(out_hidden_state, axis=-1, keepdims=True)
        normalized = (out_hidden_state - mean) / (std + 1e-5)
        out_layer_norm = layer_norm_weight * normalized + layer_norm_bias
        output = np.matmul(out_layer_norm, lm_head.T)
        
        # 使用温度采样
        temperature = 0.9
        output_scaled = output / temperature
        maxIndex = np.argmax(output_scaled)
        
        # 生成文本并发送
        if token >= input_ids.shape[-1] - 1 and maxIndex not in [2, 17, 44, 48, 27]:
            decoded_text = tokenizer.decode(maxIndex, skip_special_tokens=True)
            await send_response(websocket, "response", decoded_text)

        if maxIndex == 2:
            break

        last_id = maxIndex

    return total_time / count if count > 0 else 0

async def handle_message(websocket, message):
    try:
        await send_response(websocket, "responseStart", displayName="Python Test")
        await send_response(websocket, "response", "测试回应：")
        
        # 生成模型响应
        latency = await generate_response(message['content'], websocket)
        
        # 发送最终延迟
        await send_response(websocket, "responseEnd", latency=str(latency))
        
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()

async def connection_handler(websocket, path):
    print("Client connected")
    try:
        while True:
            message = await websocket.recv()
            await handle_message(websocket, json.loads(message))
    except websockets.ConnectionClosed:
        print("Client disconnected")

async def main():
    server = await websockets.serve(connection_handler, WS_HOST, WS_PORT)
    print(f"Server started on ws://{WS_HOST}:{WS_PORT}")
    await server.wait_closed()

if __name__ == "__main__":
    os.environ["Runtime.xrt_bo"] = "false"
    asyncio.run(main())