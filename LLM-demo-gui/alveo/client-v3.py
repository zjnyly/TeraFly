import asyncio
import websockets
import json
from xrt_py_new import *

loopLynx = LoopLynx()

# 等待并检查确认数据包
async def wait_ack(websocket, expected_type):
    print(f"Waiting for ack: {expected_type}")

    message = await websocket.recv()
    message = json.loads(message)

    if message["originalType"] != expected_type:
        raise RuntimeError(
            f"Expected message type {expected_type}, got {message['type']}. Terminating connection."
        )

# 发送回应起始数据包
async def send_response_start(websocket):
    print("Sending responseStart")

    response = {
        "type": "responseStart",
        "displayName": "Python Test",
        "avatarType": "fpga",
    }

    await websocket.send(json.dumps(response))
    await wait_ack(websocket, "responseStart")

# 发送回应文本数据包
async def send_response_text(websocket, text):
    print(f"Sending responseText: {text}")

    response = {"type": "response", "content": text}
    await websocket.send(json.dumps(response))

    await wait_ack(websocket, "response")

# 发送回应结束数据包
async def send_response_end(websocket, latency):
    print("Sending responseEnd")

    response = {"type": "responseEnd", "latency": str(latency)}
    await websocket.send(json.dumps(response))

    await wait_ack(websocket, "responseEnd")

# 处理用户消息
async def handle_message(websocket, message):
    print(f"Received message: {message['content']}")
    loopLynx.process_prompt(message['content'])
    await send_response_start(websocket)  # 发送回应起始数据包
    await send_response_text(websocket, loopLynx.word)  # 发送回应文本
    
    
    while(loopLynx.token < loopLynx.MAX_SEQ_LEN and not loopLynx.finish):
        loopLynx.process_token()
        if loopLynx.valid:
            await send_response_text(websocket, loopLynx.word)  # 发送回应文本


    await send_response_end(websocket, loopLynx.getLatency())  # 发送回应结束数据包

# 处理客户端连接
async def handle_connection(websocket, path):
    print("Client connected")
    try:
        while True:
            message = await websocket.recv()
            message = json.loads(message)
            await handle_message(websocket, message)

    except websockets.ConnectionClosed:
        print("Client disconnected")

    except RuntimeError as e:
        print(f"Runtime Error: {e}")

    except Exception as e:
        print(f"Error: {e}")

# 主函数，启动 WebSocket 服务器
async def main():
    loop = asyncio.get_event_loop()
    server = await websockets.serve(handle_connection, "localhost", 10088, loop=loop)
    print("Server started on ws://localhost:10088")
    await server.wait_closed()

if __name__ == "__main__":
    
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.run_forever()

