import asyncio
import websockets
import json


# 等待并检查确认数据包
async def wait_ack(websocket: websockets.ServerConnection, expected_type):

    print(f"Waiting for ack: {expected_type}")

    message = await websocket.recv()
    message = json.loads(message)

    if message["originalType"] != expected_type:
        raise RuntimeError(
            f"Expected message type {expected_type}, got {message['type']}. Terminating connection."
        )


# 发送回应起始数据包
async def send_response_start(websocket: websockets.ServerConnection):
    print("Sending responseStart")

    response = {
        "type": "responseStart",
        "displayName": "Python Test",
        "avatarType": "gpu",
    }

    await websocket.send(json.dumps(response))

    await wait_ack(websocket, "responseStart")


# 发送回应文本数据包
async def send_response_text(websocket: websockets.ServerConnection, text):

    print(f"Sending responseText: {text}")

    response = {"type": "response", "content": text}
    await websocket.send(json.dumps(response))

    await wait_ack(websocket, "response")


# 发送回应结束数据包
async def send_response_end(websocket: websockets.ServerConnection, latency):

    print("Sending responseEnd")

    response = {"type": "responseEnd", "latency": str(latency)}
    await websocket.send(json.dumps(response))

    await wait_ack(websocket, "responseEnd")


# 响应用户输入
async def handle_message(websocket: websockets.ServerConnection, message: dict):
    print(f"Received message: {message['content']}")

    await send_response_start(websocket)  # 发送回应起始数据包
    await send_response_text(websocket, "测试回应：")  # 发送回应文本

    await asyncio.sleep(1)
    await send_response_text(websocket, message["content"])

    for _ in range(5):
        await asyncio.sleep(1)
        await send_response_text(websocket, "不是哥们")

    await send_response_end(websocket, "15")  # 发送回应结束数据包


# 在这里处理对一个网页客户端的连接
async def handle_connection(websocket: websockets.ServerConnection):
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


# 主函数，启动WebSocket服务器
async def main():
    async with websockets.serve(handle_connection, "localhost", 10088):
        print("Server started on ws://localhost:10088")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    #asyncio.run(main())
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
