import json

import requests

#该文件提供了如何通过Python的requests库与API进行交互的示例。

#基础 URL 定义
base_url = "http://192.168.20.59:9009"#base_url 是API的基础地址，指向本地服务器。

#这段代码定义了一个用于与聊天模型进行交互的函数 create_chat_completion，并在主程序中调用该函数。代码使用了 requests 库向一个指定的 API 发送 HTTP POST 请求

def create_chat_completion(model, messages, stream=False):
    #构建请求数据
    data = {
        "model": model,  # 模型名称
        "messages": messages,  # 会话历史
        "stream": stream,  # 是否流式响应
        "max_tokens": 100,  # 最多生成字数
        "temperature": 0.8,  # 温度
        "top_p": 0.8,  # 采样概率
    }

#发送 POST 请求
    response = requests.post(f"{base_url}/v1/chat/completions", json=data, stream=stream)
    #使用 requests.post 发送 POST 请求，URL 由 base_url 和 /v1/chat/completions 组合而成。
    #json=data 将请求数据以 JSON 格式发送。
    #stream=stream 控制是否以流式方式接收响应。

    #响应处理
    if response.status_code == 200:#检查响应状态码是否为200（表示成功）。
        if stream:
            # 处理流式响应
            for line in response.iter_lines():#使用 response.iter_lines() 逐行读取响应内容。
                if line:
                    decoded_line = line.decode('utf-8')[6:]#每行数据进行解码并去掉前6个字符（通常是用于流式传输的标记）。
                    try:##将解码后的内容转换为 JSON 格式，并提取其中的聊天内容=
                        response_json = json.loads(decoded_line)
                        content = response_json.get("choices", [{}])[0].get("delta", {}).get("content", "")
                        print(content)
                    except:#如果解析失败，打印特殊标记的内容。
                        print("Special Token:", decoded_line)
        else:
            # 处理非流式响应
            decoded_line = response.json()#如果不是流式响应，直接将响应体解析为 JSON。
            print(decoded_line)
            content = decoded_line.get("choices", [{}])[0].get("message", "").get("content", "")
            #打印整个响应内容，并提取消息内容。
            print(content)
    else:#如果响应状态码不是200，打印错误信息。
        print("Error:", response.status_code)
        return None


if __name__ == "__main__":
    chat_messages = [#chat_messages 定义了一条用户消息，要求模型讲一个大约100字的故事。
        {
            "role": "user",
            "content": "你好，给我讲一个故事，大概100字"
        }
    ]
    create_chat_completion("qwen-7b-chat", chat_messages, stream=False)
#调用 create_chat_completion 函数，使用指定的模型 "qwen-7b-chat" 和消息内容，流式响应设为 False。