from openai import OpenAI

#此文件是测试聊天功能的代码
#这段代码展示了如何使用 OpenAI 客户端通过 API 与模型进行交互，具体实现了列出可用模型和进行聊天完成的功能。

#初始化 OpenAI 客户端
client = OpenAI(#创建一个 OpenAI 客户端实例
    api_key="EMPTY",#api_key 被设置为 "EMPTY"，这表明可能是在一个本地的或者无密钥的测试环境中。
    base_url="http://192.168.20.44:7861/v1/",#base_url 指向本地或私有的 API 服务。
)

#列出模型
# List models API
#调用 client.models.list() 获取可用的模型列表，并将其打印出来。
models = client.models.list()#model_dump() 方法通常用于序列化对象，以便以可读的格式输出模型信息。
print(models.model_dump())

# 聊天完成 API
# Chat completion API
chat_completion = client.chat.completions.create(#使用 client.chat.completions.create() 方法进行一次聊天完成请求。
    messages=[#messages 参数中包含一个字典，表示用户的输入信息，角色为 "user"，内容为 "感冒了怎么办"（“我感冒了，该怎么办？”）。
        {
            "role": "user",
            "content": "感冒了怎么办",
        }
    ],
    model="gpt-3.5-turbo",#指定使用的模型为 "gpt-3.5-turbo"。
)
print(chat_completion)#打印出聊天完成的结果

#流式聊天完成 API
stream = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "感冒了怎么办",
        }
    ],
    model="gpt-3.5-turbo",
    stream=True,#通过设置 stream=True，调用聊天完成 API 来获取流式响应。这种方式可以逐步接收模型的输出，而不是等待整个响应完成。
)
for part in stream:#使用一个循环来遍历 stream 对象中的每一个部分。
    print(part.choices[0].delta.content or "", end="", flush=True)#在每个循环中，打印出模型的输出内容。
#这里 part.choices[0].delta.content 获取响应内容，如果内容为空则打印一个空字符串。end="" 参数确保输出在同一行，flush=True 则确保输出立即刷新到控制台。