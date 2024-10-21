from openai import OpenAI

#此文件测试文本补全功能的代码，这段代码主要测试了使用 OpenAI 客户端的文本补全功能。

#初始化 OpenAI 客户端
client = OpenAI(#从 openai 库导入 OpenAI 类，创建一个 OpenAI 客户端实例。
    api_key="EMPTY",#api_key 被设置为 "EMPTY"，这通常表示在一个本地或无密钥的测试环境中。
    base_url="http://192.168.20.44:7861/v1/",#base_url 指向一个私有的 API 服务器，假设该服务器在本地网络中运行。
)

#普通的文本补全请求
# Chat completion API
completion = client.completions.create(#使用 client.completions.create() 方法发送文本补全请求。
    model="gpt-3.5-turbo",#model 参数指定使用的模型为 "gpt-3.5-turbo"。
    prompt="感冒了怎么办",#prompt 参数传入用户的输入，即“感冒了怎么办”。
)
print(completion)#结果存储在 completion 变量中，并打印出来。此时输出的内容将包含模型对该提示的响应。

#流式文本补全请求
stream = client.completions.create(#使用 client.completions.create() 方法进行文本补全请求，但这次设置了 stream=True。这意味着响应将以流的形式返回，可以逐步接收输出而不是一次性接收。
    model="gpt-3.5-turbo",
    prompt="感冒了怎么办",
    stream=True,#在流式文本补全请求中，设置 stream=True 的作用是允许逐步接收模型的输出，而不是等待整个响应完成后一次性返回。
)
for part in stream:#使用一个 for 循环遍历 stream 对象中的每个部分。
    print(part.choices[0].text or "", end="", flush=True)
#每次循环中，part.choices[0].text 获取模型的输出文本。使用 or "" 处理可能的空值，以确保不会引发错误。
#end="" 确保输出在同一行，flush=True 则确保输出立即显示。