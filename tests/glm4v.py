from openai import OpenAI

#测试GLM-4V模型的示例
#通过发送一条消息请求该模型对一张图片的内容进行解析

client = OpenAI(
    api_key="EMPTY",
    base_url="http://192.168.0.59:7891/v1",
)
#client = OpenAI(...): 创建一个 OpenAI 客户端实例。
# api_key="EMPTY": 这里需要填入有效的 API 密钥，当前值为 "EMPTY"，表示未填入有效的密钥。
# base_url="http://192.168.0.59:7891/v1": 指定 API 的基础 URL，这是一个本地或自定义的服务器。

#发送消息请求
stream = client.chat.completions.create(#stream = client.chat.completions.create(...): 使用 chat.completions.create() 方法生成一个聊天完成请求。
    messages=[#messages: 这个参数是一个消息列表，包含了发送给模型的消息
        {
            #{"role": "user", "content": [...]}: 这是一个用户角色的消息
            "role": "user",
            "content": [#content: 这里包含一个列表，列表中的内容包括：
                {
                    #第一个内容项：一个文本消息，内容为“这张图片是什么地方？”。
                    "type": "text",
                    "text": "这张图片是什么地方？"
                },
                {
                    #第二个内容项：一个图片消息，使用 image_url 类型来指定。
                    "type": "image_url",#image_url: 该项包含一个 URL，指向要分析的图片。
                    "image_url": {
                        # Either an url or base64
                        "url": "http://djclub.cdn.bcebos.com/uploads/images/pageimg/20230325/64-2303252115313.jpg"
                    }
                }
            ]
        }
    ],
    model="glm-4v-9b",#model="glm-4v-9b": 指定要使用的模型。
    stream=True,#stream=True: 指定请求是流式的，意味着响应将逐步返回，而不是一次性返回。
)

#处理响应
for part in stream:#for part in stream:: 迭代流中的每一部分。
    print(part.choices[0].delta.content or "", end="", flush=True)
#print(part.choices[0].delta.content or "", end="", flush=True): 打印出模型的响应内容。
#part.choices[0].delta.content: 获取当前部分的响应内容。如果该内容为空，则打印一个空字符串。
# end="": 在打印内容后不换行。
# flush=True: 确保输出立即刷新到控制台。
