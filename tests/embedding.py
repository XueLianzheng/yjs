from openai import OpenAI#从 openai 库导入 OpenAI 类。这是用来访问 OpenAI API 的客户端。

#此文件测试文本嵌入功能的代码
#这段代码的目的是使用 OpenAI 的 API 来计算文本的嵌入（embedding）。

#初始化客户端:
client = OpenAI(#client = OpenAI(...): 创建一个 OpenAI 客户端实例。这里需要传入 API 密钥和基础 URL。
    api_key="EMPTY",#api_key="EMPTY": 这里应该填入有效的 API 密钥。当前值为 "EMPTY"，表示未填入任何有效的密钥。
    base_url="http://192.168.20.159:8000/v1/",#base_url="http://192.168.20.159:8000/v1/": 指定了 API 的基础 URL，通常是 OpenAI API 的地址。
)

#计算文本嵌入
# compute the embedding of the text
embedding = client.embeddings.create(#embedding = client.embeddings.create(...): 使用 embeddings 接口计算文本嵌入
    input="你好",#input="你好": 计算文本 "你好" 的嵌入。
    model="aspire/acge_text_embedding",#model="aspire/acge_text_embedding": 指定使用的模型，这里是一个名为 aspire/acge_text_embedding 的自定义模型。
    dimensions=384,#dimensions=384: 指定生成嵌入的维度为 384。
)
#输出嵌入长度
print(len(embedding.data[0].embedding))
#embedding.data[0].embedding：假设 embedding 是一个包含嵌入数据的对象，data[0] 获取第一个（也是唯一的一个）嵌入对象，然后 .embedding 访问该对象中的实际嵌入向量。
#len(...)：返回嵌入向量的长度，应该是 384，正如在调用嵌入时所指定的维度。