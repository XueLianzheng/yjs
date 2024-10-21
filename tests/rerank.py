import cohere#导入 Cohere 客户端库，以便进行 API 调用。

#测试重排序功能的代码
#这段代码使用了 Cohere 的 API 来实现一个重排序（reranking）功能。

#初始化Cohere客户端
client = cohere.Client(api_key="none", base_url="http://192.168.20.44:7861/v1")
#client = cohere.Client(...): 创建一个 Cohere 客户端实例。
# api_key="none": 这里的 API 密钥设置为“none”，应替换为有效的 Cohere API 密钥，以便进行授权访问。
# base_url="http://192.168.20.44:7861/v1": 指定 API 的基础 URL，可能是一个本地或自定义的服务器地址。

#定义查询:
query = "人工智能"#query = "人工智能": 定义一个查询字符串，表示用户希望获得与“人工智能”相关的文档。

#定义文档集合
corpus = [
    "人工智能",
    "AI",
    "我喜欢看电影",
    "如何学习自然语言处理？",
    "what's Artificial Intelligence?",
]#corpus: 创建一个文档列表，包含多个与“人工智能”相关的字符串。这些文档将用于重排序，表示候选的返回结果。

#调用重排序功能:
results = client.rerank(model="bce", query=query, documents=corpus, return_documents=True)
#results = client.rerank(...): 调用 rerank 方法进行重排序。
#model="bce": 指定使用的模型，这里是 bce（可能是指某个特定的 Cohere 模型）。
# query=query: 将之前定义的查询字符串作为参数传入。
# documents=corpus: 将候选文档列表传入。
# return_documents=True: 指示返回排序后的文档。
print(results.json(indent=4, ensure_ascii=False))
#print(results.json(indent=4, ensure_ascii=False)): 将结果格式化为 JSON 格式，并打印出来。
# indent=4: 指定缩进级别为 4，使输出易于阅读。
# ensure_ascii=False: 允许输出中文字符，而不是以 Unicode 转义序列表示。