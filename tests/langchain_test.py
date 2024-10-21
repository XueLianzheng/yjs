from langchain.schema import HumanMessage#HumanMessage: 用于创建人类用户的消息实例。
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
#ChatOpenAI: 用于与 OpenAI 的聊天模型交互。
#OpenAIEmbeddings: 用于生成文本嵌入的类。

#测试Langchain相关功能的代码
#这段代码展示了如何使用 Langchain 和 OpenAI 的接口进行文本处理，包括发送消息和生成文本嵌入。

#创建消息
text = "你好"#
messages = [HumanMessage(content=text)]#创建一个包含用户消息的列表，HumanMessage 的内容为 text。

#初始化聊天模型
llm = ChatOpenAI(openai_api_key="xxx", openai_api_base="http://192.168.20.44:7861/v1")
#创建一个 ChatOpenAI 实例，准备与 OpenAI 的聊天模型进行交互。
#openai_api_key="xxx": 此处需要替换为有效的 OpenAI API 密钥。
# openai_api_base="http://192.168.20.44:7861/v1": 指定 API 的基础 URL，这里是一个本地或自定义的服务器地址。

#发送消息并获取响应
print(llm.invoke(messages))
#使用 invoke 方法发送消息列表 messages 并打印模型的响应。该方法会将 messages 传递给聊天模型，并返回模型生成的回复。

#生成文本嵌入
embedding = OpenAIEmbeddings(openai_api_key="xxx", openai_api_base="http://192.168.20.44:7861/v1")
#embedding = OpenAIEmbeddings(...): 创建一个 OpenAIEmbeddings 实例，用于生成文本嵌入。
#openai_api_key="xxx": 同样需要替换为有效的 API 密钥。
#openai_api_base="http://192.168.20.44:7861/v1": 指定 API 的基础 URL。
print(embedding.embed_documents(["你好"]))
#调用 embed_documents 方法生成输入文档（这里是一个包含“你好”的列表）的嵌入，并打印结果。
