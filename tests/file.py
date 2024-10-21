import requests#import requests: 导入 requests 库，用于发起 HTTP 请求。
from openai import OpenAI#from openai import OpenAI: 从 openai 库导入 OpenAI 类，以便通过 API 进行文件操作。

#此文件测试文件相关功能的代码
#这段代码的目的是测试与文件相关的功能，主要是通过 OpenAI API 上传、列出和删除文件。

#初始化客户端:
client = OpenAI(#client = OpenAI(...): 创建一个 OpenAI 客户端实例
    api_key="EMPTY",#api_key="EMPTY": 这里需要填入有效的 API 密钥，当前值为 "EMPTY"，表示未填入任何有效的密钥。
    base_url="http://192.168.0.59:7891/v1/",#base_url="http://192.168.0.59:7891/v1/": 指定了 API 的基础 URL，这里是一个本地或自定义的服务器。
)

#列出文件
print(client.files.list())
#print(client.files.list()): 调用 files.list() 方法列出当前可用的文件，并打印结果。这将返回一个包含所有文件信息的列表，通常会包括文件的 ID、名称、大小等。

#创建文件
uf = client.files.create(#uf = client.files.create(...): 使用 files.create() 方法上传文件。
    file=open("../README.md", "rb"),#file=open("../README.md", "rb"): 以二进制模式打开位于上一级目录的 README.md 文件。确保路径正确，并且文件存在。
    purpose="chat",#purpose="chat": 指定文件的用途为 chat，这个参数可能用于后端处理，以确定文件如何被使用。
)
print(uf)#print(uf): 打印上传文件后的响应，通常会返回包含文件 ID 和其他信息的对象。

#分割文件
print(
    requests.post(#requests.post(...): 发送一个 POST 请求到指定的 URL，目的是将刚上传的文件进行分割。
        url="http://192.168.0.59:7891/v1/files/split",#url="http://192.168.0.59:7891/v1/files/split": 指定请求的 URL，用于文件分割。
        json={"file_id": uf.id},#json={"file_id": uf.id}: 请求体中包含要分割的文件 ID，使用 uf.id 作为参数。
    ).json()
    #print(...).json(): 将服务器的响应转换为 JSON 格式并打印。
)

#删除文件
df = client.files.delete(file_id=uf.id)#df = client.files.delete(file_id=uf.id): 调用 files.delete() 方法删除之前上传的文件。
#file_id=uf.id: 指定要删除的文件 ID，使用刚上传文件的 ID。
print(df)#print(df): 打印删除文件后的响应，通常会确认删除操作的结果。
