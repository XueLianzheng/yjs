from typing import Optional

from fastapi import Depends, HTTPException
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from openai import AsyncOpenAI
from sse_starlette import EventSourceResponse

from api.utils.protocol import ChatCompletionCreateParams, CompletionCreateParams, EmbeddingCreateParams

#此文件展示如何使用项目API与OpenAI接口交互的示例

app = FastAPI()#创建一个 FastAPI 应用实例 app，用于定义路由和处理请求。

#添加 CORS 中间件
app.add_middleware(#
    CORSMiddleware,#通过 add_middleware 方法添加 CORSMiddleware，允许跨域请求。
    allow_origins=["*"],#允许来自所有源的请求。
    allow_credentials=True,#允许带有凭据（如 cookies）的请求。
    allow_methods=["*"],#允许所有 HTTP 方法（GET、POST 等）
    allow_headers=["*"],#允许所有请求头。
)

#API 密钥和模型配置
API_KEYS = None  # 此处设置允许访问接口的api_key列表
#API_KEYS 用于存储允许访问 API 的密钥列表，当前设为 None，需要根据需求进行设置。

# 此处设置模型和接口地址的对应关系，模型和接口地址的映射
MODEL_LIST = {#MODEL_LIST 是一个字典，映射了不同类型的模型及其配置。
#每个模型类型（ chat, completion, embedding）都是一个字典。
    "chat":
        {
            # 模型名称
            "qwen-7b-chat": {
                "addtional_names": ["gpt-3.5-turbo"],  # 其他允许访问该模型的名称，比如chatgpt-next-web使用gpt-3.5-turbo，则需要加入到此处
                "base_url": "http://192.168.20.59:7891/v1",  # 实际访问该模型的接口地址
                "api_key": "xxx"#访问该模型所需的 API 密钥（这里用 xxx 占位）。
            },
            # 模型名称
            "baichuan2-13b": {
                "addtional_names": [],  # 其他允许访问该模型的名称
                "base_url": "http://192.168.20.44:7860/v1",  # 实际访问该模型的接口地址
                "api_key": "xxx"
            },
            # 需要增加其他模型，仿照上面的例子添加即可
        },
    "completion":
        {
            "sqlcoder": {
                "addtional_names": [],  # 其他允许访问该模型的名称
                "base_url": "http://192.168.20.59:7892/v1",  # 实际访问该模型的接口地址
                "api_key": "xxx"
            },
            # 需要增加其他模型，仿照上面的例子添加即可
        },
    "embedding":
        {
            "base_url": "http://192.168.20.59:8001/v1",  # 实际访问该模型的接口地址
            "api_key": "xxx",  # api_key
        },
}

#创建模型名称映射
#CHAT_MODEL_MAP 和 COMPLETION_MODEL_MAP 生成了两个字典，映射了所有允许访问的模型名称（包括附加名称）到其原始名称。
CHAT_MODEL_MAP = {am: name for name, detail in MODEL_LIST["chat"].items() for am in (detail["addtional_names"] + [name])}
COMPLETION_MODEL_MAP = {am: name for name, detail in MODEL_LIST["completion"].items() for am in (detail["addtional_names"] + [name])}
#如果 qwen-7b-chat 的 addtional_names 包含 gpt-3.5-turbo，则 CHAT_MODEL_MAP 会包含:{'qwen-7b-chat': 'qwen-7b-chat', 'gpt-3.5-turbo': 'qwen-7b-chat'}

# API 密钥检查函数
#异步函数 check_api_key用于检查 API 请求的密钥。
async def check_api_key(
    #使用 Depends 注入 HTTPBearer 认证，auto_error=False 表示如果没有提供认证，函数不会自动引发错误。
    auth: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
):

    if API_KEYS is not None:#如果 API_KEYS 被设置（不是 None），检查 auth 是否存在。
        if auth is None or (token := auth.credentials) not in API_KEYS:
        #如果没有提供认证信息或提供的密钥不在允许的 API_KEYS 列表中，抛出 HTTPException，返回 401 状态和错误信息
            raise HTTPException(
                status_code=401,
                detail={
                    "error": {
                        "message": "",
                        "type": "invalid_request_error",
                        "param": None,
                        "code": "invalid_api_key",
                    }
                },
            )
        return token
    else:#如果 API_KEYS 未设置，允许所有请求，直接返回 None。
        # api_keys not set; allow all
        return None


#聊天完成接口
@app.post("/v1/chat/completions", dependencies=[Depends(check_api_key)])
#定义一个 POST 路由 /v1/chat/completions，使用 check_api_key 函数作为依赖，确保请求经过 API 密钥检查。
async def create_chat_completion(request: ChatCompletionCreateParams):
#request 是一个 ChatCompletionCreateParams 类型的参数，包含请求的数据。

#模型验证和客户端初始化
    if request.model not in CHAT_MODEL_MAP:#检查请求中的模型名称是否有效（即是否在 CHAT_MODEL_MAP 中）。
        raise HTTPException(status_code=404, detail="Invalid model")#如果无效，抛出 404 错误。

    model = CHAT_MODEL_MAP[request.model]#获取有效模型名称
    client = AsyncOpenAI(#创建 AsyncOpenAI 客户端实例
        api_key=MODEL_LIST["chat"][model]["api_key"],
        base_url=MODEL_LIST["chat"][model]["base_url"],
        #传入相应的 API 密钥和基础 URL。
    )
#请求参数处理
    params = request.dict(
        #将 request 对象转换为字典，排除值为 None 的字段。
        exclude_none=True,
        include={
            #只包括指定的字段，以便发送到聊天完成 API。
            "messages",
            "model",
            "frequency_penalty",
            "function_call",
            "functions",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "n",
            "presence_penalty",
            "seed",
            "stop",
            "temperature",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "user",
            "stream",
        }
    )
#响应处理
    response = await client.chat.completions.create(**params)
    #使用 client.chat.completions.create(**params) 异步调用聊天完成 API，获取响应。
    async def chat_completion_stream_generator():
        #定义一个异步生成器 chat_completion_stream_generator，用于处理响应流
        async for chunk in response:#逐块读取响应
            yield chunk.json()#并将其转换为 JSON 格式返回。
        yield "[DONE]"#最后返回一个 "[DONE]" 标志，表示流的结束。

#返回结果
    if request.stream:
        return EventSourceResponse(chat_completion_stream_generator())
#如果请求中 stream 字段为真，返回 EventSourceResponse，用于服务器发送事件（SSE）流式传输数据。
    return response
#否则，直接返回 response。

#实现了两个主要的 POST 接口：/v1/completions 和 /v1/embeddings，用于生成文本完成和创建嵌入向量，分别使用了异步 OpenAI 客户端。下面是对代码的详细分析：

#创建完成接口
@app.post("/v1/completions", dependencies=[Depends(check_api_key)])
#定义一个 POST 路由 /v1/completions，依赖于 check_api_key 函数确保请求经过 API 密钥检查。
async def create_completion(request: CompletionCreateParams):
#request 参数是一个类型为 CompletionCreateParams 的数据类，包含请求的数据。

#模型验证和客户端初始化
    if request.model not in COMPLETION_MODEL_MAP:#检查请求中的模型名称是否有效（即是否在 COMPLETION_MODEL_MAP 中）。
        raise HTTPException(status_code=404, detail="Invalid model")#如果无效，抛出 404 错误

    model = COMPLETION_MODEL_MAP[request.model]#获取有效模型名称

    client = AsyncOpenAI(#创建 AsyncOpenAI 客户端实例
        api_key=MODEL_LIST["completion"][model]["api_key"],
        base_url=MODEL_LIST["completion"][model]["base_url"],
        #传入相应的 API 密钥和基础 URL。
    )

#请求参数处理
    params = request.dict(#将 request 对象转换为字典，排除值为 None 的字段。
        exclude_none=True,
        include={
            #只包括指定的字段，以便发送到文本生成 API。
            "prompt",
            "model",
            "best_of",
            "echo",
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "n",
            "presence_penalty",
            "seed",
            "stop",
            "suffix",
            "temperature",
            "top_p",
            "user",
            "stream",
        }
    )

#响应处理
    response = await client.completions.create(**params)
#使用 client.completions.create(**params) 异步调用文本生成 API，获取响应。
    async def generate_completion_stream_generator():
    #定义一个异步生成器 generate_completion_stream_generator，用于处理响应流
        async for chunk in response:#块读取响应并将其转换为 JSON 格式返回。
            yield chunk.json()
        yield "[DONE]"#最后返回一个 "[DONE]" 标志，表示流的结束。

#返回结果
    if request.stream:
        return EventSourceResponse(generate_completion_stream_generator())
#如果请求中 stream 字段为真，返回 EventSourceResponse，用于流式传输数据。
    return response#否则，直接返回 response。

#创建嵌入接口
@app.post("/v1/embeddings", dependencies=[Depends(check_api_key)])
#定义一个 POST 路由 /v1/embeddings，同样依赖于 check_api_key 函数。
async def create_embeddings(request: EmbeddingCreateParams):
    #request 参数是一个类型为 EmbeddingCreateParams 的数据类，包含请求的数据。
    client = AsyncOpenAI(#创建 AsyncOpenAI 客户端实例，用于生成嵌入。
        api_key=MODEL_LIST["embedding"]["api_key"],
        base_url=MODEL_LIST["embedding"]["base_url"],
        ##传入相应的 API 密钥和基础 URL。
    )
    embeddings = await client.embeddings.create(**request.dict(exclude_none=True))
    return embeddings
#调用 client.embeddings.create(**request.dict(exclude_none=True)) 异步生成嵌入，返回结果。

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9009, log_level="info")
#如果脚本作为主程序运行，使用 uvicorn 启动 FastAPI 应用，监听 0.0.0.0 地址的 9009 端口，并设置日志级别为 info。