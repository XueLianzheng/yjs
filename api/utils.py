import json
from threading import Lock
from typing import (
    Optional,
    Union,
    Iterator,
    List,
    AsyncIterator,
)

import anyio
from anyio.streams.memory import MemoryObjectSendStream
from fastapi import Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.security.http import HTTPAuthorizationCredentials, HTTPBearer
from loguru import logger
from pydantic import BaseModel
from starlette.concurrency import iterate_in_threadpool

from api.common import jsonify, dictify
from api.config import SETTINGS
from api.protocol import (
    ChatCompletionCreateParams,
    CompletionCreateParams,
    ErrorResponse,
    ErrorCode
)

#此文件实用工具函数，支持其他模块的功能，用于API的请求处理和响应生成。。

llama_outer_lock = Lock()
llama_inner_lock = Lock()
#创建两个锁对象 llama_outer_lock 和 llama_inner_lock。
#这些锁通常用于在异步环境中控制对共享资源的访问，防止竞争条件。

#check_api_key 是一个异步函数
#async是一个加在函数前的修饰符，被async定义的函数会默认返回一个Promise对象resolve的值。.
async def check_api_key(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
):#使用 Depends 从 FastAPI 的依赖注入系统中提取 HTTPAuthorizationCredentials 对象。
#HTTPBearer 表示这个函数需要处理 Bearer Token 的身份验证。
# auto_error=False: 这意味着如果未提供身份验证令牌，FastAPI 不会自动抛出异常，而是允许我们在函数内部进行处理。


    if not SETTINGS.api_keys:
        # api_keys not set; allow all
        return None
#如果 SETTINGS.api_keys 未设置（可能是空列表或 None），则允许所有请求通过（返回 None）。


#auth is None: 检查是否有提供的身份验证凭证。
#(token := auth.credentials) not in SETTINGS.api_keys: 如果提供了凭证，则检查该凭证是否在设置的 API 密钥列表中。
    if auth is None or (token := auth.credentials) not in SETTINGS.api_keys:
        raise HTTPException(
            #如果没有提供有效的 API 密钥，则会抛出 HTTPException，返回状态码 401（未授权）和错误详情。
            status_code=401,
            detail={
                "error": {
                    "message": "",#错误信息，当前为空字符串
                    "type": "invalid_request_error",# 错误类型（invalid_request_error）。
                    "param": None,#无相关参数，值为 None。
                    "code": "invalid_api_key",#错误代码（invalid_api_key）。
                }
            },
        )
    return token#如果通过了上述检查，则返回有效的 API 密钥令牌。
#这段代码实现了一个简单的 API 密钥验证机制，确保只有有效的 API 密钥才能访问特定的 FastAPI 路由。使用了异步编程模型，可以高效地处理并发请求。它还考虑了灵活性，允许在未设置 API 密钥时接受所有请求，适合不同的运行环境


#create_error_response用于处理错误响应，创建一个 JSON 格式的错误响应。
def create_error_response(code: int, message: str) -> JSONResponse:
    return JSONResponse(#最终返回的 JSONResponse 包含了错误信息。
        dictify(#将 ErrorResponse 实例转换为字典格式的函数。
            ErrorResponse(
                message=message,#字符串类型，表示错误信息。
                code=code#整数类型，表示错误代码。
            )
        ),
        status_code=500#返回一个 JSONResponse 对象，状态码为 500，表示服务器内部错误。
    )


#check_completion_requests检查和处理输入的请求参数，确保它们是有效的，并为后续处理准备。
async def check_completion_requests(
    request: Union[CompletionCreateParams, ChatCompletionCreateParams],
    #请求参数，是 CompletionCreateParams 或 ChatCompletionCreateParams 的实例。
    stop: Optional[List[str]] = None,
    #可选，表示停止生成的字符串列表。
    stop_token_ids: Optional[List[int]] = None,
    #可选，表示停止生成的 token ID 列表。
    chat: bool = True,
    #布尔值，指示请求是否为聊天模式，默认为 True。
) -> Union[CompletionCreateParams, ChatCompletionCreateParams, JSONResponse]:
    #返回处理后的 request 对象，或者在遇到错误时返回 JSONResponse。

    error_check_ret = _check_completion_requests(request)
    if error_check_ret is not None:
        return error_check_ret
    #调用 _check_completion_requests 函数来验证 request 的有效性。如果返回值不为 None，表示有错误，直接返回该错误响应。

    _stop = stop or []
    _stop_token_ids = stop_token_ids or []
    #如果没有提供 stop 或 stop_token_ids，则初始化为空列表。

    request.stop = request.stop or []
    if isinstance(request.stop, str):
        request.stop = [request.stop]
    #确保 request.stop 是一个列表，如果原本是字符串，则将其转换为列表。

    if chat and (
        "qwen" in SETTINGS.model_name.lower()
        and (request.functions is not None or request.tools is not None)
    ):
        request.stop.append("Observation:")
    #当 chat 为 True 且模型名称包含 "qwen" 时，如果请求包含函数或工具，则向 request.stop 添加 "Observation:" 作为停止条件。

    request.stop = list(set(_stop + request.stop))
    request.stop_token_ids = request.stop_token_ids or []
    request.stop_token_ids = list(set(_stop_token_ids + request.stop_token_ids))
    #合并并去重 stop 和 stop_token_ids 列表，以确保没有重复项。

    request.top_p = max(request.top_p, 1e-5)
    if request.temperature <= 1e-5:
        request.top_p = 1.0
    #确保 top_p 的值不小于 1e-5。如果 temperature 小于等于 1e-5，则将 top_p 设置为 1.0。这部分代码用于控制文本生成的随机性和多样性。

    return request#返回最终处理过的请求对象，准备进行后续的处理。


#_check_completion_requests主要用于验证请求参数 request 是否符合特定的要求。
#request: 类型可以是 CompletionCreateParams 或 ChatCompletionCreateParams，用于传入生成文本所需的参数。
#返回一个可选的 JSONResponse 对象，表示可能的错误响应。如果没有错误，返回 None。
def _check_completion_requests(request: Union[CompletionCreateParams, ChatCompletionCreateParams]) -> Optional[JSONResponse]:
    #函数内部对传入的 request 参数进行了一系列的检查，确保每个参数都符合规定的范围或格式。
    # Check all params

    if request.max_tokens is not None and request.max_tokens <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.max_tokens} is less than the minimum of 1 - 'max_tokens'",
        )
    #如果 max_tokens 不为 None 且小于或等于 0，返回错误响应，提示 max_tokens 应该至少为 1。

    if request.n is not None and request.n <= 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.n} is less than the minimum of 1 - 'n'",
        )
    #如果 n 不为 None 且小于或等于 0，返回错误响应，提示 n 应该至少为 1。

    if request.temperature is not None and request.temperature < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is less than the minimum of 0 - 'temperature'",
        )
    #如果 temperature 不为 None 且小于 0，返回错误响应，提示 temperature 应该至少为 0。
    if request.temperature is not None and request.temperature > 2:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.temperature} is greater than the maximum of 2 - 'temperature'",
        )
    #如果 temperature 大于 2，返回错误响应，提示其超出最大限制。

    if request.top_p is not None and request.top_p < 0:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is less than the minimum of 0 - 'top_p'",
        )
    #如果 top_p 不为 None 且小于 0，返回错误响应，提示 top_p 应该至少为 0。
    if request.top_p is not None and request.top_p > 1:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.top_p} is greater than the maximum of 1 - 'temperature'",
        )
    #如果 top_p 大于 1，返回错误响应，提示其超出最大限制。

    if request.stop is None or isinstance(request.stop, (str, list)):
        return None
    #如果 stop 为 None 或是字符串/列表类型，则认为其有效，返回 None，表示没有错误。
    else:
        return create_error_response(
            ErrorCode.PARAM_OUT_OF_RANGE,
            f"{request.stop} is not valid under any of the given schemas - 'stop'",
        )
    #如果 stop 既不为 None，也不是字符串或列表，返回错误响应，提示其不符合有效的格式。

#异步函数 get_event_publisher，用于处理和发送事件数据到客户端。函数利用了 Python 的异步特性，以及 anyio 库来处理并发操作。
async def get_event_publisher(
    request: Request,#Request 对象，表示来自客户端的请求。
    inner_send_chan: MemoryObjectSendStream,#MemoryObjectSendStream 对象，用于向客户端发送数据。
    iterator: Union[Iterator, AsyncIterator],#同步迭代器或异步迭代器，生成要发送的数据。
):
    #上下文管理和异常处理
    async with inner_send_chan:#使用 async with 确保 inner_send_chan 在使用完后正确关闭。
        try:
            #迭代和发送数据
            if SETTINGS.engine not in ["vllm", "tgi"]:#如果 SETTINGS.engine 不是 "vllm" 或 "tgi"，则使用 iterate_in_threadpool 以异步方式迭代 iterator。
                async for chunk in iterate_in_threadpool(iterator):
                    if isinstance(chunk, BaseModel):
                    #对于每个 chunk:如果是 BaseModel 类型，使用 jsonify 转换为 JSON 格式。
                        chunk = jsonify(chunk)
                    elif isinstance(chunk, dict):
                    #如果是字典，使用 json.dumps 转换为 JSON 字符串。
                        chunk = json.dumps(chunk, ensure_ascii=False)

                    await inner_send_chan.send(dict(data=chunk))
                    #通过 inner_send_chan.send 将数据发送到客户端。

                    if await request.is_disconnected():
                        raise anyio.get_cancelled_exc_class()()
                    #检查请求是否已经断开。如果断开，抛出取消异常。

                    if SETTINGS.interrupt_requests and llama_outer_lock.locked():
                        await inner_send_chan.send(dict(data="[DONE]"))
                        raise anyio.get_cancelled_exc_class()()
                    #如果设置了中断请求并且 llama_outer_lock 被锁定，发送 [DONE] 并抛出取消异常。

            #处理特定引擎
            else:#如果 SETTINGS.engine 是 "vllm" 或 "tgi"，直接异步迭代 iterator。
                async for chunk in iterator:
                    chunk = jsonify(chunk)#将每个 chunk 转换为 JSON 格式
                    await inner_send_chan.send(dict(data=chunk))
                    #通过 inner_send_chan.send 将数据发送到客户端。

                    if await request.is_disconnected():
                        raise anyio.get_cancelled_exc_class()()
                    #检查请求是否断开。

            #发送结束标志
            await inner_send_chan.send(dict(data="[DONE]"))
            #在所有数据发送完后，发送 [DONE] 标志，表示数据传输完成。

        #异常处理
        #捕获取消异常，记录客户端断开连接的信息。
        except anyio.get_cancelled_exc_class() as e:
            logger.info("disconnected")
            with anyio.move_on_after(1, shield=True):
            #使用 anyio.move_on_after，在 1 秒后继续运行，确保所有的日志记录完成，最终抛出异常。
                logger.info(f"Disconnected from client (via refresh/close) {request.client}")
                raise e
