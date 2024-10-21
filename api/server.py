from api.config import SETTINGS
from api.models import (
    app,
    EMBEDDING_MODEL,
    LLM_ENGINE,
    RERANK_MODEL,
)

#该文件负责创建FastAPI应用和加载模型。
#这段代码是一个 FastAPI 应用的启动和路由配置部分。它根据不同的条件导入和注册不同的路由器（router），并最终启动应用。

#前缀设置
prefix = SETTINGS.api_prefix
#从 SETTINGS 中获取 API 路由的前缀，通常是一个字符串，例如 /api。

#嵌入模型路由
if EMBEDDING_MODEL is not None:
    from api.routes.embedding import embedding_router

    app.include_router(embedding_router, prefix=prefix, tags=["Embedding"])
# 如果 EMBEDDING_MODEL 被定义（非 None），则导入 embedding_router 并将其注册到 FastAPI 应用中，使用指定的 prefix 和标签 "Embedding"。

#文件路由
#尝试导入 file_router，如果成功，则注册到应用中；如果导入失败（例如文件不存在），则捕获异常并忽略。这是一种处理可选模块的方法。
    try:
        from api.routes.file import file_router

        app.include_router(file_router, prefix=prefix, tags=["File"])
    except ImportError:
        pass

#重排序模型路由:
if RERANK_MODEL is not None:
    from api.routes.rerank import rerank_router

    app.include_router(rerank_router, prefix=prefix, tags=["Rerank"])
#如果 RERANK_MODEL 被定义，则导入 rerank_router 并注册到应用中，使用指定的前缀和标签 "Rerank"。


#LLM（大语言模型）引擎路由:
if LLM_ENGINE is not None:
    from api.routes import model_router

    app.include_router(model_router, prefix=prefix, tags=["Model"])
#如果 LLM_ENGINE 被定义，则导入并注册 model_router，用来处理与大语言模型相关的路由。

    # 根据 SETTINGS.engine 的值选择导入不同的聊天和补全路由。
    if SETTINGS.engine == "vllm":#如果引擎是 "vllm"，则从 api.vllm_routes 导入路由；
        from api.vllm_routes import chat_router as chat_router
        from api.vllm_routes import completion_router as completion_router

    else:#否则，从 api.routes 导入。然后注册这些路由到应用中，分别使用标签 "Chat Completion" 和 "Completion"。
        from api.routes.chat import chat_router as chat_router
        from api.routes.completion import completion_router as completion_router

    app.include_router(chat_router, prefix=prefix, tags=["Chat Completion"])
    app.include_router(completion_router, prefix=prefix, tags=["Completion"])

#应用启动
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=SETTINGS.host, port=SETTINGS.port, log_level="info")
#当脚本作为主程序运行时，使用 uvicorn 启动 FastAPI 应用，指定主机、端口和日志级别。这使得开发者能够在本地测试应用。