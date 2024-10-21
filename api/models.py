from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.common import dictify
from api.config import SETTINGS
#此文件定义数据模型或业务模型。

#用于创建 FastAPI 应用的函数 create_app
def create_app() -> FastAPI:
    import gc#gc: Python 的垃圾回收模块，用于手动触发垃圾回收。
    import torch#torch: PyTorch 深度学习框架，用于处理张量和 GPU 操作。

    def torch_gc() -> None:#该函数用于收集 GPU 内存
        r"""
        Collects GPU memory.
        """
        gc.collect()#gc.collect(): 触发 Python 的垃圾回收，清理不再使用的内存。
        if torch.cuda.is_available():#检查是否可用 CUDA（即是否存在 GPU）。
            torch.cuda.empty_cache()#torch.cuda.empty_cache(): 清空未使用的 GPU 内存缓存，以减少内存占用。
            torch.cuda.ipc_collect()#torch.cuda.ipc_collect(): 清理 GPU 的进程间通信 (IPC) 缓存，帮助回收内存。

    @asynccontextmanager#@asynccontextmanager 装饰器定义一个异步上下文管理器，用于管理 FastAPI 应用的生命周期。
    #使用 lifespan 来处理应用的生命周期，确保在应用结束时进行清理。
    async def lifespan(app: "FastAPI"):  # collects GPU memory
        yield
        torch_gc()
        #yield 语句在这里表示在应用启动后可以进行其他操作，直到应用关闭时，执行 torch_gc()，以确保在应用结束时清理 GPU 内存。

    """ create fastapi app server """
    app = FastAPI(lifespan=lifespan) #创建 FastAPI 应用
    #创建一个 FastAPI 应用实例，并将 lifespan 上下文管理器传递给它。这确保在应用的生命周期内会调用 torch_gc()。

    #添加跨源资源共享 (CORS) 中间件，以允许来自任何源的请求
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],# 允许来自所有源的请求。
        allow_credentials=True,# 允许发送凭证（如 Cookies）。
        allow_methods=["*"],#允许所有 HTTP 方法（如 GET、POST 等）。
        allow_headers=["*"],#允许所有请求头。
    )
    return app#返回创建的 FastAPI 应用实例。

# create_rag_models函数主要用于根据配置创建 RAG（Retrieval-Augmented Generation）模型的实例。
def create_rag_models():#函数 create_rag_models 没有参数，返回一个包含 RAG 模型的列表。
    """ get rag models. """
    rag_models = []#用于存储创建的模型实例
    if "rag" in SETTINGS.tasks and SETTINGS.activate_inference:#检查设置中的任务是否包含 "rag" 并且推理是否被激活。如果满足条件，则继续创建模型
        if SETTINGS.embedding_name:#如果设置中指定了 embedding_name，则导入 RAGEmbedding 类
            from api.rag import RAGEmbedding
            rag_models.append(#创建其实例，使用指定的 embedding_name 和 embedding_device。将创建的模型实例添加到 rag_models 列表中。
               RAGEmbedding(SETTINGS.embedding_name, SETTINGS.embedding_device)
            )
        else:#如果未指定 embedding_name，则向列表中添加 None，表示没有创建嵌入模型。
            rag_models.append(None)
        if SETTINGS.rerank_name:#如果设置中指定了 rerank_name，则导入 RAGReranker 类
            from api.rag import RAGReranker
            #创建其实例，使用指定的 rerank_name 和 rerank_device。将创建的模型实例添加到 rag_models 列表中。
            rag_models.append(
                RAGReranker(SETTINGS.rerank_name, device=SETTINGS.rerank_device)
            )
        else:#如果未指定 rerank_name，则向列表中添加 None，表示没有创建重排序模型。
            rag_models.append(None)
    return rag_models if len(rag_models) == 2 else [None, None]
    #如果 rag_models 列表的长度为 2（即成功创建了两个模型），则返回该列表；否则，返回 [None, None]，表示没有成功创建任何模型。

#create_hf_llm用于创建一个 Hugging Face 语言模型（LLM）实例，用于聊天或文本生成。
def create_hf_llm():
    """ get generate model for chat or completion. """
    from api.engine.hf import HuggingFaceEngine#用于创建模型实例
    from api.adapter.loader import load_model_and_tokenizer#用于加载模型和分词器。

    include = {#集合 include，其中包含了在加载模型时可能使用的参数名称。这些参数与模型的配置有关，例如设备映射和数据类型。
        "device_map",
        "load_in_8bit",
        "load_in_4bit",
        "dtype",
        "rope_scaling",
        "flash_attn",
    }
    kwargs = dictify(SETTINGS, include=include)
    #使用 dictify 函数将设置中的参数（根据 include 集合中列出的参数）转换为字典 kwargs。这个字典将用于加载模型时传递给 load_model_and_tokenizer。
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=SETTINGS.model_path, **kwargs,
    )#调用 load_model_and_tokenizer 函数，传入模型的路径（SETTINGS.model_path）和加载参数 kwargs。该函数返回加载的模型和分词器，分别赋值给 model 和 tokenizer。

    logger.info("Using HuggingFace Engine")
    # 记录一条信息日志，指明当前正在使用 Hugging Face 引擎。这有助于在调试和运行时了解系统状态。

    #创建并返回一个 HuggingFaceEngine 实例
    return HuggingFaceEngine(
        model,#加载的模型
        tokenizer,#加载的分词器
        model_name=SETTINGS.model_name,#从设置中获取的模型名称
        max_model_length=SETTINGS.context_length if SETTINGS.context_length > 0 else None,#最大模型长度，如果设置中的上下文长度大于 0，则使用该值；否则为 None
        template_name=SETTINGS.chat_template,#从设置中获取的聊天模板名称
    )

#create_vllm_engine用于创建一个 VLLM 生成引擎的实例，适用于聊天或文本生成。
def create_vllm_engine():
    """ get vllm generate engine for chat or completion. """
    try:
        import vllm #尝试导入 vllm 模块及相关类：
        from vllm.engine.arg_utils import AsyncEngineArgs# AsyncEngineArgs: 用于构建引擎参数的类。
        from vllm.engine.async_llm_engine import AsyncLLMEngine# AsyncLLMEngine: 异步 LLM 引擎的类。
        from api.engine.vllm_engine import VllmEngine # VllmEngine: 包装 VLLM 引擎的自定义类。
    except ImportError:# 如果导入失败，抛出 ValueError，提示 VLLM 引擎不可用。
        raise ValueError("VLLM engine not available")

    vllm_version = vllm.__version__#获取当前安装的 VLLM 模块的版本。

    include = {
        "tokenizer_mode",
        "trust_remote_code",
        "tensor_parallel_size",
        "dtype",
        "gpu_memory_utilization",
        "max_num_seqs",
        "enforce_eager",
        "lora_extra_vocab_size",
        "disable_custom_all_reduce",
    }#创建一个集合 include，其中包含在加载引擎时可能会用到的参数名称。这些参数与 VLLM 的配置相关。

    #如果 VLLM 版本大于或等于 0.4.3，则动态增加两个参数 max_seq_len_to_capture 和 distributed_executor_backend 到 include 集合中。
    if vllm_version >= "0.4.3":
        include.add("max_seq_len_to_capture")
        include.add("distributed_executor_backend")

    #使用 dictify 函数从 SETTINGS 中提取 include 集合中的参数，并将其转换为字典 kwargs。这些参数将用于配置引擎。
    kwargs = dictify(SETTINGS, include=include)

    #创建一个 AsyncEngineArgs 的实例 engine_args，将需要的参数传递给它。
    engine_args = AsyncEngineArgs(
        model=SETTINGS.model_path,#模型路径
        max_num_batched_tokens=SETTINGS.max_num_batched_tokens if SETTINGS.max_num_batched_tokens > 0 else None,#最大批处理令牌数
        max_model_len=SETTINGS.context_length if SETTINGS.context_length > 0 else None,#最大模型长度
        quantization=SETTINGS.quantization_method,#量化方法
        max_cpu_loras=SETTINGS.max_cpu_loras if SETTINGS.max_cpu_loras > 0 else None,#最大 CPU LoRA 数量
        disable_log_stats=SETTINGS.vllm_disable_log_stats,#禁用日志统计的设置
        disable_log_requests=True,#禁用请求日志
        **kwargs,
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    #使用 AsyncEngineArgs 实例 engine_args 创建异步 LLM 引擎的实例 engine。

    logger.info("Using vllm engine")
    #记录一条信息日志，指明当前正在使用 VLLM 引擎。

    #创建并返回一个 VllmEngine 实例
    return VllmEngine(
        engine,# 之前创建的异步 LLM 引擎
        SETTINGS.model_name,#从设置中获取的模型名称
        SETTINGS.chat_template,# 从设置中获取的聊天模板名称
    )


# fastapi app
app = create_app()#通过调用 create_app() 函数来初始化 FastAPI 应用

# model for rag
EMBEDDING_MODEL, RERANK_MODEL = create_rag_models()
#调用 create_rag_models() 函数，创建一个 RAG（Retrieval-Augmented Generation）模型。
#Embedding Model (EMBEDDING_MODEL): 用于将输入文本转换为向量表示，以便进行相似性搜索。
#Re-rank Model (RERANK_MODEL): 用于对检索到的结果进行重新排序，以提升结果的相关性。

# llm,创建 LLM 引擎:
if "llm" in SETTINGS.tasks and SETTINGS.activate_inference:#检查 SETTINGS.tasks 中是否包含 "llm"，同时确保 SETTINGS.activate_inference 为 True。这表明应用程序要使用语言模型（LLM）并且启用推理。
    if SETTINGS.engine == "default":#如果 SETTINGS.engine 等于 "default"，则调用 create_hf_llm() 创建一个 Hugging Face 的 LLM 引擎。
        LLM_ENGINE = create_hf_llm()
    elif SETTINGS.engine == "vllm":#如果 SETTINGS.engine 等于 "vllm"，则调用 create_vllm_engine() 创建一个 VLLM 引擎。
        LLM_ENGINE = create_vllm_engine()
else:#未使用 LLM 引擎的情况:
    LLM_ENGINE = None
#如果不满足上述条件，即 SETTINGS.tasks 中没有 "llm" 或者 SETTINGS.activate_inference 为 False，则将 LLM_ENGINE 设置为 None，表示没有可用的 LLM 引擎。