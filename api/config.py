import os
from pathlib import Path
from typing import Optional, Dict, List, Union

import dotenv
from loguru import logger
from pydantic import BaseModel, Field

from api.common import jsonify, disable_warnings

dotenv.load_dotenv()
#dotenv.load_dotenv()：加载环境变量文件。
disable_warnings(BaseModel)
#disable_warnings(BaseModel)：禁用Pydantic模型的特定警告。

# 该文件定义了项目的配置和设置，包括从环境变量中加载的配置。

def get_bool_env(key, default="false"):
    return os.environ.get(key, default).lower() == "true"
# 使用 os.environ.get(key, default) 获取环境变量的值，并将其转换为小写，最后与字符串 "true" 比较。


def get_env(key, default):
    val = os.environ.get(key, "")
    return val or default
# 使用 os.environ.get(key, "") 获取环境变量的值，若其为空则返回 default。

ENGINE = get_env("ENGINE", "default").lower()
# 通过调用 get_env("ENGINE", "default") 获取环境变量 ENGINE 的值，如果不存在则使用默认值 "default"。然后将其转换为小写。
TEI_ENDPOINT = get_env("TEI_ENDPOINT", None)
# 通过调用 get_env("TEI_ENDPOINT", None) 获取环境变量 TEI_ENDPOINT 的值，如果不存在则使用默认值 None。
TASKS = get_env("TASKS", "llm").lower().split(",")  # llm, rag
# 通过调用 get_env("TASKS", "llm") 获取环境变量 TASKS 的值，如果不存在则使用默认值 "llm"。然后.lower()将其转换为小写，并用.split(",")利用逗号分隔成一个列表。

STORAGE_LOCAL_PATH = get_env(
    "STORAGE_LOCAL_PATH",
    os.path.join(Path(__file__).parents[1], "data", "file_storage")
)
os.makedirs(STORAGE_LOCAL_PATH, exist_ok=True)
# 通过调用 get_env 获取环境变量 STORAGE_LOCAL_PATH 的值，使用默认路径（基于当前脚本位置）构造。如果不存在该环境变量，则默认路径为 os.path.join(Path(__file__).parents[1], "data", "file_storage")。
# 使用 os.makedirs(STORAGE_LOCAL_PATH, exist_ok=True) 创建该路径（如果不存在的话）。


# 该类用于管理应用程序的设置。它使用 Pydantic 库的功能来进行数据验证和解析。
class BaseSettings(BaseModel): #BaseModel：来自 Pydantic 的基类，允许定义数据模型并进行验证。
    """ Settings class. """
    # Optional用于类型注解，支持可选类型
    host: Optional[str] = Field( # Field是用于定义字段属性的函数。
        default=get_env("HOST", "0.0.0.0"),
        description="Listen address.",
    ) #此host字段用于设置监听地址，通过 get_env 函数获取环境变量 HOST，如果未定义则默认为 "0.0.0.0"。

    port: Optional[int] = Field(
        default=int(get_env("PORT", 8000)),
        description="Listen port.",
    )# port字段用于设置监听端口，通过 get_env 获取环境变量 PORT，并转为整数，默认值为 8000。

    api_prefix: Optional[str] = Field(
        default=get_env("API_PREFIX", "/v1"),
        description="API prefix.",
    )#api_prefix此字段用于设置API的前缀，通过 get_env 获取环境变量 API_PREFIX，默认值为 "/v1"。

    engine: Optional[str] = Field(
        default=ENGINE,
        description="Choices are ['default', 'vllm'].",
    )#可选择的引擎类型，如 'default' 或 'vllm'

    tasks: Optional[List[str]] = Field(
        default=list(TASKS),
        description="Choices are ['llm', 'rag'].",
    )#从TASKS构建的列表，支持的任务列表，如 'llm' 和 'rag'。

    # device related
    device_map: Optional[Union[str, Dict]] = Field( #Union用于类型注解，支持联合类型。
        default=get_env("DEVICE_MAP", "auto"),
        description="Device map to load the model."
    )# device_map是模型加载的设备映射，通过 get_env 获取环境变量 DEVICE_MAP，默认值为 "auto"。

    gpus: Optional[str] = Field(
        default=get_env("GPUS", None),
        description="Specify which gpus to load the model."
    )# gpus字段指定加载模型的GPU，通过 get_env 获取环境变量 GPUS，默认为 None。


    num_gpus: Optional[int] = Field(
        default=int(get_env("NUM_GPUs", 1)),
        ge=0,
        description="How many gpus to load the model."
    )# num_gpus字段指定加载模型的GPU数量，必须大于等于0。通过 get_env 获取环境变量 NUM_GPUs，并转为整数，默认值为 1。

    activate_inference: Optional[bool] = Field(
        default=get_bool_env("ACTIVATE_INFERENCE", "true"),
        description="Whether to activate inference."
    )# activate_inference字段表示是否激活推理功能，通过 get_bool_env 函数获取环境变量 ACTIVATE_INFERENCE，默认为 True。

    model_names: Optional[List] = Field(
        default_factory=list,
        description="All available model names"
    ) # model_names是所有可用模型名称的列表，使用 default_factory，返回一个空列表。

    # support for api key check
    api_keys: Optional[List[str]] = Field(
        default=get_env("API_KEYS", "").split(",") if get_env("API_KEYS", "") else None,
        description="Support for api key check."
    ) # api_keys字段用于支持API密钥检查，通过 get_env 获取环境变量 API_KEYS，如果存在则用逗号分隔的字符串转换为列表。

# LLMSettings类用于管理与大型语言模型（LLM）相关的配置设置
class LLMSettings(BaseModel):
    # model related
    model_name: Optional[str] = Field(
        default=get_env("MODEL_NAME", None),
        description="The name of the model to use for generating completions."
    )# model_name字段指定要使用的模型名称，以便生成文本完成。
    model_path: Optional[str] = Field(
        default=get_env("MODEL_PATH", None),
        description="The path to the model to use for generating completions."
    )# model_path字段指定模型的路径，用于加载模型。
    dtype: Optional[str] = Field(
        default=get_env("DTYPE", "half"),
        description="Precision dtype."
    )# dtype字段用于指定数值的精度类型，例如 "float16" 或 "float32"。

    # quantize related
    load_in_8bit: Optional[bool] = Field(
        default=get_bool_env("LOAD_IN_8BIT"),
        description="Whether to load the model in 8 bit."
    )# load_in_8bit字段指定是否以 8 位精度加载模型，以节省内存和加速计算。
    load_in_4bit: Optional[bool] = Field(
        default=get_bool_env("LOAD_IN_4BIT"),
        description="Whether to load the model in 4 bit."
    )# load_in_4bit字段指定是否以 4 位精度加载模型，进一步节省内存。

    # context related
    context_length: Optional[int] = Field(
        default=int(get_env("CONTEXT_LEN", -1)),
        ge=-1,
        description="Context length for generating completions."
    )# context_length字段指定用于生成文本完成的上下文长度。ge=-1 表示这个值必须大于或等于 -1。
    chat_template: Optional[str] = Field(
        default=get_env("PROMPT_NAME", None),
        description="Chat template for generating completions."
    )#chat_template字段用于生成文本完成的聊天模板

    rope_scaling: Optional[str] = Field(
        default=get_env("ROPE_SCALING", None),
        description="RoPE Scaling."
    )# rope_scaling字段用于指定 RoPE（相对位置编码）缩放参数。
    flash_attn: Optional[bool] = Field(
        default=get_bool_env("FLASH_ATTN", "auto"),
        description="Use flash attention."
    )# flash_attn字段指定是否使用闪存注意力机制以加速计算。

    interrupt_requests: Optional[bool] = Field(
        default=get_bool_env("INTERRUPT_REQUESTS", "true"),
        description="Whether to interrupt requests when a new request is received.",
    )# interrupt_requests字段指定在接收到新请求时是否中断当前请求


#RAGSettings类用于管理与检索增强生成（RAG, Retrieval-Augmented Generation）相关的配置设置。
class RAGSettings(BaseModel):
    # embedding related
    embedding_name: Optional[str] = Field(
        default=get_env("EMBEDDING_NAME", None),
        description="The path to the model to use for generating embeddings."
    )# embedding_name字段指定用于生成嵌入（embeddings）的模型的路径。
    rerank_name: Optional[str] = Field(
        default=get_env("RERANK_NAME", None),
        description="The path to the model to use for reranking."
    )# rerank_name字段指定用于重新排序（reranking）的模型的路径。
    embedding_size: Optional[int] = Field(
        default=int(get_env("EMBEDDING_SIZE", -1)),
        description="The embedding size to use for generating embeddings."
    )# embedding_size字段指定用于生成嵌入的大小。
    embedding_device: Optional[str] = Field(
        default=get_env("EMBEDDING_DEVICE", "cuda:0"),
        description="Device to load the model."
    )# embedding_device字段指定加载嵌入模型的设备（例如：GPU 或 CPU）。
    rerank_device: Optional[str] = Field(
        default=get_env("RERANK_DEVICE", "cuda:0"),
        description="Device to load the model."
    )# rerank_device字段指定加载重新排序模型的设备。


class VLLMSetting(BaseModel):
    trust_remote_code: Optional[bool] = Field(
        default=get_bool_env("TRUST_REMOTE_CODE"),
        description="Whether to use remote code."
    )# 指示是否使用远程代码
    tokenize_mode: Optional[str] = Field(
        default=get_env("TOKENIZE_MODE", "auto"),
        description="Tokenize mode for vllm server."
    )# VLLM 服务器的分词模式。
    tensor_parallel_size: Optional[int] = Field(
        default=int(get_env("TENSOR_PARALLEL_SIZE", 1)),
        ge=1,
        description="Tensor parallel size for vllm server."
    )# VLLM 服务器的张量并行大小，必须大于或等于 1。
    gpu_memory_utilization: Optional[float] = Field(
        default=float(get_env("GPU_MEMORY_UTILIZATION", 0.9)),
        description="GPU memory utilization for vllm server."
    )# VLLM 服务器的 GPU 内存利用率，默认值为 0.9。
    max_num_batched_tokens: Optional[int] = Field(
        default=int(get_env("MAX_NUM_BATCHED_TOKENS", -1)),
        ge=-1,
        description="Max num batched tokens for vllm server."
    )# VLLM 服务器的最大批处理令牌数，允许为 -1。
    max_num_seqs: Optional[int] = Field(
        default=int(get_env("MAX_NUM_SEQS", 256)),
        ge=1,
        description="Max num seqs for vllm server."
    )# VLLM 服务器的最大序列数量，必须大于或等于 1。
    quantization_method: Optional[str] = Field(
        default=get_env("QUANTIZATION_METHOD", None),
        description="Quantization method for vllm server."
    )# VLLM 服务器的量化方法。
    enforce_eager: Optional[bool] = Field(
        default=get_bool_env("ENFORCE_EAGER"),
        description="Always use eager-mode PyTorch. If False, will use eager mode and CUDA graph in hybrid for maximal performance and flexibility."
    )# 始终使用急切模式（eager-mode）的 PyTorch。如果为 False，则在最大性能和灵活性下使用急切模式和 CUDA 图。
    max_seq_len_to_capture: Optional[int] = Field(
        default=int(get_env("MAX_SEQ_LEN_TO_CAPTURE", 8192)),
        description="Maximum context length covered by CUDA graphs. When a sequence has context length larger than this, we fall back to eager mode."
    )# CUDA 图覆盖的最大上下文长度。如果序列的上下文长度超过此值，将回退到急切模式。
    max_loras: Optional[int] = Field(
        default=int(get_env("MAX_LORAS", 1)),
        description="Max number of LoRAs in a single batch."
    )# 单个批次中的最大 LoRA 数量。
    max_lora_rank: Optional[int] = Field(
        default=int(get_env("MAX_LORA_RANK", 32)),
        description="Max LoRA rank."
    )# 最大 LoRA 秩。
    lora_extra_vocab_size: Optional[int] = Field(
        default=int(get_env("LORA_EXTRA_VOCAB_SIZE", 256)),
        description="Maximum size of extra vocabulary that can be present in a LoRA adapter added to the base model vocabulary."
    )# 添加到基础模型词汇表中的 LoRA 适配器的额外词汇的最大大小。
    lora_dtype: Optional[str] = Field(
        default=get_env("LORA_DTYPE", "auto"),
        description="Data type for LoRA. If auto, will default to base model dtype."
    )# LoRA 的数据类型。如果为 "auto"，则默认使用基础模型的数据类型。
    max_cpu_loras: Optional[int] = Field(
        default=int(get_env("MAX_CPU_LORAS", -1)),
        ge=-1,
    )# CPU 上的最大 LoRA 数量，允许为 -1。
    lora_modules: Optional[str] = Field(
        default=get_env("LORA_MODULES", ""),
    )# 指定 LoRA 模块。
    disable_custom_all_reduce: Optional[bool] = Field(
        default=get_bool_env("DISABLE_CUSTOM_ALL_REDUCE"),
    )# 指示是否禁用自定义全归约。
    vllm_disable_log_stats: Optional[bool] = Field(
        default=get_bool_env("VLLM_DISABLE_LOG_STATS", "true"),
    )# 指示是否禁用 VLLM 日志统计。
    distributed_executor_backend: Optional[str] = Field(
        default=get_env("DISTRIBUTED_EXECUTOR_BACKEND", None),
    )# 指定分布式执行器的后端。


#定义了一个名为 TEXT_SPLITTER_CONFIG 的字典，其中包含多个文本分割器（text splitter）的配置。
TEXT_SPLITTER_CONFIG = {
    "ChineseRecursiveTextSplitter": {
        #source: 设为 "huggingface"，指明该分割器的来源为 Hugging Face。
        "source": "huggingface",   # 选择tiktoken则使用openai的方法
        "tokenizer_name_or_path": get_env("EMBEDDING_NAME", ""),
    },
    "SpacyTextSplitter": {
        "source": "huggingface",
        "tokenizer_name_or_path": "gpt2",
    },
    "RecursiveCharacterTextSplitter": {
        "source": "tiktoken",#表明该分割器使用 Tiktoken 库。
        "tokenizer_name_or_path": "cl100k_base",
    },
    "MarkdownHeaderTextSplitter": {#该分割器专注于 Markdown 格式的文本分割。
        "headers_to_split_on":# 这是一个列表，包含多个元组，每个元组指定了一种 Markdown 头部（header）类型及其对应的分割方式：
            [
                ("#", "head1"),
                ("##", "head2"),
                ("###", "head3"),
                ("####", "head4"),
            ]
    },
}


PARENT_CLASSES = [BaseSettings]
#初始化一个列表 PARENT_CLASSES，其中包含 BaseSettings 类。这意味着 Settings 类将继承自 BaseSettings。

#根据 TASKS 和 ENGINE 的值，动态地向 PARENT_CLASSES 列表中添加不同的设置类
if "llm" in TASKS:
    if ENGINE == "default":# 如果 ENGINE 是 "default"，则向 PARENT_CLASSES 添加 LLMSettings。
        PARENT_CLASSES.append(LLMSettings)
    elif ENGINE == "vllm":#如果 ENGINE 是 "vllm"，则向 PARENT_CLASSES 添加 LLMSettings 和 VLLMSetting。
        PARENT_CLASSES.extend([LLMSettings, VLLMSetting])

if "rag" in TASKS:#如果 TASKS 中包含 "rag"（可能指检索增强生成任务），则向 PARENT_CLASSES 添加 RAGSettings。
    PARENT_CLASSES.append(RAGSettings)


class Settings(*PARENT_CLASSES): #使用 *PARENT_CLASSES 将 PARENT_CLASSES 列表中的类作为基类定义 Settings 类。这意味着 Settings 将包含 BaseSettings、LLMSettings、VLLMSetting 和 RAGSettings 中的所有设置。
    ...


SETTINGS = Settings()#创建一个 Settings 的实例，命名为 SETTINGS。
for name in ["model_name", "embedding_name", "rerank_name"]:
    if getattr(SETTINGS, name, None):
        SETTINGS.model_names.append(getattr(SETTINGS, name).split("/")[-1])
#遍历一个包含 "model_name"、"embedding_name" 和 "rerank_name" 的列表：如果在 SETTINGS 中找到相应的属性且其值不为 None，则将该属性的值分割并提取出最后一个部分（通常是模型的名称）追加到 SETTINGS.model_names 列表中。
logger.debug(f"SETTINGS: {jsonify(SETTINGS, indent=4)}")
# 记录 SETTINGS 的内容，使用 jsonify 函数将 SETTINGS 转换为 JSON 格式，格式化为 4 个空格的缩进以便于阅读。

if SETTINGS.gpus:
    #检查 SETTINGS.gpus 中的 GPU 数量是否小于 SETTINGS.num_gpus，如果是，抛出一个 ValueError，说明请求的 GPU 数量大于可用的 GPU 数量。
    if len(SETTINGS.gpus.split(",")) < SETTINGS.num_gpus:
        raise ValueError(
            f"Larger --num_gpus ({SETTINGS.num_gpus}) than --gpus {SETTINGS.gpus}!"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = SETTINGS.gpus
    #如果检查通过，将环境变量 CUDA_VISIBLE_DEVICES 设置为 SETTINGS.gpus，以指定可用的 GPU。
