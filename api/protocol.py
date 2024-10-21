import enum
from typing import (
    Optional,
    Dict,
    List,
    Union,
    Literal,
    Any,
    TypedDict,
)

from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolChoiceOptionParam,
)
from openai.types.chat.completion_create_params import FunctionCall, ResponseFormat
from openai.types.create_embedding_response import Usage
from pydantic import BaseModel

#定义API协议相关的内容

#这是控制器心跳的过期时间，单位为秒。当控制器在 90 秒内没有收到任何心跳信号时，系统可能会认为该控制器已经失效，可能会触发一些错误处理机制，例如重新启动控制器或进行故障转移。
CONTROLLER_HEART_BEAT_EXPIRATION = 90# 适用于监控服务的健康状态，以确保系统能及时发现故障并作出响应。
#这是工作节点发送心跳信号的时间间隔，单位为秒。每 30 秒，工作节点会向控制器发送一次心跳，以表明其仍在运行并且可以处理任务。
WORKER_HEART_BEAT_INTERVAL = 30#心跳机制用于保持工作节点的活跃状态，确保控制器能够及时获取节点的状态信息。
# 这是工作节点在调用 API 时的超时时间，单位为秒。如果工作节点在 20 秒内没有收到 API 的响应，则会认为该请求超时，可能会重试或者记录错误。
WORKER_API_TIMEOUT = 20#适用于管理 API 请求的响应时间，确保工作节点能够及时响应和处理请求，避免长时间等待而影响系统性能。


class Role(str, enum.Enum):#Role 枚举定义了系统中可能存在的角色类型。这些角色可以用于管理不同用户或组件之间的交互。
    USER = "user"#普通用户，通常是与系统交互的最终用户
    ASSISTANT = "assistant"#助手角色，可能用于提供支持或辅助用户
    SYSTEM = "system"# 系统角色，表示系统本身，可能用于执行系统级操作。
    FUNCTION = "function"#函数角色，可能代表特定的功能或服务。
    TOOL = "tool"#工具角色，可能表示与特定工具或功能相关的操作。

#ErrorCode 枚举定义了 API 调用中可能出现的错误代码。每个错误代码都与特定的错误类型相关联，便于识别和处理。
class ErrorCode(enum.IntEnum):
    """
    https://platform.openai.com/docs/guides/error-codes/api-errors
    """

    VALIDATION_TYPE_ERROR = 40001

    INVALID_AUTH_KEY = 40101
    INCORRECT_AUTH_KEY = 40102
    NO_PERMISSION = 40103

    INVALID_MODEL = 40301
    PARAM_OUT_OF_RANGE = 40302
    CONTEXT_OVERFLOW = 40303

    RATE_LIMIT = 42901
    QUOTA_EXCEEDED = 42902
    ENGINE_OVERLOADED = 42903

    INTERNAL_ERROR = 50001
    CUDA_OUT_OF_MEMORY = 50002
    GRADIO_REQUEST_ERROR = 50003
    GRADIO_STREAM_UNKNOWN_ERROR = 50004
    CONTROLLER_NO_WORKER = 50005
    CONTROLLER_WORKER_TIMEOUT = 50006
    # VALIDATION_TYPE_ERROR(40001): 验证类型错误，通常是输入参数的格式或类型不符合要求。
    # INVALID_AUTH_KEY(40101): 无效的授权密钥，表示提供的API密钥无效。
    # INCORRECT_AUTH_KEY(40102): 错误的授权密钥，表示提供的API密钥与账户不匹配。
    # NO_PERMISSION(40103): 没有权限，表示当前用户没有执行请求所需的权限。
    # INVALID_MODEL(40301): 无效的模型，表示请求中指定的模型不存在或不可用。
    # PARAM_OUT_OF_RANGE(40302): 参数超出范围，表示请求的参数值超出了允许的范围。
    # CONTEXT_OVERFLOW(40303): 上下文溢出，表示请求的上下文内容超出了限制。
    # RATE_LIMIT(42901): 超出速率限制，表示超过了API调用的速率限制。
    # QUOTA_EXCEEDED(42902): 超出配额，表示已达到API调用的配额上限。
    # ENGINE_OVERLOADED(42903): 引擎过载，表示服务器过载，无法处理请求。
    # INTERNAL_ERROR(50001): 内部错误，表示服务器在处理请求时发生了未预期的错误。
    # CUDA_OUT_OF_MEMORY(50002): CUDA内存不足，表示在执行深度学习操作时GPU内存不足。
    # GRADIO_REQUEST_ERROR(50003): Gradio请求错误，表示在处理Gradio请求时发生错误。
    # GRADIO_STREAM_UNKNOWN_ERROR(50004): Gradio流未知错误，表示Gradio流在处理时发生了未知错误。
    # CONTROLLER_NO_WORKER(50005): 控制器没有工作节点，表示没有可用的工作节点来处理请求。
    # CONTROLLER_WORKER_TIMEOUT(50006): 控制器工作节点超时，表示工作节点在指定时间内未响应。

class ErrorResponse(BaseModel):#ErrorResponse 类用于表示 API 错误响应的结构。它继承自 BaseModel，是某个数据验证库（如 Pydantic）的基类，通常用于数据模型。
    object: str = "error"#表示响应类型，默认为 "error"。
    message: str#包含详细的错误信息，便于开发人员理解错误原因。
    code: int#表示具体的错误代码，对应于 ErrorCode 中定义的错误代码，便于程序化处理。

#ChatCompletionCreateParams 是一个用于定义聊天补全请求参数的类。它继承自 Pydantic 的 BaseModel，允许我们轻松地进行数据验证和转换。
class ChatCompletionCreateParams(BaseModel):
    messages: List[Dict[str, Any]]
    #messages: 这是一个包含对话历史的消息列表，每条消息是一个字典。字典通常包括发送者的角色（如用户或助手）和消息内容。这些消息为模型提供上下文
    """A list of messages comprising the conversation so far.

    [Example Python code](https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models).
    """

    model: str
    #model: 要使用的模型的ID。这决定了哪个具体的语言模型将处理当前的聊天请求。
    """ID of the model to use.

    See the
    [model endpoint compatibility](https://platform.openai.com/docs/models/model-endpoint-compatibility)
    table for details on which models work with the Chat API.
    """

    frequency_penalty: Optional[float] = 0.
    #frequency_penalty: 控制生成文本时对重复词的惩罚程度。值越高，模型越不倾向于重复之前出现过的词。
    """Number between -2.0 and 2.0.

    Positive values penalize new tokens based on their existing frequency in the
    text so far, decreasing the model's likelihood to repeat the same line verbatim.

    [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)
    """

    function_call: Optional[FunctionCall] = None
    #function_call: 指定模型是否调用某个函数。这个参数已经被 tool_choice 取代。
    """Deprecated in favor of `tool_choice`.

    Controls which (if any) function is called by the model. `none` means the model
    will not call a function and instead generates a message. `auto` means the model
    can pick between generating a message or calling a function. Specifying a
    particular function via `{"name": "my_function"}` forces the model to call that
    function.

    `none` is the default when no functions are present. `auto`` is the default if
    functions are present.
    """

    functions: Optional[List] = None
    #functions: 一个函数列表，模型可以为这些函数生成输入。这个参数已经被 tools 取代。
    """Deprecated in favor of `tools`.

    A list of functions the model may generate JSON inputs for.
    """

    logit_bias: Optional[Dict[str, int]] = None
    #logit_bias: 修改特定标记出现在生成文本中的概率。通过调整标记的对数概率来影响其选择概率。
    """Modify the likelihood of specified tokens appearing in the completion.

    Accepts a JSON object that maps tokens (specified by their token ID in the
    tokenizer) to an associated bias value from -100 to 100. Mathematically, the
    bias is added to the logits generated by the model prior to sampling. The exact
    effect will vary per model, but values between -1 and 1 should decrease or
    increase likelihood of selection; values like -100 or 100 should result in a ban
    or exclusive selection of the relevant token.
    """

    logprobs: Optional[bool] = False
    #logprobs: 决定是否返回输出标记的对数概率。
    """Whether to return log probabilities of the output tokens or not.

    If true, returns the log probabilities of each output token returned in the
    `content` of `message`. This option is currently not available on the
    `gpt-4-vision-preview` model.
    """

    max_tokens: Optional[int] = None
    #max_tokens: 指定生成的最大标记数。输入和生成的标记总长度受模型的上下文长度限制。
    """The maximum number of [tokens](/tokenizer) to generate in the chat completion.

    The total length of input tokens and generated tokens is limited by the model's
    context length.
    [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
    for counting tokens.
    """

    n: Optional[int] = 1
    #n: 为每条输入消息生成的聊天补全选项数量。
    """How many chat completion choices to generate for each input message."""

    presence_penalty: Optional[float] = 0.
    #presence_penalty: 控制生成文本时对新主题的偏好程度。值越高，模型越倾向于讨论新主题。
    """Number between -2.0 and 2.0.

    Positive values penalize new tokens based on whether they appear in the text so
    far, increasing the model's likelihood to talk about new topics.

    [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)
    """

    response_format: Optional[ResponseFormat] = None
    #response_format: 指定模型输出的格式。
    """An object specifying the format that the model must output.

    Used to enable JSON mode.
    """

    seed: Optional[int] = None
    #seed: 用于随机生成的种子，以便在相同参数下生成相同结果。
    """This feature is in Beta.

    If specified, our system will make a best effort to sample deterministically,
    such that repeated requests with the same `seed` and parameters should return
    the same result. Determinism is not guaranteed, and you should refer to the
    `system_fingerprint` response parameter to monitor changes in the backend.
    """

    stop: Optional[Union[str, List[str]]] = None
    #stop: 指定最多4个序列，遇到这些序列时API将停止生成更多标记。
    """Up to 4 sequences where the API will stop generating further tokens."""

    temperature: Optional[float] = 0.9
    #temperature: 控制采样温度。值越高，输出越随机；值越低，输出越集中。
    """What sampling temperature to use, between 0 and 2.

    Higher values like 0.8 will make the output more random, while lower values like
    0.2 will make it more focused and deterministic.

    We generally recommend altering this or `top_p` but not both.
    """

    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None
    #tool_choice: 控制模型调用哪个函数。
    """
    Controls which (if any) function is called by the model. `none` means the model
    will not call a function and instead generates a message. `auto` means the model
    can pick between generating a message or calling a function. Specifying a
    particular function via
    `{"type: "function", "function": {"name": "my_function"}}` forces the model to
    call that function.

    `none` is the default when no functions are present. `auto` is the default if
    functions are present.
    """

    tools: Optional[List] = None
    #tools: 模型可能调用的工具列表。
    """A list of tools the model may call.

    Currently, only functions are supported as a tool. Use this to provide a list of
    functions the model may generate JSON inputs for.
    """

    top_logprobs: Optional[int] = None
    #top_logprobs: 指定每个标记位置返回的最可能标记数量及其对数概率。
    """
    An integer between 0 and 5 specifying the number of most likely tokens to return
    at each token position, each with an associated log probability. `logprobs` must
    be set to `true` if this parameter is used.
    """

    top_p: Optional[float] = 1.0
    #top_p: 控制模型考虑的标记概率质量。值越小，考虑的标记越少。
    """
    An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass. So 0.1
    means only the tokens comprising the top 10% probability mass are considered.

    We generally recommend altering this or `temperature` but not both.
    """

    user: Optional[str] = None
    #user: 代表终端用户的唯一标识符，有助于监控和检测滥用行为。
    """
    A unique identifier representing your end-user, which can help OpenAI to monitor
    and detect abuse.
    [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
    """

    stream: Optional[bool] = False
    #stream: 决定是否以流式方式发送部分消息。
    """If set, partial message deltas will be sent, like in ChatGPT.

    Tokens will be sent as data-only
    [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
    as they become available, with the stream terminated by a `data: [DONE]`
    message.
    [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).
    """

    # Addictional parameters
    repetition_penalty: Optional[float] = 1.03
    #repetition_penalty: 控制重复词的惩罚程度。1.0表示没有惩罚，值越高，重复词的生成概率越低。
    """The parameter for repetition penalty. 1.0 means no penalty.
    See[this paper](https://arxiv.org / pdf / 1909.05858.pdf) for more details.
    """

    typical_p: Optional[float] = None
    #typical_p: 用于典型解码的质量，控制生成文本的多样性和可读性。
    """Typical Decoding mass.
    See[Typical Decoding for Natural Language Generation](https://arxiv.org / abs / 2202.00666) for more information
    """

    watermark: Optional[bool] = False
    #watermark: 是否在生成文本中使用水印技术，以标识文本来源。
    """Watermarking with [A Watermark for Large Language Models](https://arxiv.org / abs / 2301.10226)
    """

    best_of: Optional[int] = 1
    #best_of: 生成多个候选文本并返回最佳文本的数量。
    ignore_eos: Optional[bool] = False
    #ignore_eos: 是否忽略生成文本中的结束标记
    use_beam_search: Optional[bool] = False
    #use_beam_search: 是否使用束搜索算法生成文本。
    stop_token_ids: Optional[List[int]] = None
    #stop_token_ids: 指定生成文本时停止的标记ID列表。
    skip_special_tokens: Optional[bool] = True
    #skip_special_tokens: 是否跳过生成文本中的特殊标记。
    spaces_between_special_tokens: Optional[bool] = True
    #spaces_between_special_tokens: 是否在特殊标记之间添加空格。
    min_p: Optional[float] = 0.0
    #min_p: 控制生成文本的最小概率质量。
    include_stop_str_in_output: Optional[bool] = False
    #include_stop_str_in_output: 是否在输出中包含停止字符串。
    length_penalty: Optional[float] = 1.0
    #length_penalty: 控制生成文本长度的惩罚程度。
    guided_json: Optional[Union[str, dict, BaseModel]] = None
    #guided_json: 指导生成文本的JSON规范。
    guided_regex: Optional[str] = None
    #guided_regex: 指导生成文本的正则表达式。
    guided_choice: Optional[List[str]] = None
    #guided_choice: 指导生成文本的选择列表。
    guided_grammar: Optional[str] = None
    #guided_grammar: 指导生成文本的语法规则
    guided_decoding_backend: Optional[str] = None
    #guided_decoding_backend: 指导解码的后端实现。

#CompletionCreateParams 类是用于定义文本补全请求参数的类。它继承自 Pydantic 的 BaseModel，允许我们轻松地进行数据验证和转换。
class CompletionCreateParams(BaseModel):
    model: str
    """ID of the model to use.

    You can use the
    [List models](https://platform.openai.com/docs/api-reference/models/list) API to
    see all of your available models, or see our
    [Model overview](https://platform.openai.com/docs/models/overview) for
    descriptions of them.
    """

    prompt: Union[str, List[str], List[int], List[List[int]], None]
    """
    The prompt(s) to generate completions for, encoded as a string, array of
    strings, array of tokens, or array of token arrays.

    Note that <|endoftext|> is the document separator that the model sees during
    training, so if a prompt is not specified the model will generate as if from the
    beginning of a new document.
    """

    best_of: Optional[int] = 1
    """
    Generates `best_of` completions server-side and returns the "best" (the one with
    the highest log probability per token). Results cannot be streamed.

    When used with `n`, `best_of` controls the number of candidate completions and
    `n` specifies how many to return – `best_of` must be greater than `n`.

    **Note:** Because this parameter generates many completions, it can quickly
    consume your token quota. Use carefully and ensure that you have reasonable
    settings for `max_tokens` and `stop`.
    """

    echo: Optional[bool] = False
    """Echo back the prompt in addition to the completion"""

    frequency_penalty: Optional[float] = 0.
    """Number between -2.0 and 2.0.

    Positive values penalize new tokens based on their existing frequency in the
    text so far, decreasing the model's likelihood to repeat the same line verbatim.

    [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)
    """

    logit_bias: Optional[Dict[str, int]] = None
    """Modify the likelihood of specified tokens appearing in the completion.

    Accepts a JSON object that maps tokens (specified by their token ID in the GPT
    tokenizer) to an associated bias value from -100 to 100. You can use this
    [tokenizer tool](/tokenizer?view=bpe) (which works for both GPT-2 and GPT-3) to
    convert text to token IDs. Mathematically, the bias is added to the logits
    generated by the model prior to sampling. The exact effect will vary per model,
    but values between -1 and 1 should decrease or increase likelihood of selection;
    values like -100 or 100 should result in a ban or exclusive selection of the
    relevant token.

    As an example, you can pass `{"50256": -100}` to prevent the <|endoftext|> token
    from being generated.
    """

    logprobs: Optional[int] = None
    """
    Include the log probabilities on the `logprobs` most likely tokens, as well the
    chosen tokens. For example, if `logprobs` is 5, the API will return a list of
    the 5 most likely tokens. The API will always return the `logprob` of the
    sampled token, so there may be up to `logprobs+1` elements in the response.

    The maximum value for `logprobs` is 5.
    """

    max_tokens: Optional[int] = 16
    """The maximum number of [tokens](/tokenizer) to generate in the completion.

    The token count of your prompt plus `max_tokens` cannot exceed the model's
    context length.
    [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
    for counting tokens.
    """

    n: Optional[int] = 1
    """How many completions to generate for each prompt.

    **Note:** Because this parameter generates many completions, it can quickly
    consume your token quota. Use carefully and ensure that you have reasonable
    settings for `max_tokens` and `stop`.
    """

    presence_penalty: Optional[float] = 0.
    """Number between -2.0 and 2.0.

    Positive values penalize new tokens based on whether they appear in the text so
    far, increasing the model's likelihood to talk about new topics.

    [See more information about frequency and presence penalties.](https://platform.openai.com/docs/guides/gpt/parameter-details)
    """

    seed: Optional[int] = None
    """
    If specified, our system will make a best effort to sample deterministically,
    such that repeated requests with the same `seed` and parameters should return
    the same result.

    Determinism is not guaranteed, and you should refer to the `system_fingerprint`
    response parameter to monitor changes in the backend.
    """

    stop: Optional[Union[str, List[str]]] = None
    """Up to 4 sequences where the API will stop generating further tokens.

    The returned text will not contain the stop sequence.
    """

    suffix: Optional[str] = None
    """The suffix that comes after a completion of inserted text."""

    temperature: Optional[float] = 1.
    """What sampling temperature to use, between 0 and 2.

    Higher values like 0.8 will make the output more random, while lower values like
    0.2 will make it more focused and deterministic.

    We generally recommend altering this or `top_p` but not both.
    """

    tool_choice: Optional[ChatCompletionToolChoiceOptionParam] = None
    """
    Controls which (if any) function is called by the model. `none` means the model
    will not call a function and instead generates a message. `auto` means the model
    can pick between generating a message or calling a function. Specifying a
    particular function via
    `{"type: "function", "function": {"name": "my_function"}}` forces the model to
    call that function.

    `none` is the default when no functions are present. `auto` is the default if
    functions are present.
    """

    tools: Optional[List] = None
    """A list of tools the model may call.

    Currently, only functions are supported as a tool. Use this to provide a list of
    functions the model may generate JSON inputs for.
    """

    top_p: Optional[float] = 1.
    """
    An alternative to sampling with temperature, called nucleus sampling, where the
    model considers the results of the tokens with top_p probability mass. So 0.1
    means only the tokens comprising the top 10% probability mass are considered.

    We generally recommend altering this or `temperature` but not both.
    """

    user: Optional[str] = None
    """
    A unique identifier representing your end-user, which can help OpenAI to monitor
    and detect abuse.
    [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
    """

    stream: Optional[bool] = False
    """If set, partial message deltas will be sent, like in ChatGPT.

    Tokens will be sent as data-only
    [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format)
    as they become available, with the stream terminated by a `data: [DONE]`
    message.
    [Example Python code](https://cookbook.openai.com/examples/how_to_stream_completions).
    """

    # Addictional parameters
    repetition_penalty: Optional[float] = 1.03
    """The parameter for repetition penalty. 1.0 means no penalty.
    See[this paper](https://arxiv.org / pdf / 1909.05858.pdf) for more details.
    """

    typical_p: Optional[float] = None
    """Typical Decoding mass.
    See[Typical Decoding for Natural Language Generation](https://arxiv.org / abs / 2202.00666) for more information
    """

    watermark: Optional[bool] = False
    """Watermarking with [A Watermark for Large Language Models](https://arxiv.org / abs / 2301.10226)
    """

    response_format: Optional[ResponseFormat] = None
    """An object specifying the format that the model must output.

    Used to enable JSON mode.
    """

    ignore_eos: Optional[bool] = False

    use_beam_search: Optional[bool] = False

    stop_token_ids: Optional[List[int]] = None

    skip_special_tokens: Optional[bool] = True

    spaces_between_special_tokens: Optional[bool] = True

    min_p: Optional[float] = 0.0

    include_stop_str_in_output: Optional[bool] = False

    length_penalty: Optional[float] = 1.0

    guided_json: Optional[Union[str, dict, BaseModel]] = None

    guided_regex: Optional[str] = None

    guided_choice: Optional[List[str]] = None

    guided_grammar: Optional[str] = None

    guided_decoding_backend: Optional[str] = None

#EmbeddingCreateParams用于创建嵌入（embedding）的参数。这个类继承自 BaseModel，可能是使用 Pydantic 或类似库来进行数据验证和序列化。
class EmbeddingCreateParams(BaseModel):
    input: Union[str, List[str], List[int], List[List[int]]]#支持多种格式，以便灵活地处理不同类型的输入。
    #这是要嵌入的输入文本，可以是单个字符串、字符串数组、整数数组或嵌套的整数数组（例如，token 的数组）。嵌入的输入不能超过模型的最大输入限制（对于 text-embedding-ada-002 是 8192 tokens），且不能为空字符串。
    """Input text to embed, encoded as a string or array of tokens.

    To embed multiple inputs in a single request, pass an array of strings or array
    of token arrays. The input must not exceed the max input tokens for the model
    (8192 tokens for `text-embedding-ada-002`) and cannot be an empty string.
    [Example Python code](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken)
    for counting tokens.
    """

    model: str#指定要使用的嵌入模型。
    #要使用的模型的 ID。用户可以通过 API 获取可用模型的列表，或查看模型概述以了解模型的具体描述
    """ID of the model to use.

    You can use the
    [List models](https://platform.openai.com/docs/api-reference/models/list) API to
    see all of your available models, or see our
    [Model overview](https://platform.openai.com/docs/models/overview) for
    descriptions of them.
    """

    encoding_format: Literal["float", "base64"] = "float"#根据需求选择合适的格式，便于后续处理。
    #指定返回嵌入的格式，可以是浮点数或 base64 编码格式。
    """The format to return the embeddings in.

    Can be either `float` or [`base64`](https://pypi.org/project/pybase64/).
    """

    dimensions: Optional[int] = None#允许用户指定想要的嵌入维度，从而获取符合需求的嵌入表示。
    #输出嵌入的维度数，仅在 text-embedding-3 及以后的模型中支持。
    """The number of dimensions the resulting output embeddings should have.

    Only supported in `text-embedding-3` and later models.
    """

    user: Optional[str] = None#在某些情况下，提供用户标识符可以增强安全性和可追溯性。
    #代表最终用户的唯一标识符，有助于 OpenAI 监控和检测滥用行为。
    """
    A unique identifier representing your end-user, which can help OpenAI to monitor
    and detect abuse.
    [Learn more](https://platform.openai.com/docs/guides/safety-best-practices/end-user-ids).
    """

#Embedding 类是一个用于定义嵌入向量的类。它继承自 Pydantic 的 BaseModel，用于结构化地表示嵌入数据。
class Embedding(BaseModel):
    embedding: Any
    #embedding: 这是嵌入向量，通常是一个浮点数列表。向量的长度取决于所使用的模型。嵌入向量用于表示输入文本在模型中的特征空间位置。
    """The embedding vector, which is a list of floats.

    The length of vector depends on the model as listed in the
    [embedding guide](https://platform.openai.com/docs/guides/embeddings).
    """

    index: int
    #index: 嵌入在嵌入列表中的索引。用于标识该嵌入在批处理请求中的位置。
    """The index of the embedding in the list of embeddings."""

    object: Literal["embedding"]
    #object: 对象类型，恒定为"embedding"。用于明确地标识对象的类型为嵌入。
    """The object type, which is always "embedding"."""

#CreateEmbeddingResponse 类是用于定义嵌入响应的类。它继承自 Pydantic 的 BaseModel，用于结构化地表示生成嵌入后的响应数据。
class CreateEmbeddingResponse(BaseModel):
    data: List[Embedding]
    #data: 这是一个 Embedding 对象的列表，表示由模型生成的嵌入。每个 Embedding 包含向量数据和相关的元信息。
    """The list of embeddings generated by the model."""

    model: str
    #model: 生成嵌入时所使用的模型名称。这可以帮助确定嵌入的来源和特性。
    """The name of the model used to generate the embedding."""

    object: Literal["list"]
    #object: 对象类型，恒定为"list"。用于明确地标识该响应的类型为一个列表。
    """The object type, which is always "list"."""

    usage: Usage
    #usage: 请求的使用信息，通常包括请求中用到的资源量或次数等指标。这可以帮助用户了解和监控API的使用情况。
    """The usage information for the request."""

#RerankRequest 类的作用是定义向 OpenAI API 发送的重排序请求的结构。该类包含了执行文档重排序所需的所有信息，包括要使用的模型、查询文本、待重排序的文档列表以及其他可选参数。
class RerankRequest(BaseModel):
    model: str#指定用于重排序的模型名称。
    # 明确使用的算法或模型，这对于结果的准确性和性能至关重要。
    """The name of the model used to rerank."""

    query: str#提供用于重排序的查询文本。
    #这个查询是重排序的基础，模型将根据此查询对给定文档进行排名。
    """The query for rerank."""

    documents: List[str]#包含要进行重排序的文档列表。
    #提供待重排序的文档内容，以便模型分析和评估相关性。
    """The documents for rerank."""

    top_n: Optional[int] = None#可选参数，指示希望返回的最佳文档数量。
    #允许用户指定想要获取的结果数量，方便在处理大型文档集时进行优化。

    return_documents: Optional[bool] = False#可选参数，指示是否在响应中返回文档。
    #根据用户需要控制返回内容的详细程度，提升灵活性。


class Document(TypedDict):#定义了一个文档的基本结构。
    text: str#表示文档的文本内容。
#这个类封装了单个文档的文本信息。它用于在其他数据结构中表示具体的文档，尤其是在重排序操作中需要处理的文档。

class DocumentObj(TypedDict):# 定义一个包含重排序结果的文档对象的结构。
    index: int#表示文档在原始输入列表中的索引位置。
    relevance_score: float#表示文档与查询的相关性得分，通常是模型计算得出的分数。
    document: Optional[Document]#可选属性，包含一个 Document 对象，代表具体的文档信息。
    #该类用于在重排序的结果中表示每个文档的详细信息，包括其在原始列表中的位置、与查询的相关性以及文档的内容。这对于评估和分析重排序结果非常有用。


class RerankResponse(TypedDict):#定义重排序操作的响应结构。
    id: Optional[str]#可选属性，表示请求的唯一标识符，便于跟踪和调试。
    results: List[DocumentObj]#包含重排序结果的文档对象列表，每个对象都是 DocumentObj 类型。
    #这个类用于表示 API 返回的重排序结果。通过 results 属性，用户可以获取到所有重排序的文档及其相关信息，而 id 属性则可以用于识别特定请求。这种结构化响应使得处理和分析重排序结果变得更加清晰和方便。
