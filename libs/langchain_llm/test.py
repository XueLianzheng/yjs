from langchain_llm import HuggingFaceLLM, ChatHuggingFace, VLLM, ChatVLLM

#该文件提供了如何使用HuggingFace和VLLM进行测试的示例。

def test_huggingface():#该函数测试 Hugging Face 语言模型的不同调用方式。
    #创建 Hugging Face 语言模型实例
    llm = HuggingFaceLLM(#HuggingFaceLLM 是用于加载 Hugging Face 模型的类。
        model_name="qwen-7b-chat",#model_name 指定了使用的模型名称。
        model_path="/data/checkpoints/Qwen-7B-Chat",#model_path 指向模型的存储路径。
        load_model_kwargs={"device_map": "auto"},
        #load_model_kwargs 是加载模型时的参数，这里设置了 device_map 为 auto，这意味着模型会自动分配到可用的设备上（如 GPU）。
    )

    #调用模型

    # invoke method
    #普通调用
    #使用 invoke 方法直接调用模型，传入一个带有用户问题的 prompt，并指定结束标记。
    prompt = "<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n"
    print(llm.invoke(prompt, stop=["<|im_end|>"]))

    # Token Streaming
    #使用 stream 方法进行流式输出，逐块打印响应。
    for chunk in llm.stream(prompt, stop=["<|im_end|>"]):
        print(chunk, end="", flush=True)

    #OpenAI 风格的调用
    # openai usage
    print(llm.call_as_openai(prompt, stop=["<|im_end|>"]))
    #使用 call_as_openai 方法模拟 OpenAI 的调用风格，传入相同的 prompt。

    #OpenAI 流式调用
    # Streaming
    for chunk in llm.call_as_openai(prompt, stop=["<|im_end|>"], stream=True):
        print(chunk.choices[0].text, end="", flush=True)
    #同样使用 call_as_openai，但启用流式响应，逐块输出生成的文本。

    #使用 ChatHuggingFace
    chat_llm = ChatHuggingFace(llm=llm)
    #创建一个 ChatHuggingFace 实例，传入刚刚创建的 llm。

    #ChatHuggingFace 调用
    # invoke method
    query = "你是谁？"
    print(chat_llm.invoke(query))
    #直接调用聊天模型的 invoke 方法。

    # Token Streaming
    for chunk in chat_llm.stream(query):#使用 stream 方法逐块输出响应。
        print(chunk.content, end="", flush=True)

    #OpenAI 风格的调用
    # openai usage
    #构建消息格式，调用 call_as_openai 方法。
    messages = [
        {"role": "user", "content": query}
    ]
    print(chat_llm.call_as_openai(messages))

    #OpenAI 流式调用
    # Streaming
    for chunk in chat_llm.call_as_openai(messages, stream=True):
        print(chunk.choices[0].delta.content or "", end="", flush=True)
    #流式调用与前面的类似，逐块输出聊天内容。

def test_vllm():#该函数测试 VLLM 语言模型的不同调用方式。
    llm = VLLM(#VLLM 是用于加载 VLLM 模型的类。
        model_name="qwen",
        model="/data/checkpoints/Qwen-7B-Chat",
        #model_name 和 model 参数指定了模型的名称和路径。
        trust_remote_code=True,
        #trust_remote_code=True 表示信任从远程加载的代码，可能用于模型加载时的配置。
    )

    #调用模型
    # invoke method
    #普通调用
    prompt = "<|im_start|>user\n你是谁？<|im_end|>\n<|im_start|>assistant\n"
    print(llm.invoke(prompt, stop=["<|im_end|>"]))
    #类似于 Hugging Face 的调用，直接传入 prompt。

    #OpenAI 风格的调用
    # openai usage
    print(llm.call_as_openai(prompt, stop=["<|im_end|>"]))
    #同样调用 call_as_openai。

    #使用 ChatVLLM
    chat_llm = ChatVLLM(llm=llm)
    #创建一个 ChatVLLM 实例。

    #ChatVLLM 调用
    # invoke method
    query = "你是谁？"
    print(chat_llm.invoke(query))
    #直接调用聊天模型的 invoke 方法。

    #OpenAI 风格的调用
    # openai usage
    messages = [
        {"role": "user", "content": query}
    ]
    print(chat_llm.call_as_openai(messages))
    #使用消息格式调用 call_as_openai。

if __name__ == "__main__":#在脚本直接运行时，调用 test_huggingface() 函数。
    test_huggingface()
