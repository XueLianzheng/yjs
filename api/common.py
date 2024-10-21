from __future__ import annotations

import os
from typing import Any, Dict, Type

import pydantic
from pydantic import BaseModel

PYDANTIC_V2 = pydantic.VERSION.startswith("2.")
#该变量用于检测 pydantic 的版本，检查Pydantic版本是否为2.x，以决定后续代码中使用不同版本特性。

# 该文件包含了一些通用函数，用于处理数据模型和环境变量。

def dictify(data: "BaseModel", **kwargs) -> Dict[str, Any]:
    try:  # pydantic v2
        return data.model_dump(**kwargs)
    except AttributeError:  # pydantic v1
        return data.dict(**kwargs)
# 该函数将 pydantic 模型实例转换为字典。它根据 pydantic 的版本调用不同的方法。
# model_dump 是 pydantic v2 中的方法，而 dict 是 v1 中的方法。

def jsonify(data: "BaseModel", **kwargs) -> str:
    try:  # pydantic v2
        return data.model_dump_json(**kwargs)
    except AttributeError:  # pydantic v1
        return data.json(**kwargs)
# 该函数将 pydantic 模型实例转换为 JSON 字符串，同样地根据 pydantic 版本选择调用的方法。
# model_dump_json 是 v2 中的方法，json 是 v1 中的方法。

def model_validate(data: Type["BaseModel"], obj: Any) -> "BaseModel":
    try:  # pydantic v2
        return data.model_validate(obj)
    except AttributeError:  # pydantic v1
        return data.parse_obj(obj)
# 该函数用于验证和解析输入数据对象，返回一个 pydantic 模型实例。
# model_validate 和 parse_obj 分别是 v2 和 v1 中的方法。

def disable_warnings(model: Type["BaseModel"]):
    # Disable warning for model_name settings
    if PYDANTIC_V2:
        model.model_config["protected_namespaces"] = ()
# 该函数用于禁用 pydantic 模型的名称设置相关的警告。
# 在 pydantic v2 中，通过设置 model_config 的 protected_namespaces 属性来实现。

def get_bool_env(key, default="false"):
    return os.environ.get(key, default).lower() == "true"
# 该函数从环境变量中获取布尔值。若环境变量为 "true" 则返回 True，否则返回 False。
# os.environ.get用于访问环境变量。它的作用是获取指定环境变量的值，如果该环境变量不存在，则返回一个默认值（如果提供了的话）。
# lower()用于将字符串中的所有大写字母转换为小写字母。该方法不会修改原始字符串，而是返回一个新的字符串。

def get_env(key, default):
    val = os.environ.get(key, "")
    return val or default
#该函数从环境变量中获取指定键的值。如果没有找到该键，则返回默认值。
