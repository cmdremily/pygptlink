from typing import Any, Sequence
from pygptlink.logging import logger
import tiktoken
import lmstudio as lms
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition, FunctionParameters
from lmstudio import Chat

# Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb


def message_token_constants(model: str) -> tuple[int, int]:
    # For now, all relevant models have the same token constants.
    tokens_per_message = 3
    tokens_per_name = 1
    return tokens_per_message, tokens_per_name


def tool_token_constants(model: str) -> tuple[int, int, int, int, int, int]:
    # Initialize function settings to 0
    func_init = 0
    prop_init = 0
    prop_key = 0
    enum_init = 0
    enum_item = 0
    func_end = 0

    if model in ["gpt-4o", "gpt-4o-mini"]:
        # Set function settings for the above models
        func_init = 7
        prop_init = 3
        prop_key = 3
        enum_init = -3
        enum_item = 3
        func_end = 12
    elif model in ["gpt-3.5-turbo", "gpt-4"]:
        # Set function settings for the above models
        func_init = 10
        prop_init = 3
        prop_key = 3
        enum_init = -3
        enum_item = 3
        func_end = 12
    else:
        # We pick the model with the most tokens as the fallback.
        logger.error(f"tool_token_constants() is not implemented for model {model}. Assuming gpt-4 token counts.")
        return tool_token_constants(model="gpt-4o")
    return func_init, prop_init, prop_key, enum_init, enum_item, func_end


def lms_model_loaded(model: str) -> bool:
    """Check if the specified model is loaded in LMS."""
    loaded = {k.get_info().model_key for k in lms.list_loaded_models("llm")}
    return model in loaded


def num_tokens_for_messages_lms(chat: Chat, model: str) -> int:
    assert lms_model_loaded(model), f"Model {model} must be loaded before counting tokens."
    llm = lms.llm(model)
    formatted = llm.apply_prompt_template(chat)
    return len(llm.tokenize(formatted))


def _get_openai_model(model: str) -> str:
    # Tiktoken hasn't updated for gpt-4.1 yet, so we need to map it to gpt-4o.
    if "gpt-4.1" in model:
        return "gpt-4o"
    if "gpt-5" in model:
        return "gpt-4o"
    return model


def num_messages_tokens_openai(messages: Sequence[ChatCompletionMessageParam], model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(_get_openai_model(model))
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")

    tokens_per_message, tokens_per_name = message_token_constants(_get_openai_model(model))

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message

        if "tool_calls" in message:
            for tool_call in message.get("tool_calls", []):
                num_tokens += 3  # Stab in the dark guess at token count for tool calls.
                num_tokens += len(encoding.encode(tool_call["id"]))
                num_tokens += len(encoding.encode(tool_call["type"]))
                num_tokens += len(encoding.encode(tool_call["function"]["name"]))
                num_tokens += len(encoding.encode(tool_call["function"]["arguments"]))

        for key, value in message.items():
            if key == "tool_calls":
                continue
            if not value:
                continue
            if not isinstance(value, str):
                raise ValueError(
                    f"Expected string value for key '{key}' in message, got {type(value).__name__} instead."
                )
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def num_tool_tokens_openai(tools: list[ChatCompletionToolParam], model: str) -> int:
    func_init, prop_init, prop_key, enum_init, enum_item, func_end = tool_token_constants(_get_openai_model(model))
    enc = tiktoken.encoding_for_model(_get_openai_model(model))

    ans = 0
    if len(tools or []) > 0:
        for tool in tools:
            f: FunctionDefinition = tool["function"]
            ans += func_init
            ans += len(enc.encode(f"{f["name"]}:{f.get("description", "").removesuffix(".")}"))
            f_param: FunctionParameters = f.get("parameters", {})
            f_properties: dict[str, Any] = f_param.get("properties", {})  # type: ignore
            if len(f_properties) > 0:
                ans += prop_init  # Add tokens for start of each property
                for key in list(f_properties.keys()):
                    ans += prop_key  # Add tokens for each set property
                    p_name = key
                    p_type = f_properties[key]["type"]
                    p_desc = f_properties[key]["description"]
                    if "enum" in f_properties[key].keys():
                        ans += enum_init  # Add tokens if property has enum list
                        for item in f_properties[key]["enum"]:
                            ans += enum_item
                            ans += len(enc.encode(item))
                    if p_desc.endswith("."):
                        p_desc = p_desc[:-1]
                    line = f"{p_name}:{p_type}:{p_desc}"
                    ans += len(enc.encode(line))
        ans += func_end
    return ans
