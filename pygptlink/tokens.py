from typing import Any, Sequence
from pygptlink.logging import logger
import tiktoken
import lmstudio as lms
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from openai.types.shared_params import FunctionDefinition, FunctionParameters
from lmstudio import AnyChatMessageDict, ChatHistoryDataDict

# Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb


def message_token_constants(model: str) -> tuple[int, int]:
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06",
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        return message_token_constants(model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        return message_token_constants(model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        return message_token_constants(model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        return message_token_constants(model="gpt-4-0613")
    else:
        # Print error and assume the values are unchanged. Don't stop production just to make an uneducated guess.
        logger.error(f"message_token_constants() is not implemented for model {model}, assuming same as gpt-4o.")
        return message_token_constants("gpt-4o")
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
        return tool_token_constants(model="gpt-4")
    return func_init, prop_init, prop_key, enum_init, enum_item, func_end


def num_tokens_for_messages_lms(messages: Sequence[AnyChatMessageDict], model: str) -> int:
    loaded = {k.get_info().model_key for k in lms.list_loaded_models("llm")}
    if model not in loaded:
        raise ValueError(f"Model {model} is not loaded in LMS. Please load the model before counting tokens.")
    llm = lms.llm(model)
    history: ChatHistoryDataDict = {"messages": messages}
    formatted = llm.apply_prompt_template(history)
    return len(llm.tokenize(formatted))


def num_messages_tokens_openai(messages: Sequence[ChatCompletionMessageParam], model: str) -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")

    tokens_per_message, tokens_per_name = message_token_constants(model)

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
    func_init, prop_init, prop_key, enum_init, enum_item, func_end = tool_token_constants(model)
    enc = tiktoken.encoding_for_model(model)

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
