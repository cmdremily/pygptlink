from pygptlink.gpt_logging import logger
import tiktoken
import lmstudio as lms

# Adapted from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb


def message_token_constants(model: str) -> tuple[int, int]:
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
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
        logger.error(
            f"""message_token_constants() is not implemented for model {model}, assuming same as gpt-4o.""")
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

    if model in [
        "gpt-4o",
        "gpt-4o-mini"
    ]:
        # Set function settings for the above models
        func_init = 7
        prop_init = 3
        prop_key = 3
        enum_init = -3
        enum_item = 3
        func_end = 12
    elif model in [
        "gpt-3.5-turbo",
        "gpt-4"
    ]:
        # Set function settings for the above models
        func_init = 10
        prop_init = 3
        prop_key = 3
        enum_init = -3
        enum_item = 3
        func_end = 12
    else:
        # We pick the model with the most tokens as the fallback.
        logger.error(
            f"""tool_token_constants() is not implemented for model {model}. Assuming gpt-4 token counts."""
        )
        return tool_token_constants(model="gpt-4")
    return func_init, prop_init, prop_key, enum_init, enum_item, func_end


def num_tokens_for_messages(messages: list[dict], model: str) -> int:
    loaded = {k.get_info().model_key for k in lms.list_loaded_models("llm")}
    if model in loaded:
        llm = lms.llm(model)
        formatted = llm.apply_prompt_template({"messages": messages})
        return len(llm.tokenize(formatted))

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")

    tokens_per_message, tokens_per_name = message_token_constants(model)

    def recursive_iterate(dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                yield from recursive_iterate(value)
            else:
                yield key, value

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            if key == "tool_calls":
                for tool_call in value:
                    for key, value in recursive_iterate(tool_call):
                        num_tokens += len(encoding.encode(value))
            else:
                num_tokens += len(encoding.encode(value))
                if key == "name":
                    num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


def num_tokens_for_tools(functions, model) -> int:
    loaded = {k.get_info().model_key for k in lms.list_loaded_models("llm")}
    if model in loaded:
        return 0

    func_init, prop_init, prop_key, enum_init, enum_item, func_end = tool_token_constants(
        model)

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")

    func_token_count = 0
    if len(functions or []) > 0:
        for f in functions:
            func_token_count += func_init  # Add tokens for start of each function
            function = f["function"]
            f_name = function["name"]
            f_desc = function["description"]
            f_param = function.get("parameters", {})
            if f_desc.endswith("."):
                f_desc = f_desc[:-1]
            line = f_name + ":" + f_desc
            # Add tokens for set name and description
            func_token_count += len(encoding.encode(line))
            if len(f_param) and len(f_param["properties"]) > 0:
                func_token_count += prop_init  # Add tokens for start of each property
                for key in list(f_param["properties"].keys()):
                    func_token_count += prop_key  # Add tokens for each set property
                    p_name = key
                    p_type = f_param["properties"][key]["type"]
                    p_desc = f_param["properties"][key]["description"]
                    if "enum" in f_param["properties"][key].keys():
                        func_token_count += enum_init  # Add tokens if property has enum list
                        for item in f_param["properties"][key]["enum"]:
                            func_token_count += enum_item
                            func_token_count += len(encoding.encode(item))
                    if p_desc.endswith("."):
                        p_desc = p_desc[:-1]
                    line = f"{p_name}:{p_type}:{p_desc}"
                    func_token_count += len(encoding.encode(line))
        func_token_count += func_end
    return func_token_count
