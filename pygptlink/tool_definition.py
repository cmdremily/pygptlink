import inspect
import json
from typing import Any, Callable, Literal, NotRequired, TypedDict

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
from pygptlink.logging import logger
from pygptlink.no_response_desired import NoResponseDesired


class ArgSpec(TypedDict):
    name: str
    type: str
    description: str


class FunctionParameters(TypedDict):
    type: Literal["object"]
    properties: dict[str, dict[str, str]]
    required: list[str]
    additionalProperties: Literal[False]


class ToolDescriptionOpenAI(TypedDict):
    type: Literal["function"]
    name: str
    description: str
    strict: Literal[True]
    parameters: NotRequired[FunctionParameters]


class ToolDefinition:
    def __init__(
        self,
        name: str,
        callback: Callable[..., Any],
        description: str,
        required_args: list[ArgSpec] | None = None,
        optional_args: list[ArgSpec] | None = None,
    ):
        self.name = name
        self.description = description
        self.callback = callback
        self.required_args = required_args or []
        self.optional_args = optional_args or []

    async def invoke(self, tool_call: ChatCompletionMessageToolCall) -> str | NoResponseDesired:
        assert self.name == tool_call.function.name

        try:
            kwargs = json.loads(tool_call.function.arguments)
            missing_args = [arg["name"] for arg in self.required_args if arg["name"] not in kwargs]
            if missing_args:
                raise ValueError(f"Missing required arguments: {missing_args}")

            result = self.callback(**kwargs)
            if inspect.isawaitable(result):
                ans = await result
            else:
                ans = result

            if ans is None:
                rv = "The tool call completed successfully."
            elif isinstance(ans, NoResponseDesired):
                return ans
            else:
                rv = f"The tool call returned: {ans}"
        except Exception as e:
            rv = f"The tool call threw an exception: {e}"

        logger.debug(f"Tool invocation completed with: {rv}")
        return rv

    def describe_openai(self) -> ChatCompletionToolParam:
        desc: ChatCompletionToolParam = {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "strict": True},
        }
        if self.required_args or self.optional_args:
            all_args = self.required_args + self.optional_args
            desc["function"]["parameters"] = {
                "type": "object",
                "properties": {t["name"]: {"type": t["type"], "description": t["description"]} for t in all_args},
                "required": [t["name"] for t in (self.required_args or [])],
                "additionalProperties": False,
            }
        return desc
