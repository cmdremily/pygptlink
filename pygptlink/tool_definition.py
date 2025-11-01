import inspect
import json
from typing import Any, Callable, Literal, NotRequired, TypedDict

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from openai.types.chat.chat_completion_tool_param import ChatCompletionToolParam
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

    async def invoke(self, tool_call: ChatCompletionMessageToolCall) -> tuple[str, bool]:
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

            if not ans:
                return ("Success.", True)
            elif isinstance(ans, NoResponseDesired):
                return ("Success. Do not respond to this tool call.", False)
            else:
                return (ans, True)
        except Exception as e:
            return (f"An exception ocurred: {e}", True)

    def describe_openai(self) -> ChatCompletionToolParam:
        # OpenAI API requires the "additionalProperties" field to be False when
        # strict is True, meaning that the "Parameters" object must be present.
        all_args = self.required_args + self.optional_args
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "strict": True,
                "parameters": {
                    "type": "object",
                    "properties": {t["name"]: {"type": t["type"], "description": t["description"]} for t in all_args},
                    "required": [t["name"] for t in (self.required_args or [])],
                    "additionalProperties": False,
                },
            },
        }
