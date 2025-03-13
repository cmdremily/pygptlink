import json
from typing import Any, Callable

from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall
from pygptlink.gpt_logging import logger
from pygptlink.gpt_no_response_desired import GPTNoResponseDesired


class GPTToolDefinition:
    def __init__(self, name: str, callback: Callable, description: str,
                 required_args: list[dict[str, str]] = [],
                 optional_args: list[dict[str, str]] = []):
        self.name = name
        self.description = description
        self.callback = callback
        self.required_args = required_args
        self.optional_args = optional_args

    async def invoke(self, tool_call: ChatCompletionMessageToolCall) -> str:
        assert self.name == tool_call.function.name

        try:
            kwargs = json.loads(tool_call.function.arguments)
            missing_args = [arg["name"]
                            for arg in self.required_args if arg["name"] not in kwargs]
            if missing_args:
                raise ValueError(f"Missing required arguments: {missing_args}")

            ans = await self.callback(**kwargs)
            if ans is None:
                rv = "The tool call completed successfully."
            elif isinstance(ans, GPTNoResponseDesired):
                return ans
            else:
                rv = f"The tool call returned: {ans}"
        except Exception as e:
            rv = f"The tool call threw an exception: {e}"

        logger.debug(f"Tool invocation completed with: {rv}")
        return rv

    def describe(self) -> dict[str, Any]:
        if self.required_args or self.optional_args:
            all_args = self.required_args + self.optional_args
            return {
                "type": "function",
                "function": {
                    "name": self.name,
                    "description": self.description,
                    "parameters": {
                        "type": "object",
                        "properties": {t["name"]: {"type": t["type"], "description": t["description"]} for t in all_args},
                        "required": [t["name"] for t in (self.required_args or [])],
                    },
                }
            }
        else:
            return {
                "type": "function",
                "function": {
                        "name": self.name,
                        "description": self.description
                }
            }
