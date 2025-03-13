import asyncio
from collections.abc import Callable
from typing import Any, Optional

from pygptlink.gpt_context import GPTContext
from pygptlink.gpt_no_response_desired import GPTNoResponseDesired
from pygptlink.gpt_tokens import num_tokens_for_tools
from pygptlink.gpt_tool_definition import GPTToolDefinition
from pygptlink.gpt_logging import logger
from pygptlink.sentenceextractor import SentenceExtractor


import lmstudio as lms


class LMSCompletion:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.sentence_extractor = SentenceExtractor()

    def complete(self, context: GPTContext,
                 callback: Callable[[str, bool], None] | None = None,
                 extra_system_prompt: str | None = None,
                 tools: list[GPTToolDefinition] = [],
                 force_tool: bool = False,
                 allowed_tools: list[str] | None = None,
                 no_append: bool = False) -> str:
        """Generates a response to the current context.

        Args:
            context (GPTContext): A context to perform a completion on.
            extra_system_prompt (Optional[str]): An extra system prompt to be injected into the context, can be None.
            callback (Optional[Callable[[str], None]]): A callback to call with the response, line by line.
            force_tool (str | bool | None): The name of a tool that must be called by the model. Defaults to None. Can be True to force any tool.
            allowed_tools (List[str], optional): A list of tools that the model may call. None means any tool it
                                                 knows of, [] means no tools may be called, or a list of named
                                                 tools. Defaults to None.
            no_append (bool): False by default. If True, then the result of the completion will not be added to the context.

        Raises:
            ValueError: When inputs are invalid.

        Returns:
            Nothing
        """

        tools_map = {tool.name: tool for tool in tools}
        completion_settings = {
            "maxTokens": context.max_response_tokens,
            "temperature": 0.4,
            "cpuThreads": 20,
        }

        if allowed_tools == None:
            tool_defs = [
                tool.describe() for tool in tools_map.values()] if tools_map else None
        elif allowed_tools == []:
            tool_defs = None
        else:
            if not all(tool_name in tools_map for tool_name in allowed_tools):
                raise ValueError(
                    "allowed_tools={allowed_tools} contains unknown tool")
            tool_defs = [tools_map[tool_name].describe()
                         for tool_name in allowed_tools]

        if isinstance(force_tool, bool) and force_tool:
            tool_choice = "required"
            if not tool_defs:
                raise ValueError(
                    "Tool call forced but no tools allowed or available!")
        else:
            tool_choice = "auto" if tools_map else None

        # Prepare arguments for completion
        # Load model first so the token count will work
        model = lms.llm(context.model, config={
            "contextLength": context.max_tokens,
            "gpuOffload": {
                "ratio": 1.0,
                "splitStrategy": "favorMainGpu"
            }
        })
        tool_tokens = num_tokens_for_tools(
            functions=tool_defs, model=context.model)
        messages = context.messages(
            sticky_system_message=extra_system_prompt, reserved_tokens=tool_tokens, lms=True)

        def on_fragment(fragment: lms.LlmPredictionFragment, round_index: int = 0):
            if callback:
                callback(fragment.content, False)

        def on_message(msg: lms.AssistantResponse | lms.ToolResultMessage):
            if not no_append:
                context.append_lms_completion(msg)

        logger.debug(f"Prompting {model} with: {messages}")

        result = model.respond(history={"messages": messages}, config=completion_settings,
                               on_prediction_fragment=on_fragment, on_message=on_message)
        if callback:
            callback("", True)
        return result.content
