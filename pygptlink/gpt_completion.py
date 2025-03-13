import asyncio
from collections.abc import Callable
import time
from typing import Any, Optional

import openai
from openai import AsyncOpenAI
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk
from pygptlink.gpt_context import GPTContext
from pygptlink.gpt_no_response_desired import GPTNoResponseDesired
from pygptlink.gpt_tokens import num_tokens_for_tools
from pygptlink.gpt_tool_definition import GPTToolDefinition
from pygptlink.gpt_logging import logger
from pygptlink.sentenceextractor import SentenceExtractor


class GPTCompletion:
    def __init__(self, api_key: str):
        self.lock = asyncio.Lock()
        self.oai = AsyncOpenAI(api_key=api_key)
        self.sentence_extractor = SentenceExtractor()

    async def complete(self, context: GPTContext,
                       callback: Callable[[str, bool], None] = None,
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
            'model': context.model,
            'max_tokens': context.max_response_tokens,
            'stream': True,
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
        tool_tokens = num_tokens_for_tools(
            functions=tool_defs, model=context.model)
        messages = context.messages(
            sticky_system_message=extra_system_prompt, reserved_tokens=tool_tokens)
        logger.debug(f"Prompting with: {messages}")

        # Give up after about a minute
        attempts = 5
        delay_s = 1
        stream = None
        for attempt in range(attempts):
            try:
                stream = await self.oai.chat.completions.create(messages=messages, **completion_settings, tools=tool_defs, tool_choice=tool_choice)
                break
            except Exception as e:
                if isinstance(e, openai.APIStatusError):
                    if e.status_code != 429 and e.status_code >= 400 and e.status_code < 500:
                        # Client error other than 429, retrying is unlikely to succeed
                        raise e
                # Remaining errors are 429, 5xx and APIConnectionError. We retry all of those.
                if attempt < attempts-1:
                    time.sleep(delay_s)
                    delay_s *= 2
                else:
                    raise Exception(
                        f"Giving up after {attempts} attempts, final error was: ") from e

        partial_sentence = ""
        full_response: dict[str, str] = {}
        chunk: ChatCompletionChunk
        async for chunk in stream:
            GPTCompletion._merge_dicts(full_response, chunk.model_dump())
            partial_sentence += chunk.choices[0].delta.content or ""
            lines, partial_sentence = self.sentence_extractor.extract_partial(
                partial_sentence)
            for line in lines:
                if callback:
                    callback(line, False)
        partial_sentence = partial_sentence.strip()
        if callback:
            callback(partial_sentence, True)

        # Look for any function calls in the finished completion.
        chat_completion = GPTCompletion._to_chat_completion(full_response)
        logger.debug(f"Received object: {chat_completion}")
        if not no_append:
            context.append_completion(chat_completion)
        should_respond_to_tool = False
        for choice in chat_completion.choices:
            for tool_call in choice.message.tool_calls or []:
                tool = tools_map.get(tool_call.function.name, None)
                if tool is None:
                    logger.warning(
                        f"Invalid tool invocation, tool: {tool_call.function.name} doesn't exist.")
                    response = f"Error: No such tool: {tool_call.function.name}."
                    if not no_append:
                        context.append_tool_response(
                            tool_call.id, response)
                    # Let the LLM know so it can try to fix
                    should_respond_to_tool = True
                else:
                    logger.info(f"Tool invocation: {tool_call.function}")
                    response = await tool.invoke(tool_call)
                    if isinstance(response, GPTNoResponseDesired):
                        response = ""
                    else:
                        should_respond_to_tool = True
                    context.append_tool_response(
                        tool_call.id, response)
            logger.info(
                f" -- LLM Response Ended ({choice.finish_reason}) -- ")

        response = None
        if chat_completion.choices[0].message:
            response = chat_completion.choices[0].message.content

        if should_respond_to_tool:
            sub_response = await self.complete(context=context,
                                               extra_system_prompt=extra_system_prompt,
                                               callback=callback,
                                               allowed_tools=allowed_tools,
                                               force_tool=force_tool,
                                               tools=tools,
                                               no_append=no_append)

            if not response:
                return sub_response
            elif sub_response:
                return response + " " + sub_response
            else:
                return response
        else:
            return response

    @staticmethod
    def _to_chat_completion(merged_chunks: dict[str, Any]) -> ChatCompletion:
        merged_chunks["object"] = "chat.completion"
        for choice in merged_chunks["choices"]:
            choice["message"] = choice["delta"]
            choice.pop("delta", None)
            choice.pop("logprobs", None)
        return ChatCompletion(**merged_chunks)

    @staticmethod
    def _merge_dicts(current: dict[str, Any], delta: dict[str, Any]) -> None:
        for key, value in delta.items():
            if value is None:
                continue
            elif key not in current:
                current[key] = value
            elif key in ["id", "model", "created", "index", "object", "system_fingerprint"]:
                continue
            else:
                if isinstance(value, str):
                    if current[key] is None:
                        current[key] = value
                    else:
                        current[key] += value
                elif isinstance(value, dict):
                    if current[key] is None:
                        current[key] = {}
                    GPTCompletion._merge_dicts(current[key], value)
                elif isinstance(value, list):
                    if current[key] is None:
                        current[key] = []
                    for entry in value:
                        index = entry.get("index")
                        if index is not None:
                            existing_entry = next(
                                (e for e in current[key] if e.get("index") == index), None)
                            if existing_entry is None:
                                current[key].append(entry)
                            else:
                                GPTCompletion._merge_dicts(
                                    existing_entry, entry)
