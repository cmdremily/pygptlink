from datetime import datetime, timezone
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import jsonlines
from lmstudio import AssistantResponse, Chat, ChatHistoryDataDict, ToolResultMessage
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pygptlink.gpt_logging import logger
from pygptlink.gpt_tokens import num_tokens_for_messages


import lmstudio as lms


class GPTContext:
    """Represents a context history usable for chat completion.

    In this class "message" has a very distinct meaning, as one structured dict element in the "messages" list of dicts that is required by the OpenAI API for completions.    
    """

    def __init__(self, model: str, max_tokens: int, max_response_tokens: int, persona_file: str | Path | None = None,
                 context_file: str | Path | None = None, completion_log_file: str | Path | None = None, temperature: float | None = None) -> None:
        """Creates a new GPTContext.

        The `max_tokens` and `max_response_tokens` parameters are interdependent. The total number of tokens that can be returned from messages() is `max_tokens - max_response_tokens`. This is designed to prevent the response from being prematurely truncated due to the max token limit.

        Parameters:
            model (str): Name of an OpenAI model supported currently, such as "gpt-4o".
            max_tokens (int): Defines the maximum token length for the generated "messages" output. Due to some margin of error in token estimation, it's advised not to set this parameter equal to the model's maximum token length but to allow some room for discrepancy.
            max_response_tokens (int): Specifies the maximum response length to be passed to the API. The LLM response will be forcefully clipped after reaching this token limit. This serves as a damage control measure; for shorter, uninterrupted replies, consider giving instructions to the model instead.
            persona_file (Optional[str], optional): This file includes text that primes the agent to behave in a specific way. If no value is provided, it defaults to None and the model behaves like a "helpful AI assistant".
            context_file (Optional[str], optional): Contains a JSONL file with past messages. It's used to maintain the persisting context and can also aid in post-processing the potentially extensive history for fine-tuning. If not specified, defaults to None and means that the context is not persisted on disk.
            completion_log_file (Optional[str], optional): This debug log contains all received and sent message objects. Unlike the context file, it holds the full completion response instead of just the necessary parts for the next chat completion. Defaults to None if left unspecified.
            temperature (Optional[float], optional): The sampling temperature [0.0, 1.0] lower is more deterministic output. Null for model default.
        """
        self.model = model
        self.persona_file = persona_file
        self.completion_log_file = completion_log_file or None
        self.max_tokens = max_tokens
        self.max_response_tokens = max_response_tokens
        self.temperature = temperature

        self.context: list[dict[str, Any]] = []
        self.context_file = context_file or None
        if self.context_file:
            try:
                # Generously assume that one token is always 8 characters (it's ~4 on average in English)
                # to get an upper bound on the number of characters to load into memory. This is not
                # intended to be accurate, but rather just a way to bound the amount of memory and work
                # required to load the context.
                max_chars_to_read = max_tokens*8

                with open(self.context_file, 'r', encoding='utf-8') as file:
                    start_offs = max(0, os.path.getsize(
                        self.context_file) - max_chars_to_read)
                    file.seek(start_offs)
                    if start_offs != 0:
                        file.readline()  # Discard until next endline.
                    with jsonlines.Reader(file) as reader:
                        for message in reader.iter():
                            assert isinstance(message, dict)
                            self.context.append(message)
            except FileNotFoundError:
                logger.warning(
                    f"The context file does not exist: {self.context_file}")
            except OSError as e:
                logger.error(f"An error occurred while opening the file: {e}")

    def copy(self) -> 'GPTContext':
        """Creates a deep(er) copy of this context without a backing file

        The context messages themselves aren't copied, but their containing array is, allowing you to add and remove but not change events.
        """
        new_ctx = GPTContext(model=self.model, max_tokens=self.max_tokens, max_response_tokens=self.max_response_tokens,
                             persona_file=self.persona_file)
        new_ctx.context = self.context.copy()
        return new_ctx

    def append_lms_completion(self, msg: AssistantResponse | ToolResultMessage):
        if self.completion_log_file:
            with open(self.completion_log_file, 'a', encoding='utf-8') as file:
                file.write(msg.__str__() + '\n')

        message: dict[str, Any] = {"role": msg.role}
        if isinstance(msg, ToolResultMessage):
            # This is a bit special, to maintain compatibility we need to split these into multiple
            # messages.
            for content in msg.content:
                assert content.type == "toolCallResult"
                assert content.tool_call_id
                self.__append_message(message={"role": msg.role,
                                               "tool_call_id": content.tool_call_id,
                                               "content": content.content})
            return

        for content in msg.content:
            if content.type == "text":
                message['content'] = content.text
            elif content.type == "toolCallRequest":
                tools: list[dict[str, Any]] = message.setdefault("tool_calls", [])
                tools.append(
                    {
                        "id": content.tool_call_request.id,
                        "type": content.tool_call_request.type,
                        "function": {
                            "name": content.tool_call_request.name,
                            "arguments": content.tool_call_request.arguments
                        }
                    })
        self.__append_message(message=message)

    def append_completion(self, completion: ChatCompletion, choice: int = 0) -> None:
        """Parses a provided ChatCompletion object and appends suitable entries to the context to represent that this completion is part of the context history.

        Args:
            completion (ChatCompletion): The completion itself.
            choice (int, optional): Which of the Choice objects in the completion to pick. Defaults to 0.
        """
        if self.completion_log_file:
            with open(self.completion_log_file, 'a', encoding='utf-8') as file:
                file.write(completion.model_dump_json(indent=2) + '\n')

        if len(completion.choices) < 1:
            logger.warning("Empty choices for completion!")
            return
        completion_message: ChatCompletionMessage = completion.choices[choice].message

        message: dict[str, Any] = {"role": completion_message.role}
        if completion_message.content is not None:
            message["content"] = completion_message.content
        if completion_message.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tool_call.id,
                    "type": tool_call.type,
                    "function": {
                        "name": tool_call.function.name,
                        "arguments": tool_call.function.arguments
                    }
                } for tool_call in completion_message.tool_calls
            ]
        self.__append_message(message)

    def append_assistant_prompt(self, content: str) -> None:
        """Appends a user message onto the context.

        Typically used before completing the context to respond to the provided user query.

        Args:
            user (str): The name of the user that is prompting. Must match [a-zA-Z0-9].
            content (str): The contents of the prompt.
        """
        message = {"role": "assistant", "content": content}
        self.__append_message(message)

    def append_user_prompt(self, content: str, user: str = "User") -> None:
        """Appends a user message onto the context.

        Typically used before completing the context to respond to the provided user query.

        Args:
            user (str): The name of the user that is prompting. Must match [a-zA-Z0-9].
            content (str): The contents of the prompt.
        """
        message = {"role": "user", "name": user, "content": content}
        self.__append_message(message)

    def append_tool_response(self, id: str, content: str) -> None:
        """Appends a tool response message onto the context.

        There is usually no need to call this directly, the GPTCompletion.complete method will do that automatically.

        Args:
            id (str): The name of the user that is prompting. Must match [a-zA-Z0-9].
            content (str): The contents of the prompt.
        """
        message = {"role": "tool", "content": content, "tool_call_id": id}
        self.__append_message(message)

    def append_system_message(self, content: str) -> None:
        """Appends a system message onto the context.

        The models typically treats system messages slightly differently. They know that system messages are from the operator.

        Use system messages when you need to influence the models behaviour.

        Args:
            content (str): The contents of the prompt.
        """
        self.__append_message(self.__system_message(content))

    def lms_messages(self, sticky_system_message: str | None = None, reserved_tokens: int = 0) -> ChatHistoryDataDict:
        """Generates a list of dicts that can be passed to OpenAIs completion API.

        There is usually no need to call this directly.

        Args:
            sticky_system_message (str, optional): The following text will be included as a system message early in the output messages but without adding it to the context. Use this for important system messages that should never "roll off" the context window or that you need to provide but don't want in the context permanently. Defaults to None.

        Returns:
            list[dict]: A "messages" structure, a list of "message" dicts.
        """
        assert sticky_system_message == None or isinstance(
            sticky_system_message, str)

        if self.persona_file:
            with open(self.persona_file, 'r') as file:
                persona = file.read()
        else:
            persona = None

         # TODO: LMStudio doesn't support consecutive assistant messages, we should attempt to detect and merge them.
        messages = [m for m in self.context]
        available_tokens = self.max_tokens - self.max_response_tokens - reserved_tokens
        while len(messages) and (messages[0]["role"] == "tool" or num_tokens_for_messages(messages, self.model) > available_tokens):
            if len(self.context) > 0:
                self.context.pop(0)
            messages.pop(0)

        # LMStudio doesn't support multiple system prompts, we combine sticky system and persona.
        merged_system = ((persona or "") + "\n\n" +
                         (sticky_system_message or "")).strip()
        if merged_system:
            sys_msg = self.__system_message(merged_system)
            while len(messages) and (messages[0]["role"] == "tool" or num_tokens_for_messages([sys_msg] + messages, self.model) > available_tokens):
                if len(self.context) > 0:
                    self.context.pop(0)
                messages.pop(0)
            messages = [sys_msg] + messages

        for msg in messages:
            if not msg['content']:
                raise ValueError("Empty content in context!")

        return ChatHistoryDataDict(messages=messages)  # type: ignore

    def oai_messages(self, sticky_system_message: str | None = None, reserved_tokens: int = 0, lms: bool = False) -> list[dict[str, str]]:
        """Generates a list of dicts that can be passed to OpenAIs completion API.

        There is usually no need to call this directly.

        Args:
            sticky_system_message (str, optional): The following text will be included as a system message early in the output messages but without adding it to the context. Use this for important system messages that should never "roll off" the context window or that you need to provide but don't want in the context permanently. Defaults to None.

        Returns:
            list[dict]: A "messages" structure, a list of "message" dicts.
        """
        assert sticky_system_message == None or isinstance(
            sticky_system_message, str)

        if self.persona_file:
            with open(self.persona_file, 'r') as file:
                persona = file.read()
        else:
            persona = None

        # Find split point
        left_split: list[dict[str, str]] = []
        right_split: list[dict[str, str]] = []
        for entry in reversed(self.context):
            entry = self.__strip_internal_tags(entry)
            if not right_split:
                right_split.append(entry)
            elif right_split[-1]["role"] == "tool":
                right_split.append(entry)
            else:
                left_split.append(entry)
        right_split.reverse()
        left_split.reverse()

        messages = left_split
        if persona:
            messages.append(self.__system_message(persona))
        if sticky_system_message:
            messages.append(self.__system_message(sticky_system_message))
        messages += right_split

        available_tokens = self.max_tokens - self.max_response_tokens - reserved_tokens
        while len(messages) and (messages[0]["role"] == "tool" or num_tokens_for_messages(messages, self.model) > available_tokens):
            if len(self.context) > 0:
                self.context.pop(0)
            messages.pop(0)

        for msg in messages:
            if not msg['content']:
                raise ValueError("Empty content in context!")

        return messages

    def clear(self) -> None:
        """Deletes everything from this context. INCLUDING DELETING THE CONTEXT ON DISK.
        """
        self.context = []
        if self.context_file and os.path.exists(self.context_file):
            os.remove(self.context_file)

    def __strip_internal_tags(self, message: dict[str, str]) -> dict[str, str]:
        return {k: v for k, v in message.items() if k != "msg_id" and k != "msg_timestamp"}

    def __append_message(self, message: dict[str, str]):
        message["msg_id"] = str(uuid4())
        message["msg_timestamp"] = datetime.now(tz=timezone.utc).isoformat()
        self.context.append(message)
        if self.context_file:
            with jsonlines.open(self.context_file, 'a') as file:
                file.write(message)

    def __system_message(self, content: str):
        return {"role": "system", "content": content}
