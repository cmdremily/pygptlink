from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Literal, NotRequired, TypedDict, Union
from uuid import uuid4

import jsonlines
from lmstudio import AnyChatMessageDict, AssistantResponse, ToolResultMessage
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pygptlink.logging import logger
from pygptlink.tokens import (
    num_tokens_for_messages_lms,
    num_messages_tokens_openai,
)


class MessageBase(TypedDict):
    msg_id: NotRequired[str]
    msg_timestamp: NotRequired[str]


class MessageAssistant(MessageBase):
    role: Literal["assistant"]
    content: str


class MessageUser(MessageBase):
    role: Literal["user"]
    name: str
    content: str


class MessageSystem(MessageBase):
    role: Literal["system"]
    content: str


class MessageToolReturn(MessageBase):
    role: Literal["tool"]
    content: str
    tool_call_id: str


class Function(TypedDict):
    name: str
    arguments: str  # JSON string of arguments


class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: Function


class MessageToolCalls(MessageBase):
    role: Literal["assistant"]
    tool_calls: list[ToolCall]


MessageType = Union[MessageAssistant | MessageUser | MessageSystem | MessageToolReturn | MessageToolCalls]


class Context:
    """Represents a context history usable for chat completion.

    In this class "message" has a very distinct meaning, as one structured dict element in the "messages" list of dicts that is required by the OpenAI API for completions.
    """

    def __init__(
        self,
        model: str,
        max_tokens: int,
        max_response_tokens: int,
        persona_file: str | Path | None = None,
        context_file: str | Path | None = None,
        completion_log_file: str | Path | None = None,
    ) -> None:
        """Creates a new GPTContext.

        The `max_tokens` and `max_response_tokens` parameters are interdependent. The total number of tokens that can be returned from messages() is `max_tokens - max_response_tokens`. This is designed to prevent the response from being prematurely truncated due to the max token limit.

        Parameters:
            model (str): Name of an OpenAI model supported currently, such as "gpt-4o".
            max_tokens (int): Defines the maximum token length for the generated "messages" output. Due to some margin of error in token estimation, it's advised not to set this parameter equal to the model's maximum token length but to allow some room for discrepancy.
            max_response_tokens (int): Specifies the maximum response length to be passed to the API. The LLM response will be forcefully clipped after reaching this token limit. This serves as a damage control measure; for shorter, uninterrupted replies, consider giving instructions to the model instead.
            persona_file (Optional[str], optional): This file includes text that primes the agent to behave in a specific way. If no value is provided, it defaults to None and the model behaves like a "helpful AI assistant".
            context_file (Optional[str], optional): Contains a JSONL file with past messages. It's used to maintain the persisting context and can also aid in post-processing the potentially extensive history for fine-tuning. If not specified, defaults to None and means that the context is not persisted on disk.
            completion_log_file (Optional[str], optional): This debug log contains all received and sent message objects. Unlike the context file, it holds the full completion response instead of just the necessary parts for the next chat completion. Defaults to None if left unspecified.
        """
        self.model = model
        self.persona_file = persona_file
        self.completion_log_file = completion_log_file or None
        self.max_tokens = max_tokens
        self.max_response_tokens = max_response_tokens

        self.context: list[MessageType] = []
        self.context_file = context_file or None
        if self.context_file:
            try:
                # Generously assume that one token is always 8 characters (it's ~4 on average in English)
                # to get an upper bound on the number of characters to load into memory. This is not
                # intended to be accurate, but rather just a way to bound the amount of memory and work
                # required to load the context.
                max_chars_to_read = max_tokens * 8

                with open(self.context_file, "r", encoding="utf-8") as file:
                    start_offs = max(0, os.path.getsize(self.context_file) - max_chars_to_read)
                    file.seek(start_offs)
                    if start_offs != 0:
                        file.readline()  # Discard until next endline.
                    with jsonlines.Reader(file) as reader:
                        for message in reader.iter():
                            self.context.append(message)  # type: ignore
            except FileNotFoundError:
                logger.warning(f"The context file does not exist: {self.context_file}")
            except OSError as e:
                logger.error(f"An error occurred while opening the file: {e}")

    def copy(self) -> "Context":
        """Creates a deep(er) copy of this context without a backing file

        The context messages themselves aren't copied, but their containing array is, allowing you to add and remove but not change events.
        """
        new_ctx = Context(
            model=self.model,
            max_tokens=self.max_tokens,
            max_response_tokens=self.max_response_tokens,
            persona_file=self.persona_file,
        )
        new_ctx.context = self.context.copy()
        return new_ctx

    def append_completion_lms(self, msg: AssistantResponse | ToolResultMessage):
        if self.completion_log_file:
            with open(self.completion_log_file, "a", encoding="utf-8") as file:
                file.write(msg.__str__() + "\n")

        if isinstance(msg, ToolResultMessage):
            # This is a bit special, to maintain compatibility we need to split these into multiple
            # messages.
            for content in msg.content:
                assert content.type == "toolCallResult"
                assert content.tool_call_id is not None, "Tool call ID is missing!"
                self.__append_message(
                    MessageToolReturn(
                        role="tool",
                        tool_call_id=content.tool_call_id,
                        content=content.content,
                    )
                )
            return

        tool_call_accumulator = MessageToolCalls(
            role="assistant",
            tool_calls=[],
        )
        for content in msg.content:
            if content.type == "text":
                self.append_assistant_prompt(content.text)
            elif content.type == "toolCallRequest":
                assert content.tool_call_request.id is not None, "Tool call ID is missing!"
                tool_call_accumulator["tool_calls"].append(
                    ToolCall(
                        id=content.tool_call_request.id,
                        type="function",
                        function=Function(
                            name=content.tool_call_request.name,
                            arguments=json.dumps(content.tool_call_request.arguments),
                        ),
                    )
                )
        if tool_call_accumulator["tool_calls"]:
            self.__append_message(tool_call_accumulator)

    def append_completion_openai(self, completion: ChatCompletion, choice: int = 0) -> None:
        """Parses a provided ChatCompletion object and appends suitable entries to the context to represent that this completion is part of the context history.

        Args:
            completion (ChatCompletion): The completion itself.
            choice (int, optional): Which of the Choice objects in the completion to pick. Defaults to 0.
        """
        if self.completion_log_file:
            with open(self.completion_log_file, "a", encoding="utf-8") as file:
                file.write(completion.model_dump_json(indent=2) + "\n")

        if len(completion.choices) < 1:
            logger.warning("Empty choices for completion!")
            return
        completion_message: ChatCompletionMessage = completion.choices[choice].message

        if completion_message.content is not None:
            self.__append_message(
                message={
                    "role": completion_message.role,
                    "content": completion_message.content,
                }
            )
        if completion_message.tool_calls:
            tool_calls: list[ToolCall] = []
            for tool_call in completion_message.tool_calls:
                tool_calls.append(
                    {
                        "id": tool_call.id,
                        "type": tool_call.type,
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                )
            self.__append_message(message={"role": completion_message.role, "tool_calls": tool_calls})

    def append_assistant_prompt(self, content: str) -> None:
        message: MessageAssistant = {"role": "assistant", "content": content}
        self.__append_message(message)

    def append_user_prompt(self, content: str, user: str = "User") -> None:
        if not user or not user.isalnum():
            raise ValueError(f"Invalid user name: {user}. Must match [a-zA-Z0-9].")
        message: MessageUser = {"role": "user", "name": user, "content": content}
        self.__append_message(message)

    def append_tool_response(self, id: str, content: str) -> None:
        message: MessageToolReturn = {
            "role": "tool",
            "content": content,
            "tool_call_id": id,
        }
        self.__append_message(message)

    def append_system_message(self, content: str) -> None:
        self.__append_message(self.__system_message(content))

    def messages_lms(
        self, sticky_system_message: str | None = None, reserved_tokens: int = 0
    ) -> list[AnyChatMessageDict]:
        """Generates a list of dicts that can be passed to OpenAIs completion API.

        There is usually no need to call this directly.

        Args:
            sticky_system_message (str, optional): The following text will be included as a system message early in the output messages but without adding it to the context. Use this for important system messages that should never "roll off" the context window or that you need to provide but don't want in the context permanently. Defaults to None.

        Returns:
            list[dict]: A "messages" structure, a list of "message" dicts.
        """
        assert sticky_system_message == None or isinstance(sticky_system_message, str)

        if self.persona_file:
            with open(self.persona_file, "r") as file:
                persona = file.read()
        else:
            persona = None

        # TODO: LMStudio doesn't support consecutive assistant messages, we should attempt to detect and merge them.
        messages: list[AnyChatMessageDict] = [self.__msg_to_lms(m) for m in self.context]
        available_tokens = self.max_tokens - self.max_response_tokens - reserved_tokens
        while len(messages) and (
            messages[0]["role"] == "tool" or num_tokens_for_messages_lms(messages, self.model) > available_tokens
        ):
            if len(self.context) > 0:
                self.context.pop(0)
            messages.pop(0)

        # LMStudio doesn't support multiple system prompts, we combine sticky system and persona.
        merged_system = ((persona or "") + "\n\n" + (sticky_system_message or "")).strip()
        if merged_system:
            sys_msg = self.__msg_to_lms(self.__system_message(merged_system))
            while len(messages) and (
                messages[0]["role"] == "tool"
                or num_tokens_for_messages_lms([sys_msg] + messages, self.model) > available_tokens
            ):
                if len(self.context) > 0:
                    self.context.pop(0)
                messages.pop(0)
            messages = [sys_msg] + messages

        for msg in messages:
            if not msg["content"]:
                raise ValueError("Empty content in context!")

        return messages

    def messages_openai(
        self, sticky_system_message: str | None = None, reserved_tokens: int = 0
    ) -> list[ChatCompletionMessageParam]:
        """Generates a list of dicts that can be passed to OpenAIs completion API.

        There is usually no need to call this directly.

        Args:
            sticky_system_message (str, optional): The following text will be included as a system message early in the output messages but without adding it to the context. Use this for important system messages that should never "roll off" the context window or that you need to provide but don't want in the context permanently. Defaults to None.

        Returns:
            list[dict]: A "messages" structure, a list of "message" dicts.
        """
        if self.persona_file:
            with open(self.persona_file, "r") as file:
                persona = file.read()
        else:
            persona = None

        # Find split point
        left_split: list[ChatCompletionMessageParam] = []
        right_split: list[ChatCompletionMessageParam] = []
        for entry in reversed(self.context):
            if not right_split:
                right_split.append(self.__msg_to_openai(entry))
            elif right_split[-1]["role"] == "tool":
                right_split.append(self.__msg_to_openai(entry))
            else:
                left_split.append(self.__msg_to_openai(entry))
        right_split.reverse()
        left_split.reverse()

        messages = left_split
        if persona:
            messages.append(self.__msg_to_openai(self.__system_message(persona)))
        if sticky_system_message:
            messages.append(self.__msg_to_openai(self.__system_message(sticky_system_message)))
        messages += right_split

        available_tokens = self.max_tokens - self.max_response_tokens - reserved_tokens
        while len(messages) and (
            messages[0]["role"] == "tool" or num_messages_tokens_openai(messages, self.model) > available_tokens
        ):
            if len(self.context) > 0:
                self.context.pop(0)
            messages.pop(0)
        return messages

    def clear(self) -> None:
        """Deletes everything from this context. INCLUDING DELETING THE CONTEXT ON DISK."""
        self.context = []
        if self.context_file and os.path.exists(self.context_file):
            os.remove(self.context_file)

    def __append_message(self, message: MessageType) -> None:
        if not "tool_calls" in message and not message.get("content"):
            raise ValueError(f"Empty content in context for message: {message}!")
        message["msg_id"] = str(uuid4())
        message["msg_timestamp"] = datetime.now(tz=timezone.utc).isoformat()
        self.context.append(message)
        if self.context_file:
            with jsonlines.open(self.context_file, "a") as file:  # type: ignore
                file.write(message)

    def __system_message(self, content: str) -> MessageSystem:
        return {"role": "system", "content": content}

    @staticmethod
    def __msg_to_openai(message: MessageType) -> ChatCompletionMessageParam:
        if message["role"] == "assistant":
            if "tool_calls" in message:
                return {
                    "role": "assistant",
                    "tool_calls": message.get("tool_calls", []),
                }
            return {"role": "assistant", "content": message["content"]}
        elif message["role"] == "user":
            return {
                "role": "user",
                "name": message["name"],
                "content": message["content"],
            }
        elif message["role"] == "system":
            return {"role": "system", "content": message["content"]}
        elif message["role"] == "tool":
            return {
                "role": "tool",
                "content": message["content"],
                "tool_call_id": message.get("tool_call_id", ""),
            }

    @staticmethod
    def __msg_to_lms(message: MessageType) -> AnyChatMessageDict:
        if message["role"] == "assistant":
            if "tool_calls" in message:
                return {
                    "role": "assistant",
                    "content": message.get("content"),
                }
            return {"role": "assistant", "content": message["content"]}
        elif message["role"] == "user":
            return {
                "role": "user",
                "name": message["name"],
                "content": message["content"],
            }
        elif message["role"] == "system":
            return {"role": "system", "content": message["content"]}
        elif message["role"] == "tool":
            return {
                "role": "tool",
                "content": message["content"],
                "tool_call_id": message.get("tool_call_id", ""),
            }
