from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Generator, Literal, NotRequired, TypedDict, Union
from uuid import uuid4

import jsonlines
from lmstudio import (
    AssistantResponse,
    Chat,
    ToolCallRequest,
    ToolCallResultData,
    ToolResultMessage,
)
import lmstudio
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from pygptlink.logging import logger
from pygptlink.tokens import (
    lms_model_loaded,
    num_messages_tokens_openai,
)


class Function(TypedDict):
    name: str
    arguments: str  # JSON string of arguments


class ToolCall(TypedDict):
    id: str
    type: Literal["function"]
    function: Function


class MessageBase(TypedDict):
    msg_id: NotRequired[str]
    msg_timestamp: NotRequired[str]


class MessageAssistant(MessageBase):
    role: Literal["assistant"]
    content: NotRequired[str]
    tool_calls: NotRequired[list[ToolCall]]


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


MessageType = Union[MessageAssistant | MessageUser | MessageSystem | MessageToolReturn]


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
        temperature: float = 1.0,
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
        self._model = model
        self._persona_file = persona_file
        self._completion_log_file = completion_log_file or None
        self._max_tokens = max_tokens
        self._max_response_tokens = max_response_tokens
        self._temperature = temperature

        self._context: list[MessageType] = []
        self._context_file = context_file or None
        if self._context_file:
            try:
                # Generously assume that one token is always 8 characters (it's ~4 on average in English)
                # to get an upper bound on the number of characters to load into memory. This is not
                # intended to be accurate, but rather just a way to bound the amount of memory and work
                # required to load the context.
                max_chars_to_read = max_tokens * 8

                with open(self._context_file, "r", encoding="utf-8") as file:
                    start_offs = max(
                        0, os.path.getsize(self._context_file) - max_chars_to_read
                    )
                    file.seek(start_offs)
                    if start_offs != 0:
                        file.readline()  # Discard until next endline.
                    with jsonlines.Reader(file) as reader:
                        for message in reader.iter():
                            self._context.append(message)  # type: ignore
            except FileNotFoundError:
                logger.warning(f"The context file does not exist: {self._context_file}")
            except OSError as e:
                logger.error(f"An error occurred while opening the file: {e}")

    @property
    def temperature(self) -> float:
        return self._temperature

    @property
    def model(self) -> str:
        return self._model

    @property
    def max_response_tokens(self) -> int:
        return self._max_response_tokens

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def copy(self) -> "Context":
        """Creates a deep(er) copy of this context without a backing file

        The context messages themselves aren't copied, but their containing array is, allowing you to add and remove but not change events.
        """
        new_ctx = Context(
            model=self._model,
            max_tokens=self._max_tokens,
            max_response_tokens=self._max_response_tokens,
            persona_file=self._persona_file,
        )
        new_ctx._context = self._context.copy()
        return new_ctx

    def append_completion_lms(self, msg: AssistantResponse | ToolResultMessage) -> None:
        if self._completion_log_file:
            with open(self._completion_log_file, "a", encoding="utf-8") as file:
                file.write(msg.__str__() + "\n")

        if isinstance(msg, ToolResultMessage):
            # This is a bit special, to maintain compatibility we need to split these into multiple
            # messages.
            for tool_call_result in msg.content:
                assert tool_call_result.type == "toolCallResult"
                assert tool_call_result.tool_call_id is not None, (
                    "Tool call ID is missing!"
                )
                self.__append_message(
                    MessageToolReturn(
                        role="tool",
                        tool_call_id=tool_call_result.tool_call_id,
                        content=tool_call_result.content,
                    )
                )
            return

        tool_call_accumulator = MessageAssistant(
            role="assistant",
            tool_calls=[],
        )
        assert "tool_calls" in tool_call_accumulator, (
            "Tool calls accumulator is missing!"
        )
        for content in msg.content:
            if content.type == "text":
                self.append_assistant_prompt(content.text)
            elif content.type == "toolCallRequest":
                assert content.tool_call_request.id is not None, (
                    "Tool call ID is missing!"
                )
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

    def append_completion_openai(
        self, completion: ChatCompletion, choice: int = 0
    ) -> None:
        """Parses a provided ChatCompletion object and appends suitable entries to the context to represent that this completion is part of the context history.

        Args:
            completion (ChatCompletion): The completion itself.
            choice (int, optional): Which of the Choice objects in the completion to pick. Defaults to 0.
        """
        if self._completion_log_file:
            with open(self._completion_log_file, "a", encoding="utf-8") as file:
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
            self.__append_message(
                message={"role": completion_message.role, "tool_calls": tool_calls}
            )

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
    ) -> Chat:
        """Generates a list of dicts that can be passed to OpenAIs completion API.

        There is usually no need to call this directly.

        Args:
            sticky_system_message (str, optional): The following text will be included as a system message early in the output messages but without adding it to the context. Use this for important system messages that should never "roll off" the context window or that you need to provide but don't want in the context permanently. Defaults to None.

        Returns:
            list[dict]: A "messages" structure, a list of "message" dicts.
        """
        chat = Chat()

        assert lms_model_loaded(self._model), (
            f"Model {self._model} must be loaded before converting context."
        )
        model = lmstudio.llm(self._model)
        assert model is not None, f"Model {self._model} not found in LMStudio!"
        if self._persona_file:
            with open(self._persona_file, "r") as file:
                persona = file.read()
        else:
            persona = ""
        merged_system = (persona + "\n\n" + (sticky_system_message or "")).strip()

        chat.add_system_prompt(merged_system)
        for msg in self._combine_messages(self._context):
            if msg["role"] == "assistant":
                tool_calls: list[ToolCallRequest] = []
                if "tool_calls" in msg:
                    for tool_call in msg["tool_calls"]:
                        tool_calls.append(
                            ToolCallRequest(
                                type="function",
                                id=tool_call["id"],
                                name=tool_call["function"]["name"],
                                arguments=json.loads(
                                    tool_call["function"]["arguments"]
                                ),
                            )
                        )
                chat.add_assistant_response(
                    response=msg.get("content", ""), tool_call_requests=tool_calls
                )
            elif msg["role"] == "user":
                chat.add_user_message(msg["content"])
            elif msg["role"] == "system":
                chat.add_system_prompt(msg["content"])
            elif msg["role"] == "tool":
                chat.add_tool_result(
                    ToolCallResultData(
                        content=msg["content"], tool_call_id=msg["tool_call_id"]
                    )
                )
        return chat

    def _splice_sticky_messages(
        self, sticky_system_message: str | None = None
    ) -> list[MessageType]:
        if self._persona_file:
            with open(self._persona_file, "r") as file:
                persona = file.read()
        else:
            persona = None

        # Find split point
        left_split: list[MessageType] = []
        right_split: list[MessageType] = []
        for entry in reversed(self._context):
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
        return messages

    def _combine_messages(
        self, messages: list[MessageType]
    ) -> Generator[MessageType, None, None]:
        prev_msg: MessageType | None = None
        for msg in messages:
            if not prev_msg:
                prev_msg = msg
                continue
            merged_msg = self.__merge_messages(prev_msg, msg)
            if merged_msg:
                prev_msg = merged_msg
            else:
                yield prev_msg
                prev_msg = msg
        if prev_msg:
            yield prev_msg

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
        spliced_messages = self._splice_sticky_messages(sticky_system_message)
        openai_messages: list[ChatCompletionMessageParam] = []
        for x in self._combine_messages(spliced_messages):
            for y in self.__msg_to_openai(x):
                openai_messages.append(y)

        available_tokens = (
            self._max_tokens - self._max_response_tokens - reserved_tokens
        )
        while len(openai_messages) and (
            openai_messages[0]["role"] == "tool"
            or num_messages_tokens_openai(openai_messages, self._model)
            > available_tokens
        ):
            if len(self._context) > 0:
                self._context.pop(0)
            openai_messages.pop(0)
        return openai_messages

    def clear(self) -> None:
        """Deletes everything from this context. INCLUDING DELETING THE CONTEXT ON DISK."""
        self._context = []
        if self._context_file and os.path.exists(self._context_file):
            os.remove(self._context_file)

    def __merge_messages(
        self, left: MessageType, right: MessageType
    ) -> MessageType | None:
        if left["role"] == "assistant" and right["role"] == "assistant":
            merged_content = (
                left.get("content", "") + "\n\n" + right.get("content", "")
            ).strip()
            merged_tool_calls = left.get("tool_calls", []) + right.get("tool_calls", [])
            return MessageAssistant(
                role="assistant", content=merged_content, tool_calls=merged_tool_calls
            )
        elif left["role"] == "user" and right["role"] == "user":
            if left["name"] != right["name"]:
                return None  # Cannot merge user messages with different names.
            return MessageUser(
                role="user",
                name=left["name"],
                content=left["content"] + "\n\n" + right["content"],
            )
        elif left["role"] == "system" and right["role"] == "system":
            return MessageSystem(
                role="system", content=left["content"] + "\n\n" + right["content"]
            )
        else:
            # Either different roles or tool messages, cannot merge.
            return None

    def __append_message(self, message: MessageType) -> None:
        if "tool_calls" not in message and not message.get("content"):
            raise ValueError(f"Empty content in context for message: {message}!")
        message["msg_id"] = str(uuid4())
        message["msg_timestamp"] = datetime.now(tz=timezone.utc).isoformat()
        self._context.append(message)
        if self._context_file:
            with jsonlines.open(self._context_file, "a") as file:
                _ = file.write(message)

    def __system_message(self, content: str) -> MessageSystem:
        return {"role": "system", "content": content}

    @staticmethod
    def __msg_to_openai(
        message: MessageType,
    ) -> Generator[ChatCompletionMessageParam, None, None]:
        if message["role"] == "assistant":
            assert "tool_calls" in message or "content" in message, (
                "Assistant message must have either tool_calls or content."
            )
            if "tool_calls" in message and "content" in message:
                yield {
                    "role": "assistant",
                    "content": message["content"],
                    "tool_calls": message["tool_calls"],
                }
            elif "tool_calls" in message:
                yield {
                    "role": "assistant",
                    "tool_calls": message["tool_calls"],
                }
            elif "content" in message:
                yield {
                    "role": "assistant",
                    "content": message["content"],
                }
        elif message["role"] == "user":
            yield {
                "role": "user",
                "name": message["name"],
                "content": message["content"],
            }
        elif message["role"] == "system":
            yield {"role": "system", "content": message["content"]}
        elif message["role"] == "tool":
            yield {
                "role": "tool",
                "content": message["content"],
                "tool_call_id": message.get("tool_call_id", ""),
            }
