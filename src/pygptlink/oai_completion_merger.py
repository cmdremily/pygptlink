from typing import Optional
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageToolCall,
    Function,
)
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk


class OpenAICompletionMerger:
    def __init__(self) -> None:
        self.__completion: ChatCompletion | None = None

    def result(self) -> ChatCompletion:
        if self.__completion is None:
            raise ValueError("No completion has been appended yet.")
        return self.__completion

    def append(self, chunk: ChatCompletionChunk) -> ChatCompletion:
        if self.__completion is None:
            self.__completion = ChatCompletion(
                id=chunk.id,
                choices=[],
                created=chunk.created,
                model=chunk.model,
                object="chat.completion",
                service_tier=chunk.service_tier,
                system_fingerprint=chunk.system_fingerprint,
                usage=chunk.usage,
            )

        if self.__completion.usage is None:
            self.__completion.usage = chunk.usage

        for chunk_choice in chunk.choices:
            assert (
                not chunk_choice.delta.role or chunk_choice.delta.role == "assistant"
            ), f"Expected role 'assistant', got {chunk_choice.delta.role} in chunk {chunk.id}."
            if len(self.__completion.choices) <= chunk_choice.index:
                # New choice, create minimal structure, filled in later
                self.__completion.choices.append(
                    Choice(
                        finish_reason=chunk_choice.finish_reason or "stop",
                        index=chunk_choice.index,
                        message=ChatCompletionMessage(
                            role="assistant",
                        ),
                    )
                )

            result_choice = self.__completion.choices[chunk_choice.index]
            result_choice.finish_reason = chunk_choice.finish_reason or "stop"
            result_choice.message.content = self.__append_or_not(
                result_choice.message.content, chunk_choice.delta.content
            )
            result_choice.message.refusal = self.__append_or_not(
                result_choice.message.refusal, chunk_choice.delta.refusal
            )

            for delta_toolcall in chunk_choice.delta.tool_calls or []:
                if not result_choice.message.tool_calls:
                    result_choice.message.tool_calls = []
                if len(result_choice.message.tool_calls) <= delta_toolcall.index:
                    # New tool call, create minimal structure, filled in later
                    result_choice.message.tool_calls.append(
                        ChatCompletionMessageToolCall(
                            id="",
                            function=Function(name="", arguments=""),
                            type="function",
                        )
                    )
                result_tool_call = result_choice.message.tool_calls[delta_toolcall.index]
                result_tool_call.id = self.__append_or_not(result_tool_call.id, delta_toolcall.id) or ""
                if delta_toolcall.function:
                    result_tool_call.function.name = (
                        self.__append_or_not(result_tool_call.function.name, delta_toolcall.function.name) or ""
                    )
                    result_tool_call.function.arguments = (
                        self.__append_or_not(
                            result_tool_call.function.arguments,
                            delta_toolcall.function.arguments,
                        )
                        or ""
                    )

        return self.__completion

    @staticmethod
    def __append_or_not(old: Optional[str], new: Optional[str]) -> Optional[str]:
        if not old:
            return new
        if not new:
            return old
        return old + new
