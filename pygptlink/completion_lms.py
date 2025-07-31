import asyncio
from collections.abc import Callable
from pygptlink.context import Context
from pygptlink.logging import logger
from pygptlink.sentenceextractor import SentenceExtractor
import lmstudio as lms


class CompletionLMS:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.sentence_extractor = SentenceExtractor()

    def complete(
        self,
        context: Context,
        callback: Callable[[str, bool], None] | None = None,
        extra_system_prompt: str | None = None,
        no_append: bool = False,
    ) -> str:
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

        model = lms.llm(
            context.model,
            config={
                "contextLength": context.max_tokens,
            },
        )
        messages = context.messages_lms(
            sticky_system_message=extra_system_prompt,
            reserved_tokens=0,
        )

        def on_fragment(fragment: lms.LlmPredictionFragment, _round_index: int = 0):
            if callback:
                callback(fragment.content, False)

        def on_message(msg: lms.AssistantResponse | lms.ToolResultMessage):
            if not no_append:
                context.append_completion_lms(msg)

        logger.debug(f"Prompting {model} with: {messages}")

        result = model.respond(
            history=messages,
            config={
                "maxTokens": context.max_response_tokens,
                "temperature": 0.4,
                "cpuThreads": 20,
            },
            on_prediction_fragment=on_fragment,
            on_message=on_message,
        )
        if callback:
            callback("", True)
        return result.content
