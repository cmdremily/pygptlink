# coding=utf-8
from collections.abc import Callable

import lmstudio as lms

from pygptlink.context_history import ContextHistory
from pygptlink.logging import logger


class LMSCompletion:
    @staticmethod
    def complete(context: ContextHistory,
                 callback: Callable[[str, bool], None] | None = None,
                 extra_system_prompt: str | None = None,
                 no_append: bool = False) -> str:
        """Generates a response to the current context.

        Args:
            context (ContextHistory): A context to perform a completion on.
            extra_system_prompt (Optional[str]): An extra system prompt to be injected into the context, can be None.
            callback (Optional[Callable[[str], None]]): A callback to call with the response, line by line.
            no_append (bool): False by default. If True, then the result of the completion will not be added to the context.

        Raises:
            ValueError: When inputs are invalid.

        Returns:
            Nothing
        """
        def on_fragment(fragment: lms.LlmPredictionFragment, __round_index: int = 0):
            if callback:
                callback(fragment.content, False)

        def on_message(msg: lms.AssistantResponse | lms.ToolResultMessage):
            if not no_append:
                context.append_completion_lms(msg)


        # Model must be loaded before attempting to generate the context as that will trigger token counting
        # which will load the model with the default config unless loaded manually first.
        config = lms.LlmLoadModelConfig()
        config.context_length = context.max_tokens
        config.flash_attention = False
        config.try_mmap = False
        model = lms.llm(context.model, config=config)

        completion_settings = lms.LlmPredictionConfig()
        completion_settings.max_tokens = context.max_response_tokens
        completion_settings.temperature = context.temperature
        completion_settings.cpu_threads = 20

        messages = context.context_for_lms(sticky_system_message=extra_system_prompt)

        logger.debug(f"Prompting {model} with: {messages}")

        result = model.respond(messages, config=completion_settings, on_prediction_fragment=on_fragment, on_message=on_message)
        if callback:
            callback("", True)
        return result.content
