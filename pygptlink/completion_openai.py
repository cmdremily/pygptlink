from collections.abc import Callable
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionToolParam
from pygptlink.context import Context
from pygptlink.no_response_desired import NoResponseDesired
from pygptlink.tokens import num_tool_tokens_openai
from pygptlink.tool_definition import ToolDefinition
from pygptlink.logging import logger
from pygptlink.oai_completion_merger import OpenAICompletionMerger
from pygptlink.sentenceextractor import SentenceExtractor


class CompletionOpenAI:
    def __init__(self, api_key: str):
        self.__client = AsyncOpenAI(api_key=api_key, max_retries=5, timeout=60)
        self.__sentence_extractor = SentenceExtractor()

    async def complete(
        self,
        context: Context,
        callback: Callable[[str, bool], None] | None = None,
        extra_system_prompt: str | None = None,
        tools: list[ToolDefinition] | None = None,
        force_tool: bool = False,
        allowed_tools: list[str] | None = None,
        no_append: bool = False,
    ) -> str | None:
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

        tools_map = {tool.name: tool for tool in tools or []}
        tool_defs: list[ChatCompletionToolParam] = []
        if allowed_tools:
            if not all(tool_name in tools_map for tool_name in allowed_tools):
                raise ValueError(
                    f"Allowed_tools={allowed_tools} contains unknown tool. Known tools: {list(tools_map.keys())}"
                )
            tool_defs = [tools_map[tool_name].describe_openai() for tool_name in allowed_tools]
        elif allowed_tools is None:
            tool_defs = [tool.describe_openai() for tool in tools_map.values()]

        if tool_defs:
            tool_tokens = num_tool_tokens_openai(tools=tool_defs, model=context.model)
            tool_choice = "required" if force_tool else "auto"
        else:
            tool_tokens = 0
            tool_choice = "none"
            if force_tool:
                raise ValueError("Tool call forced but no tools allowed or available!")

        messages = context.messages_openai(sticky_system_message=extra_system_prompt, reserved_tokens=tool_tokens)

        logger.debug(f"Prompting with: {messages}")
        stream = await self.__client.chat.completions.create(
            model=context.model,
            temperature=context.temperature,
            max_tokens=context.max_response_tokens,
            stream=True,
            stream_options={"include_usage": True},
            messages=messages,
            tools=tool_defs,
            tool_choice=tool_choice,
            store=False,
        )

        partial_sentence = ""
        chunk_merger = OpenAICompletionMerger()
        async for chunk in stream:
            chunk_merger.append(chunk)
            if not chunk.choices or not chunk.choices[0].delta:
                continue
            partial_sentence += chunk.choices[0].delta.content or ""
            lines, partial_sentence = self.__sentence_extractor.extract_partial(partial_sentence)
            for line in lines:
                if callback:
                    callback(line, False)
        partial_sentence = partial_sentence.strip()
        if callback:
            callback(partial_sentence, True)

        # Look for any function calls in the finished completion.
        chat_completion = chunk_merger.result()
        logger.debug(f"Received object: {chat_completion}")
        usage = chat_completion.usage
        if usage is None:
            logger.warning("No usage information in the completion response, assuming 0 tokens used.")
        else:
            logger.debug(f"Prompt tokens: {usage.prompt_tokens}")
            logger.debug(f"Completion tokens: {usage.completion_tokens}")
            logger.debug(f"Total tokens: {usage.total_tokens}")
        if not no_append:
            context.append_completion_openai(chat_completion)
        should_respond_to_tool = False
        for choice in chat_completion.choices:
            for tool_call in choice.message.tool_calls or []:
                tool = tools_map.get(tool_call.function.name, None)
                if tool is None:
                    logger.warning(f"Invalid tool invocation, tool: {tool_call.function.name} doesn't exist.")
                    response = f"Error: No such tool: {tool_call.function.name}."
                    if not no_append:
                        context.append_tool_response(tool_call.id, response)
                    # Let the LLM know so it can try to fix
                    should_respond_to_tool = True
                else:
                    logger.info(f"Tool invocation: {tool_call.function}")
                    response = await tool.invoke(tool_call)
                    if isinstance(response, NoResponseDesired):
                        response = ""
                    else:
                        should_respond_to_tool = True
                    context.append_tool_response(tool_call.id, response)
            logger.info(f" -- LLM Response Ended ({choice.finish_reason}) -- ")

        response = None
        if chat_completion.choices[0].message:
            response = chat_completion.choices[0].message.content

        if should_respond_to_tool:
            sub_response = await self.complete(
                context=context,
                extra_system_prompt=extra_system_prompt,
                callback=callback,
                allowed_tools=allowed_tools,
                force_tool=force_tool,
                tools=tools,
                no_append=no_append,
            )

            if not response:
                return sub_response
            elif sub_response:
                return response + " " + sub_response
            else:
                return response
        else:
            return response
