import unittest
from unittest.mock import Mock

from openai.types.chat.chat_completion_chunk import ChoiceDeltaToolCall, ChoiceDeltaToolCallFunction
from pygptlink.gpt_tool_definition import GPTToolDefinition


class TestToolDefinition(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.valid_tool_call = ChoiceDeltaToolCall(
            index=0,
            id="123",
            function=ChoiceDeltaToolCallFunction(
                name="ExampleName", arguments='{"arg1": "value"}')
        )

    async def test_invoke_success_void_return(self):
        # Example tool definition with required argument
        async def callback_example(arg1: str):
            return None

        tool_def = GPTToolDefinition(name="ExampleName", description="Example Tool", callback=callback_example,
                                     required_args=[{"name": "arg1", "description": "Argument 1", "type": "string"}])

        result = await tool_def.invoke(self.valid_tool_call)
        self.assertEqual(result, "The tool call completed successfully.")

    async def test_invoke_success_string_return(self):
        # Example tool definition with required argument
        async def callback_example(arg1: str):
            return "Hello World!"

        tool_def = GPTToolDefinition(name="ExampleName", description="Example Tool", callback=callback_example,
                                     required_args=[{"name": "arg1", "description": "Argument 1", "type": "string"}])

        result = await tool_def.invoke(self.valid_tool_call)
        self.assertEqual(result, "The tool call returned: Hello World!")

    async def test_invoke_missing_required_argument(self):
        # Example tool definition with missing required argument
        async def callback_example(arg1: str):
            return f"Processed: {arg1}"

        tool_def = GPTToolDefinition(name="ExampleName", description="Example Tool", callback=callback_example,
                                     required_args=[{"name": "arg1", "description": "Argument 1", "type": "string"}])

        # Creating a tool call without the required argument
        invalid_tool_call = ChoiceDeltaToolCall(
            index=0,
            id="456",
            function=ChoiceDeltaToolCallFunction(
                name="ExampleName", arguments='{}')
        )

        result = await tool_def.invoke(invalid_tool_call)
        self.assertEqual(
            result, "The tool call threw an exception: Missing required arguments: ['arg1']")

    async def test_invoke_exception(self):
        # Example tool definition with an exception in the callback
        def callback_example(arg1: str):
            raise ValueError("Example exception")

        tool_def = GPTToolDefinition(name="ExampleName", description="Example Tool", callback=callback_example,
                                     required_args=[{"name": "arg1", "description": "Argument 1", "type": "string"}])

        result = await tool_def.invoke(self.valid_tool_call)
        self.assertEqual(
            result, "The tool call threw an exception: Example exception")

    async def test_describe_with_arguments(self):
        # Example tool definition with arguments
        tool_def = GPTToolDefinition(
            name="ExampleName",
            description="Example Tool",
            callback=None,
            required_args=[
                {"name": "arg1", "description": "Argument 1", "type": "string"},
                {"name": "arg2", "description": "Argument 2", "type": "integer"}
            ],
            optional_args=[
                {"name": "arg3", "description": "Argument 3", "type": "boolean"}
            ]
        )

        result = tool_def.describe()
        expected_result = {
            "type": "function",
            "function": {
                "name": "ExampleName",
                "description": "Example Tool",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string", "description": "Argument 1"},
                        "arg2": {"type": "integer", "description": "Argument 2"},
                        "arg3": {"type": "boolean", "description": "Argument 3"}
                    },
                    "required": ["arg1", "arg2"]
                }
            }
        }
        self.assertEqual(result, expected_result)

    async def test_describe_without_arguments(self):
        # Example tool definition without arguments
        tool_def = GPTToolDefinition(
            name="ExampleName", description="Example Tool", callback=lambda: None)

        result = tool_def.describe()
        expected_result = {
            "type": "function",
            "function": {
                "name": "ExampleName",
                "description": "Example Tool"
            }
        }
        self.assertEqual(result, expected_result)


if __name__ == '__main__':
    unittest.main()
