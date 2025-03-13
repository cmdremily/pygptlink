import os
import tempfile
import unittest

import tiktoken
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import ChatCompletionMessageToolCall, Function
from pygptlink.gpt_context import GPTContext


class TestGPTContext(unittest.TestCase):

    def setUp(self):
        # Create temporary files for testing
        self.persona_file = tempfile.mktemp()
        self.context_file = tempfile.mktemp()
        self.completion_log_file = tempfile.mktemp()

        # Initialize GPTContext with some sample values
        with open(self.persona_file, 'w') as persona_file:
            persona_file.write("Sample persona content")

        with open(self.context_file, 'w') as persona_file:
            persona_file.write("")

        # Initialize GPTContext with some sample values
        self.cut = GPTContext(
            persona_file=self.persona_file,
            context_file=self.context_file,
            model="gpt-3.5-turbo-0613",
            max_tokens=1024,
            max_response_tokens=100,
            completion_log_file=self.completion_log_file
        )

    def tearDown(self):
        # Clean up temporary files if they exist
        if os.path.exists(self.persona_file):
            os.remove(self.persona_file)
        if os.path.exists(self.context_file):
            os.remove(self.context_file)
        if os.path.exists(self.completion_log_file):
            os.remove(self.completion_log_file)

    def test_empty_context(self):
        actual = self.cut.oai_messages()
        expected = [{
            "role": "system",
            "content": "Sample persona content"
        }]
        self.assertEqual(actual, expected)

    def test_empty_context_additional_system_prompt(self):
        actual = self.cut.oai_messages("Other prompt")
        expected = [{
            "role": "system",
            "content": "Sample persona content"
        }, {
            "role": "system",
            "content": "Other prompt"
        }]
        self.assertEqual(actual, expected)

    def test_context_persistence(self):
        self.cut.append_user_prompt(user="user1", content="test1")
        self.cut.append_user_prompt(user="user2", content="test2")

        cut = GPTContext(
            persona_file=self.persona_file,
            context_file=self.context_file,
            model="gpt-3.5-turbo-0613",
            max_tokens=1024,
            max_response_tokens=100,
            completion_log_file=self.completion_log_file
        )
        actual = cut.oai_messages("Extra")

        expected = [{
            "role": "user",
            "name": "user1",
            "content": "test1"
        }, {
            "role": "system",
            "content": "Sample persona content"
        }, {
            "role": "system",
            "content": "Extra"
        }, {
            "role": "user",
            "name": "user2",
            "content": "test2"
        }]
        self.assertEqual(actual, expected)

    def test_append_completion(self):
        # Create a mock ChatCompletion for testing
        completion = ChatCompletion(
            model="gpt-3.5-turbo-0613",
            id="completion_id",
            object="chat.completion",
            created=1234567890,
            choices=[Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None
                ),
            )
            ]
        )

        self.cut.append_completion(completion)

        actual = self.cut.oai_messages("Extra")
        expected = [{
            "role": "system",
            "content": "Sample persona content"
        }, {
            "role": "system",
            "content": "Extra"
        }, {
            "role": "assistant"
        }]
        self.assertEqual(actual, expected)

    def test_append_completion_with_content(self):
        # Create a mock ChatCompletion for testing
        completion = ChatCompletion(
            model="gpt-3.5-turbo-0613",
            id="completion_id",
            object="chat.completion",
            created=1234567890,
            choices=[Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content="Hello world!",
                    role="assistant",
                ),
            )
            ]
        )

        self.cut.append_completion(completion)

        actual = self.cut.oai_messages("Extra")
        expected = [{
            "role": "system",
            "content": "Sample persona content"
        }, {
            "role": "system",
            "content": "Extra"
        }, {
            "role": "assistant",
            "content": "Hello world!"
        }]
        self.assertEqual(actual, expected)

    def test_append_completion_tool_call(self):
        # Create a mock ChatCompletion for testing
        completion = ChatCompletion(
            model="gpt-3.5-turbo-0613",
            id="completion_id",
            object="chat.completion",
            created=1234567890,
            choices=[Choice(
                finish_reason="stop",
                index=0,
                message=ChatCompletionMessage(
                    content=None,
                    role="assistant",
                    tool_calls=[ChatCompletionMessageToolCall(
                        id="toolid", type="function", function=Function(name="fun", arguments="args"))]
                ),
            )
            ]
        )

        self.cut.append_completion(completion)

        actual = self.cut.oai_messages("Extra")
        expected = [{
            "role": "system",
            "content": "Sample persona content"
        }, {
            "role": "system",
            "content": "Extra"
        }, {
            "role": "assistant",
            "tool_calls": [
                {"id": "toolid",
                 "type": "function",
                 "function": {"name": "fun", "arguments": "args"}
                 }
            ]
        }]
        self.assertEqual(actual, expected)

    def test_single_prompt(self):
        self.cut.append_user_prompt(
            user="User123", content="User prompt content")

        actual = self.cut.oai_messages()
        expected = [{
            "role": "system",
            "content": "Sample persona content"
        }, {
            "role": "user",
            "name": "User123",
            "content": "User prompt content"
        }]
        self.assertEqual(actual, expected)

    def test_longer_history(self):
        self.cut.append_user_prompt(user="user1", content="content1")
        self.cut.append_user_prompt(user="user2", content="content2")
        self.cut.append_user_prompt(user="user3", content="content3")

        actual = self.cut.oai_messages()
        expected = [{
            "role": "user",
            "name": "user1",
            "content": "content1"
        }, {
            "role": "user",
            "name": "user2",
            "content": "content2"
        }, {
            "role": "system",
            "content": "Sample persona content"
        }, {
            "role": "user",
            "name": "user3",
            "content": "content3"
        }]
        self.assertEqual(actual, expected)

    def test_last_message_is_tool(self):
        self.cut.append_tool_response("id", "content")
        self.cut.append_user_prompt(user="user2", content="content2")
        self.cut.append_user_prompt(user="user3", content="content3")

        actual = self.cut.oai_messages()
        expected = [{
            "role": "user",
            "name": "user2",
            "content": "content2"
        }, {
            "role": "system",
            "content": "Sample persona content"
        }, {
            "role": "user",
            "name": "user3",
            "content": "content3"
        }]
        self.assertEqual(actual, expected)

    def test_last_messages_are_tool(self):
        self.cut.append_tool_response("id", "content")
        self.cut.append_tool_response("id2", "content2")
        self.cut.append_user_prompt(user="user2", content="content2")
        self.cut.append_user_prompt(user="user3", content="content3")

        actual = self.cut.oai_messages()
        expected = [{
            "role": "user",
            "name": "user2",
            "content": "content2"
        }, {
            "role": "system",
            "content": "Sample persona content"
        }, {
            "role": "user",
            "name": "user3",
            "content": "content3"
        }]
        self.assertEqual(actual, expected)

    def test_single_promp_additional(self):
        self.cut.append_user_prompt(
            user="User123", content="User prompt content")

        actual = self.cut.oai_messages("Other prompt")
        expected = [{
            "role": "system",
            "content": "Sample persona content"
        }, {
            "role": "system",
            "content": "Other prompt"
        }, {
            "role": "user",
            "name": "User123",
            "content": "User prompt content"
        }]
        self.assertEqual(actual, expected)

    def test_append_tool(self):
        self.cut.append_tool_response("tool_id", "Tool content")

        actual = self.cut.oai_messages("Other prompt")
        expected = [{
            "role": "system",
            "content": "Sample persona content"
        }, {
            "role": "system",
            "content": "Other prompt"
        }, {
            "role": "tool",
            "tool_call_id": "tool_id",
            "content": "Tool content"
        }]
        self.assertEqual(actual, expected)

    def test_dont_split_toolcalls(self):
        self.cut.append_completion(ChatCompletion(
            **{
                "id": "chatcmpl-8hnsHMg1kfd96VCz7WRRQSEkehWS6",
                "choices": [{
                    "finish_reason": "stop",
                    "index": 0,
                    "message": {
                            "content": None,
                            "role": "assistant",
                            "function_call": None,
                            "tool_calls": [
                                {
                                    "id": "call1",
                                    "function": {
                                        "arguments": "args1",
                                        "name": "fun1"
                                    },
                                    "type": "function",
                                    "index": 0},
                                {
                                    "id": "call2",
                                    "function": {
                                        "arguments": "args2",
                                        "name": "fun2"
                                    },
                                    "type": "function",
                                    "index": 0},
                                {
                                    "id": "call3",
                                    "function": {
                                        "arguments": "args3",
                                        "name": "fun3"
                                    },
                                    "type": "function",
                                    "index": 0}]
                            }}],
                "created": 1705450513,
                "model": "gpt-3.5-turbo-0613",
                "object": "chat.completion",
                "system_fingerprint": None,
                "usage": None
            }
        ))
        self.cut.append_tool_response("call1", "ret1")
        self.cut.append_tool_response("call2", "ret2")
        self.cut.append_tool_response("call3", "ret3")

        actual = self.cut.oai_messages("Other prompt")
        expected = [
            {
                "role": "system",
                "content": "Sample persona content"
            },
            {
                "role": "system",
                "content": "Other prompt"
            },
            {
                "role": "assistant", "tool_calls": [
                    {
                        "id": "call1",
                        "type": "function",
                        "function": {
                            "name": "fun1",
                            "arguments": "args1"
                        }
                    },
                    {
                        "id": "call2",
                        "type": "function",
                        "function": {
                            "name": "fun2",
                            "arguments": "args2"
                        }
                    },
                    {
                        "id": "call3",
                        "type": "function",
                        "function": {
                            "name": "fun3",
                            "arguments": "args3"
                        }
                    }
                ]
            },
            {
                "role": "tool",
                "tool_call_id": "call1",
                "content": "ret1"
            },
            {
                "role": "tool",
                "tool_call_id": "call2",
                "content": "ret2"
            },
            {
                "role": "tool",
                "tool_call_id": "call3",
                "content": "ret3"
            }]
        self.assertEqual(actual, expected)
