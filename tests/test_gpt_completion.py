import unittest

from pygptlink.gpt_completion import GPTCompletion


class TestGPTCompletion(unittest.TestCase):

    def test_merge_dicts_with_new_key(self):
        current = {"key1": "value1"}
        delta = {"key2": "value2"}
        GPTCompletion._merge_dicts(current, delta)
        self.assertEqual(current, {"key1": "value1", "key2": "value2"})

    def test_merge_dicts_with_string_value(self):
        current = {"key1": "hello"}
        delta = {"key1": "world"}
        GPTCompletion._merge_dicts(current, delta)
        self.assertEqual(current, {"key1": "helloworld"})

    def test_merge_dicts_with_dict_value(self):
        current = {"key1": {"subkey1": "value1"}}
        delta = {"key1": {"subkey2": "value2"}}
        GPTCompletion._merge_dicts(current, delta)
        self.assertEqual(
            current, {"key1": {"subkey1": "value1", "subkey2": "value2"}})

    def test_merge_dicts_with_list_value(self):
        current = {"key1": [{"index": 1, "value": "v1"}]}
        delta = {"key1": [{"index": 2, "value": "v2"}]}
        GPTCompletion._merge_dicts(current, delta)
        self.assertEqual(
            current, {"key1": [{"index": 1, "value": "v1"}, {"index": 2, "value": "v2"}]})

    def test_merge_dicts_x(self):
        current = {'id': 'chatcmpl-8i0TCMGMjvA2meWFU6opHa0aMbtrG',
                   'choices': [
                       {'delta': {'content': None,
                                  'function_call': None,
                                  'role': 'assistant',
                                  'tool_calls': None},
                        'finish_reason': None,
                        'index': 0,
                        'logprobs': None}],
                   'created': 1705498930,
                   'model': 'gpt-3.5-turbo-1106',
                   'object': 'chat.completion.chunk',
                   'system_fingerprint': 'fp_fe56e538d5'}
        delta = {'id': 'chatcmpl-8i0TCMGMjvA2meWFU6opHa0aMbtrG',
                 'choices': [
                     {'delta': {'content': None,
                                'function_call': None,
                                'role': None,
                                'tool_calls': [{'index': 0,
                                                'id': 'call_hUpofswY2nmZnBXF3KmPHIc7',
                                                'function': {'arguments': '', 'name': 'ban'},
                                                'type': 'function'}]
                                },
                      'finish_reason': None,
                      'index': 0,
                      'logprobs': None}],
                 'created': 1705498930,
                 'model': 'gpt-3.5-turbo-1106',
                 'object': 'chat.completion.chunk',
                 'system_fingerprint': 'fp_fe56e538d5'}
        GPTCompletion._merge_dicts(current, delta)
        self.assertEqual(current, {'id': 'chatcmpl-8i0TCMGMjvA2meWFU6opHa0aMbtrG',
                                   'choices': [
                                       {'delta': {
                                           'content': None,
                                           'function_call': None,
                                           'role': 'assistant',
                                           'tool_calls': [{'index': 0,
                                                           'id': 'call_hUpofswY2nmZnBXF3KmPHIc7',
                                                           'function': {'arguments': '', 'name': 'ban'},
                                                           'type': 'function'}]},
                                        'finish_reason': None,
                                        'index': 0,
                                        'logprobs': None}],
                                   'created': 1705498930,
                                   'model': 'gpt-3.5-turbo-1106',
                                   'object': 'chat.completion.chunk',
                                   'system_fingerprint': 'fp_fe56e538d5'})
