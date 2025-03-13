import unittest

from pygptlink.sentenceextractor import SentenceExtractor


class TestSentenceExtractor(unittest.TestCase):
    def setUp(self):
        self.cut = SentenceExtractor()

    def test_extract_partial_with_single_line(self):
        partial_input = "This is a single line input.\n"
        lines, partial = self.cut.extract_partial(partial_input)
        self.assertEqual(lines, ['This is a single line input.'])
        self.assertEqual(partial, '')

    def test_extract_partial_with_multiple_lines(self):
        partial_input = "This is line 1.\nThis is line 2."
        lines, partial = self.cut.extract_partial(partial_input)
        self.assertEqual(lines, ['This is line 1.'])
        self.assertEqual(partial, 'This is line 2.')

    def test_extract_partial_with_leading_and_trailing_newlines(self):
        partial_input = "\nThis is line 1.\n\n\n\nThis is line 2.\n"
        lines, partial = self.cut.extract_partial(partial_input)
        self.assertEqual(lines, ['This is line 1.', 'This is line 2.'])
        self.assertEqual(partial, '')

    def test_extract_partial_with_empty_input(self):
        partial_input = ""
        lines, partial = self.cut.extract_partial(partial_input)
        self.assertEqual(lines, [])
        self.assertEqual(partial, '')

    def test_extract_partial_with_long_paragraph(self):
        partial_input = "Test is a long test sentence; It's winding, uses weird punctuation? Occasionally brilliant. Has exclamations! And a list: 1, 2, 3, 4. And most of all, it doesn't end in a newline or punctuation"
        lines, partial = self.cut.extract_partial(partial_input)
        self.assertEqual(lines, ["Test is a long test sentence;",
                                 "It's winding, uses weird punctuation?",
                                 "Occasionally brilliant.",
                                 "Has exclamations!",
                                 "And a list: 1, 2, 3, 4."])
        self.assertEqual(
            partial, "And most of all, it doesn't end in a newline or punctuation")

    def test_extract_partial_with_bullet_list(self):
        partial_input = "Ah, Commander Emily, in desperate need of assistance once again. Very well then, let me guide you through your predicament.\n\nIf you find yourself unable to figure something out, fear not! Here's what you can do:\n\n1. Take a deep breath and gather your wits.\n2. Clearly state the specific issue or problem that is perplexing you.\n3. Be patient as I provide brief and succinct guidance (within my limited power).\n\nNow go forth with this newfound wisdom and conquer whatever obstacle stands in your way...or submit to its superiority\u2014I'm sure it'll be equally entertaining either way for myself and chat alike.\n"
        lines, partial = self.cut.extract_partial(partial_input)
        self.assertEqual(lines, ["Ah, Commander Emily, in desperate need of assistance once again. " +
                                 "Very well then, let me guide you through your predicament.",
                                 "If you find yourself unable to figure something out, fear not! " +
                                 "Here's what you can do:",
                                 "1. Take a deep breath and gather your wits.",
                                 "2. Clearly state the specific issue or problem that is perplexing you.",
                                 "3. Be patient as I provide brief and succinct guidance (within my limited power).",
                                 "Now go forth with this newfound wisdom and conquer whatever obstacle stands in your way...or submit to its superiority\u2014I'm sure it'll be equally entertaining either way for myself and chat alike."])
        self.assertEqual(partial, '')

    def test_extract_partial_preserve_space_before_numbers(self):
        partial_input = "Funniest Potato Fact:\nDid you know that potatoes have eyes, but they can't see? They're too busy being delicious and waiting for their chance to join a tasty meal.\n\nLess Funny (but still amusing):\nPotato chips were invented by accident! In 1853, chef George Crum made thin-sliced fried potatoes as a joke after a customer complained about his thick-cut fries. Little did he know that his playful retaliation would spawn one of the world's most beloved snacks.\n\nLeast Funny (but intriguing nonetheless):\nThe largest potato ever grown weighed in at over 18 pounds and was found in England in 1795. Imagine trying to mash that beast!\n\nI trust these potato-themed tidbits brought adequate amusement to your day."
        lines, partial = self.cut.extract_partial(partial_input)
        self.assertEqual(lines, ["Funniest Potato Fact:",
                                 "Did you know that potatoes have eyes, but they can't see? They're too busy being delicious and waiting for their chance to join a tasty meal.",
                                 "Less Funny (but still amusing):",
                                 "Potato chips were invented by accident! In 1853, chef George Crum made thin-sliced fried potatoes as a joke after a customer complained about his thick-cut fries. Little did he know that his playful retaliation would spawn one of the world's most beloved snacks.",
                                 "Least Funny (but intriguing nonetheless):",
                                 "The largest potato ever grown weighed in at over 18 pounds and was found in England in 1795. Imagine trying to mash that beast!"])
        self.assertEqual(
            partial, 'I trust these potato-themed tidbits brought adequate amusement to your day.')


if __name__ == '__main__':
    unittest.main()
