import unittest
from utf8_tokenizer import UTF8Tokenizer


class TestUTF8Tokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = UTF8Tokenizer()

    def test_encode(self):
        # Test encoding of a specific string
        result = self.tokenizer.encode('arithmetic_training_data.jsonl')

        # Add assertions here to check the result
        # For example:
        self.assertIsNotNone(result)
        self.assertIsInstance(result, list)  # Assuming encode returns a list

        # You might want to add more specific assertions based on what you expect
        # the result to be. For example:
        # self.assertEqual(result, [expected_token_ids])

        # If you want to print the result (not typically done in unit tests, but can be useful for debugging)
        print(result)


if __name__ == '__main__':
    unittest.main()