import sys
import unittest

sys.path.append('../nlutil')

from nlutil.nlutil import tfidf, tokenize

class TestNLUtil(unittest.TestCase):
    def test_tokenize(self):
        # Normal test
        test_input_1 = "the quick brown fox jumped over the lazy dog"
        expected_output_1 = ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]
        test_output_1 = tokenize(test_input_1)
        self.assertEqual(test_output_1, expected_output_1)
        # Test mixed uppercase/lowercase
        test_input_2 = "tHe quick BROwn fox juMPed OVER the lazy dog"
        expected_output_2 = ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]
        test_output_2 = tokenize(test_input_2)
        self.assertEqual(test_output_2, expected_output_2)
        # Test random punctuation
        test_input_3 = "the quick! brown fox jumped over the; lazy dog."
        expected_output_3 = ["the", "quick", "brown", "fox", "jumped", "over", "the", "lazy", "dog"]
        test_output_3 = tokenize(test_input_3)
        self.assertEqual(test_output_3, expected_output_3)

    def test_tfidf(self):
        document_1 = "an apple is a delicious fruit"
        corpus_1 = ["a pear is a tasty fruit", "an ant is an animal", "a strawberry is a delicious fruit"]
        expected_output_1 = {"an": 1/4, "apple": 1/2, "is": 1/5, "a": 1/6, "delicious": 1/3, "fruit": 1/4}
        output_1 = tfidf(document_1, corpus_1)
        self.assertEqual(output_1, expected_output_1)
        document_2 = "an apple is a delicious fruit"
        corpus_2 = ["a pear is a tasty fruit; an ant is an animal; a strawberry is a delicious fruit"]
        expected_output_2 = {"an": 1/4, "apple": 1/2, "is": 1/5, "a": 1/6, "delicious": 1/3, "fruit": 1/4}
        output_2 = tfidf(document_2, corpus_2)
        self.assertEqual(output_2, expected_output_2)


if __name__ == "__main__":
    unittest.main()
