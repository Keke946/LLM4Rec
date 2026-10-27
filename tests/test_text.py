import unittest

from llm4rec.text import extract_interest_tags, normalize_tags


class TextExtractionTest(unittest.TestCase):
    def test_extracts_chinese_aliases(self):
        tags = extract_interest_tags("我想看猫猫搞笑视频", vocabulary=["cat", "funny", "sports"])
        self.assertEqual(tags, ["cat", "funny"])

    def test_normalizes_list_input(self):
        self.assertEqual(normalize_tags([" Cat ", "搞笑", "cat"]), ["cat", "funny"])


if __name__ == "__main__":
    unittest.main()
