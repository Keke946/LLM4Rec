import tempfile
import unittest
from pathlib import Path

import pandas as pd

from llm4rec.data import load_videos, parse_list_cell, validate_video_dataset


class DataLoadingTest(unittest.TestCase):
    def test_parse_list_cell_uses_literal_eval(self):
        self.assertEqual(parse_list_cell("['funny', 'cat']"), ["funny", "cat"])

    def test_load_videos_normalizes_columns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "videos.csv"
            pd.DataFrame(
                [
                    {
                        "video_id": "vid_001",
                        "title": "Funny Cat",
                        "description": "cats",
                        "tags": "['funny', 'cat']",
                        "labels": "['Comedy']",
                    }
                ]
            ).to_csv(path, index=False)

            videos = load_videos(path)
            self.assertEqual(videos.loc[0, "tags"], ["funny", "cat"])
            self.assertIn("full_text", videos.columns)
            self.assertEqual(validate_video_dataset(videos)["rows"], 1)


if __name__ == "__main__":
    unittest.main()
