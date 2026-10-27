import tempfile
from pathlib import Path
import unittest
from llm4rec.datasets.captions import read_kuairec_captions

HEADER='video_id,manual_cover_text,caption,topic_tag,first_level_category_id,first_level_category_name,second_level_category_id,second_level_category_name,third_level_category_id,third_level_category_name\n'


class CaptionParserTest(unittest.TestCase):
    def test_reconstructs_only_known_newline_shape(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'captions.csv'
            path.write_text(HEADER+'0,cover,"quoted, caption",[],1,a,2,b,3,c\n1,cover,first line\nsecond line,[],1,a,2,b,3,c\n')
            frame,repairs=read_kuairec_captions(path)
            self.assertEqual(frame.caption.tolist(),['quoted, caption','first line\nsecond line'])
            self.assertEqual(frame.video_id.tolist(),[0,1])
            self.assertEqual(len(repairs),1)

    def test_unrecognized_malformed_row_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'captions.csv';path.write_text(HEADER+'0,incomplete\n')
            with self.assertRaises(ValueError):read_kuairec_captions(path)


if __name__=='__main__':unittest.main()
