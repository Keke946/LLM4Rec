import tempfile
import unittest
from pathlib import Path
import pandas as pd

from llm4rec.features.candidate_writer import CandidateWriter


class CandidateWriterTest(unittest.TestCase):
    def test_multiple_batches_keep_empty_and_nonempty_source_lists(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'candidates.parquet'
            columns = ['user_id', 'item_id', 'label', 'is_injected_positive',
                       'is_retrieved_positive', 'retrieval_sources']
            writer = CandidateWriter(path, columns, batch_users=2)
            for index in range(5):
                writer.append([dict(user_id=str(index), item_id='i', label=1,
                                    is_injected_positive=int(index < 2),
                                    is_retrieved_positive=int(index >= 2),
                                    retrieval_sources=[] if index < 2 else ['trend'])])
            stats = writer.finish()
            frame = pd.read_parquet(path)
            self.assertEqual(len(frame), 5)
            self.assertEqual(stats['retrieved_positives'], 3)
            self.assertAlmostEqual(stats['candidate_positive_recall'], .6)
            self.assertEqual(list(frame.iloc[-1].retrieval_sources), ['trend'])
