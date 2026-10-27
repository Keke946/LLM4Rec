import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from llm4rec.retrieval import ItemCFRetriever, PopularRetriever
from llm4rec.retrieval.content_faiss import ContentFaissRetriever
from llm4rec.retrieval.base import BaseRetriever


class RetrieverArtifactTest(unittest.TestCase):
    def test_content_history_survives_save_and_load(self):
        with tempfile.TemporaryDirectory() as directory:
            items = pd.DataFrame(dict(item_id=["a", "b", "c"],
                                      full_text=["pets dogs", "pets cats", "cooking food"]))
            train = pd.DataFrame([dict(user_id="u", item_id="c", label=1, timestamp=1)])
            model = ContentFaissRetriever().fit(train, items)
            path = Path(directory)/"content"
            model.save(path)
            restored = BaseRetriever.load(path)
            self.assertEqual(restored._profile_text("u"), "cooking food")
            self.assertEqual([c.item_id for c in model.retrieve("u", top_k=2)],
                             [c.item_id for c in restored.retrieve("u", top_k=2)])

    def test_saved_retriever_loads_in_fresh_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            items = pd.DataFrame(
                [
                    {"item_id": "i1", "full_text": "action"},
                    {"item_id": "i2", "full_text": "drama"},
                    {"item_id": "i3", "full_text": "comedy"},
                ]
            )
            train = pd.DataFrame(
                [
                    {"user_id": "u1", "item_id": "i1", "label": 1, "timestamp": 1, "split": "train"},
                    {"user_id": "u2", "item_id": "i2", "label": 1, "timestamp": 1, "split": "train"},
                    {"user_id": "u2", "item_id": "i3", "label": 1, "timestamp": 2, "split": "train"},
                ]
            )
            for retriever in [PopularRetriever(), ItemCFRetriever()]:
                retriever.fit(train, items)
                retriever.save(root / retriever.name)

            code = (
                "from llm4rec.retrieval.base import BaseRetriever; "
                f"r=BaseRetriever.load(r'{root / 'popular'}'); "
                "print(len(r.retrieve('u1', top_k=2)))"
            )
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(result.stdout.strip(), "2")


if __name__ == "__main__":
    unittest.main()
