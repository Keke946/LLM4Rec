import unittest

import numpy as np

from llm4rec.ranking.gbdt_features import sources_from_value


class GBDTFeatureTest(unittest.TestCase):
    def test_sources_from_value_handles_parquet_like_values(self):
        self.assertEqual(sources_from_value(np.array(["trend"], dtype=object)), ["trend"])
        self.assertEqual(sources_from_value("['trend']"), ["trend"])
        self.assertEqual(sources_from_value("trend+metadata"), ["trend", "metadata"])


if __name__ == "__main__":
    unittest.main()
