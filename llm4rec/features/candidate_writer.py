"""Bound candidate generation memory while retaining all supervised/audit rows."""
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


class CandidateWriter:
    def __init__(self, path: Path, columns: list[str], batch_users: int = 20):
        self.path, self.columns, self.batch_users = path, columns, batch_users
        self.buffer = []
        self.users = 0
        self.writer = None
        self.counts = dict(rows=0, positives=0, retrieved_positives=0, injected_positives=0, negatives=0)

    def append(self, rows):
        self.buffer.extend(rows)
        self.users += 1
        if self.users % self.batch_users == 0:
            self.flush()

    def flush(self):
        if not self.buffer:
            return
        frame = pd.DataFrame(self.buffer, columns=self.columns)
        self.counts['rows'] += len(frame)
        self.counts['positives'] += int(frame.label.sum())
        self.counts['retrieved_positives'] += int(frame.is_retrieved_positive.sum())
        self.counts['injected_positives'] += int(frame.is_injected_positive.sum())
        self.counts['negatives'] += int((frame.label == 0).sum())
        table = pa.Table.from_pandas(frame, preserve_index=False)
        # List types must stay stable even in batches with zero source memberships.
        for name in ['retrieval_sources']:
            idx = table.schema.get_field_index(name)
            table = table.set_column(idx, name, table[name].cast(pa.list_(pa.string())))
        if self.writer is None:
            self.writer = pq.ParquetWriter(self.path, table.schema, compression='zstd')
        self.writer.write_table(table.cast(self.writer.schema))
        self.buffer.clear()

    def finish(self):
        self.flush()
        if self.writer is not None:
            self.writer.close()
        else:
            pd.DataFrame(columns=self.columns).to_parquet(self.path, index=False)
        positives = self.counts['positives']
        return {**self.counts, 'users': self.users,
                'positive_injection_rate': self.counts['injected_positives'] / positives if positives else 0.0,
                'candidate_positive_recall': self.counts['retrieved_positives'] / positives if positives else 0.0}
