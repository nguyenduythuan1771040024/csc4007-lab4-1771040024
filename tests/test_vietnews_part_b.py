import tempfile
import unittest
from pathlib import Path
import pandas as pd
from scripts.run_vietnews_part_b import heldout_rows, select_articles, hashfile
class AuditTests(unittest.TestCase):
    def test_sampling_reproducible_and_privacy(self):
        df=pd.DataFrame({"content":["tin "*300 for _ in range(40)],"category":["a","b"]*20})
        a=heldout_rows(df,"category",42,30)
        b=heldout_rows(df,"category",42,30)
        self.assertEqual(a["_source_row"].tolist(),b["_source_row"].tolist())
        selected=select_articles(a,"content","category",lambda x:x.split())
        self.assertEqual(len(selected),3)
        self.assertTrue(all("content" not in row for row in selected))
        self.assertTrue(all(r["removed_at_256"]==44 for r in selected))
    def test_hash_changes(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/"a.csv"
            path.write_text("A")
            first=hashfile(path)
            path.write_text("B")
            self.assertNotEqual(first,hashfile(path))
if __name__=="__main__":unittest.main()
