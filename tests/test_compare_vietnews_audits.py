import csv
import tempfile
import unittest
from pathlib import Path

from scripts.compare_vietnews_audits import compare_audits, save_results


FIELDS = ["tokenizer", "n_total", "n_train", "n_test", "n_classes", "vocab_size", "max_len", "test_len_p95", "test_unk_rate", "test_truncation_rate", "test_avg_pad_ratio"]


def csv_file(path, max_len, *, n_test=20, unk_rate=0.08, truncation=None):
    row = {
        "tokenizer": "whitespace", "n_total": 100, "n_train": 80, "n_test": n_test,
        "n_classes": 3, "vocab_size": 150, "max_len": max_len,
        "test_len_p95": 312.0, "test_unk_rate": unk_rate,
        "test_truncation_rate": (0.7 if max_len == 128 else 0.3) if truncation is None else truncation,
        "test_avg_pad_ratio": 0.16 if max_len == 128 else 0.31,
    }
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(row)


class VietNewsComparisonTests(unittest.TestCase):
    def test_compare_and_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            a, b = base / "len128.csv", base / "len256.csv"
            csv_file(a, 128)
            csv_file(b, 256)
            rows = compare_audits(a, b)
            self.assertEqual([r["max_len"] for r in rows], [128, 256])
            self.assertEqual(rows[1]["test_len_p95"], 312.0)
            save_results(rows, base / "out")
            self.assertTrue((base / "out" / "vietnews_length_comparison.csv").exists())
            self.assertIn("không phụ thuộc max_len", (base / "out" / "vietnews_length_comparison.md").read_text(encoding="utf-8"))

    def test_detect_mismatched_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            a, b = base / "len128.csv", base / "len256.csv"
            csv_file(a, 128)
            csv_file(b, 256, n_test=22)
            with self.assertRaisesRegex(ValueError, "n_test"):
                compare_audits(a, b)

    def test_detect_mismatched_unk_rate(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            a, b = base / "len128.csv", base / "len256.csv"
            csv_file(a, 128)
            csv_file(b, 256, unk_rate=0.10)
            with self.assertRaisesRegex(ValueError, "test_unk_rate"):
                compare_audits(a, b)


if __name__ == "__main__":
    unittest.main()
