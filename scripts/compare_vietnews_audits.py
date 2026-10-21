"""Compare VietNewsSense sequence audit outputs (max_len=128 vs 256).

Uses the CSVs produced by Lab 3's run_vietnews_sequence_audit.py.
Does not train a model or require access to the raw news corpus.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

REQUIRED = {
    "tokenizer", "n_total", "n_train", "n_test", "n_classes",
    "vocab_size", "max_len", "test_len_p95", "test_unk_rate",
    "test_truncation_rate", "test_avg_pad_ratio",
}
IDENTICAL = ("n_total", "n_train", "n_test", "n_classes", "vocab_size")
METRICS = ("test_len_p95", "test_unk_rate", "test_truncation_rate", "test_avg_pad_ratio")


def load_audit(path: Path, expected_max_len: int) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Không tìm thấy file: {path}")
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path}: thiếu cột {sorted(missing)}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            tokenizer = row["tokenizer"].strip()
            if not tokenizer or tokenizer in rows:
                raise ValueError(f"{path}: tokenizer bị trống/trùng: {tokenizer!r}")
            if int(row["max_len"]) != expected_max_len:
                raise ValueError(f"{path}: {tokenizer} phải có max_len={expected_max_len}")
            for field in IDENTICAL:
                int(row[field])
            for field in METRICS:
                float(row[field])
            rows[tokenizer] = row
    if not rows:
        raise ValueError(f"{path}: file không có kết quả")
    return rows


def compare_audits(file128: Path, file256: Path) -> list[dict[str, str | float | int]]:
    audit128 = load_audit(file128, 128)
    audit256 = load_audit(file256, 256)
    if set(audit128) != set(audit256):
        raise ValueError("Hai lần chạy dùng tập tokenizer khác nhau")
    combined = []
    for tokenizer in sorted(audit128):
        a, b = audit128[tokenizer], audit256[tokenizer]
        for field in IDENTICAL:
            if int(a[field]) != int(b[field]):
                raise ValueError(f"{tokenizer}: {field} khác nhau; kiểm tra dataset/seed/split")
        # Lab 3 tính p95 và UNK trên toàn bộ token gốc (trước truncation).
        # Do đó hai số này phải giữ nguyên nếu dữ liệu/split/vocab/tokenizer như nhau.
        for field in ("test_len_p95", "test_unk_rate"):
            if abs(float(a[field]) - float(b[field])) > 1e-6:
                raise ValueError(f"{tokenizer}: {field} thay đổi; kiểm tra tính công bằng")
        if float(b["test_truncation_rate"]) > float(a["test_truncation_rate"]) + 1e-6:
            raise ValueError(f"{tokenizer}: truncation tăng bất thường khi tăng max_len")
        for row in (a, b):
            combined.append({
                "tokenizer": tokenizer,
                "max_len": int(row["max_len"]),
                "n_test": int(row["n_test"]),
                "test_len_p95": float(row["test_len_p95"]),
                "test_unk_rate": float(row["test_unk_rate"]),
                "test_truncation_rate": float(row["test_truncation_rate"]),
                "test_avg_pad_ratio": float(row["test_avg_pad_ratio"]),
            })
    return combined


def save_results(rows: list[dict[str, str | float | int]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = ["tokenizer", "max_len", "n_test", "test_len_p95", "test_unk_rate", "test_truncation_rate", "test_avg_pad_ratio"]
    csv_path = output_dir / "vietnews_length_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    header = "| Tokenizer | max_len | N test | p95 (token) | UNK rate | Truncation rate | Avg pad ratio |"
    sep = "|---|---:|---:|---:|---:|---:|---:|"
    data = [
        f"| {r['tokenizer']} | {r['max_len']} | {r['n_test']} | {r['test_len_p95']:.1f} | "
        f"{r['test_unk_rate']:.4f} | {r['test_truncation_rate']:.4f} | {r['test_avg_pad_ratio']:.4f} |"
        for r in rows
    ]
    lines = [
        "# VietNewsSense — Long-Context Transfer Check (Lab 4)", "", header, sep, *data, "",
        "## Hướng dẫn diễn giải", "",
        "- Chỉ so sánh **max_len=128 và 256 của cùng một tokenizer**; không trộn tác động của tokenizer và max_len.",
        "- test_len_p95 và test_unk_rate được Lab 3 đo trên **toàn bộ token gốc**, nên không phụ thuộc max_len.",
        "- Truncation giảm chỉ có nghĩa là ít văn bản bị cắt hơn, **không chứng minh** accuracy/F1 tốt hơn.",
        "- Không thể kết luận LSTM/GRU tốt hơn RNN trên VietNewsSense nếu chưa huấn luyện và đánh giá.",
        "- Không commit corpus gốc hoặc dữ liệu nhạy cảm lên GitHub nếu chưa được phép.", "",
    ]
    (output_dir / "vietnews_length_comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Created: {csv_path} and {output_dir / 'vietnews_length_comparison.md'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tổng hợp audit 128/256 token từ Lab 3 (không huấn luyện mô hình)")
    parser.add_argument("--len128", type=Path, required=True, help="transfer_comparison.csv với max_len=128")
    parser.add_argument("--len256", type=Path, required=True, help="transfer_comparison.csv với max_len=256")
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/vietnews_lab4"))
    args = parser.parse_args()
    save_results(compare_audits(args.len128, args.len256), args.output_dir)


if __name__ == "__main__":
    main()
