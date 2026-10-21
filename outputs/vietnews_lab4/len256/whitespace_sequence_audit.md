# VietNewsSense Sequence Audit — whitespace

## Train-side vocabulary
- vocab_size: 20000
- max_len: 256
- train_orig_len_median: 612.00
- train_orig_len_p95: 1456.95

## Held-out sequence behavior
- n_test: 200
- test_orig_len_median: 563.50
- test_orig_len_p95: 1399.65
- test_unk_rate: 0.0393
- test_truncation_rate: 0.9350
- test_avg_pad_ratio: 0.0197

## Gợi ý đọc kết quả
- UNK rate cao: vocabulary hoặc tokenization chưa bao phủ tốt dữ liệu held-out.
- Truncation rate cao: max_len có thể quá nhỏ.
- Padding ratio cao: max_len có thể lớn hơn cần thiết.
- So sánh hai tokenizer bằng cùng split, vocab_size và max_len.
