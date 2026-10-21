# VietNewsSense — Long-Context Transfer Check (Lab 4)

| Tokenizer | max_len | N test | p95 (token) | UNK rate | Truncation rate | Avg pad ratio |
|---|---:|---:|---:|---:|---:|---:|
| underthesea | 128 | 200 | 1173.0 | 0.0424 | 0.9900 | 0.0045 |
| underthesea | 256 | 200 | 1173.0 | 0.0424 | 0.9150 | 0.0262 |
| whitespace | 128 | 200 | 1399.6 | 0.0393 | 0.9900 | 0.0042 |
| whitespace | 256 | 200 | 1399.6 | 0.0393 | 0.9350 | 0.0197 |

## Hướng dẫn diễn giải

- Chỉ so sánh **max_len=128 và 256 của cùng một tokenizer**; không trộn tác động của tokenizer và max_len.
- test_len_p95 và test_unk_rate được Lab 3 đo trên **toàn bộ token gốc**, nên không phụ thuộc max_len.
- Truncation giảm chỉ có nghĩa là ít văn bản bị cắt hơn, **không chứng minh** accuracy/F1 tốt hơn.
- Không thể kết luận LSTM/GRU tốt hơn RNN trên VietNewsSense nếu chưa huấn luyện và đánh giá.
- Không commit corpus gốc hoặc dữ liệu nhạy cảm lên GitHub nếu chưa được phép.
