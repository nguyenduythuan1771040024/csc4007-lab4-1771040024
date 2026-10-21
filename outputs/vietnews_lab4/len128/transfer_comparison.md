# VietNewsSense — Sequence Transfer Comparison

Cùng dataset, cùng split, cùng vocab limit và cùng max_len; chỉ thay tokenization.

| tokenizer   |   vocab_size |   test_len_median |   test_len_p95 |   test_unk_rate |   test_truncation_rate |   test_avg_pad_ratio |
|:------------|-------------:|------------------:|---------------:|----------------:|-----------------------:|---------------------:|
| whitespace  |        20000 |          563.5000 |      1399.6500 |          0.0393 |                 0.9900 |               0.0042 |
| underthesea |        20000 |          480.5000 |      1173.0000 |          0.0424 |                 0.9900 |               0.0045 |

## Reflection prompts
1. Tokenization làm thay đổi vocabulary và sequence length như thế nào?
2. Với cùng max_len, cách nào tạo nhiều truncation hoặc padding hơn?
3. UNK rate held-out thay đổi ra sao?
4. Những thay đổi này có thể ảnh hưởng Embedding + RNN như thế nào?

## Lưu ý
Đây là sequence transfer check, không phải cuộc thi accuracy giữa hai tokenizer.
Không cần train thêm một RNN đầy đủ trên VietNewsSense trong Lab 3.
