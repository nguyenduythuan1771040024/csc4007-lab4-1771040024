# VietNewsSense Long-Context Transfer — Lab 4

- Họ tên / MSSV / lớp: Nguyễn Duy Thuận / 1771040024 / KHMT-1701
- Nguồn dữ liệu, giấy phép, schema: VietNewsSense CSV thật từ giảng viên, columns `content` và `category`; không công khai toàn văn hay commit raw corpus.
- Đường dẫn manifest và dataset fingerprint: `outputs/vietnews_lab4/manifest.json`; dataset_sha256 = `ed2d3fdc37dbfb2e7be390cb70814c34d7acdab54b017b645e506bb9eb8eb06d`; test_rows_sha256 = `c4721ec24b129063fa2401b0bce27a78599e55492d380a13cc3ae37dc3165258`
- Seed / max_rows / vocab_size: 42 / 1000 / 20000; tokenizer chính là `underthesea`

## Bảng số liệu

| Tokenizer | max_len | N test | p95 (token) | UNK rate | Truncation rate | Padding ratio |
|---|---:|---:|---:|---:|---:|---:|
| whitespace | 128 | 200 | 1399.6 | 0.0393 | 0.9900 | 0.0042 |
| whitespace | 256 | 200 | 1399.6 | 0.0393 | 0.9350 | 0.0197 |
| underthesea | 128 | 200 | 1173.0 | 0.0424 | 0.9900 | 0.0045 |
| underthesea | 256 | 200 | 1173.0 | 0.0424 | 0.9150 | 0.0262 |

## Ba bài báo dài (theo article_candidates.csv)

| source_csv_row | article_sha256 | Chuyên mục | Tổng token | Bằng chứng A/B/C và ảnh hưởng 128/256 |
|---:|---|---|---:|---|
| 306 | `bf4d0840e0ef4a6b4b2a96e87a363973eb6722e77096233ce6781b8be3315efd` | Kinh doanh | 1935 | max_len=128 bỏ 1807 token; max_len=256 bỏ 1679 token |
| 1381 | `e118738fa27b8635cada33c28902e0ee98382626bb34c14d5bfbeb31abcd2e7d` | Kinh doanh | 1727 | max_len=128 bỏ 1599 token; max_len=256 bỏ 1471 token |
| 954 | `3f5aa84e20926b3d993b5255d4ebf7b743b53d89d4bf1ff14a4bf18c25be3d` | Kinh doanh | 1619 | max_len=128 bỏ 1491 token; max_len=256 bỏ 1363 token |

## Nhận xét kỹ thuật

With max_len=256, truncation decreases from 0.9900 to 0.9150 for underthesea, while average padding ratio remains low (0.0262); with max_len=128, nearly every document is cut. UNK rate does not change when max_len changes because Lab 3 computes UNK on the full original string. Even so, max_len=256 retains only about 15–18% of a long business article; lowering max_len to 128 is more aggressive and keeps less context, but increasing it to 256 does not restore tokens beyond position 256. Therefore truncation is a coverage loss, whereas long-range dependency concerns interactions between tokens that both still remain. For this dataset, 256 is preferable to 128 under the tested setup, yet a production Vietnamese news classifier should consider higher max_len or chunking/hierarchical aggregation rather than assume 256 tokens are sufficient.

## Checklist

- [x] Dữ liệu thật và fingerprint có thể truy vết
- [x] Không công khai toàn văn
- [x] Ba bài có source row và SHA-256
- [x] Kết luận dựa trên số liệu thật