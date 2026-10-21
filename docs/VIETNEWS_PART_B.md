# CSC4007 Lab 4 – VietNewsSense Part B (bản cập nhật)

Phần A vẫn dùng IMDB và LSTM/GRU. Phần B chỉ đánh giá rủi ro cắt chuỗi, chưa huấn luyện phân loại nhiều lớp tiếng Việt.

## Chuẩn bị
Clone Lab 3 và Lab 4 cạnh nhau. Cài requirements của Lab 3 bao gồm underthesea. Sử dụng CSV VietNewsSense THẬT do giảng viên cung cấp; ví dụ chỉ giả định tên cột content và category.

## Chạy một lệnh tại repo Lab 4

    python scripts/run_vietnews_part_b.py --lab3_repo ../csc4007_lab3_starter_kit --data_path ../csc4007_lab3_starter_kit/data/raw/vietnewssense.csv --text_col content --label_col category --seed 42 --max_rows 1000 --output_dir outputs/vietnews_lab4

Script dùng cùng dữ liệu và seed cho hai cấu hình 128/256, ghi manifest.json chứa dataset SHA-256 và fingerprint các dòng test, bảng so sánh và article_candidates.csv.
article_candidates.csv có tối đa 3 bài dài thuộc held-out, chỉ chứa số dòng CSV tính cả header (bắt đầu từ 2), nhãn, SHA-256 nội dung và số token. Không phát hành toàn văn.

## Minh chứng
Điền reports/vietnews_transfer_lab4.md, có 3 bài báo (vị trí token A=1–128, B=129–256, C=257+), và giải thích sự khác nhau giữa truncation và long-range dependency. Nếu không đủ 3 bài dài, cần đề nghị bổ sung dữ liệu chứ không bịa.

Không dùng kết quả IMDB để kết luận khả năng phân loại VietNewsSense. Không commit dữ liệu thô. Phần B được phân bổ 10/100 điểm trong rubric đề xuất; giảng viên phải công bố trước khi áp dụng.

## Giới hạn
Chưa chạy kiểm thử end-to-end trên VietNewsSense thật cho đến khi có CSV của giảng viên. CI chỉ chạy kiểm tra với dữ liệu mô phỏng.