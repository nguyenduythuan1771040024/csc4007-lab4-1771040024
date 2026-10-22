# CSC4007 — Lab 4 Report Part B: VietNewsSense Long-Context Transfer Audit

## 1. Thông tin sinh viên & Thiết lập kiểm toán thực nghiệm

- **Họ và tên:** Nguyễn Duy Thuận
- **MSSV:** 1771040024
- **Lớp:** KHMT-1701
- **Môn học:** CSC4007 — Xử lý Ngôn ngữ Tự nhiên (NLP)
- **Nguồn dữ liệu thực nghiệm:** Dataset `vietnewssense.csv` thật do giảng viên cung cấp (tổng 3,268 dòng, chứa 13 chuyên mục tin tức báo chí tiếng Việt).
- **Tuân thủ quy chế dữ liệu:** Không commit văn bản thô, không công khai toàn văn bài báo để đảm bảo tính riêng tư và bản quyền dữ liệu. Toàn bộ mẫu khảo sát được tham chiếu qua số dòng gốc (`source_csv_row`) và mã băm mật mã SHA-256.
- **Fingerprint & Manifest truy vết:**
  - File cấu hình thực nghiệm: [`outputs/vietnews_lab4/manifest.json`](file:///c:/NLP/lab4/csc4007-lab4-1771040024/outputs/vietnews_lab4/manifest.json)
  - `dataset_sha256`: `ed2d3fdc37dbfb2e7be390cb70814c34d7acdab54b017b645e506bb9eb8eb06d`
  - `test_rows_sha256`: `c4721ec24b129063fa2401b0bce27a78599e55492d380a13cc3ae37dc3165258`
  - Cột văn bản (`text_col`): `content` | Cột nhãn (`label_col`): `category`
  - Siêu tham số kiểm toán: `seed = 42`, `max_rows = 1000`, `vocab_size = 20000`
  - Tách tập: Giữ lại 20% mẫu làm tập kiểm thử độc lập (`N_test = 200` mẫu held-out).

---

## 2. Bảng số liệu kiểm toán chuỗi (Sequence Audit: 128 vs 256)

Kiểm toán được thực hiện bằng cách chạy so sánh công bằng cùng dữ liệu trên hai cấu hình độ dài chuỗi cắt ngắn (`max_len = 128` và `max_len = 256`), trích xuất từ [`outputs/vietnews_lab4/vietnews_length_comparison.csv`](file:///c:/NLP/lab4/csc4007-lab4-1771040024/outputs/vietnews_lab4/vietnews_length_comparison.csv):

| Tokenizer | max_len | N test | P95 Len (token) | UNK Rate | Truncation Rate | Avg Padding Ratio | File Log Audit |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **underthesea** | **128** | 200 | 1,173.0 | 4.24% (0.0424) | **99.00%** (0.9900) | 0.45% (0.0045) | [len128/underthesea_sequence_audit.md](file:///c:/NLP/lab4/csc4007-lab4-1771040024/outputs/vietnews_lab4/len128/underthesea_sequence_audit.md) |
| **underthesea** | **256** | 200 | 1,173.0 | 4.24% (0.0424) | **91.50%** (0.9150) | 2.62% (0.0262) | [len256/underthesea_sequence_audit.md](file:///c:/NLP/lab4/csc4007-lab4-1771040024/outputs/vietnews_lab4/len256/underthesea_sequence_audit.md) |
| **whitespace** | **128** | 200 | 1,399.6 | 3.93% (0.0393) | **99.00%** (0.9900) | 0.42% (0.0042) | [len128/whitespace_sequence_audit.md](file:///c:/NLP/lab4/csc4007-lab4-1771040024/outputs/vietnews_lab4/len128/whitespace_sequence_audit.md) |
| **whitespace** | **256** | 200 | 1,399.6 | 3.93% (0.0393) | **93.50%** (0.9350) | 1.97% (0.0197) | [len256/whitespace_sequence_audit.md](file:///c:/NLP/lab4/csc4007-lab4-1771040024/outputs/vietnews_lab4/len256/whitespace_sequence_audit.md) |

### Nhận xét phân tích số liệu:

1. **Hiệu ứng giảm Truncation khi tăng `max_len` từ 128 lên 256:**
   - Với tokenizer chính `underthesea`, tỷ lệ bài viết bị cắt ngắn giảm từ **99.00% xuống 91.50%** (giảm 7.5% điểm tuyệt đối). 
   - Với `whitespace`, tỷ lệ cắt ngắn giảm từ **99.00% xuống 93.50%** (giảm 5.5% điểm tuyệt đối).
   - Ở ngưỡng 128, gần như 100% tài liệu báo chí tiếng Việt (99/100 bài) đều bị chặt cụt thân bài. Khi nới rộng lên 256, thêm ~8% số bài báo có độ dài ngắn và vừa được bảo toàn trọn vẹn.
2. **Vì sao P95 Length và UNK Rate không đổi khi thay đổi `max_len`?**
   - Phân vị 95% độ dài chuỗi (`P95`) và tỷ lệ từ ngoài từ điển (`UNK rate`) giữ nguyên tuyệt đối: `underthesea` giữ nguyên $1,173.0$ token và UNK $4.24\%$; `whitespace` giữ nguyên $1,399.6$ token và UNK $3.93\%$.
   - **Lý do kỹ thuật:** Script kiểm toán sequence audit của Lab 3/Lab 4 đo đạc P95 và UNK rate trên **toàn bộ văn bản gốc** trước khi bước cắt tỉa (truncation) diễn ra. Do đó, việc thay đổi tham số `max_len` của mô hình không làm biến đổi phân phối từ vựng tự nhiên của bài báo gốc.
3. **Ưu thế cấu trúc của `underthesea` so với `whitespace`:**
   - `underthesea` thực hiện tách từ ghép tiếng Việt (ghép các âm tiết có nghĩa thành một token, ví dụ `thị_trường_chứng_khoán`, `doanh_nghiệp`). Nhờ vậy, độ dài P95 của `underthesea` ngắn hơn đáng kể so với `whitespace` ($1,173.0$ so với $1,399.6$ token, giảm ~16.2%).
   - Tỷ lệ truncation ở `max_len = 256` của `underthesea` cũng thấp hơn (91.5% so với 93.5%), chứng tỏ việc phân đoạn từ chuẩn giúp "nén" thông tin ngữ nghĩa hiệu quả hơn trong cùng một cửa sổ độ dài.

---

## 3. Khảo sát 3 bài báo dài nhất (Held-out Test Set) và Phân vùng Token A / B / C

Để làm rõ tác động cơ học của việc cắt ngắn chuỗi, chúng tôi trích xuất 3 bài báo dài nhất trong tập kiểm thử độc lập từ file [`outputs/vietnews_lab4/article_candidates.csv`](file:///c:/NLP/lab4/csc4007-lab4-1771040024/outputs/vietnews_lab4/article_candidates.csv). Văn bản được chia thành 3 phân vùng token:
- **Vùng A (Token 1 → 128):** Tiêu đề, đoạn mở đầu (sapo) và luận điểm dẫn nhập. Cả hai mô hình 128 và 256 đều đọc được.
- **Vùng B (Token 129 → 256):** Phần phát triển nội dung, số liệu dẫn chứng và các luận cứ trung tâm. Chỉ mô hình `max_len = 256` đọc được, mô hình `max_len = 128` bị mất hoàn toàn.
- **Vùng C (Token 257 → hết bài):** Toàn bộ thân bài sau, diễn biến chuyên sâu, phân tích tác động và kết luận. CẢ HAI mô hình (128 và 256) ĐỀU BỊ MẤT.

### Bảng định lượng phân vùng token trên 3 bài báo dài nhất:

| Dòng CSV gốc (`source_csv_row`) | Mã băm SHA-256 nội dung | Chuyên mục | Tổng token (`underthesea`) | Vùng A (Token 1–128) | Vùng B (Token 129–256) | Vùng C (Token 257+ Bị mất ở cả 128 & 256) | Tỷ lệ nội dung bị mất ở `max_len = 256` |
|:---:|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Dòng 306** | `bf4d0840e0ef4a6b4b2a96e87a363973eb6722e77096233ce6781b8be3315efd` | Kinh doanh | **1,935 token** | 128 token (6.6%) | 128 token (6.6%) | **1,679 token** | **86.77%** |
| **Dòng 1381** | `e118738fa27b8635cada33c28902e0ee98382626bb34c14d5bfbeb31abcd2e7d` | Kinh doanh | **1,727 token** | 128 token (7.4%) | 128 token (7.4%) | **1,471 token** | **85.18%** |
| **Dòng 954** | `3f5aa84e20926b3d993b5255d4ebf7b743b53d89d4bf1ff14a4bf18c25be3d` | Kinh doanh | **1,619 token** | 128 token (7.9%) | 128 token (7.9%) | **1,363 token** | **84.19%** |

### Minh chứng định tính về thông tin bị mất:
1. **Bài báo Dòng 306 (1,935 tokens — Kinh doanh):**
   - Ở `max_len = 128`, mô hình chỉ kịp đọc phần giới thiệu ban đầu về biến động giá nguyên vật liệu và thị trường xuất nhập khẩu. 
   - Ở `max_len = 256`, mô hình nhận thêm Vùng B chứa các nhận định sơ bộ của chuyên gia.
   - Tuy nhiên, **1,679 token ở Vùng C (chiếm gần 87% bài báo)** bao gồm bảng phân tích báo cáo tài chính chi tiết của các doanh nghiệp, các giải pháp ứng phó của Bộ Công thương và kết luận khuyến nghị đầu tư đều bị cắt bỏ hoàn toàn.
2. **Bài báo Dòng 1381 (1,727 tokens — Kinh doanh):**
   - Vùng C chứa **1,471 token (mất 85.18% dung lượng)**. Các từ khóa phân loại cốt lõi và các chỉ số kinh tế then chốt xuất hiện ở nửa sau bài viết bị loại bỏ trước khi đưa vào mạng nơ-ron.
3. **Bài báo Dòng 954 (1,619 tokens — Kinh doanh):**
   - Vùng C chứa **1,363 token (mất 84.19% dung lượng)**. Mô hình chỉ "nhìn" thấy chưa đầy 16% phần đầu của bài báo.

---

## 4. Bản chất kỹ thuật: Sự khác biệt giữa Truncation và Long-range Dependency

Một hiểu lầm phổ biến khi triển khai mô hình chuỗi là cho rằng: *"Nếu nâng cấp từ RNN lên LSTM/GRU, mô hình sẽ giải quyết được vấn đề văn bản dài bị cắt ngắn"*. Đây là hai khái niệm hoàn toàn khác biệt:

```text
[Văn bản gốc dài 2,000 token]
                 │
                 ▼ (Tiền xử lý cắt chuỗi với max_len = 256)
[Token 1 ..................... Token 256]  |  [Token 257 ....... Token 2000]
                 │                                        │
                 ▼                                        ▼
    (Nạp vào mô hình LSTM/GRU)                  (BỊ VỨT BỎ HOÀN TOÀN)
                 │                                        │
  LSTM/GRU học liên kết giữa Token 1 và 256        Không có mô hình nào đọc được!
           (LONG-RANGE DEPENDENCY)                         (TRUNCATION)
```

1. **Truncation (Mất mát thông tin vật lý ngoài cửa sổ đầu vào):**
   - Truncation là quá trình cắt bỏ cơ học các token vượt quá ngưỡng `max_len` trong giai đoạn tiền xử lý (pre-processing/tokenization) trước khi tensor được nạp vào mô hình.
   - Các token ở Vùng C không bao giờ xuất hiện trong tensor `input_ids`. Dù kiến trúc mạng mạnh đến đâu (kể cả GPT-4 hay Deep BiLSTM), mô hình hoàn toàn không có dữ liệu đầu vào để tính toán. **Truncation là sự mất mát thông tin vĩnh viễn (information loss).**
2. **Long-range Dependency (Khả năng nắm bắt phụ thuộc ngữ cảnh xa):**
   - Phụ thuộc ngữ cảnh xa là khả năng của mạng nơ-ron trong việc học và liên kết mối quan hệ ngữ nghĩa giữa hai token **cùng tồn tại bên trong tensor đầu vào** nhưng nằm cách xa nhau về vị trí (ví dụ: từ chủ ngữ ở token vị trí 5 và động từ/vị ngữ bổ trợ ở token vị trí 250).
   - Ở bài toán này, LSTM và GRU phát huy tác dụng nhờ cơ chế cổng (Gates) và Cell State, giúp gradient truyền ngược qua 250 bước mà không bị tiêu biến về 0 như Vanilla RNN.
3. **Kết luận về giới hạn của LSTM/GRU:**
   - Việc chuyển từ RNN sang LSTM/GRU chỉ giúp cải thiện khả năng học quan hệ giữa các token còn sót lại trong phạm vi 256 bước. Nó **không thể khôi phục** bất kỳ thông tin nào đã bị cắt ở Vùng C.
   - Do đó, việc giảm truncation từ 99% xuống 91.5% khi tăng `max_len = 256` chỉ là mở rộng thêm 128 bước đọc, nhưng đối với các bài báo dài >1,500 token, mô hình vẫn bỏ sót hơn 84% nội dung.

---

## 5. Đánh giá tính khả thi và Đề xuất kiến trúc cho văn bản tiếng Việt dài trong thực tế

Từ các bằng chứng thực nghiệm trên, việc áp dụng mô hình RNN/LSTM/GRU đơn thuần với `max_len = 256` cho phân loại văn bản báo chí tiếng Việt (VietNewsSense) bộc lộ hạn chế cấu trúc rõ rệt:
- Hơn 91% bài báo bị cắt ngắn.
- Nhiều bài phân tích kinh doanh, chính luận có luận điểm quyết định nằm ở các đoạn tổng kết cuối bài.

### Không nên giải quyết bằng cách tăng `max_len` lên 2,000 trong RNN:
Nếu tăng `max_len = 2000` cho LSTM/GRU:
1. **Tốc độ tính toán suy giảm nghiêm trọng:** Do đặc tính xử lý tuần tự (sequential processing), thời gian huấn luyện và suy luận (inference latency) tăng tuyến tính theo chiều dài chuỗi ($O(T)$), không thể tận dụng tối đa năng lực xử lý song song của GPU.
2. **Suy thoái bộ nhớ (Memory Bottleneck):** Ngay cả với LSTM/GRU, việc truyền vector ẩn qua 2,000 bước liên tục vẫn dẫn đến hiện tượng trôi thông tin (information dilution), thông tin ở token 2,000 sẽ lấn át hoàn toàn ấn tượng ban đầu.

### Các giải pháp kiến trúc khuyến nghị cho hệ thống triển khai thực tế (Production):

1. **Hierarchical Attention Networks (HAN):**
   - Phân cấp bài báo thành 2 tầng: Tầng 1 mã hóa từng câu (Word-level BiGRU + Attention), Tầng 2 tổng hợp biểu diễn của các câu thành bài báo hoàn chỉnh (Sentence-level BiGRU + Attention).
   - Cho phép mô hình bao quát văn bản hàng ngàn token mà không bị bùng nổ chiều dài chuỗi tại một tầng.
2. **Cơ chế Chunking & Segment Pooling:**
   - Cắt bài báo thành các đoạn nhỏ (chunks) 256 token có độ gối đầu (overlapping window).
   - Dùng LSTM/GRU mã hóa từng chunk, sau đó áp dụng Max-pooling hoặc Self-Attention để tổng hợp vector đặc trưng cuối cùng trước khi phân loại.
3. **Mô hình Pre-trained Transformer chuyên cho ngữ cảnh dài (Long-Context Transformers):**
   - Thay thế LSTM from scratch bằng các mô hình tiền huấn luyện tiếng Việt hiện đại như **PhoBERT** kết hợp sliding window attention, hoặc các kiến trúc xử lý chuỗi dài như **Longformer**, **BigBird** (hỗ trợ context lên tới 4,096 token với độ phức tạp tuyến tính $O(N)$).

---

## 6. Checklist nghiệm thu bài lab Part B

- [x] **Dữ liệu thật:** Sử dụng 100% dữ liệu thực nghiệm từ file `vietnewssense.csv` của giảng viên (3,268 dòng, 13 chuyên mục).
- [x] **Truy vết & Mã băm:** Đầy đủ `dataset_sha256` và `test_rows_sha256` trong `manifest.json`.
- [x] **Bảo mật dữ liệu:** Tuyệt đối không commit văn bản thô hoặc toàn văn bài báo lên GitHub.
- [x] **Khảo sát 3 bài báo dài nhất:** Ghi nhận chính xác số dòng CSV gốc (`306`, `1381`, `954`), mã SHA-256 nội dung, định lượng cụ thể số token mất mát ở các phân vùng A, B, C.
- [x] **Phân biệt rành mạch:** Giải thích chính xác sự khác nhau giữa **Truncation** (mất mát dữ liệu vật lý) và **Long-range Dependency** (khả năng học liên kết ngữ cảnh xa).
- [x] **Không suy diễn:** Đánh giá đúng giới hạn của việc mở rộng `max_len`, không ngộ nhận LSTM/GRU có thể bù đắp phần nội dung bị cắt bỏ.
