# Báo cáo kết quả — Day 17: Memory Systems for AI Agent

## 1. Cách chạy

```bash
python src/benchmark.py          # offline, kết quả lặp lại được, không cần API key
python src/benchmark.py --live   # gọi LLM thật theo .env (xem .env.example)
pytest src/test_agents.py -v     # 16 test
```

## 2. Kiến trúc đã triển khai

| Lớp memory | Nằm ở đâu | Sống bao lâu | Chứa gì |
|---|---|---|---|
| Short-term | `CompactMemoryManager.state[thread]["messages"]` | trong một thread | vài message gần nhất, nguyên văn |
| Persistent | `state/profiles/<user>/User.md` (`UserProfileStore`) | qua mọi thread, qua cả lần restart | fact ổn định: tên, nơi ở, nghề, style, đồ uống, món ăn, thú cưng, mối quan tâm |
| Compact | `CompactMemoryManager.state[thread]["summary"]` | trong một thread | bản tóm tắt có giới hạn của các message cũ |

- `BaselineAgent`: chỉ có danh sách message theo `thread_id`, mỗi lượt gửi lại toàn bộ thread. Đổi thread là quên.
- `AdvancedAgent`: mỗi lượt `extract → ghi User.md → append + compact → prompt = User.md + summary + recent`.
- Hai agent dùng chung bộ trích fact và chung hàm trả lời (`answer_from_facts`). Khác biệt duy nhất là **fact nào được nhìn thấy**: baseline chỉ thấy fact trong thread hiện tại, advanced thấy fact trong `User.md`. Nhờ vậy phép so sánh công bằng: chênh lệch đến từ memory, không đến từ việc một agent "thông minh" hơn.
- Chế độ live nằm trong `src/live_agents.py`: `create_agent` + `InMemorySaver`, 3 tool đọc/ghi/sửa `User.md`, `dynamic_prompt` chèn profile, `SummarizationMiddleware` để compact.

## 3. Kết quả benchmark (offline, threshold 800 token, giữ 4 message)

### Standard Benchmark — 10 hội thoại, 101 lượt, 14 câu recall

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 2666 | 15220 | 0.00 | 0.15 | 0 | 0 |
| Advanced | 2574 | 22484 | 1.00 | 1.00 | 1666 | 0 |

Prompt tokens của Advanced so với Baseline: **+47.7%**.

### Long-Context Stress Benchmark — 1 hội thoại, 16 lượt rất dài, 3 câu recall

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
|---|---|---|---|---|---|---|
| Baseline | 2593 | 21990 | 0.00 | 0.15 | 0 | 0 |
| Advanced | 2672 | 10178 | 1.00 | 1.00 | 825 | 4 |

Prompt tokens của Advanced so với Baseline: **−53.7%**.

Độ nhạy theo ngưỡng compact trên stress test:

| Threshold | Compactions | Prompt tokens (Advanced) | So với Baseline |
|---|---|---|---|
| 600 | 8 | 9105 | −58.6% |
| 800 (mặc định) | 4 | 10178 | −53.7% |
| 1000 | 3 | 11309 | −48.6% |
| 1200 | 2 | 12821 | −41.7% |

## 4. Phân tích

**Vì sao Advanced recall tốt hơn Baseline.** Câu recall được hỏi ở thread mới. Baseline chỉ có message của thread đó nên không có gì để trả lời (0.00). Advanced đọc `User.md`, thứ không phụ thuộc thread. Baseline không hỏng: trong cùng thread nó vẫn trả lời đúng (test `test_cross_session_recall` kiểm cả hai chiều).

**Vì sao Advanced tốn hơn ở hội thoại ngắn.** Mỗi thread chuẩn chỉ khoảng 250 token, không bao giờ chạm ngưỡng 800 nên compact không chạy (0 lần). Trong khi đó Advanced vẫn phải mang profile (~75 token) và system prompt dài hơn vào **mọi lượt**, kể cả lượt không cần đến. 115 lượt × phần cộng thêm đó ra +47.7%. Compact không tạo ra lợi ích nào khi chưa có gì để nén; nó chỉ là chi phí cố định.

**Vì sao compact thắng ở hội thoại dài.** Baseline gửi lại toàn bộ lịch sử mỗi lượt, nên prompt của lượt thứ *n* tỉ lệ với *n* và tổng chi phí tăng theo *n²*: lượt cuối của stress test Baseline phải xử lý khoảng 2500 token, còn Advanced bị chặn quanh ngưỡng 800. Tổng cộng 21990 so với 10178.

**Vì sao compact chủ yếu tối ưu `Prompt tokens processed`.** Cột `Agent tokens only` gần như bằng nhau ở cả hai agent (2593 vs 2672): người dùng vẫn nói từng ấy chữ và agent vẫn trả lời từng ấy chữ. Compact không thay đổi nội dung hội thoại, nó chỉ thay đổi **lượng ngữ cảnh cũ bị gửi lại**. Chi phí thật của agent nhiều lượt nằm ở phần gửi lại này, không nằm ở phần sinh ra.

**Memory file tăng trưởng thế nào và rủi ro.** `User.md` tăng 1666 byte sau 10 phiên và 825 byte sau stress test. Vì profile được chèn vào mọi lượt, mỗi byte thêm vào file là chi phí lặp lại mãi mãi. Các biện pháp đã dùng để giữ file có giới hạn:
- fact là các field cố định, correction **ghi đè** chứ không nối thêm;
- `interests` giới hạn 5 mục, `History` giới hạn 8 dòng;
- prompt chỉ nhận `prompt_view()` (bỏ metadata và history), không nhận cả file.

Rủi ro còn lại: fact lưu sai sẽ bị nhắc lại ở mọi phiên sau (sai một lần, sai mãi); file chứa thông tin cá nhân dạng plain text; bản tóm tắt compact là mất mát có chủ đích — chỉ giữ câu đầu của mỗi message và 8 dòng gần nhất, nên chi tiết ở giữa hội thoại dài (ví dụ con số trong các mẩu tin) sẽ mất nếu không được đưa vào `User.md`.

## 5. Bonus

| Bonus | Giải quyết vấn đề gì | Cải thiện | Rủi ro thêm |
|---|---|---|---|
| **Confidence threshold** (`profile_confidence_threshold = 0.6`) | Câu đùa "hay là chuyển sang product manager" hoặc "lúc đầu mình nói ở Huế" khớp mẫu nhưng không phải fact hiện tại | Giữ recall đúng ở stress test: nghề vẫn là MLOps engineer; bỏ gate thì câu đùa bị ghi thành nghề (có test chứng minh) | Ngưỡng cao quá sẽ bỏ sót fact thật; điểm confidence là heuristic do người viết đặt, không được hiệu chỉnh từ dữ liệu |
| **Conflict handling** | Đà Nẵng → Huế, backend → MLOps | Câu phủ định ("không còn ở X") thành *retraction*, giá trị mới ghi đè, thay đổi được ghi vào `History` | Latest-wins tin tuyệt đối vào phát biểu mới nhất; một câu bị trích sai sẽ xoá mất fact đúng |
| **Entity extraction có cấu trúc** | `User.md` dài và lộn xộn nếu lưu câu thô | 8 field cố định; `response_style` gộp theo slot (độ dài / format / ví dụ / nhấn mạnh) nên "3 bullet" không bị "bullet" ghi đè | Fact nằm ngoài schema không được lưu; regex gắn chặt với cách diễn đạt tiếng Việt trong dataset |
| **Memory decay** (`fact_score`) | File và prompt phình theo thời gian | Điểm = confidence × 0.98^tuổi × hệ số lặp lại; dùng để chọn fact khi `prompt_view(max_facts=…)` có ngân sách, và loại mối quan tâm cũ nhất | Fact đúng nhưng ít được nhắc (món ăn, thú cưng) bị xếp thấp; vì vậy decay chỉ **giảm ưu tiên**, không tự xoá fact |
| **Không lưu câu hỏi** | "Mình còn ở Huế không?" không phải lời khẳng định | Câu kết thúc bằng `?` bị bỏ qua hoàn toàn khi trích fact | Câu hỏi tu từ chứa fact thật sẽ bị bỏ sót |

## 6. Giới hạn cần biết khi đọc kết quả

- Recall 1.00 của Advanced đến từ bộ trích fact **dựa trên luật**, được viết để xử lý các cách diễn đạt có trong hai dataset này. Với câu chữ khác, recall sẽ thấp hơn; đó là lý do chế độ live giao thêm việc ghi nhớ cho LLM qua tool.
- Token là ước lượng `len/4`, không phải tokenizer thật. Số tuyệt đối không chính xác, nhưng tỉ lệ giữa hai agent thì có ý nghĩa vì cùng một thước đo.
- `Response quality` offline là heuristic (70% độ phủ fact, 15% ngắn gọn, 15% có trả lời), không đo độ tự nhiên của câu văn.
- Chế độ live mới được kiểm tra bằng model giả (xác nhận profile được chèn vào prompt và summarization có chạy), **chưa chạy với API thật** vì repo không có key.
