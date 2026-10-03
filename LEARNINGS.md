# Kiến thức rút ra từ lab Day 17 — Memory Systems for AI Agent

## 1. Ba loại memory trả lời ba câu hỏi khác nhau

| Loại | Câu hỏi nó trả lời | Phạm vi | Cái giá |
|---|---|---|---|
| Short-term | "Vừa nãy mình nói gì?" | một thread | prompt tăng theo độ dài thread |
| Persistent (`User.md`) | "Người này là ai?" | mọi thread, mọi phiên | cộng thêm token vào mọi lượt; lưu sai thì sai mãi |
| Compact | "Trước đó đã bàn những gì?" | một thread dài | mất chi tiết, không lấy lại được |

LLM không có trí nhớ. "Nhớ" nghĩa là hệ thống chọn đưa lại thông tin nào vào prompt. Thiết kế memory là thiết kế việc chọn đó.

## 2. Chi phí thật nằm ở ngữ cảnh gửi lại, không nằm ở câu trả lời

- Agent gửi lại cả lịch sử mỗi lượt thì chi phí tổng tăng theo bình phương số lượt.
- Phải đo hai con số riêng: token của chính hội thoại (`Agent tokens only`) và token ngữ cảnh bị xử lý lại (`Prompt tokens processed`). Chỉ nhìn con số đầu sẽ kết luận sai rằng hai agent tốn như nhau.
- Trong lab: hai agent chênh nhau ~3% ở cột đầu, nhưng chênh −54% ở cột sau trên hội thoại dài.

## 3. Memory không miễn phí và không phải lúc nào cũng thắng

- Hội thoại ngắn: Advanced tốn hơn Baseline 48% prompt token vì mang profile vào mọi lượt mà compact chưa có gì để nén.
- Hội thoại dài: Advanced rẻ hơn 54%.
- Có một điểm hoà vốn. Ngưỡng compact là một nút vặn đánh đổi: ngưỡng thấp tiết kiệm hơn nhưng nén sớm và mất nhiều chi tiết hơn.

## 4. Ghi vào memory khó hơn đọc từ memory

Phần khó nhất của lab không phải lưu file mà là quyết định **cái gì đáng lưu**:

- Câu hỏi không phải fact ("mình còn ở Huế không?").
- Nhắc đến không phải khẳng định ("Hà Nội chỉ là nơi đi họp").
- Câu đùa và giả định không phải fact ("hay là chuyển sang product manager").
- Quá khứ không phải hiện tại ("lúc đầu mình nói ở Huế").
- Thông tin mới nhất không phải lúc nào cũng đúng nhất: bám máy móc vào đoạn text mới nhất sẽ lưu nhầm nhiễu.

Từ đó sinh ra các guardrail: confidence threshold, xử lý phủ định theo từng mệnh đề, và correction phải **ghi đè** fact cũ chứ không để hai giá trị cùng tồn tại.

## 5. Cấu trúc hoá fact giúp kiểm soát cả độ đúng lẫn kích thước

- Field cố định (`name`, `location`, …) làm cho correction trở thành thao tác ghi đè đơn giản và file không phình.
- Preference nên gộp theo slot thay vì ghi đè nguyên câu: người dùng nói "trả lời thành bullet" không có nghĩa là họ rút lại "ngắn gọn".
- Mọi danh sách trong memory đều cần giới hạn (số mối quan tâm, số dòng history, số dòng summary). Thứ gì không có giới hạn sẽ thành chi phí tăng mãi.
- Markdown dễ đọc, dễ sửa tay, dễ giải thích; đổi lại phải tự viết parser và không có truy vấn.

## 6. Tóm tắt là nén có mất mát

- Fact ổn định phải được rút ra `User.md` **trước khi** message bị compact, nếu không sẽ mất cùng với message.
- Một bản tóm tắt tệ nguy hiểm hơn không có tóm tắt: agent trông như còn nhớ nhưng đã mất phần quan trọng.
- Memory decay nên hạ ưu tiên chứ không tự xoá: fact đúng nhưng hiếm khi được nhắc (món ăn yêu thích) vẫn là fact đúng.

## 7. Cách benchmark một hệ thống memory

- So sánh công bằng: cùng input, cùng bộ đo, cùng năng lực "đọc hiểu"; chỉ khác phần memory.
- Baseline phải ngây thơ nhưng không hỏng: nó vẫn phải nhớ trong cùng thread.
- Hỏi recall ở **thread mới**, nếu không thì chỉ đang đo short-term memory.
- Cần hai bộ dữ liệu: một bộ bình thường để lộ chi phí của memory, một bộ stress để lộ lợi ích của compact. Một bộ duy nhất sẽ kể sai một nửa câu chuyện.
- Dữ liệu phải có correction và nhiễu, nếu không mọi cách trích fact đều đạt điểm tối đa.
- Chế độ offline xác định (deterministic) cho phép test và benchmark chạy lặp lại được, không tốn API.
- Test phải kiểm cả hành vi **không được xảy ra**: baseline không được nhớ qua thread, câu đùa không được ghi vào profile, fact cũ không được xuất hiện sau khi đính chính.

## 8. Kỹ thuật và công cụ

- Tách lớp: provider (`model_provider`) / cấu hình (`config`) / memory (`memory_store`) / agent / benchmark. Memory layer không phụ thuộc provider nào.
- Một interface cho 6 provider (OpenAI, OpenAI-compatible, Gemini, Anthropic, Ollama, OpenRouter), import SDK kiểu lazy để chế độ offline không cần cài gì thêm.
- LangChain 1.x: `create_agent`, `InMemorySaver` (short-term theo `thread_id`), tool nhận `ToolRuntime` để biết user hiện tại, `dynamic_prompt` để chèn profile, `SummarizationMiddleware` để compact.
- Xử lý tiếng Việt: chuẩn hoá Unicode NFC trước khi so khớp, luôn đọc/ghi file bằng UTF-8, so khớp có phân biệt hoa thường để "AI" không khớp nhầm vào "hai".
- Làm sạch `user_id` trước khi ghép thành đường dẫn file.

## 9. Câu chuyện tổng thể

1. Baseline không nhớ dài hạn.
2. Thêm `User.md` thì recall tăng từ 0 lên 1.
3. Hội thoại dài làm chi phí prompt tăng rất nhanh.
4. Compact kéo chi phí đó xuống hơn một nửa.
5. Hệ thống mạnh hơn nhưng phức tạp hơn, và mỗi lớp memory thêm vào đều cần guardrail riêng.
