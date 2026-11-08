# Islay Othello / Reversi (WZebra Professional Desktop GUI)

Phần mềm giao diện desktop cờ Othello/Reversi chuyên nghiệp theo phong cách **WZebra** (Gunnar Andersson) với **Theme Trắng (Light Theme)** thanh lịch, các điều khiển desktop chuẩn (Menu bar, Toolbar, Move Table, Advantage Curve, Position Setup), phần bàn cờ được vẽ bằng đồ họa 2D Canvas mượt mà, và toàn bộ logic cùng engine kết nối bằng **Lua**.

---

## Điểm nổi bật & Triết lý thiết kế (Design Architecture)

1. **Giao diện Desktop chuẩn (Native UI Controls) & Theme Trắng (White Theme)**:
   - Toàn bộ giao diện sử dụng **Theme Trắng sáng** chuẩn macOS/Windows hiện đại (`#ffffff`, `#f1f5f9`, chữ than đậm `#0f172a`), không tối tăm như game arcade.
   - Các điều khiển là **UI chuẩn**: Menu bar, Toolbar, bảng biên bản nước đi có thanh cuộn, thẻ điểm, bảng phân tích engine, bảng công cụ xếp cờ.

2. **Bàn cờ Canvas đồ họa 2D (Graphics Canvas)**:
   - Chỉ có phần bàn cờ là khu vực đồ họa Canvas chuyên biệt: nỉ xanh thi đấu emerald, viền gỗ walnut, chữ số tọa độ A–H, 1–8 và 4 điểm sao.
   - Quân cờ hiệu ứng 3D đổ bóng chân thực.
   - **Đánh số thứ tự nước đi trên quân cờ (Move Numbers)**: 1, 2, 3... chuẩn phong cách WZebra.
   - **Điểm đánh giá trực tiếp trên ô đi (Evaluation Overlay)**: Hiển thị badge điểm số (+1.8, -0.4, =0.0) trên từng ô hợp lệ.
   - Xem trước số quân bị lật khi rê chuột.

3. **Biểu đồ đường cong lợi thế (Advantage / Evaluation Curve Graph)**:
   - Đồ thị đường cong thể hiện sự biến thiên lợi thế từ nước 1 đến hiện tại.
   - Click vào bất kỳ điểm nào trên biểu đồ để chuyển ngay đến thế cờ ở nước đi đó.

4. **Bảng biên bản nước đi & Điều khiển VCR Navigation**:
   - Bảng 4 cột: `#`, `Black`, `White`, `Advantage`. Click vào dòng bất kỳ để xem lại thế cờ.
   - Các nút VCR: `|<<` (Về đầu), `<` (Nước trước), `>` (Nước sau), `>>|` (Về cuối).

5. **Chế độ xếp cờ tự do (Position Setup Palette)**:
   - Dễ dàng dựng các thế cờ tàn cuộc, giải đố, nhập/xuất FEN để phân tích cùng AI.

6. **Backend viết bằng Lua & Bitboard Validation Engine**:
   - `gui/bitboard.lua`: Port trực tiếp cấu trúc dữ liệu `islay::Board`, `Bitboard` 64-bit và `GameStatus` từ core C++ của engine sang Lua. Đảm bảo tính toán lật quân, toạ độ, luật cờ (Othello/Reversi) và Perft(1..5) khớp $100\%$ với engine gốc.
   - `gui/board.lua`: Quản lý thế cờ, kiểm tra luật, FEN, lật quân và đồng bộ với Bitboard model.
   - `gui/openings.lua`: Nhận diện khai cuộc kinh điển Othello (Tiger, Rose, Cow, Buffalo...).
   - `gui/engine.lua`: Giao tiếp bất đồng bộ non-blocking với engine `./build/islay` qua giao thức UCI.
   - `gui/bridge.lua`: Điều phối trạng thái ván cờ, kiểm định tính đúng đắn của nước đi người dùng và nước đi từ AI engine trước khi áp dụng.

---

## Cấu trúc thư mục (Folder Layout)

```text
gui/
├── bin/
│   └── islay_gui     # Ứng dụng Desktop Native macOS (Light Theme, < 60KB)
├── src/
│   └── app.m         # Mã nguồn wrapper Native Cocoa + WebKit (C/Obj-C)
├── web/
│   ├── index.html    # Giao diện Desktop WZebra (Toolbar, Scoreboard, Canvas, Inspector)
│   ├── style.css     # Theme Trắng chuyên nghiệp (Light Theme)
│   └── app.js        # Logic vẽ đồ họa bàn cờ 2D Canvas & biểu đồ lợi thế
├── bitboard.lua      # Port bitboard 64-bit, movegen và luật cờ chuẩn từ C++ islay
├── board.lua         # Logic bàn cờ Othello/Reversi, cây nước đi, move numbers, FEN
├── openings.lua      # Thư viện nhận diện các thế khai cuộc Othello kinh điển
├── engine.lua        # Giao tiếp UCI non-blocking với engine islay
├── bridge.lua        # Cầu nối điều phối logic & kiểm tra tính hợp lệ của nước đi
├── json.lua          # Bộ giải mã/mã hóa JSON thuần Lua
├── server.py         # Server web cục bộ dự phòng
├── run.sh            # Script khởi chạy thông minh
└── README.md         # Hướng dẫn chi tiết
```

---

## Hướng dẫn khởi chạy (How to Run)

### Cách 1: Khởi chạy ứng dụng Desktop (Khuyên dùng)

Từ thư mục gốc dự án:

```sh
./gui/run.sh
```

Lệnh này sẽ tự động khởi chạy ứng dụng desktop native macOS với Theme Trắng chuyên nghiệp.

### Cách 2: Khởi chạy qua trình duyệt mặc định

```sh
./gui/run.sh --web
# hoặc:
python3 gui/server.py
```

### Cách 3: Khởi chạy bằng LÖVE 2D (cũng hỗ trợ Theme Trắng)

```sh
./gui/run.sh --love
# hoặc:
love gui
```

---

## Danh sách phím tắt (Keyboard Shortcuts)

| Phím tắt | Tác vụ |
|---|---|
| `Cmd+N` | Bắt đầu ván mới (New Game) |
| `Cmd+Z` | Đi lại nước trước (Undo) |
| `Cmd+Y` | Tiến tới nước sau (Redo) |
| `H` | Gợi ý nước đi tối ưu (Hint Best Move) |
| `P` | Chạy Perft Benchmark |
| `Space` | Pass lượt khi bị ép pass trong luật Othello |
| `Cmd+C` | Sao chép thế cờ FEN |
| `Cmd+V` | Dán thế cờ FEN |
