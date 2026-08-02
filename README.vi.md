# islay

[English](README.md) · [Giao thức UCI](UCI.md)

`islay` là một engine Othello/Reversi hướng tới thi đấu, được viết bằng C++20.
Dự án kết hợp sinh nước đi bằng bitboard có tối ưu SIMD, tìm kiếm negamax/PVS
có chọn lọc, giải chính xác tàn cuộc và bộ đánh giá kiểu NNUE gọn nhẹ được huấn
luyện bằng self-play. Engine cung cấp giao thức văn bản kiểu UCI đã được tài
liệu hóa để tích hợp với GUI, trình chạy trận đấu và công cụ dòng lệnh.

## Điểm nổi bật

- Hỗ trợ cả luật Othello và Reversi.
- Dùng backend sinh nước đi scalar portable, AVX2 hoặc ARM NEON được chọn khi
  biên dịch.
- Tìm kiếm bằng iterative deepening, PVS, transposition table, ProbCut, late
  move reduction, futility pruning, killer move và continuation history.
- Sắp xếp tàn cuộc chặt bằng parity của vùng liên thông và trả về điểm lý
  thuyết trò chơi chính xác khi độ sâu đạt số ô trống còn lại.
- Nạp opening book nhận biết đối xứng D4 và ánh xạ nước đi canonical về đúng
  hướng bàn cờ ban đầu.
- Phát hành `weights/v20.nnue`, mạng NNUE-lite theo từng giai đoạn với embedding
  int16 và accumulator ẩn gồm tám giá trị.
- Có pipeline self-play xác định, bootstrap NNUE sang NNUE, đo trận đấu theo cặp,
  perft và bộ kiểm tra tính đúng tích hợp.

## Công nghệ và Kiến trúc

| Thành phần | Cách triển khai |
|---|---|
| Ngôn ngữ và build | C++20, CMake 3.16+, tùy chọn tối ưu CPU native và LTO |
| Biểu diễn bàn cờ | Bitboard với trạng thái tương đối player/opponent |
| Sinh nước đi | Scalar fallback, AVX2 trên x86-64, integer/NEON hybrid trên ARM64 |
| Tìm kiếm | Negamax/PVS iterative-deepening, TT, cắt tỉa chọn lọc, sắp xếp theo history |
| Tàn cuộc | Tìm kiếm chính xác có xử lý pass, stability bound, parity vùng liên thông |
| Đánh giá | Heuristic tích hợp, trọng số pattern đã huấn luyện hoặc NNUE-lite lượng tử hóa |
| Opening book | Khóa D4 canonical với ánh xạ nước đi an toàn qua phép xoay/đối xứng |
| Giao diện | Giao thức dòng kiểu UCI bất đồng bộ qua standard input/output |
| Kiểm chứng | Self-test theo module, perft chuẩn, search oracle, trận Elo theo cặp |

Bộ đánh giá NNUE-lite tái sử dụng các pattern feature incremental của engine.
Các hàng feature đang hoạt động được cộng vào tám giá trị ẩn, sau đó được đánh
giá bởi các head tuyến tính và ReLU theo từng giai đoạn. Phiên bản 20 được
warm-start từ mạng v19 đã lượng tử hóa và huấn luyện bằng khoảng năm triệu ván
self-play mới.

## Elo ước tính

`islay` chưa có **chỉ số Elo tuyệt đối** đáng tin cậy vì chưa được hiệu chuẩn
với một tập engine bên ngoài ổn định. Các số dưới đây là ước tính theo cặp, đảo
màu, và chỉ mang ý nghĩa tương đối so với các bộ đánh giá `islay` cũ hơn.

| So sánh | Phép thử | Mẫu | Elo ước tính | CI 95% | LOS |
|---|---:|---:|---:|---:|---:|
| NNUE v19 với pattern v18 | 200k nodes | ledger của dự án | +45,9 | [33,7; 58,2] | 100% |
| NNUE v19 với pattern v18 | equal time 100 ms | ledger của dự án | +31,4 | [19,8; 42,9] | 100% |
| NNUE v20 với NNUE v19 | 200k nodes | 4.000 | +8,08 | [2,24; 13,92] | 99,67% |
| NNUE v20 với NNUE v19 | equal time 100 ms | 2.280 | +3,51 | [-4,25; 11,27] | 81,23% |

Phép đo equal-time của v20 đã dừng trước mức dự kiến 4.000 ván, vì vậy
confidence interval vẫn bao gồm 0.

Vì vậy, ước tính thực tế hiện tại là **v20 ≈ +3,5 Elo so với v19 ở equal
time**. Cộng hai ước tính equal-time nối tiếp cho kết quả gần đúng
**v20 ≈ +35 Elo so với v18**. Đây là ước tính định hướng của dự án, không phải
confidence interval tổng hợp chính thức hay rating thi đấu tuyệt đối.

## Build

Yêu cầu:

- Trình biên dịch C++20 như Clang, GCC hoặc MSVC phiên bản gần đây
- CMake 3.16 trở lên

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Build Release bật tối ưu CPU native và link-time optimization khi toolchain hỗ
trợ. Để tạo binary portable, dùng:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DISLAY_NATIVE=OFF -DISLAY_LTO=OFF
cmake --build build -j
```

## Chạy engine

Khởi động một phiên engine tương tác:

```sh
./build/islay
```

Nạp bộ đánh giá được khuyến nghị và tìm kiếm từ thế cờ ban đầu:

```sh
printf 'uci\nsetoption name EvalFile value weights/v20.nnue\nisready\nposition startpos\ngo depth 10\nquit\n' \
  | ./build/islay
```

Các option thường dùng:

| Option | Mục đích |
|---|---|
| `Rule` | Chọn `Othello` hoặc `Reversi` |
| `EvalFile` | Nạp mạng `.nnue` hoặc trọng số pattern `.pat` |
| `Hash` | Đặt kích thước transposition table tìm kiếm theo MiB |
| `Threads` | Đặt số thread tìm kiếm |
| `OwnBook` / `BookFile` | Bật và nạp opening book |

Xem [UCI.md](UCI.md) để biết đầy đủ giao thức, định dạng thế cờ, giới hạn tìm
kiếm, phản hồi và cách xử lý lỗi.

## Kiểm thử và Phát triển

Chạy toàn bộ bộ kiểm tra tính đúng:

```sh
printf 'debug on\ntest\nquit\n' | ./build/islay
```

Dòng cuối phải là `ALL TESTS PASSED`. Bộ test bao phủ sinh nước đi, đánh giá,
search oracle, trạng thái pattern incremental, đối xứng opening book, các giá
trị perft chuẩn, tính nhất quán của cache, đối xứng và hành vi pass theo từng
bộ luật.

Chạy riêng một phép kiểm tra perft có thể tái lập:

```sh
printf 'position startpos\ngo perft 8\nquit\n' | ./build/islay
```

Với thay đổi về sức mạnh, hãy dùng opening theo cặp có đảo màu và báo cáo Elo,
confidence interval 95% cùng likelihood of superiority. Giảm node không tự nó
chứng minh engine chơi mạnh hơn.

## Cấu trúc Repository

```text
main.cpp       Điểm khởi động engine
src/           Bàn cờ, sinh nước đi, tìm kiếm, đánh giá, huấn luyện và UCI
weights/       Các file đánh giá pattern và NNUE theo phiên bản
UCI.md         Tài liệu tham chiếu giao thức phát hành
CMakeLists.txt Cấu hình build và lựa chọn kiến trúc
```
