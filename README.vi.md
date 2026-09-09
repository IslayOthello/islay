# islay

[English](README.md) · [Giao thức UCI](UCI.md)

`islay` là engine sinh nước đi và perft cho Othello/Reversi, viết bằng C++20
với giao diện văn bản kiểu UCI. Dự án hỗ trợ thiết lập thế cờ, hiển thị bàn cờ,
đếm chuỗi nước đi hợp lệ, benchmark perft và self-test.
Engine hiện không tìm nước đi tốt nhất hay tự chơi ván đấu.

## Chức năng

- Luật Othello: pass tiêu thụ một ply; luật Reversi: dừng khi bên đi hết nước hợp lệ.
- Sinh nước đi bằng bitboard với backend scalar, AVX2 hoặc ARM NEON chọn khi biên dịch.
- Perft có bulk counting và cache nhận biết đối xứng bàn cờ.
- Nạp thế cờ, kiểm tra tính hợp lệ của nước đi và pass.
- Self-test cho movegen, perft, cache, đối xứng và khác biệt giữa hai bộ luật.

## Build

Yêu cầu trình biên dịch C++20 và CMake 3.16 trở lên.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Build Release bật tối ưu CPU native và LTO khi được hỗ trợ.
Dùng `-DISLAY_NATIVE=OFF` để build portable, `-DISLAY_LTO=OFF` để tắt LTO.

## Sử dụng

```sh
./build/islay
printf 'uci\nisready\nposition startpos\ngo perft 8\nquit\n' | ./build/islay
```

Perft depth 8 từ thế cờ ban đầu trả về `390216` node.
Thêm `nocache` để đếm không dùng bảng cache:

```text
go perft 8 nocache
```

Chỉ còn hai option: `Rule` (`Othello` hoặc `Reversi`) và `PerftHash`
(kích thước cache theo MiB, mặc định 256). Perft chạy đồng bộ.
Các lệnh search như `go depth` và option evaluator/book đã được loại bỏ.

Xem [UCI.md](UCI.md) để biết cú pháp lệnh, định dạng thế cờ và hành vi perft.

## Kiểm thử

```sh
printf 'debug on\ntest\nquit\n' | ./build/islay
printf 'debug on\nbench 8\nquit\n' | ./build/islay
```

Self-test phải kết thúc bằng `ALL TESTS PASSED`.
Benchmark đếm perft không cache từ depth 1 đến depth được chỉ định.

Bộ test còn kiểm tra chín thế khai cuộc/trung cuộc/tàn cuộc/pass cố định với
cả hai luật, tám phép đối xứng và cache 1 MiB.
Xem [PERFT_BENCHMARK.md](PERFT_BENCHMARK.md) để biết median NPS trên nhiều vị trí
và cách chạy phép đo A/B đơn luồng.

## Cấu trúc

```text
main.cpp       Điểm khởi động
src/           Bàn cờ, bitboard, sinh nước đi, perft, options và UCI
UCI.md         Tài liệu giao thức
CMakeLists.txt Cấu hình build và lựa chọn kiến trúc
```
