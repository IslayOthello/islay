#include <algorithm>
#include <atomic>
#include <bit>
#include <cerrno>
#include <charconv>
#include <cmath>
#include <csignal>
#include <cstring>
#include <iostream>
#include <map>
#include <poll.h>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include "selfplay.hpp"

namespace {
  // Private local protocol v1; all descriptors are inherited, no public listener or shm name.
  constexpr std::size_t kInputBytes = 128 * sizeof(float), kOutputBytes = 66 * sizeof(float);
  constexpr std::size_t kFrameBytes = 32 + islay::kMctsMaxPly * islay::kReplayRecordBytes;
  constexpr std::size_t kSlotBytes  = kInputBytes + kOutputBytes + kFrameBytes + 16;
  static_assert(std::endian::native == std::endian::little && sizeof(float) == 4);

  std::uint64_t integer(const std::string &s, std::uint64_t low, std::uint64_t high) {
    std::uint64_t value;
    const auto [end, error] = std::from_chars(s.data(), s.data() + s.size(), value);
    if (error != std::errc{} || end != s.data() + s.size() || value < low || value > high)
      throw std::invalid_argument("invalid worker integer");
    return value;
  }
  double real(const std::string &s) {
    std::size_t  end;
    const double value = std::stod(s, &end);
    if (end != s.size() || !std::isfinite(value))
      throw std::invalid_argument("invalid worker real");
    return value;
  }
  template<class T>
  void put(std::ostream &out, T value) {
    for (std::size_t i = 0; i < sizeof(T); ++i)
      out.put(static_cast<char>((value >> (8 * i)) & 255));
  }

  class SharedEvaluator final : public islay::Evaluator {
  public:
    SharedEvaluator(int memory, int control, std::size_t workers, std::size_t slot, int timeout) :
        control_(control), size_(workers * kSlotBytes), timeout_(timeout) {
      struct stat status{};
      // Darwin reports shm lengths rounded to a VM page; Linux may retain the exact length.
      const auto page = sysconf(_SC_PAGESIZE);
      if (page <= 0 || fstat(memory, &status) || status.st_size < static_cast<off_t>(size_) ||
          status.st_size >= static_cast<off_t>(size_) + page)
        throw std::invalid_argument("worker shm size mismatch");
      memory_ = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, memory, 0);
      if (memory_ == MAP_FAILED)
        throw std::runtime_error("worker mmap failed");
      auto *base = static_cast<char *>(memory_);
      input_     = reinterpret_cast<float *>(base + slot * kInputBytes);
      output_    = reinterpret_cast<float *>(base + workers * kInputBytes + slot * kOutputBytes);
      frame_     = base + workers * (kInputBytes + kOutputBytes) + slot * kFrameBytes;
      metrics_   = base + workers * (kInputBytes + kOutputBytes + kFrameBytes) + slot * 16;
    }
    ~SharedEvaluator() override { munmap(memory_, size_); }
    SharedEvaluator(const SharedEvaluator &)            = delete;
    SharedEvaluator &operator=(const SharedEvaluator &) = delete;

    void send(char value) {
      ssize_t size;
      do {
        size = ::send(control_, &value, 1, 0);
      } while (size < 0 && errno == EINTR);
      if (size != 1)
        throw std::runtime_error("worker control send failed");
    }
    bool receive(void *destination, std::size_t size, bool idle = false) {
      auto       *out      = static_cast<char *>(destination);
      std::size_t received = 0;
      while (received < size) {
        pollfd descriptor{control_, POLLIN, 0};
        int    ready;
        do {
          ready = poll(&descriptor, 1, idle ? -1 : timeout_);
        } while (ready < 0 && errno == EINTR);
        if (ready <= 0)
          throw std::runtime_error("worker inference timeout/poll failure");
        const auto count = recv(control_, out + received, size - received, 0);
        if (count < 0 && errno == EINTR)
          continue;
        if (count == 0 && idle && received == 0)
          return false;
        if (count <= 0)
          throw std::runtime_error("worker server exited mid-request");
        received += count;
      }
      return true;
    }
    void evaluate(std::span<const islay::Board> boards, std::span<islay::Evaluation> outputs,
                  std::stop_token stop) override {
      if (boards.size() != 1 || outputs.size() != 1)
        throw std::invalid_argument("worker evaluator expects one outstanding leaf");
      if (stop.stop_requested())
        return;
      const auto board = boards[0];
      if (board.player & board.opponent)
        throw std::invalid_argument("worker overlapping input board");
      for (int square = 0; square < 64; ++square) {
        input_[square]      = static_cast<float>((board.player >> square) & 1);
        input_[64 + square] = static_cast<float>((board.opponent >> square) & 1);
      }
      std::atomic_thread_fence(std::memory_order_release);
      send('R');
      char answer;
      receive(&answer, 1);
      if (answer != 'A')
        throw std::runtime_error("worker invalid inference acknowledgement");
      std::atomic_thread_fence(std::memory_order_acquire);
      if (!std::all_of(output_, output_ + 66, [](float v) { return std::isfinite(v); }) || std::abs(output_[65]) > 1)
        throw std::runtime_error("worker invalid inference output");
      std::copy_n(output_, 65, outputs[0].logits.begin());
      outputs[0].value = output_[65];
    }
    void publish(std::uint64_t id, const islay::SelfplayConfig &config, std::span<const islay::ReplayRecord> records,
                 const islay::CachedEvaluator &cache) {
      std::ostringstream frame(std::ios::out | std::ios::binary);
      frame.write("ISLAGM01", 8);
      put(frame, id);
      put(frame, islay::selfplay_seed(config.seed, id));
      put(frame, static_cast<std::uint32_t>(records.size()));
      put(frame, std::uint32_t{0});
      for (const auto &record: records)
        islay::write_replay_record(frame, record);
      const auto bytes = frame.str();
      if (bytes.size() > kFrameBytes)
        throw std::runtime_error("worker replay frame exceeds slot");
      std::memcpy(frame_, bytes.data(), bytes.size());
      std::memcpy(metrics_, &cache.hits, 8);
      std::memcpy(metrics_ + 8, &cache.misses, 8);
      std::atomic_thread_fence(std::memory_order_release);
      send('D');
    }

  private:
    int         control_;
    std::size_t size_;
    int         timeout_;
    void       *memory_{};
    float      *input_{}, *output_{};
    char       *frame_{}, *metrics_{};
  };
} // namespace

int main(int argc, char **argv) {
  std::signal(SIGPIPE, SIG_IGN);
  try {
    if (argc == 2 && std::string(argv[1]) == "--identity") {
      std::cout << "{\"protocol\":\"islay-shm-v1\",\"slot_bytes\":" << kSlotBytes << "}\n";
      return 0;
    }
    std::map<std::string, std::string> options{
            {"--shm-fd", ""},    {"--control-fd", ""},          {"--workers", ""},
            {"--slot", ""},      {"--seed", "20260909"},        {"--simulations", "128"},
            {"--tree-mib", "1"}, {"--cache-mib", "1"},          {"--epsilon", "0.25"},
            {"--alpha", "0.3"},  {"--temperature-plies", "16"}, {"--timeout-ms", "30000"}};
    std::map<std::string, bool> seen;
    for (int i = 1; i < argc; i += 2) {
      if (i + 1 == argc || !options.contains(argv[i]) || seen[argv[i]])
        throw std::invalid_argument("expected unique worker --option value pairs");
      seen[argv[i]]    = true;
      options[argv[i]] = argv[i + 1];
    }
    islay::SelfplayConfig config;
    config.seed              = integer(options["--seed"], 0, UINT64_MAX);
    config.simulations       = integer(options["--simulations"], 1, 1000000);
    config.tree_bytes        = integer(options["--tree-mib"], 1, 4096) * 1024 * 1024;
    config.epsilon           = real(options["--epsilon"]);
    config.alpha             = real(options["--alpha"]);
    config.temperature_plies = integer(options["--temperature-plies"], 0, 128);
    const auto cache_bytes   = integer(options["--cache-mib"], 0, 4096) * 1024 * 1024;
    const auto workers       = integer(options["--workers"], 1, 128);
    const auto slot          = integer(options["--slot"], 0, workers - 1);
    const int  memory        = integer(options["--shm-fd"], 0, INT32_MAX);
    const int  control       = integer(options["--control-fd"], 0, INT32_MAX);
    if (memory == control)
      throw std::invalid_argument("worker descriptors must differ");
    SharedEvaluator evaluator(memory, control, workers, slot, integer(options["--timeout-ms"], 100, 120000));
    close(memory); // The mapping owns the shared pages for the remainder of this process.
    evaluator.send('V');
    char command;
    while (evaluator.receive(&command, 1, true)) {
      if (command == 'Q')
        break;
      if (command != 'G')
        throw std::runtime_error("worker expected game command");
      std::uint64_t id;
      evaluator.receive(&id, sizeof(id));
      // Reset at game boundaries: a resumed cohort must not inherit prior-game cache entries.
      islay::CachedEvaluator cache(evaluator, cache_bytes);
      const auto             records = islay::selfplay_game(id, cache, config);
      evaluator.publish(id, config, records, cache);
    }
    close(control);
  } catch (const std::exception &failure) {
    std::cerr << "selfplay worker error: " << failure.what() << '\n';
    return 1;
  }
}
