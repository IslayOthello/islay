#ifndef ISLAY_TRAIN_HPP
#define ISLAY_TRAIN_HPP

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "nn.hpp"
#include "selfplay.hpp"

namespace islay {

  struct StepMetrics {
    std::uint64_t step{0};
    std::uint64_t target_steps{0};
    float         total_loss{0.0f};
    float         policy_loss{0.0f};
    float         value_loss{0.0f};
    float         grad_norm{0.0f};
    std::uint64_t games_completed{0};
    std::size_t   buffer_size{0};
    double        steps_per_sec{0.0};
    double        elapsed_sec{0.0};
    double        eta_sec{0.0};
  };

  using ProgressCallback = std::function<void(const StepMetrics &)>;

  class ReplayBuffer {
  public:
    explicit ReplayBuffer(std::size_t max_samples = 100000);

    void        push_game(std::span<const ReplayRecord> game);
    std::size_t load_file(const std::string &path);
    bool        sample_batch(std::size_t batch_size, std::vector<TrainingSample> &out, std::mt19937_64 &rng,
                             bool augment = true) const;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept { return max_samples_; }
    void                      clear();

  private:
    std::size_t                 max_samples_;
    std::vector<TrainingSample> buffer_;
    std::size_t                 head_{0};
    bool                        full_{false};
    std::array<std::size_t, 3>  phase_counts_{};
    mutable std::mutex          mutex_;

    void store_sample(TrainingSample sample);
  };

  struct PipelineConfig {
    std::string     dataset_file     = "";
    std::uint32_t   num_workers      = 4;
    std::uint64_t   target_steps     = 1000;
    std::uint64_t   target_epochs    = 0;
    std::uint32_t   batch_size       = 64;
    std::uint32_t   checkpoint_every = 100;
    std::uint32_t   warmup_games     = 10;
    std::string     output_file      = "model.bin";
    std::string     output_dir       = "checkpoints";
    std::string     init_checkpoint  = "";
    OptimizerConfig optimizer;
    SelfplayConfig  selfplay;
  };

  class AlphaZeroTrainer {
  public:
    explicit AlphaZeroTrainer(const PipelineConfig &config);
    ~AlphaZeroTrainer();

    // Run training (offline from dataset_file or online self-play + training)
    void run(ProgressCallback on_progress = nullptr);
    void stop();

    [[nodiscard]] std::uint64_t                    total_games() const noexcept { return games_completed_.load(); }
    [[nodiscard]] std::uint64_t                    current_step() const noexcept { return step_.load(); }
    [[nodiscard]] std::shared_ptr<NativeNeuralNet> net() const noexcept { return net_; }

  private:
    PipelineConfig                   config_;
    std::shared_ptr<NativeNeuralNet> net_;
    ReplayBuffer                     buffer_;
    std::atomic<bool>                stop_requested_{false};
    std::atomic<std::uint64_t>       games_completed_{0};
    std::atomic<std::uint64_t>       step_{0};

    void worker_loop(int worker_id);
  };

  [[nodiscard]] bool train_selftest();

} // namespace islay

#endif // ISLAY_TRAIN_HPP
