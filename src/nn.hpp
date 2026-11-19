#ifndef ISLAY_NN_HPP
#define ISLAY_NN_HPP

#include <array>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "board.hpp"
#include "evaluator.hpp"
#include "game.hpp"

namespace islay {

  inline constexpr int kNumResBlocks     = 8;
  inline constexpr int kChannels         = 64;
  inline constexpr int kBoardSpatial     = 64; // 8x8
  inline constexpr int kKernelSpatial    = 9; // 3x3
  inline constexpr int kNumParams        = 606025;
  inline constexpr int kNumBnStatsFloats = 2182; // running mean and var

  // Weight layout for the versioned b8c64-v2 WDL+margin contract.
  struct ModelWeights {
    // Stem: Conv 64x2x3x3 + BN
    std::vector<float> conv_stem_w; // 64 * 2 * 9 = 1152
    std::vector<float> bn_stem_gamma; // 64
    std::vector<float> bn_stem_beta; // 64
    std::vector<float> bn_stem_mean; // 64
    std::vector<float> bn_stem_var; // 64

    // 8 Residual Blocks
    struct ResBlockWeights {
      std::vector<float> conv1_w; // 64 * 64 * 9 = 36864
      std::vector<float> bn1_gamma; // 64
      std::vector<float> bn1_beta; // 64
      std::vector<float> bn1_mean; // 64
      std::vector<float> bn1_var; // 64

      std::vector<float> conv2_w; // 64 * 64 * 9 = 36864
      std::vector<float> bn2_gamma; // 64
      std::vector<float> bn2_beta; // 64
      std::vector<float> bn2_mean; // 64
      std::vector<float> bn2_var; // 64
    };
    std::array<ResBlockWeights, kNumResBlocks> blocks;

    // Policy Head: Conv 2x64x1x1 + BN + Linear 128->65
    std::vector<float> conv_pol_w; // 2 * 64 * 1 = 128
    std::vector<float> bn_pol_gamma; // 2
    std::vector<float> bn_pol_beta; // 2
    std::vector<float> bn_pol_mean; // 2
    std::vector<float> bn_pol_var; // 2
    std::vector<float> fc_pol_w; // 65 * 128 = 8320
    std::vector<float> fc_pol_b; // 65

    // Value Head: Conv 1x64x1x1 + BN + Linear 64->64 + two Linear 64->1 outputs
    std::vector<float> conv_val_w; // 1 * 64 * 1 = 64
    std::vector<float> bn_val_gamma; // 1
    std::vector<float> bn_val_beta; // 1
    std::vector<float> bn_val_mean; // 1
    std::vector<float> bn_val_var; // 1
    std::vector<float> fc_val1_w; // 64 * 64 = 4096
    std::vector<float> fc_val1_b; // 64
    std::vector<float> fc_val2_w; // 2 * 64 (WDL, margin)
    std::vector<float> fc_val2_b; // 2 (WDL, margin)

    ModelWeights();
    void                      init_random(std::uint64_t seed = 20260909);
    [[nodiscard]] std::size_t total_trainable_params() const noexcept;
    void                      apply_ema(const ModelWeights &current, float decay) noexcept;
  };

  // Pre-fused weights for zero-overhead inference: Conv + BN folded into Conv + Bias
  struct FusedWeights {
    // Stem
    std::vector<float> stem_w; // 64 * 2 * 9
    std::vector<float> stem_w_packed; // [64/4, 2, 9, 4] for packed inference
    std::vector<float> stem_b; // 64

    // 8 ResBlocks
    struct FusedBlock {
      std::vector<float> conv1_w; // 64 * 64 * 9
      std::vector<float> conv1_w_packed; // [64/4, 64, 9, 4]
      std::vector<float> conv1_b; // 64
      std::vector<float> conv2_w; // 64 * 64 * 9
      std::vector<float> conv2_w_packed; // [64/4, 64, 9, 4]
      std::vector<float> conv2_b; // 64
    };
    std::array<FusedBlock, kNumResBlocks> blocks;

    // Policy Head
    std::vector<float> pol_conv_w; // 2 * 64 * 1
    std::vector<float> pol_conv_b; // 2
    std::vector<float> pol_fc_w; // 65 * 128
    std::vector<float> pol_fc_b; // 65

    // Value Head
    std::vector<float> val_conv_w; // 1 * 64 * 1
    std::vector<float> val_conv_b; // 1
    std::vector<float> val_fc1_w; // 64 * 64
    std::vector<float> val_fc1_b; // 64
    std::vector<float> val_fc2_w; // 2 * 64
    std::vector<float> val_fc2_b; // 2

    FusedWeights();
    void fuse_from(const ModelWeights &src);
  };

  // Training sample input
  struct TrainingSample {
    Board                          board;
    Bitboard                       legal_moves{0};
    bool                           forced_pass{false};
    std::array<float, kPolicySize> target_policy{};
    float                          target_wdl{0.0f};
    float                          target_margin{0.0f};
  };

  // Optimizer hyperparameters
  struct OptimizerConfig {
    float learning_rate = 0.01f;
    float momentum      = 0.9f;
    float weight_decay  = 0.0001f;
    float grad_clip     = 5.0f;
  };

  class NativeNeuralNet {
  public:
    NativeNeuralNet();
    explicit NativeNeuralNet(const std::string &checkpoint_path);
    ~NativeNeuralNet();

    // High-performance inference (uses fused weights + SIMD kernels)
    void evaluate_board(const Board &board, Evaluation &output, bool mask_legal = true) const;
    void evaluate_batch(std::span<const Board> boards, std::span<Evaluation> outputs, bool mask_legal = true) const;

    // Training pass: Forward, Loss with Legal Masking, Backward, and Optimizer Step
    struct TrainMetrics {
      float total_loss{0.0f};
      float policy_loss{0.0f};
      float value_loss{0.0f};
      float grad_norm{0.0f};
    };
    TrainMetrics train_batch(std::span<const TrainingSample> batch, const OptimizerConfig &opt_cfg);

    // Serialization
    void save(const std::string &path, std::uint64_t training_steps = 0, bool use_ema = false) const;
    void load(const std::string &path);
    void update_fused_weights();

    void update_ema(float decay = 0.999f) noexcept;
    void fuse_ema() noexcept;

    [[nodiscard]] std::uint64_t training_steps() const noexcept { return training_steps_; }
    void                        set_training_steps(std::uint64_t s) noexcept { training_steps_ = s; }

    [[nodiscard]] const ModelWeights &weights() const noexcept { return weights_; }
    [[nodiscard]] ModelWeights       &weights() noexcept { return weights_; }
    [[nodiscard]] const ModelWeights &ema_weights() const noexcept { return ema_weights_; }

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    ModelWeights          weights_;
    ModelWeights          ema_weights_;
    bool                  ema_initialized_{false};
    FusedWeights          fused_;
    std::uint64_t         training_steps_{0};
  };

  // Evaluator adapter implementing the Evaluator interface
  class NativeNeuralEvaluator final : public Evaluator {
  public:
    explicit NativeNeuralEvaluator(std::shared_ptr<NativeNeuralNet> net, bool mask_legal = true);
    void evaluate(std::span<const Board> boards, std::span<Evaluation> outputs, std::stop_token stop) override;
    [[nodiscard]] std::string description() const override;

  private:
    std::shared_ptr<NativeNeuralNet> net_;
    bool                             mask_legal_{true};
  };

  [[nodiscard]] bool nn_selftest();

} // namespace islay

#endif // ISLAY_NN_HPP
