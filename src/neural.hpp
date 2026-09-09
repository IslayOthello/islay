#ifndef ISLAY_NEURAL_HPP
#define ISLAY_NEURAL_HPP

#include <memory>
#include <string>

#include "evaluator.hpp"

namespace islay {
  inline constexpr int      kNeuralInputSize = 128;
  inline constexpr int      kNeuralMaxBatch  = 128;
  void                      encode_neural_board(const Board &board, std::span<float, kNeuralInputSize> planes);
  [[nodiscard]] bool        neural_selftest();
  [[nodiscard]] const char *neural_backend() noexcept;

  // CPU reference backend; one owner/call at a time, no internal worker pool.
  // Metadata declares a checkpoint identity, not an authenticity guarantee.
  class NeuralEvaluator final : public Evaluator {
  public:
    explicit NeuralEvaluator(const std::string &path, bool optimize = true);
    ~NeuralEvaluator() override;
    void evaluate(std::span<const Board> boards, std::span<Evaluation> outputs, std::stop_token stop) override;
    [[nodiscard]] const std::string &description() const noexcept { return description_; }

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    std::string           description_;
  };
} // namespace islay

#endif // ISLAY_NEURAL_HPP
