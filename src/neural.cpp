#include "neural.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <vector>

#if defined(ISLAY_ONNX_ENABLED)
#include <onnxruntime_cxx_api.h>
#endif

namespace islay {

  void encode_neural_board(const Board &board, std::span<float, kNeuralInputSize> planes) {
    if (board.player & board.opponent)
      throw std::invalid_argument("neural input has overlapping discs");
    for (int sq = 0; sq < 64; ++sq) {
      planes[sq]      = static_cast<float>((board.player >> sq) & 1);
      planes[64 + sq] = static_cast<float>((board.opponent >> sq) & 1);
    }
  }

  bool neural_selftest() {
    std::array<float, kNeuralInputSize> input{};
    for (int side = 0; side < 2; ++side) {
      for (int sq = 0; sq < 64; ++sq) {
        const Board board = side ? Board{0, square_bb(sq)} : Board{square_bb(sq), 0};
        for (int symmetry = 0; symmetry < 8; ++symmetry) {
          encode_neural_board(board.symmetry(symmetry), input);
          for (int i = 0; i < kNeuralInputSize; ++i) {
            if (input[i] != (i == side * 64 + symmetry_action(sq, symmetry) ? 1.0f : 0.0f))
              return false;
          }
        }
      }
    }
    try {
      encode_neural_board(Board{1, 1}, input);
    } catch (const std::invalid_argument &) {
      return true;
    }
    return false;
  }

#if defined(ISLAY_ONNX_ENABLED)
  struct NeuralEvaluator::Impl {
    Ort::Env           env{ORT_LOGGING_LEVEL_ERROR, "islay"};
    Ort::Session       session{nullptr};
    Ort::MemoryInfo    memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<float> input;

    Impl(const std::string &path, bool optimize) {
      std::ifstream file(path, std::ios::binary | std::ios::ate);
      if (!file)
        throw std::runtime_error("cannot open EvalFile");
      const auto bytes = file.tellg();
      if (bytes <= 0 || bytes > 64 * 1024 * 1024)
        throw std::runtime_error("EvalFile must be a single ONNX file of at most 64 MiB");
      std::vector<char> data(static_cast<std::size_t>(bytes));
      file.seekg(0);
      if (!file.read(data.data(), static_cast<std::streamsize>(data.size())))
        throw std::runtime_error("cannot read EvalFile");
      Ort::SessionOptions options;
      options.SetIntraOpNumThreads(1);
      options.SetInterOpNumThreads(1);
      options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
      options.SetGraphOptimizationLevel(optimize ? ORT_ENABLE_ALL : ORT_DISABLE_ALL);
      options.AddConfigEntry("session.intra_op.allow_spinning", "0");
      options.AddConfigEntry("session.inter_op.allow_spinning", "0");
      session = Ort::Session(env, data.data(), data.size(), options);
      if (session.GetInputCount() != 1 || session.GetOutputCount() != 2)
        throw std::runtime_error("expected one input and two outputs");
      Ort::AllocatorWithDefaultOptions allocator;
      if (std::string(session.GetInputNameAllocated(0, allocator).get()) != "board" ||
          std::string(session.GetOutputNameAllocated(0, allocator).get()) != "policy_logits" ||
          std::string(session.GetOutputNameAllocated(1, allocator).get()) != "value")
        throw std::runtime_error("unexpected neural input/output names");
      const auto check = [](const Ort::TypeInfo &type, const std::vector<std::int64_t> &shape) {
        const auto tensor = type.GetTensorTypeAndShapeInfo();
        if (tensor.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT || tensor.GetShape() != shape)
          throw std::runtime_error("unexpected neural input/output shape or dtype");
      };
      check(session.GetInputTypeInfo(0), {-1, 2, 8, 8});
      check(session.GetOutputTypeInfo(0), {-1, 65});
      check(session.GetOutputTypeInfo(1), {-1, 1});
    }

    std::string metadata(const char *key) {
      Ort::AllocatorWithDefaultOptions allocator;
      const auto value = session.GetModelMetadata().LookupCustomMetadataMapAllocated(key, allocator);
      if (!value)
        throw std::runtime_error(std::string("missing neural metadata: ") + key);
      return value.get();
    }
  };

  const char *neural_backend() noexcept { return "onnx-cpu"; }
  const char *neural_runtime_version() noexcept { return OrtGetApiBase()->GetVersionString(); }

  NeuralEvaluator::NeuralEvaluator(const std::string &path, bool optimize) :
      impl_(std::make_unique<Impl>(path, optimize)) {
    if (impl_->metadata("islay.arch") != "b8c64-v1" || impl_->metadata("islay.encoding") != "relative-2x8x8-a1-v1" ||
        impl_->metadata("islay.policy") != "a1-h8-pass65-v1" || impl_->metadata("islay.value") != "relative-wdl-v1")
      throw std::runtime_error("unsupported neural architecture or encoding");
    const auto identity = impl_->metadata("islay.checkpoint_sha256");
    const auto steps    = impl_->metadata("islay.training_steps");
    if (identity.size() != 64 || identity.find_first_not_of("0123456789abcdef") != std::string::npos || steps.empty() ||
        steps.size() > 20 || steps.find_first_not_of("0123456789") != std::string::npos)
      throw std::runtime_error("invalid neural checkpoint identity/training steps");
    description_ = "onnx-cpu b8c64-v1 checkpoint " + identity + " training_steps " + steps;
    if (steps.find_first_not_of('0') == std::string::npos)
      description_ += " (untrained initialization)";
    const Board board = Board::start();
    Evaluation  output;
    evaluate(std::span(&board, 1), std::span(&output, 1), {});
  }

  NeuralEvaluator::~NeuralEvaluator() = default;

  void NeuralEvaluator::evaluate(std::span<const Board> boards, std::span<Evaluation> outputs, std::stop_token stop) {
    if (boards.size() != outputs.size() || boards.size() > kNeuralMaxBatch)
      throw std::invalid_argument("neural batch must have matching spans and at most 128 boards");
    if (boards.empty() || stop.stop_requested())
      return;
    impl_->input.resize(boards.size() * kNeuralInputSize);
    for (std::size_t i = 0; i < boards.size(); ++i)
      encode_neural_board(boards[i], std::span<float, kNeuralInputSize>(impl_->input.data() + i * kNeuralInputSize,
                                                                        kNeuralInputSize));
    const std::array<std::int64_t, 4> shape{static_cast<std::int64_t>(boards.size()), 2, 8, 8};
    auto                              input =
            Ort::Value::CreateTensor<float>(impl_->memory, impl_->input.data(), impl_->input.size(), shape.data(), 4);
    const char              *input_names[]  = {"board"};
    const char              *output_names[] = {"policy_logits", "value"};
    Ort::RunOptions          run;
    const std::stop_callback cancel(stop, [&run]() noexcept {
      // ORT explicitly permits terminating a Run from another thread.
      auto *status = Ort::GetApi().RunOptionsSetTerminate(run);
      if (status)
        Ort::GetApi().ReleaseStatus(status);
    });
    try {
      auto result = impl_->session.Run(run, input_names, &input, 1, output_names, 2);
      if (stop.stop_requested())
        return;
      if (result[0].GetTensorTypeAndShapeInfo().GetShape() !=
                  std::vector<std::int64_t>{static_cast<std::int64_t>(boards.size()), 65} ||
          result[1].GetTensorTypeAndShapeInfo().GetShape() !=
                  std::vector<std::int64_t>{static_cast<std::int64_t>(boards.size()), 1})
        throw std::runtime_error("neural runtime output shape mismatch");
      const auto *policy = result[0].GetTensorData<float>();
      const auto *value  = result[1].GetTensorData<float>();
      for (std::size_t i = 0; i < boards.size(); ++i) {
        if (!std::isfinite(value[i]) || std::abs(value[i]) > 1 ||
            !std::all_of(policy + i * 65, policy + (i + 1) * 65, [](float x) { return std::isfinite(x); }))
          throw std::runtime_error("neural runtime returned non-finite logits or invalid value");
      }
      for (std::size_t i = 0; i < boards.size(); ++i) {
        std::copy_n(policy + i * 65, 65, outputs[i].logits.begin());
        outputs[i].value = value[i];
      }
    } catch (const Ort::Exception &) {
      if (!stop.stop_requested())
        throw;
    }
  }
#else
  struct NeuralEvaluator::Impl {};
  const char *neural_backend() noexcept { return "disabled"; }
  const char *neural_runtime_version() noexcept { return "none"; }
  NeuralEvaluator::NeuralEvaluator(const std::string &, bool) {
    throw std::runtime_error("neural inference unavailable; rebuild with ISLAY_ONNX=ON");
  }
  NeuralEvaluator::~NeuralEvaluator() = default;
  void NeuralEvaluator::evaluate(std::span<const Board>, std::span<Evaluation>, std::stop_token) {
    throw std::runtime_error("neural inference unavailable");
  }
#endif
} // namespace islay
