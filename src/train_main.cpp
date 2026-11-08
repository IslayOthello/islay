#include <iostream>
#include <string>

#include "train.hpp"

using namespace islay;

int main(int argc, char **argv) {
  std::cout << "islay-train: Native C++ AlphaZero Self-Play & Training Engine\n";

  PipelineConfig config;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--data" && i + 1 < argc) {
      config.dataset_file = argv[++i];
    } else if (arg == "--workers" && i + 1 < argc) {
      config.num_workers = std::stoul(argv[++i]);
    } else if (arg == "--steps" && i + 1 < argc) {
      config.target_steps = std::stoul(argv[++i]);
    } else if (arg == "--batch" && i + 1 < argc) {
      config.batch_size = std::stoul(argv[++i]);
    } else if (arg == "--sims" && i + 1 < argc) {
      config.selfplay.simulations = std::stoul(argv[++i]);
    } else if (arg == "--warmup" && i + 1 < argc) {
      config.warmup_games = std::stoul(argv[++i]);
    } else if (arg == "--checkpoint-every" && i + 1 < argc) {
      config.checkpoint_every = std::stoul(argv[++i]);
    } else if (arg == "--output" && i + 1 < argc) {
      const std::string out = argv[++i];
      if (out.ends_with(".bin"))
        config.output_file = out;
      else
        config.output_dir = out;
    } else if (arg == "--init" && i + 1 < argc) {
      config.init_checkpoint = argv[++i];
    } else if (arg == "--lr" && i + 1 < argc) {
      config.optimizer.learning_rate = std::stof(argv[++i]);
    } else if (arg == "--help" || arg == "-h") {
      std::cout << "Usage: islay-train [options]\n"
                << "  --data <file>           Replay dataset file for offline training\n"
                << "  --workers <N>           Number of self-play threads (default: 4)\n"
                << "  --steps <N>             Target optimization steps (default: 1000)\n"
                << "  --batch <N>             Batch size (default: 64)\n"
                << "  --sims <N>              MCTS simulations per move (default: 128)\n"
                << "  --warmup <N>            Warmup games before training (default: 10)\n"
                << "  --checkpoint-every <N>  Save checkpoint every N steps (default: 100)\n"
                << "  --output <path>         Output model file or checkpoints dir\n"
                << "  --init <path>           Initial .bin checkpoint to load\n"
                << "  --lr <float>            Learning rate (default: 0.01)\n";
      return 0;
    }
  }

  AlphaZeroTrainer trainer(config);
  trainer.run();

  return 0;
}
