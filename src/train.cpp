#include "train.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <thread>

namespace islay {

  namespace {
    std::size_t sample_phase(const TrainingSample &sample) noexcept {
      const int discs = sample.board.count();
      return discs <= 20 ? 0 : discs <= 44 ? 1 : 2;
    }

    TrainingSample make_sample_from_record(const ReplayRecord &r) {
      TrainingSample sample;
      sample.board        = r.board;
      sample.legal_moves  = r.legal;
      sample.forced_pass  = (r.pass != 0);
      sample.target_wdl    = replay_wdl_target(r);
      sample.target_margin = replay_margin_target(r);

      std::uint64_t total_visits = 0;
      for (int a = 0; a < kPolicySize; ++a)
        total_visits += r.visits[a];

      if (total_visits > 0) {
        const float inv_v = 1.0f / static_cast<float>(total_visits);
        for (int a = 0; a < kPolicySize; ++a)
          sample.target_policy[a] = static_cast<float>(r.visits[a]) * inv_v;
      } else {
        sample.target_policy = r.priors;
      }
      return sample;
    }

    TrainingSample apply_symmetry(const TrainingSample &src, int s) {
      if (s == 0)
        return src;
      TrainingSample dst;
      dst.board        = src.board.symmetry(s);
      dst.legal_moves  = Board::symmetry_bb(src.legal_moves, s);
      dst.forced_pass  = src.forced_pass;
      dst.target_wdl    = src.target_wdl;
      dst.target_margin = src.target_margin;

      dst.target_policy.fill(0.0f);
      for (int a = 0; a < kPolicySize; ++a) {
        const Square sa       = symmetry_action(a, s);
        dst.target_policy[sa] = src.target_policy[a];
      }
      return dst;
    }
  } // namespace

  ReplayBuffer::ReplayBuffer(std::size_t max_samples) : max_samples_(max_samples) { buffer_.resize(max_samples_); }

  void ReplayBuffer::store_sample(TrainingSample sample) {
    if (full_)
      --phase_counts_[sample_phase(buffer_[head_])];
    buffer_[head_] = std::move(sample);
    ++phase_counts_[sample_phase(buffer_[head_])];
    head_ = (head_ + 1) % max_samples_;
    if (head_ == 0)
      full_ = true;
  }

  void ReplayBuffer::push_game(std::span<const ReplayRecord> game) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto &r: game) {
      const TrainingSample base = make_sample_from_record(r);
      for (int s = 0; s < 8; ++s)
        store_sample(apply_symmetry(base, s));
    }
  }

  std::size_t ReplayBuffer::load_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    if (!file)
      throw std::runtime_error("cannot open replay file: " + path);

    ReplayRecord                record;
    std::vector<TrainingSample> loaded;
    while (read_replay_record(file, record)) {
      const TrainingSample base = make_sample_from_record(record);
      for (int s = 0; s < 8; ++s)
        loaded.push_back(apply_symmetry(base, s));
    }

    if (loaded.empty())
      return 0;

    std::lock_guard<std::mutex> lock(mutex_);
    if (loaded.size() > max_samples_) {
      max_samples_ = loaded.size();
      buffer_.resize(max_samples_);
    }
    for (auto &s: loaded)
      store_sample(std::move(s));
    return loaded.size();
  }

  std::size_t ReplayBuffer::size() const noexcept {
    std::lock_guard<std::mutex> lock(mutex_);
    return full_ ? max_samples_ : head_;
  }

  void ReplayBuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    head_ = 0;
    full_ = false;
    phase_counts_.fill(0);
  }

  bool ReplayBuffer::sample_batch(std::size_t batch_size, std::vector<TrainingSample> &out, std::mt19937_64 &rng,
                                  bool augment) const {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::size_t           current_size = full_ ? max_samples_ : head_;
    if (current_size == 0)
      return false;

    out.resize(batch_size);
    std::uniform_int_distribution<std::size_t> dist_idx(0, current_size - 1);
    std::uniform_int_distribution<int>         dist_sym(0, 7);
    constexpr std::array<int, 8> phase_schedule{0, 0, 1, 1, 1, 2, 2, 2};
    std::uniform_int_distribution<std::size_t> retry_dist(0, current_size - 1);

    for (std::size_t i = 0; i < batch_size; ++i) {
      const int requested_phase = phase_schedule[i % phase_schedule.size()];
      std::size_t selected       = dist_idx(rng);
      if (phase_counts_[requested_phase] > 0) {
        for (int attempt = 0; attempt < 64; ++attempt) {
          const std::size_t candidate = retry_dist(rng);
          if (sample_phase(buffer_[candidate]) == static_cast<std::size_t>(requested_phase)) {
            selected = candidate;
            break;
          }
        }
      }
      const auto &src = buffer_[selected];
      if (!augment) {
        out[i] = src;
      } else {
        const int s      = dist_sym(rng);
        auto     &dst    = out[i];
        dst.board        = src.board.symmetry(s);
        dst.legal_moves  = Board::symmetry_bb(src.legal_moves, s);
        dst.forced_pass  = src.forced_pass;
        dst.target_wdl    = src.target_wdl;
        dst.target_margin = src.target_margin;

        dst.target_policy.fill(0.0f);
        for (int a = 0; a < kPolicySize; ++a) {
          const Square sa       = symmetry_action(a, s);
          dst.target_policy[sa] = src.target_policy[a];
        }
      }
    }
    return true;
  }

  // ---------------------------------------------------------------------------
  //  AlphaZeroTrainer Implementation
  // ---------------------------------------------------------------------------

  AlphaZeroTrainer::AlphaZeroTrainer(const PipelineConfig &config) :
      config_(config), net_(std::make_shared<NativeNeuralNet>()), buffer_(100000) {
    if (!config_.init_checkpoint.empty()) {
      net_->load(config_.init_checkpoint);
      step_ = net_->training_steps();
    }
    if (!config_.output_dir.empty())
      std::filesystem::create_directories(config_.output_dir);
  }

  AlphaZeroTrainer::~AlphaZeroTrainer() { stop(); }

  void AlphaZeroTrainer::stop() { stop_requested_.store(true); }

  void AlphaZeroTrainer::worker_loop(int worker_id) {
    NativeNeuralEvaluator eval(net_, true);
    CachedEvaluator       cached(eval, 16 * 1024 * 1024);
    std::mt19937_64       rng(config_.selfplay.seed + worker_id * 7919);

    while (!stop_requested_.load()) {
      const std::uint64_t game_id = (games_completed_.load() * 1024) + worker_id;
      try {
        const auto records = selfplay_game(game_id, cached, config_.selfplay);
        buffer_.push_game(records);
        games_completed_.fetch_add(1);
      } catch (const std::exception &e) {
        // Continue worker on transient search exception
      }
    }
  }

  void AlphaZeroTrainer::run(ProgressCallback on_progress) {
    stop_requested_.store(false);

    const bool is_offline = !config_.dataset_file.empty();

    std::vector<std::thread> workers;

    if (is_offline) {
      std::stringstream ss(config_.dataset_file);
      std::string       path;
      std::size_t       total_loaded = 0;
      while (std::getline(ss, path, ',')) {
        if (!path.empty()) {
          total_loaded += buffer_.load_file(path);
        }
      }
      if (total_loaded == 0)
        throw std::runtime_error("dataset file is empty or cannot be read: " + config_.dataset_file);
      if (config_.target_epochs > 0) {
        const std::uint64_t batches_per_epoch =
                (static_cast<std::uint64_t>(buffer_.size()) + config_.batch_size - 1) / config_.batch_size;
        if (batches_per_epoch == 0 ||
            config_.target_epochs > std::numeric_limits<std::uint64_t>::max() / batches_per_epoch)
          throw std::runtime_error("epoch count is too large");
        config_.target_steps = config_.target_epochs * batches_per_epoch;
      }
    } else {
      // 1. Launch self-play worker threads
      workers.reserve(config_.num_workers);
      for (std::uint32_t w = 0; w < config_.num_workers; ++w)
        workers.emplace_back(&AlphaZeroTrainer::worker_loop, this, static_cast<int>(w));

      // 2. Wait for warmup games
      while (!stop_requested_.load() && games_completed_.load() < config_.warmup_games)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::mt19937_64             train_rng(1337);
    std::vector<TrainingSample> batch;
    const auto                  start_time = std::chrono::steady_clock::now();

    // 3. Training Loop
    const std::uint64_t start_step = step_.load();
    const std::uint64_t end_step   = start_step + config_.target_steps;

    while (!stop_requested_.load() && step_.load() < end_step) {
      if (!buffer_.sample_batch(config_.batch_size, batch, train_rng, true)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        continue;
      }

      OptimizerConfig opt = config_.optimizer;
      if (config_.target_steps > 0) {
        const double progress =
                static_cast<double>(step_.load() - start_step) / static_cast<double>(config_.target_steps);
        const float lr_mult = static_cast<float>(0.1 + 0.9 * 0.5 * (1.0 + std::cos(progress * 3.14159265358979323846)));
        opt.learning_rate   = config_.optimizer.learning_rate * lr_mult;
      }
      const auto metrics  = net_->train_batch(batch, opt);
      const auto cur_step = net_->training_steps();
      net_->update_ema(0.999f);
      step_.store(cur_step);

      const auto          now        = std::chrono::steady_clock::now();
      const auto          elapsed    = std::chrono::duration<double>(now - start_time).count();
      const std::uint64_t steps_done = cur_step - start_step;
      const double        sps        = elapsed > 0 ? static_cast<double>(steps_done) / elapsed : 0.0;
      const double        eta        = sps > 0 ? static_cast<double>(end_step - cur_step) / sps : 0.0;

      StepMetrics m;
      m.step            = cur_step;
      m.target_steps    = end_step;
      m.total_loss      = metrics.total_loss;
      m.policy_loss     = metrics.policy_loss;
      m.value_loss      = metrics.value_loss;
      m.grad_norm       = metrics.grad_norm;
      m.games_completed = games_completed_.load();
      m.buffer_size     = buffer_.size();
      m.steps_per_sec   = sps;
      m.elapsed_sec     = elapsed;
      m.eta_sec         = eta;

      if (on_progress) {
        on_progress(m);
      } else {
        if (steps_done % 10 == 0 || cur_step == end_step) {
          std::cout << std::fixed << std::setprecision(4) << "[Step " << std::setw(6) << cur_step << "/" << end_step
                    << "] "
                    << "Loss: " << metrics.total_loss << " (Pol: " << metrics.policy_loss
                    << ", Val: " << metrics.value_loss << ") "
                    << "GradNorm: " << metrics.grad_norm << " "
                    << "Games: " << games_completed_.load() << " "
                    << "Buf: " << buffer_.size() << " "
                    << "Speed: " << std::setprecision(1) << sps << " steps/s\n";
        }
      }

      if (config_.checkpoint_every > 0 && cur_step % config_.checkpoint_every == 0) {
        if (!config_.output_dir.empty()) {
          std::ostringstream ss;
          ss << config_.output_dir << "/checkpoint_step_" << std::setw(7) << std::setfill('0') << cur_step << ".bin";
          net_->save(ss.str(), cur_step, true);
          if (!on_progress)
            std::cout << "Saved checkpoint (EMA): " << ss.str() << '\n';
        }
      }
    }

    if (!is_offline) {
      stop_requested_.store(true);
      for (auto &w: workers) {
        if (w.joinable())
          w.join();
      }
    }

    // Fold EMA weights into active fused model and save
    net_->fuse_ema();
    if (!config_.output_file.empty()) {
      net_->save(config_.output_file, step_.load(), true);
      if (!on_progress)
        std::cout << "Model saved to: " << config_.output_file << " (EMA smoothed)\n";
    } else if (!config_.output_dir.empty()) {
      const std::string final_path = config_.output_dir + "/model_final.bin";
      net_->save(final_path, step_.load(), true);
      if (!on_progress)
        std::cout << "Final model saved to: " << final_path << " (EMA smoothed)\n";
    }
  }

  // ---------------------------------------------------------------------------
  //  Selftest
  // ---------------------------------------------------------------------------

  bool train_selftest() {
    try {
      ReplayBuffer buf(1000);
      if (buf.size() != 0 || buf.capacity() != 1000)
        return false;

      // 1. Generate one self-play game with random net
      auto                  net = std::make_shared<NativeNeuralNet>();
      NativeNeuralEvaluator eval(net, true);

      SelfplayConfig cfg;
      cfg.simulations       = 4; // tiny budget for instant test
      cfg.tree_bytes        = 1024 * 1024;
      cfg.temperature_plies = 4;

      const auto game = selfplay_game(1001, eval, cfg);
      if (game.empty())
        return false;

      buf.push_game(game);
      if (buf.size() != game.size() * 8)
        return false;

      // 2. Sample batch with symmetry augmentation
      std::mt19937_64             rng(42);
      std::vector<TrainingSample> batch;
      if (!buf.sample_batch(4, batch, rng, true) || batch.size() != 4)
        return false;

      // 3. Perform 1 training step
      OptimizerConfig opt;
      opt.learning_rate  = 0.01f;
      const auto metrics = net->train_batch(batch, opt);
      if (!std::isfinite(metrics.total_loss) || metrics.total_loss <= 0.0f)
        return false;

      // 4. Test replay serialization & load_file
      const std::string tmp_file = "_build/selftest_replay.bin";
      {
        std::ofstream out(tmp_file, std::ios::binary);
        for (const auto &r: game)
          write_replay_record(out, r);
      }
      ReplayBuffer buf2(1000);
      if (buf2.load_file(tmp_file) != game.size() * 8 || buf2.size() != game.size() * 8)
        return false;
      std::filesystem::remove(tmp_file);

      return true;
    } catch (...) {
      return false;
    }
  }

} // namespace islay
