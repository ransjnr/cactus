#include "bench_driver.h"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>

namespace bench {

static std::vector<BackendVariant>& backend_registry() {
    static std::vector<BackendVariant> backends;
    return backends;
}

void register_backend(BackendVariant v) {
    backend_registry().push_back(v);
}

const std::vector<BackendVariant>& get_backends() {
    return backend_registry();
}

static bool framework_matches_filter(const char* framework, const std::string& filter) {
    if (filter.empty()) return true;
    std::string f = filter;
    std::string fw(framework);
    size_t pos = 0;
    while (pos < f.size()) {
        size_t comma = f.find(',', pos);
        if (comma == std::string::npos) comma = f.size();
        std::string token = f.substr(pos, comma - pos);
        if (token == fw) return true;
        pos = comma + 1;
    }
    return false;
}

bool run_comparison(TestUtils::TestRunner& runner, const BenchOptions& opt_in) {
    BenchOptions opt = opt_in;
    if (opt.num_threads <= 0)
        opt.num_threads = static_cast<int>(std::thread::hardware_concurrency());

    const auto& all_backends = get_backends();

    std::vector<const BackendVariant*> active;
    for (const auto& b : all_backends) {
        if (framework_matches_filter(b.framework, opt.backends_filter))
            active.push_back(&b);
    }

    if (active.empty()) {
        runner.log_performance("Error", "No backends matched filter");
        return false;
    }

    {
        std::ostringstream cfg;
        cfg << "warmup=" << opt.warmup
            << ", iterations=" << opt.iterations
            << ", threads=" << opt.num_threads
            << ", backends=";
        for (size_t i = 0; i < active.size(); ++i) {
            if (i > 0) cfg << ",";
            cfg << active[i]->name;
        }
        runner.log_performance("Config", cfg.str());
    }

    std::mt19937 gen(270270u);

    for (size_t M : opt.batch_sizes) {
        runner.log_performance("",
            "─────────────────────────────────────────────────────────────────────────────────────────────────────────");

        std::vector<double> backend_totals(active.size(), 0.0);

        struct AccuracyTracker {
            float worst_nrmse = 0.0f;
            float worst_max_err = 0.0f;
            bool any_failed = false;
            bool ran = false;
        };
        std::vector<AccuracyTracker> acc_trackers(active.size());

        for (size_t p = 0; p < kProjectionSpecs.size(); ++p) {
            const auto& spec = kProjectionSpecs[p];

            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            std::vector<float> fp32_weights(spec.N * spec.K);
            for (auto& v : fp32_weights) v = dist(gen);

            std::mt19937 agen(42 + M + p);
            auto act = prepare_cactus_activations(M, spec.K, agen);

            struct BackendState {
                void* weights = nullptr;
                void* activations = nullptr;
            };
            std::vector<BackendState> states(active.size());

            for (size_t bi = 0; bi < active.size(); ++bi) {
                states[bi].weights = active[bi]->prepare_weights(
                    fp32_weights.data(), spec.N, spec.K);
                if (active[bi]->prepare_activations) {
                    states[bi].activations = active[bi]->prepare_activations(
                        act.fp32.data(), M, spec.K, states[bi].weights);
                }
            }

            // Accuracy check: compute fp32 reference once, then each backend
            // populates capture_output during its warmup
            std::vector<float> reference(M * spec.N);
            reference_matmul_fp32(act.fp32.data(), fp32_weights.data(),
                                  reference.data(), M, spec.K, spec.N);

            const size_t out_count = M * spec.N;
            std::vector<float> captured(out_count, 0.0f);
            std::vector<float> backend_ref(out_count, 0.0f);
            opt.capture_output = captured.data();
            opt.capture_reference = backend_ref.data();

            std::ostringstream details;
            details << std::fixed;

            for (size_t bi = 0; bi < active.size(); ++bi) {
                if (active[bi]->max_M > 0 && M > active[bi]->max_M) {
                    if (bi > 0) details << "  ";
                    details << active[bi]->name << "=skipped(M>" << active[bi]->max_M << ")";
                    continue;
                }

                std::fill(captured.begin(), captured.end(), 0.0f);
                std::fill(backend_ref.begin(), backend_ref.end(), 0.0f);

                auto result = active[bi]->bench_fn(
                    M, states[bi].weights, states[bi].activations,
                    act.int8.data(), act.scales.data(), opt);

                bool has_output = false;
                for (size_t i = 0; i < out_count && !has_output; i++)
                    if (captured[i] != 0.0f) has_output = true;

                if (has_output) {
                    bool has_backend_ref = false;
                    for (size_t i = 0; i < out_count && !has_backend_ref; i++)
                        if (backend_ref[i] != 0.0f) has_backend_ref = true;

                    const float* ref = has_backend_ref ? backend_ref.data() : reference.data();
                    float tol = has_backend_ref ? 0.01f
                              : (active[bi]->category == QuantCategory::INT8) ? 0.05f : 0.20f;
                    auto acc = check_accuracy(ref, captured.data(), out_count, tol);

                    acc_trackers[bi].ran = true;
                    if (acc.nrmse > acc_trackers[bi].worst_nrmse)
                        acc_trackers[bi].worst_nrmse = acc.nrmse;
                    if (acc.max_abs_error > acc_trackers[bi].worst_max_err)
                        acc_trackers[bi].worst_max_err = acc.max_abs_error;
                    if (!acc.passed) acc_trackers[bi].any_failed = true;
                }

                backend_totals[bi] += result.avg_us;

                if (bi > 0) details << "  ";
                details << active[bi]->name << "="
                        << std::setprecision(1) << result.avg_us << "us";
            }

            std::string name = std::string(spec.name) + " M=" + std::to_string(M)
                + " " + std::to_string(M) + "x" + std::to_string(spec.K)
                + "x" + std::to_string(spec.N);
            runner.log_performance(name, details.str());

            for (size_t bi = 0; bi < active.size(); ++bi) {
                if (active[bi]->cleanup)
                    active[bi]->cleanup(states[bi].weights, states[bi].activations);
            }
        }

        // Print accuracy summary for this batch size
        bool has_accuracy = false;
        for (size_t bi = 0; bi < active.size(); ++bi)
            if (acc_trackers[bi].ran) { has_accuracy = true; break; }

        if (has_accuracy) {
            std::ostringstream acc_line;
            acc_line << std::fixed;
            bool first = true;
            for (size_t bi = 0; bi < active.size(); ++bi) {
                if (!acc_trackers[bi].ran) continue;
                if (!first) acc_line << "  ";
                first = false;
                acc_line << active[bi]->name << "="
                         << (acc_trackers[bi].any_failed ? "FAIL" : "PASS")
                         << " nrmse=" << std::setprecision(4) << acc_trackers[bi].worst_nrmse
                         << " max=" << std::setprecision(2) << acc_trackers[bi].worst_max_err;
            }
            runner.log_performance("Accuracy M=" + std::to_string(M), acc_line.str());
        }

        std::ostringstream summary;
        summary << std::fixed;
        for (size_t bi = 0; bi < active.size(); ++bi) {
            if (bi > 0) summary << "  ";
            summary << active[bi]->name << "="
                    << std::setprecision(1) << backend_totals[bi] << "us";
        }
        runner.log_performance("TOTAL M=" + std::to_string(M), summary.str());
    }

    return true;
}

bool run_stack(TestUtils::TestRunner& runner, const BenchOptions& opt_in) {
    // TODO: add activation quantization timing (per-projection quant_ms vs kernel_ms)
    // TODO: add softmax timing

    BenchOptions opt = opt_in;
    if (opt.num_threads <= 0)
        opt.num_threads = static_cast<int>(std::thread::hardware_concurrency());

    const size_t L = static_cast<size_t>(opt.layers);
    const auto& all_backends = get_backends();

    std::vector<const BackendVariant*> active;
    for (const auto& b : all_backends) {
        if (!b.run_once) continue;
        if (framework_matches_filter(b.framework, opt.backends_filter))
            active.push_back(&b);
    }

    if (active.empty()) {
        runner.log_performance("Error", "No backends with run_once matched filter");
        return false;
    }

    {
        std::ostringstream cfg;
        cfg << "layers=" << L
            << ", warmup=" << opt.warmup
            << ", iterations=" << opt.iterations
            << ", threads=" << opt.num_threads
            << ", group_size=" << kGroupSize
            << ", backends=";
        for (size_t i = 0; i < active.size(); ++i) {
            if (i > 0) cfg << ",";
            cfg << active[i]->name;
        }
        runner.log_performance("Stack Config", cfg.str());
    }

    std::mt19937 gen(270270u);

    for (size_t M : opt.batch_sizes) {
        runner.log_performance("",
            "─────────────────────────────────────────────────────────────────────────────────────────────────────────");

        for (size_t bi = 0; bi < active.size(); ++bi) {
            const auto* backend = active[bi];

            if (backend->max_M > 0 && M > backend->max_M) {
                runner.log_performance("SKIP " + std::string(backend->name) + " M=" + std::to_string(M),
                    "only M<=" + std::to_string(backend->max_M) + " supported");
                continue;
            }

            // Prepare weights: L layers x 7 projections
            struct WeightEntry {
                void* weights = nullptr;
                void* activations = nullptr;
                size_t proj_idx = 0;
            };
            std::vector<WeightEntry> entries(L * kProjectionSpecs.size());

            // Prepare activations per projection shape (keyed by K dimension)
            // We need one activation set per unique K value
            struct ActivationSet {
                CactusActivations act;
                size_t K;
            };
            std::vector<ActivationSet> act_sets;
            auto get_act = [&](size_t K) -> size_t {
                for (size_t i = 0; i < act_sets.size(); ++i)
                    if (act_sets[i].K == K) return i;
                std::mt19937 agen(static_cast<uint32_t>(42 + M + K));
                act_sets.push_back({prepare_cactus_activations(M, K, agen), K});
                return act_sets.size() - 1;
            };

            std::array<std::vector<float>, 7> layer0_fp32;

            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (size_t l = 0; l < L; ++l) {
                for (size_t p = 0; p < kProjectionSpecs.size(); ++p) {
                    const auto& spec = kProjectionSpecs[p];
                    std::vector<float> fp32_weights(spec.N * spec.K);
                    for (auto& v : fp32_weights) v = dist(gen);

                    if (l == 0) layer0_fp32[p] = fp32_weights;

                    size_t idx = l * kProjectionSpecs.size() + p;
                    entries[idx].proj_idx = p;
                    entries[idx].weights = backend->prepare_weights(
                        fp32_weights.data(), spec.N, spec.K);
                    if (backend->prepare_activations) {
                        size_t ai = get_act(spec.K);
                        entries[idx].activations = backend->prepare_activations(
                            act_sets[ai].act.fp32.data(), M, spec.K, entries[idx].weights);
                    }
                }
            }

            // Accuracy: run layer 0 projections once via bench_fn, compare to FP32 reference
            {
                float worst_nrmse = 0.0f;
                float worst_max_err = 0.0f;
                bool any_failed = false;
                BenchOptions acc_opt = opt;
                acc_opt.warmup = 1;
                acc_opt.iterations = 1;

                for (size_t p = 0; p < kProjectionSpecs.size(); ++p) {
                    const auto& spec = kProjectionSpecs[p];
                    size_t ai = get_act(spec.K);
                    const size_t out_count = M * spec.N;

                    std::vector<float> captured(out_count, 0.0f);
                    std::vector<float> backend_ref(out_count, 0.0f);
                    acc_opt.capture_output = captured.data();
                    acc_opt.capture_reference = backend_ref.data();

                    backend->bench_fn(M, entries[p].weights, entries[p].activations,
                                      act_sets[ai].act.int8.data(),
                                      act_sets[ai].act.scales.data(), acc_opt);

                    bool has_output = false;
                    for (size_t i = 0; i < out_count && !has_output; i++)
                        if (captured[i] != 0.0f) has_output = true;

                    if (has_output) {
                        bool has_backend_ref = false;
                        for (size_t i = 0; i < out_count && !has_backend_ref; i++)
                            if (backend_ref[i] != 0.0f) has_backend_ref = true;

                        const float* ref_ptr;
                        float tol;
                        std::vector<float> reference(out_count);
                        if (has_backend_ref) {
                            ref_ptr = backend_ref.data();
                            tol = 0.01f;
                        } else {
                            reference_matmul_fp32(act_sets[ai].act.fp32.data(),
                                layer0_fp32[p].data(), reference.data(), M, spec.K, spec.N);
                            ref_ptr = reference.data();
                            tol = (backend->category == QuantCategory::INT8) ? 0.05f : 0.20f;
                        }

                        auto acc = check_accuracy(ref_ptr, captured.data(), out_count, tol);
                        if (acc.nrmse > worst_nrmse) worst_nrmse = acc.nrmse;
                        if (acc.max_abs_error > worst_max_err) worst_max_err = acc.max_abs_error;
                        if (!acc.passed) any_failed = true;
                    }
                }

                std::ostringstream acc_line;
                acc_line << std::fixed
                         << (any_failed ? "FAIL" : "PASS")
                         << " nrmse=" << std::setprecision(4) << worst_nrmse
                         << " max=" << std::setprecision(2) << worst_max_err;
                runner.log_performance(
                    std::string("Accuracy ") + backend->name + " M=" + std::to_string(M),
                    acc_line.str());

                layer0_fp32 = {};
            }

            CactusThreading::set_gemm_threads(opt.num_threads);

            // Warmup
            for (int w = 0; w < opt.warmup; ++w) {
                for (size_t l = 0; l < L; ++l) {
                    for (size_t p = 0; p < kProjectionSpecs.size(); ++p) {
                        size_t idx = l * kProjectionSpecs.size() + p;
                        size_t ai = get_act(kProjectionSpecs[p].K);
                        backend->run_once(M, entries[idx].weights, entries[idx].activations,
                                          act_sets[ai].act.int8.data(),
                                          act_sets[ai].act.scales.data());
                    }
                }
            }

            // Timed iterations
            std::array<double, 7> per_proj_ms{};
            std::array<size_t, 7> per_proj_calls{};
            double step_total_ms = 0.0;

            for (int iter = 0; iter < opt.iterations; ++iter) {
                double step_t0 = now_ms();
                for (size_t l = 0; l < L; ++l) {
                    for (size_t p = 0; p < kProjectionSpecs.size(); ++p) {
                        size_t idx = l * kProjectionSpecs.size() + p;
                        size_t ai = get_act(kProjectionSpecs[p].K);

                        double t0 = now_ms();
                        backend->run_once(M, entries[idx].weights, entries[idx].activations,
                                          act_sets[ai].act.int8.data(),
                                          act_sets[ai].act.scales.data());
                        per_proj_ms[p] += now_ms() - t0;
                        per_proj_calls[p]++;
                    }
                }
                step_total_ms += now_ms() - step_t0;
            }

            CactusThreading::reset_gemm_threads();

            // Report per-projection stats
            for (size_t p = 0; p < kProjectionSpecs.size(); ++p) {
                const auto& spec = kProjectionSpecs[p];
                if (per_proj_calls[p] == 0) continue;

                double avg_us = (per_proj_ms[p] * 1000.0) / static_cast<double>(per_proj_calls[p]);
                double gops = compute_gops(M, spec.K, spec.N,
                    static_cast<int>(per_proj_calls[p]), per_proj_ms[p]);

                std::ostringstream details;
                details << std::fixed << std::setprecision(1)
                        << backend->name << "=" << avg_us << "us"
                        << "  gops=" << std::setprecision(2) << gops
                        << "  calls=" << per_proj_calls[p];

                std::string name = std::string(spec.name) + " M=" + std::to_string(M)
                    + " " + std::to_string(M) + "x" + std::to_string(spec.K)
                    + "x" + std::to_string(spec.N);
                runner.log_performance(name, details.str());
            }

            // Report per-step stats
            {
                double avg_step_ms = step_total_ms / static_cast<double>(opt.iterations);
                double rows_per_sec = (static_cast<double>(opt.iterations) * static_cast<double>(M))
                    / (step_total_ms / 1000.0);

                std::ostringstream details;
                details << std::fixed << std::setprecision(3)
                        << backend->name << " step=" << avg_step_ms << "ms"
                        << ", rows/s=" << std::setprecision(1) << rows_per_sec;
                runner.log_performance("TOTAL M=" + std::to_string(M), details.str());
            }

            // Cleanup
            for (auto& e : entries) {
                if (backend->cleanup)
                    backend->cleanup(e.weights, e.activations);
            }
        }
    }

    return true;
}

} // namespace bench
