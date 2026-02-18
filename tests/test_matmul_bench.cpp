#include "test_utils.h"
#include "bench/bench_common.h"
#include "bench/bench_driver.h"

#include <iostream>

int main(int argc, char** argv) {
    TestUtils::TestRunner runner("Matmul Benchmark Suite");

    bench::BenchOptions opt;
    std::string err;
    if (!bench::parse_bench_args(argc, argv, opt, err)) {
        std::cerr << "Error: " << err << "\n"
                  << "Usage: " << argv[0]
                  << " [--iterations N] [--warmup N] [--m N] [--threads N]"
                  << " [--backends fw1,fw2] [--layers N] [--mode comparison|stack|both]\n";
        return 1;
    }

    const auto& backends = bench::get_backends();
    runner.log_performance("Backends registered", std::to_string(backends.size()));
    for (const auto& b : backends)
        runner.log_performance("  Backend", std::string(b.name) + " (" + b.framework + ")");

    // Determine mode: default is stack, explicit --mode overrides
    bool do_comparison = false;
    bool do_stack = true;
    if (opt.mode == "comparison") {
        do_comparison = true;
        do_stack = false;
    } else if (opt.mode == "both") {
        do_comparison = true;
        do_stack = true;
    }

    if (do_stack && opt.layers <= 0)
        opt.layers = 18;

    if (do_comparison)
        runner.run_test("Matmul Comparison", bench::run_comparison(runner, opt));
    if (do_stack)
        runner.run_test("Matmul Stack (" + std::to_string(opt.layers) + " layers)",
                        bench::run_stack(runner, opt));

    runner.print_summary();
    return runner.all_passed() ? 0 : 1;
}
