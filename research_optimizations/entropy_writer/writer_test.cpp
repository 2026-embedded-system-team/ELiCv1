// Existing 447-case frozen-reference suite supplies sample generation helpers.
#define main frozen_compatibility_main
#include "../../benchmarks/arithmetic_test.cpp"
#undef main
#define elic elic_before
#include "before_arithmetic_core.h"
#undef elic
#include <iomanip>

static void verify_writers() {
    std::mt19937 random(20261020);
    elic::BitWriter after;
    elic_before::BitWriter before;
    for (unsigned run = 0; run < 100; ++run) {
        after.clear(); before.clear();
        for (unsigned step = 0; step < 1000; ++step) {
            if (step % 17 == 0) {
                const unsigned bit = random() & 1U, width = random() % 4097;
                before.append_repeated(bit, width); after.append_repeated(bit, width);
            } else {
                const unsigned bits = random(), width = random() % 33;
                before.append_bits(bits, width); after.append_bits(bits, width);
            }
            require(before.out == after.out, "intermediate writer buffer differs");
            require(before.count == after.count, "writer buffered bit count differs");
            require(before.cache == after.cache, "writer cached bits differ");
        }
        const unsigned bit = random() & 1U, pending = random() % 4097;
        before.finish(bit, pending); after.finish(bit, pending);
        require(before.out == after.out, "finished writer buffer differs");
        require(before.count == after.count && before.cache == after.cache, "finished writer state differs");
    }
}

static void benchmark_writer(const Sample& sample, const char* name, bool comma) {
    const size_t n = sample.symbols.size();
    elic_before::ArithmeticEncoder16 before;
    elic::ArithmeticEncoder16 after;
    before.encode_chunk(sample.cdf.data(), sample.symbols.data(), n); before.finalize();
    after.encode_chunk(sample.cdf.data(), sample.symbols.data(), n); after.finalize();
    require(before.output.out == after.output.out, "timing sample encoded bytes differ");
    const size_t compressed = before.output.out.size();
    auto run_before = [&] {
        before.reset(); before.encode_chunk(sample.cdf.data(), sample.symbols.data(), n); before.finalize();
        checksum += before.output.out.size();
    };
    auto run_after = [&] {
        after.reset(); after.encode_chunk(sample.cdf.data(), sample.symbols.data(), n); after.finalize();
        checksum += after.output.out.size();
    };
    // paired_time adds another warmup pair, for three total.
    for (unsigned warmup = 0; warmup < 2; ++warmup) { run_before(); run_after(); }
    const auto timing = paired_time(run_before, run_after, 15);
    if (comma) std::cout << ',';
    std::cout << "{\"case\":\"" << name << "\",\"symbols\":" << n
              << ",\"compressed_bytes\":" << compressed
              << ",\"before_ms\":" << timing.before * 1000
              << ",\"after_ms\":" << timing.after * 1000
              << ",\"speedup\":" << timing.before / timing.after;
    sample_array("before_samples_seconds", timing.before_samples);
    sample_array("after_samples_seconds", timing.after_samples);
    std::cout << '}';
}

int main(int argc, char** argv) {
    try {
        verify_writers();
        if (argc == 1) {
            std::cout << "{\"correctness\":\"passed\",\"writer_sequences\":100,\"writer_append_checks\":100000}\n";
            return 0;
        }
        std::cout << std::setprecision(10) << "{\"correctness\":\"passed\",\"writer_append_checks\":100000,\"rows\":[";
        for (unsigned mode = 0; mode < 4; ++mode)
            benchmark_writer(make_sample(1000000, 20261020, mode),
                (std::vector<const char*>{"uniform", "concentrated", "random_weighted", "random_rare_symbols"})[mode], mode != 0);
        for (int i = 2; i < argc; ++i) {
            std::ifstream input(argv[i], std::ios::binary);
            uint64_t n = 0;
            input.read(reinterpret_cast<char*>(&n), sizeof(n));
            require(n > 0 && n < 100000000, "invalid geometry benchmark size");
            Sample sample;
            sample.cdf.resize(static_cast<size_t>(n) * 17);
            sample.symbols.resize(static_cast<size_t>(n));
            input.read(reinterpret_cast<char*>(sample.cdf.data()), static_cast<std::streamsize>(n * 34));
            input.read(reinterpret_cast<char*>(sample.symbols.data()), static_cast<std::streamsize>(n * 2));
            require(input.good(), "truncated geometry benchmark data");
            std::string name = argv[i];
            name = name.substr(name.find_last_of("/\\") + 1);
            name = name.substr(0, name.find_last_of('.'));
            benchmark_writer(sample, name.c_str(), true);
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
