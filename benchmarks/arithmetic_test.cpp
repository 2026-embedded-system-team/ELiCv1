#include "arithmetic_core.h"
#include "reference_core.h"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

struct Sample {
    std::vector<uint16_t> cdf;
    std::vector<int16_t> symbols;
};

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static Sample make_sample(size_t n, unsigned seed, unsigned mode) {
    std::mt19937 random(seed);
    Sample sample;
    sample.cdf.resize(n * 17);
    sample.symbols.resize(n);
    for (size_t i = 0; i < n; ++i) {
        uint16_t* row = sample.cdf.data() + 17 * i;
        row[0] = 0;
        uint64_t weights[16], total = 0, cumulative = 0;
        for (unsigned j = 0; j < 16; ++j) {
            weights[j] = mode == 0 ? 1 : mode == 1 ? (j == i % 16 ? 65521 : 1) : 1 + random() % 10000;
            total += weights[j];
        }
        for (unsigned j = 1; j < 16; ++j) {
            cumulative += weights[j - 1];
            row[j] = static_cast<uint16_t>((cumulative * 65520) / total + j);
        }
        row[16] = 0;  // 65536 wraps in the public int16 CDF representation.
        if (mode == 1) sample.symbols[i] = static_cast<int16_t>(i % 16);
        else if (mode == 3) sample.symbols[i] = static_cast<int16_t>(random() % 16);
        else {
            unsigned target = random() & 65535U, symbol = 0;
            while (symbol < 15 && row[symbol + 1] <= target) ++symbol;
            sample.symbols[i] = static_cast<int16_t>(symbol);
        }
    }
    return sample;
}

static std::vector<size_t> chunks(size_t n, unsigned seed) {
    std::mt19937 random(seed);
    std::vector<size_t> result{0};
    size_t done = 0;
    while (done < n) {
        size_t take = std::min(n - done, static_cast<size_t>(1 + random() % 113));
        result.push_back(take);
        if (random() % 3 == 0) result.push_back(0);
        done += take;
    }
    result.push_back(0);
    return result;
}

static void verify(const Sample& sample, unsigned seed) {
    size_t n = sample.symbols.size();
    reference::ArithmeticEncoder16 old_encoder;
    elic::ArithmeticEncoder16 new_encoder;
    auto partitions = chunks(n, seed);
    size_t offset = 0;
    for (size_t count : partitions) {
        // data()+0 may be null for empty input, so no pointer arithmetic there.
        const auto* cdf = n ? sample.cdf.data() + offset * 17 : nullptr;
        const auto* symbols = n ? sample.symbols.data() + offset : nullptr;
        old_encoder.encode_chunk(cdf, symbols, static_cast<int64_t>(count));
        new_encoder.encode_chunk(cdf, symbols, static_cast<int64_t>(count));
        require(old_encoder.get_buffer_bytes() == new_encoder.output.out, "buffer bytes differ before finish");
        require(old_encoder.buffered_bitcount() == static_cast<int>(new_encoder.output.count), "buffered bit count differs");
        offset += count;
    }
    std::string baseline = old_encoder.finish();
    new_encoder.finalize();
    std::string optimized = new_encoder.output.out;
    require(baseline == optimized, "encoded bytes differ");

    reference::ArithmeticEncoder16 old_one;
    elic::ArithmeticEncoder16 new_one;
    old_one.encode_chunk(sample.cdf.data(), sample.symbols.data(), static_cast<int64_t>(n));
    new_one.encode_chunk(sample.cdf.data(), sample.symbols.data(), static_cast<int64_t>(n));
    new_one.finalize();
    require(old_one.finish() == baseline && new_one.output.out == baseline, "chunk partition changes stream");

    // Decode each version's stream with the other version and different chunks.
    reference::ArithmeticDecoder16 old_decoder;
    elic::ArithmeticDecoder16 new_decoder;
    old_decoder.reset(optimized);
    new_decoder.reset(baseline);
    offset = 0;
    for (size_t count : chunks(n, seed + 17)) {
        const auto* cdf = n ? sample.cdf.data() + offset * 17 : nullptr;
        auto decoded_old = old_decoder.decode_chunk(cdf, static_cast<int64_t>(count));
        std::vector<int16_t> decoded_new(count);
        new_decoder.decode_chunk(cdf, decoded_new.data(), static_cast<int64_t>(count));
        require(decoded_old == decoded_new, "old/new decoded symbols differ");
        for (size_t j = 0; j < count; ++j)
            require(decoded_new[j] == sample.symbols[offset + j], "decoded symbol differs from input");
        require(old_decoder.bytes_consumed() == new_decoder.bytes_consumed(), "input byte count differs");
        require(old_decoder.bytes_remaining() == new_decoder.bytes_remaining(), "remaining byte count differs");
        offset += count;
    }
    // Reuse the same objects: reset must remove all interval, bit and buffer state.
    old_encoder.reset();
    new_encoder.reset();
    old_encoder.encode_chunk(sample.cdf.data(), sample.symbols.data(), static_cast<int64_t>(n));
    new_encoder.encode_chunk(sample.cdf.data(), sample.symbols.data(), static_cast<int64_t>(n));
    new_encoder.finalize();
    require(old_encoder.finish() == baseline && new_encoder.output.out == baseline, "encoder reset failed");
    old_decoder.reset(baseline);
    new_decoder.reset(optimized);
    auto decoded_old = old_decoder.decode_chunk(sample.cdf.data(), static_cast<int64_t>(n));
    std::vector<int16_t> decoded_new(n);
    new_decoder.decode_chunk(sample.cdf.data(), decoded_new.data(), static_cast<int64_t>(n));
    require(decoded_old == sample.symbols && decoded_new == sample.symbols, "decoder reset failed");
}

static volatile uint64_t checksum = 0;
struct Timings {
    double before, after;
    std::vector<double> before_samples, after_samples;
};
template<class Before, class After> static Timings paired_time(Before before, After after, unsigned repeats = 9) {
    std::vector<double> old_times, new_times;
    before(); after();
    auto measure = [](auto& function) {
        auto start = std::chrono::steady_clock::now();
        function();
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    };
    for (unsigned i = 0; i < repeats; ++i) {
        if (i % 2 == 0) {
            old_times.push_back(measure(before)); new_times.push_back(measure(after));
        } else {
            new_times.push_back(measure(after)); old_times.push_back(measure(before));
        }
    }
    auto old_sorted = old_times, new_sorted = new_times;
    std::sort(old_sorted.begin(), old_sorted.end());
    std::sort(new_sorted.begin(), new_sorted.end());
    return {old_sorted[repeats / 2], new_sorted[repeats / 2], old_times, new_times};
}

static void sample_array(const char* name, const std::vector<double>& values) {
    std::cout << ",\"" << name << "\":[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << ']';
}

static void benchmark_sample(const Sample& sample, const char* name, bool comma) {
    const size_t n = sample.symbols.size();
    reference::ArithmeticEncoder16 old_encoder;
    elic::ArithmeticEncoder16 new_encoder;
    old_encoder.encode_chunk(sample.cdf.data(), sample.symbols.data(), n);
    std::string bytes = old_encoder.finish();
    std::vector<int16_t> output(n);
    reference::ArithmeticDecoder16 old_decoder;
    elic::ArithmeticDecoder16 new_decoder;
    auto encoding = paired_time([&] {
        old_encoder.reset(); old_encoder.encode_chunk(sample.cdf.data(), sample.symbols.data(), n);
        checksum += old_encoder.finish().size();
    }, [&] {
        new_encoder.reset(); new_encoder.encode_chunk(sample.cdf.data(), sample.symbols.data(), n);
        new_encoder.finalize();
        std::string copied_bytes = new_encoder.output.out;
        checksum += copied_bytes.size();
    });
    auto decoding = paired_time([&] {
        old_decoder.reset(bytes); auto decoded = old_decoder.decode_chunk(sample.cdf.data(), n);
        checksum += decoded[n / 2];
    }, [&] {
        new_decoder.reset(bytes);
        std::vector<int16_t> decoded(n);
        new_decoder.decode_chunk(sample.cdf.data(), decoded.data(), n);
        checksum += decoded[n / 2];
    });
    if (comma) std::cout << ',';
    std::cout << "{\"distribution\":\"" << name << "\",\"symbols\":" << n
              << ",\"repeats\":9,\"order\":\"alternating paired\",\"baseline_encode_seconds\":" << encoding.before
              << ",\"optimized_encode_seconds\":" << encoding.after
              << ",\"encode_speedup\":" << encoding.before / encoding.after
              << ",\"baseline_decode_seconds\":" << decoding.before
              << ",\"optimized_decode_seconds\":" << decoding.after
              << ",\"decode_speedup\":" << decoding.before / decoding.after;
    sample_array("baseline_encode_samples_seconds", encoding.before_samples);
    sample_array("optimized_encode_samples_seconds", encoding.after_samples);
    sample_array("baseline_decode_samples_seconds", decoding.before_samples);
    sample_array("optimized_decode_samples_seconds", decoding.after_samples);
    std::cout << '}';
}

int main(int argc, char** argv) {
    try {
        unsigned cases = 0;
        for (unsigned mode = 0; mode < 4; ++mode) {
            for (size_t n : {0U, 1U, 2U, 16U, 31U, 257U, 4096U}) {
                verify(make_sample(n, 11 + cases, mode), 29 + cases); ++cases;
            }
            for (unsigned seed = 0; seed < 100; ++seed) {
                verify(make_sample((seed * 113U) % 2049U, 1000 + seed, mode), 9000 + seed); ++cases;
            }
        }
        // Exact unit-frequency intervals, including values adjacent to 0/65536.
        for (unsigned dominant = 0; dominant < 16; ++dominant) {
            Sample edge;
            edge.cdf.resize(16 * 17);
            edge.symbols.resize(16);
            for (unsigned i = 0; i < 16; ++i) {
                edge.symbols[i] = static_cast<int16_t>(i);
                uint32_t cumulative = 0;
                for (unsigned j = 0; j < 17; ++j) {
                    edge.cdf[i * 17 + j] = static_cast<uint16_t>(cumulative);
                    if (j < 16) cumulative += j == dominant ? 65521U : 1U;
                }
            }
            verify(edge, 111 + dominant); ++cases;
        }
        // Repeated middle-half intervals accumulate thousands of E3 bits.
        for (unsigned final_symbol : {0U, 1U, 15U}) {
            Sample pending;
            pending.cdf.resize(4097 * 17);
            pending.symbols.assign(4097, 1);
            pending.symbols.back() = static_cast<int16_t>(final_symbol);
            for (unsigned i = 0; i < 4097; ++i) {
                auto* row = pending.cdf.data() + i * 17;
                row[0] = 0; row[1] = 16384; row[2] = 49152;
                for (unsigned j = 3; j < 16; ++j) row[j] = static_cast<uint16_t>(49150 + j);
                row[16] = 0;
            }
            verify(pending, 999 + final_symbol); ++cases;
        }
        std::cout << "{\"correctness\":\"passed\",\"cases\":" << cases
                  << ",\"checks\":[\"byte equality\",\"intermediate buffers\",\"all 16 symbols\","
                  << "\"empty and single-symbol streams\",\"random chunk boundaries\",\"bidirectional decode\","
                  << "\"encoder and decoder reset\",\"bytes consumed/remaining\","
                  << "\"unit-frequency boundary CDFs\",\"long E3 pending-bit runs\"],\"benchmarks\":[";
        if (argc > 1) {
            benchmark_sample(make_sample(500000, 20261015, 0), "uniform", false);
            benchmark_sample(make_sample(500000, 20261015, 1), "concentrated", true);
            benchmark_sample(make_sample(500000, 20261015, 2), "random_weighted", true);
            benchmark_sample(make_sample(500000, 20261015, 3), "random_rare_symbols", true);
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
                verify(sample, 20261015);
                std::string name = argv[i];
                name = name.substr(name.find_last_of("/\\") + 1);
                name = name.substr(0, name.find_last_of('.'));
                benchmark_sample(sample, name.c_str(), true);
            }
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
