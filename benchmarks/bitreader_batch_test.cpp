// Compare word-refill loads against the immediately preceding production core.
// bench_bitreader_batch.py supplies before_arithmetic_core.h from a snapshot.
#include "arithmetic_core.h"
#define elic elic_before
#include "before_arithmetic_core.h"
#undef elic
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

static void verify_readers() {
    std::mt19937 random(20261020);
    elic::BitReader after;
    elic_before::BitReader before;
    for (size_t size = 0; size < 258; ++size) {
        std::string bytes(size, '\0');
        for (char& byte : bytes) byte = static_cast<char>(random());
        for (unsigned first = 0; first <= 32; ++first) {
            after.reset(&bytes);
            before.reset(&bytes);
            for (unsigned i = 0; i < 700; ++i) {
                unsigned width = i == 0 ? first : random() % 33;
                require(after.get_bits(width) == before.get_bits(width), "bit reader value differs");
                require(after.position == before.position, "bit reader logical position differs");
            }
        }
    }
    for (unsigned width = 0; width <= 32; ++width) {
        after.reset(nullptr);
        before.reset(nullptr);
        require(after.get_bits(width) == before.get_bits(width), "null stream padding differs");
        require(after.position == before.position, "null stream position differs");
    }
}

struct Sample {
    std::vector<uint16_t> cdf;
    std::vector<int16_t> symbols;
};

static Sample make_sample(size_t n, unsigned mode) {
    std::mt19937 random(20261020 + mode);
    Sample sample;
    sample.cdf.resize(n * 17);
    sample.symbols.resize(n);
    for (size_t i = 0; i < n; ++i) {
        auto* row = sample.cdf.data() + 17 * i;
        uint64_t weights[16], total = 0, cumulative = 0;
        row[0] = 0;
        for (unsigned j = 0; j < 16; ++j) {
            weights[j] = mode == 0 ? 1 : mode == 1 ? (j == i % 16 ? 65521 : 1) : 1 + random() % 10000;
            total += weights[j];
        }
        for (unsigned j = 1; j < 16; ++j) {
            cumulative += weights[j - 1];
            row[j] = static_cast<uint16_t>((cumulative * 65520) / total + j);
        }
        row[16] = 0;
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

static std::string encode(const Sample& sample) {
    elic_before::ArithmeticEncoder16 encoder;
    encoder.encode_chunk(sample.cdf.data(), sample.symbols.data(), sample.symbols.size());
    encoder.finalize();
    return encoder.output.out;
}

static void verify_decoder(const Sample& sample) {
    std::string bytes = encode(sample);
    elic_before::ArithmeticDecoder16 before;
    elic::ArithmeticDecoder16 after;
    std::mt19937 random(29);
    for (unsigned run = 0; run < 3; ++run) {
        // Also compare truncated streams and the original zero-padding behavior.
        before.reset(bytes);
        after.reset(bytes);
        size_t offset = 0;
        do {
            size_t count = std::min(sample.symbols.size() - offset, static_cast<size_t>(random() % 114));
            std::vector<int16_t> expected(count), actual(count);
            const auto* cdf = sample.cdf.empty() ? nullptr : sample.cdf.data() + offset * 17;
            before.decode_chunk(cdf, expected.data(), count);
            after.decode_chunk(cdf, actual.data(), count);
            require(actual == expected, "decoder symbols differ");
            if (!run) for (size_t i = 0; i < count; ++i)
                require(actual[i] == sample.symbols[offset + i], "round trip symbols differ");
            require(before.bytes_consumed() == after.bytes_consumed(), "decoder consumed bytes differ");
            require(before.bytes_remaining() == after.bytes_remaining(), "decoder remaining bytes differ");
            offset += count;
        } while (offset < sample.symbols.size());
        bytes.resize(bytes.size() / 2);
    }
}

static volatile uint64_t checksum = 0;

template<class Before, class After>
static void paired(const char* name, size_t symbols, size_t compressed_bytes, Before before, After after) {
    constexpr unsigned repeats = 15;
    std::vector<double> old_samples, new_samples;
    auto measure = [](auto& function) {
        const auto start = std::chrono::steady_clock::now();
        function();
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };
    for (unsigned i = 0; i < 3; ++i) { before(); after(); }
    for (unsigned i = 0; i < repeats; ++i) {
        if (i % 2) { new_samples.push_back(measure(after)); old_samples.push_back(measure(before)); }
        else { old_samples.push_back(measure(before)); new_samples.push_back(measure(after)); }
    }
    auto a = old_samples, b = new_samples;
    std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
    std::cout << "{\"case\":\"" << name << "\",\"symbols\":" << symbols
              << ",\"cdf_bytes\":" << symbols * 17 * sizeof(uint16_t)
              << ",\"symbol_bytes\":" << symbols * sizeof(int16_t)
              << ",\"compressed_bytes\":" << compressed_bytes
              << ",\"before_ms\":" << a[repeats / 2] << ",\"after_ms\":" << b[repeats / 2]
              << ",\"speedup\":" << a[repeats / 2] / b[repeats / 2];
    auto values = [](const char* label, const std::vector<double>& samples) {
        std::cout << ",\"" << label << "\":[";
        for (size_t i = 0; i < samples.size(); ++i) { if (i) std::cout << ','; std::cout << samples[i]; }
        std::cout << ']';
    };
    values("before_samples_ms", old_samples); values("after_samples_ms", new_samples);
    std::cout << '}';
}

static void benchmark_decoder(const Sample& sample, const char* name, bool comma) {
    const auto bytes = encode(sample);
    elic_before::ArithmeticDecoder16 before;
    elic::ArithmeticDecoder16 after;
    std::vector<int16_t> old_output(sample.symbols.size()), new_output(sample.symbols.size());
    if (comma) std::cout << ',';
    paired(name, sample.symbols.size(), bytes.size(), [&] {
        before.reset(bytes); before.decode_chunk(sample.cdf.data(), old_output.data(), sample.symbols.size());
        checksum += old_output[sample.symbols.size() / 2];
    }, [&] {
        after.reset(bytes); after.decode_chunk(sample.cdf.data(), new_output.data(), sample.symbols.size());
        checksum += new_output[sample.symbols.size() / 2];
    });
}

int main(int argc, char** argv) {
    try {
        verify_readers();
        unsigned cases = 0;
        for (unsigned mode = 0; mode < 4; ++mode) {
            for (size_t n : {size_t{0}, size_t{1}, size_t{2}, size_t{16}, size_t{257}, size_t{4096}}) {
                verify_decoder(make_sample(n, mode)); ++cases;
            }
        }
        std::cout << std::setprecision(9) << "{\"correctness\":\"passed\",\"reader_checks\":5959833,"
                  << "\"decoder_cases\":" << cases << ",\"timing\":\"alternating paired median; 3 warmups; 15 repeats\",\"rows\":[";
        if (argc > 1) {
            for (unsigned mode = 0; mode < 4; ++mode) {
                const auto sample = make_sample(1000000, mode);
                benchmark_decoder(sample, (std::vector<const char*>{"uniform", "concentrated", "random_weighted", "random_rare_symbols"})[mode], mode != 0);
            }
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
                verify_decoder(sample);
                std::string name = argv[i];
                name = name.substr(name.find_last_of("/\\") + 1);
                name = name.substr(0, name.find_last_of('.'));
                benchmark_decoder(sample, name.c_str(), true);
            }
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
