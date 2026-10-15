// Optional C ABI for CPU tests with real hierarchy occupancies and supplied CDFs.
// Both paths use the same adapters/cores as arithmetic_test.cpp.
#include "arithmetic_core.h"
#include "reference_core.h"
#include <cstring>
#include <exception>
#include <memory>

#if defined(_WIN32)
#define ELIC_EXPORT extern "C" __declspec(dllexport)
#else
#define ELIC_EXPORT extern "C"
#endif

struct EncoderHandle {
    reference::ArithmeticEncoder16 original;
    elic::ArithmeticEncoder16 optimized;
    std::string bytes;
    std::string error;
    bool use_original;
    explicit EncoderHandle(bool reference) : use_original(reference) {}
};
struct DecoderHandle {
    reference::ArithmeticDecoder16 original;
    elic::ArithmeticDecoder16 optimized;
    std::string error;
    bool use_original;
    explicit DecoderHandle(bool reference) : use_original(reference) {}
};

ELIC_EXPORT void* elic_encoder_create(int original) {
    try { return new EncoderHandle(original != 0); } catch (...) { return nullptr; }
}
ELIC_EXPORT void elic_encoder_destroy(void* handle) { delete static_cast<EncoderHandle*>(handle); }
ELIC_EXPORT void elic_encoder_reset(void* handle, int clear_output) {
    auto& encoder = *static_cast<EncoderHandle*>(handle);
    if (encoder.use_original) encoder.original.reset(clear_output != 0);
    else encoder.optimized.reset(clear_output != 0);
}
ELIC_EXPORT int elic_encoder_chunk(void* handle, const uint16_t* cdf, const int16_t* symbols, int64_t n) {
    auto& encoder = *static_cast<EncoderHandle*>(handle);
    try {
        if (encoder.use_original) encoder.original.encode_chunk(cdf, symbols, n);
        else encoder.optimized.encode_chunk(cdf, symbols, n);
        return 0;
    } catch (const std::exception& error) { encoder.error = error.what(); return -1; }
}
ELIC_EXPORT int64_t elic_encoder_finish_options(void* handle, int reset_after, int clear_output_after) {
    auto& encoder = *static_cast<EncoderHandle*>(handle);
    try {
        if (encoder.use_original) encoder.bytes = encoder.original.finish(reset_after != 0, clear_output_after != 0);
        else {
            encoder.optimized.finalize(); encoder.bytes = encoder.optimized.output.out;
            if (reset_after) encoder.optimized.reset(clear_output_after != 0);
        }
        return static_cast<int64_t>(encoder.bytes.size());
    } catch (const std::exception& error) { encoder.error = error.what(); return -1; }
}
ELIC_EXPORT int64_t elic_encoder_finish(void* handle) { return elic_encoder_finish_options(handle, 1, 1); }
ELIC_EXPORT const char* elic_encoder_bytes(void* handle) {
    return static_cast<EncoderHandle*>(handle)->bytes.data();
}
ELIC_EXPORT const char* elic_encoder_error(void* handle) {
    return static_cast<EncoderHandle*>(handle)->error.c_str();
}

ELIC_EXPORT void* elic_decoder_create(int original) {
    try { return new DecoderHandle(original != 0); } catch (...) { return nullptr; }
}
ELIC_EXPORT void elic_decoder_destroy(void* handle) { delete static_cast<DecoderHandle*>(handle); }
ELIC_EXPORT int elic_decoder_reset(void* handle, const char* bytes, int64_t size) {
    auto& decoder = *static_cast<DecoderHandle*>(handle);
    try {
        std::string stream(bytes, static_cast<size_t>(size));
        if (decoder.use_original) decoder.original.reset(stream);
        else decoder.optimized.reset(stream);
        return 0;
    } catch (const std::exception& error) { decoder.error = error.what(); return -1; }
}
ELIC_EXPORT int elic_decoder_chunk(void* handle, const uint16_t* cdf, int16_t* symbols, int64_t n) {
    auto& decoder = *static_cast<DecoderHandle*>(handle);
    try {
        if (decoder.use_original) {
            auto decoded = decoder.original.decode_chunk(cdf, n);
            if (n) std::memcpy(symbols, decoded.data(), static_cast<size_t>(n) * sizeof(int16_t));
        } else decoder.optimized.decode_chunk(cdf, symbols, n);
        return 0;
    } catch (const std::exception& error) { decoder.error = error.what(); return -1; }
}
ELIC_EXPORT const char* elic_decoder_error(void* handle) {
    return static_cast<DecoderHandle*>(handle)->error.c_str();
}
