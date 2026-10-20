// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026, Electronics and Telecommunications Research Institute (ETRI)
// Adapted from benchmarks/arithmetic_ctypes.cpp. See the distributed LICENSE.
#include "arithmetic_core.h"
#include "reference_core.h"
#include <cstring>
#include <exception>

#if defined(_WIN32)
#define ELIC_EXPORT extern "C" __declspec(dllexport)
#else
#define ELIC_EXPORT extern "C" __attribute__((visibility("default")))
#endif

struct EncoderHandle {
    reference::ArithmeticEncoder16 reference;
    elic::ArithmeticEncoder16 candidate;
    std::string bytes, error;
    bool original;
    explicit EncoderHandle(bool use_reference) : original(use_reference) {}
};
struct DecoderHandle {
    reference::ArithmeticDecoder16 reference;
    elic::ArithmeticDecoder16 candidate;
    std::string error;
    bool original;
    explicit DecoderHandle(bool use_reference) : original(use_reference) {}
};

ELIC_EXPORT void* elic_encoder_create(int original) {
    try { return new EncoderHandle(original != 0); } catch (...) { return nullptr; }
}
ELIC_EXPORT void elic_encoder_destroy(void* handle) { delete static_cast<EncoderHandle*>(handle); }
ELIC_EXPORT void elic_encoder_reset(void* handle, int clear_output) {
    auto& e = *static_cast<EncoderHandle*>(handle);
    if (e.original) e.reference.reset(clear_output != 0);
    else e.candidate.reset(clear_output != 0);
}
ELIC_EXPORT int elic_encoder_chunk(void* handle, const uint16_t* cdf, const int16_t* symbols, int64_t n) {
    auto& e = *static_cast<EncoderHandle*>(handle);
    try {
        if (e.original) e.reference.encode_chunk(cdf, symbols, n);
        else e.candidate.encode_chunk(cdf, symbols, n);
        return 0;
    } catch (const std::exception& error) { e.error = error.what(); return -1; }
}
ELIC_EXPORT int64_t elic_encoder_finish_options(void* handle, int reset_after, int clear_output_after) {
    auto& e = *static_cast<EncoderHandle*>(handle);
    try {
        if (e.original) e.bytes = e.reference.finish(reset_after != 0, clear_output_after != 0);
        else {
            e.candidate.finalize(); e.bytes = e.candidate.output.out;
            if (reset_after) e.candidate.reset(clear_output_after != 0);
        }
        return static_cast<int64_t>(e.bytes.size());
    } catch (const std::exception& error) { e.error = error.what(); return -1; }
}
ELIC_EXPORT const char* elic_encoder_bytes(void* handle) {
    return static_cast<EncoderHandle*>(handle)->bytes.data();
}
ELIC_EXPORT const char* elic_encoder_error(void* handle) {
    return static_cast<EncoderHandle*>(handle)->error.c_str();
}
ELIC_EXPORT int64_t elic_encoder_buffer(void* handle) {
    auto& e = *static_cast<EncoderHandle*>(handle);
    e.bytes = e.original ? e.reference.out_.out : e.candidate.output.out;
    return static_cast<int64_t>(e.bytes.size());
}
ELIC_EXPORT int elic_encoder_bitcount(void* handle) {
    auto& e = *static_cast<EncoderHandle*>(handle);
    return e.original ? e.reference.buffered_bitcount() : static_cast<int>(e.candidate.output.count);
}

ELIC_EXPORT void* elic_decoder_create(int original) {
    try { return new DecoderHandle(original != 0); } catch (...) { return nullptr; }
}
ELIC_EXPORT void elic_decoder_destroy(void* handle) { delete static_cast<DecoderHandle*>(handle); }
ELIC_EXPORT int elic_decoder_reset(void* handle, const char* bytes, int64_t size) {
    auto& d = *static_cast<DecoderHandle*>(handle);
    try {
        const std::string stream(bytes, static_cast<size_t>(size));
        if (d.original) d.reference.reset(stream);
        else d.candidate.reset(stream);
        return 0;
    } catch (const std::exception& error) { d.error = error.what(); return -1; }
}
ELIC_EXPORT int elic_decoder_chunk(void* handle, const uint16_t* cdf, int16_t* symbols, int64_t n) {
    auto& d = *static_cast<DecoderHandle*>(handle);
    try {
        if (d.original) {
            auto decoded = d.reference.decode_chunk(cdf, n);
            if (n) std::memcpy(symbols, decoded.data(), static_cast<size_t>(n) * sizeof(int16_t));
        } else d.candidate.decode_chunk(cdf, symbols, n);
        return 0;
    } catch (const std::exception& error) { d.error = error.what(); return -1; }
}
ELIC_EXPORT const char* elic_decoder_error(void* handle) {
    return static_cast<DecoderHandle*>(handle)->error.c_str();
}
ELIC_EXPORT int64_t elic_decoder_consumed(void* handle) {
    auto& d = *static_cast<DecoderHandle*>(handle);
    return d.original ? d.reference.bytes_consumed() : d.candidate.bytes_consumed();
}
ELIC_EXPORT int64_t elic_decoder_remaining(void* handle) {
    auto& d = *static_cast<DecoderHandle*>(handle);
    return d.original ? d.reference.bytes_remaining() : d.candidate.bytes_remaining();
}
