#pragma once

// Integer-only core shared by the extension and standalone compatibility tests.
// The 16-way arithmetic-coding bitstream is unchanged.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#if defined(_MSC_VER)
#include <intrin.h>
#endif

namespace elic {
inline unsigned leading_zeros(uint32_t v) {
    if (!v) return 32;
#if defined(_MSC_VER)
    unsigned long index;
    _BitScanReverse(&index, v);
    return 31U - static_cast<unsigned>(index);
#else
    return static_cast<unsigned>(__builtin_clz(v));
#endif
}
inline uint32_t low_mask(unsigned n) {
    return n == 32 ? 0xffffffffU : ((uint32_t{1} << n) - 1);
}

class BitWriter {
public:
    std::string out;
    uint8_t cache{0};
    unsigned count{0};
    // Append the width least significant bits, most significant bit first.
    inline void append_bits(uint32_t bits, unsigned width) {
        while (width) {
            unsigned take = std::min(8U - count, width);
            width -= take;
            cache = static_cast<uint8_t>((cache << take) | ((bits >> width) & low_mask(take)));
            count += take;
            if (count == 8) {
                out.push_back(static_cast<char>(cache));
                cache = 0;
                count = 0;
            }
        }
    }
    inline void append_repeated(unsigned bit, uint64_t width) {
        if (count && width) {
            unsigned take = static_cast<unsigned>(std::min<uint64_t>(8U - count, width));
            append_bits(bit ? low_mask(take) : 0U, take);
            width -= take;
        }
        if (width >= 8) {
            out.append(static_cast<size_t>(width / 8), bit ? '\xff' : '\0');
            width %= 8;
        }
        if (width) append_bits(bit ? low_mask(static_cast<unsigned>(width)) : 0U,
                               static_cast<unsigned>(width));
    }
    inline void append_prefix(uint32_t low, unsigned width, uint64_t& pending) {
        if (!pending) append_bits(low >> (32 - width), width);
        else {
            unsigned first = low >> 31;
            append_bits(first, 1);
            append_repeated(first ^ 1U, pending);
            pending = 0;
            if (width > 1) append_bits(low >> (32 - width), width - 1);
        }
    }
    inline void finish(unsigned bit, uint64_t pending) {
        append_bits(bit, 1);
        append_repeated(bit ^ 1U, pending);
        if (count) append_bits(0, 8 - count);
    }
    void clear() { out.clear(); cache = 0; count = 0; }
};

class BitReader {
public:
    size_t position{0};
    void reset(const std::string* bytes) {
        bytes_ = bytes; position = 0; cursor_ = 0; cached_bits_ = 0; cache_ = 0;
    }
    // Original format pads the stream with zeros after the last byte.
    inline uint32_t get_bits(unsigned width) {
        if (cached_bits_ < width) {
            const size_t remaining = bytes_ ? bytes_->size() - cursor_ : 0;
            if (remaining >= sizeof(uint32_t)) {
                // memcpy permits unaligned addresses and reads exactly four
                // adjacent bytes. Interpret their stream order explicitly.
                uint32_t word;
                std::memcpy(&word, bytes_->data() + cursor_, sizeof(word));
#if defined(_MSC_VER)
                word = _byteswap_ulong(word);
#elif defined(__BYTE_ORDER__) && __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
                word = __builtin_bswap32(word);
#elif !defined(__BYTE_ORDER__) || __BYTE_ORDER__ != __ORDER_BIG_ENDIAN__
                const auto* p = reinterpret_cast<const unsigned char*>(bytes_->data() + cursor_);
                word = (uint32_t{p[0]} << 24) | (uint32_t{p[1]} << 16) |
                       (uint32_t{p[2]} << 8) | uint32_t{p[3]};
#endif
                cache_ = (cache_ << 32) | word;
                cached_bits_ += 32;
                cursor_ += sizeof(word);
            } else {
                // At most three bytes remain; never read past the string.
                for (size_t i = 0; i < remaining; ++i) {
                    cache_ = (cache_ << 8) | static_cast<uint8_t>((*bytes_)[cursor_++]);
                    cached_bits_ += 8;
                }
            }
        }
        uint32_t result;
        if (cached_bits_ < width) {
            result = static_cast<uint32_t>(cache_ << (width - cached_bits_)) & low_mask(width);
            cache_ = 0;
            cached_bits_ = 0;
        } else {
            cached_bits_ -= width;
            result = static_cast<uint32_t>(cache_ >> cached_bits_) & low_mask(width);
        }
        // Count bytes logically consumed, excluding complete prefetched bytes.
        position = cursor_ - cached_bits_ / 8;
        return result;
    }
private:
    const std::string* bytes_{nullptr};
    size_t cursor_{0};
    uint64_t cache_{0};
    unsigned cached_bits_{0};
};

class ArithmeticEncoder16 {
public:
    BitWriter output;
    void reset(bool clear_output = true) {
        low_ = 0; high_ = 0xffffffffU; pending_ = 0;
        if (clear_output) output.clear();
    }
    void encode_chunk(const uint16_t* cdf, const int16_t* symbols, int64_t size) {
        uint32_t low = low_, high = high_;
        uint64_t pending = pending_;
        output.out.reserve(output.out.size() + static_cast<size_t>(size / 2) + 16);
        for (int64_t i = 0; i < size; ++i, cdf += 17) {
            unsigned s = static_cast<unsigned>(symbols[i]);
            if (s > 15) throw std::out_of_range("sym out of range; expected 0..15");
            uint64_t span = static_cast<uint64_t>(high) - low + 1;
            uint32_t c_low = cdf[s], c_high = s == 15 ? 65536U : cdf[s + 1];
            high = low - 1 + static_cast<uint32_t>((span * c_high) >> 16);
            low += static_cast<uint32_t>((span * c_low) >> 16);
            // Batch all E1/E2 steps using the common high-order interval bits.
            if (((low ^ high) & 0x80000000U) == 0) {
                unsigned common = leading_zeros(low ^ high);
                output.append_prefix(low, common, pending);
                low = static_cast<uint32_t>(static_cast<uint64_t>(low) << common);
                high = static_cast<uint32_t>(static_cast<uint64_t>(high) << common) | low_mask(common);
            }
            // Repeated E3 steps require low bits 1 and high bits 0 below bit 31.
            if (low >= 0x40000000U && high < 0xc0000000U) {
                unsigned underflow = std::min(31U, leading_zeros((~low | high) << 1));
                pending += underflow;
                low = (low << underflow) & 0x7fffffffU;
                high = (high << underflow) | 0x80000000U | low_mask(underflow);
            }
        }
        low_ = low; high_ = high; pending_ = pending;
    }
    void finalize() { output.finish(low_ < 0x40000000U ? 0U : 1U, pending_ + 1); }
private:
    uint32_t low_{0}, high_{0xffffffffU};
    uint64_t pending_{0};
};

class ArithmeticDecoder16 {
public:
    void reset(const std::string& bytes) {
        storage_ = bytes; input_.reset(&storage_);
        low_ = 0; high_ = 0xffffffffU; value_ = 0; initialized_ = false;
    }
    void decode_chunk(const uint16_t* cdf, int16_t* symbols, int64_t size) {
        uint32_t low = low_, high = high_;
        uint32_t value = initialized_ ? value_ : input_.get_bits(32);
        initialized_ = true;
        for (int64_t i = 0; i < size; ++i, cdf += 17) {
            uint64_t span = static_cast<uint64_t>(high) - low + 1;
            uint64_t target = ((static_cast<uint64_t>(value) - low + 1) << 16) - 1;
            unsigned s;
            if (cdf[8] < 256 || cdf[8] > 65280) {
                // Extremely skewed CDFs have highly predictable search branches.
                // Retain the original division/search there instead of forcing
                // four dependent multiplications for almost-certain symbols.
                uint16_t count = static_cast<uint16_t>(target / span);
                unsigned left = 0, right = 16;
                while (left + 1 < right) {
                    unsigned middle = (left + right) / 2;
                    if (cdf[middle] < count) left = middle;
                    else if (cdf[middle] > count) right = middle;
                    else { left = middle; break; }
                }
                s = left;
            } else {
                // cdf[s] <= floor(target/span) iff cdf[s]*span <= target.
                // Four fixed comparisons avoid a per-symbol 64-bit division.
                s = span * cdf[8] <= target ? 8U : 0U;
                s += span * cdf[s + 4] <= target ? 4U : 0U;
                s += span * cdf[s + 2] <= target ? 2U : 0U;
                s += span * cdf[s + 1] <= target ? 1U : 0U;
            }
            symbols[i] = static_cast<int16_t>(s);
            uint32_t c_low = cdf[s], c_high = s == 15 ? 65536U : cdf[s + 1];
            high = low - 1 + static_cast<uint32_t>((span * c_high) >> 16);
            low += static_cast<uint32_t>((span * c_low) >> 16);
            if (((low ^ high) & 0x80000000U) == 0) {
                unsigned common = leading_zeros(low ^ high);
                low = static_cast<uint32_t>(static_cast<uint64_t>(low) << common);
                high = static_cast<uint32_t>(static_cast<uint64_t>(high) << common) | low_mask(common);
                value = static_cast<uint32_t>(static_cast<uint64_t>(value) << common) | input_.get_bits(common);
            }
            if (low >= 0x40000000U && high < 0xc0000000U) {
                unsigned underflow = std::min(31U, leading_zeros((~low | high) << 1));
                low = (low << underflow) & 0x7fffffffU;
                high = (high << underflow) | 0x80000000U | low_mask(underflow);
                value = ((value << underflow) ^ 0x80000000U) | input_.get_bits(underflow);
            }
        }
        low_ = low; high_ = high; value_ = value;
    }
    int64_t bytes_consumed() const { return static_cast<int64_t>(input_.position); }
    int64_t bytes_remaining() const {
        return static_cast<int64_t>(storage_.size() - std::min(storage_.size(), input_.position));
    }
private:
    uint32_t low_{0}, high_{0xffffffffU}, value_{0};
    std::string storage_;
    BitReader input_;
    bool initialized_{false};
};
}  // namespace elic
