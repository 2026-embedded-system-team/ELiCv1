#include <cassert>
#include <cstdint>
#include <string>
#include <vector>
namespace reference { using cdf_t = uint16_t;
static constexpr int kLp=17, kMaxSym=15, kPrecision=16;
static constexpr uint32_t kFull=65536U;
class OutCacheString {
public:
    std::string out{}; // 완성된 바이트들(8비트가 꽉 찬 것들) 누적
    uint8_t cache{0};  // 아직 out에 넣지 못한 "미완성 바이트"를 비트 단위로 쌓는 버퍼
    uint8_t count{0};  // cache에 현재 몇 비트가 들어있는지(0~7)

    // 비트 1개를 출력 스트림에 추가
    inline void append(int bit) {
        cache <<= 1; // 기존 비트들을 왼쪽으로 밀고
        cache |= static_cast<uint8_t>(bit & 1); // LSB에 새 비트를 넣음
        count += 1;
        if (count == 8) { // count == 8이면 cache를 out에 1바이트로 append하고 cache/count 초기화
            out.append(reinterpret_cast<const char*>(&cache), 1);
            count = 0;
            cache = 0;
        }
    }

    // 산술코딩 renormalization에서 쓰는 "출력 비트 + 보류(pending) 비트 처리"
    inline void append_bit_and_pending(int bit, uint64_t& pending_bits) {
        append(bit); // 결정된 비트 1개를 출력
        while (pending_bits > 0) { // 0이 될 때까지 append(!bit)를 반복, 산술코딩의 E3 조건(underflow) 때문에 "나중에 반전되어 나가야 하는 비트들"을 여기서 한꺼번에 처리
            append(!bit);
            pending_bits--;
        }
    }

    // 마지막에 바이트 경계로 정렬(byte align)
    inline void flush_to_byte_boundary() {
        if (count > 0) {
            for (int i = count; i < 8; ++i) append(0); // count > 0이면, 남은 비트 수(1~7)를 채우기 위해 append(0)을 반복해서 count==0이 되도록 만듦
            assert(count == 0);
        }
    }

    // 출력 버퍼 초기화
    inline void clear() {
        out.clear();
        cache = 0;
        count = 0;
    }
};

class InCacheString {
private:
    const std::string* in_{nullptr}; // 입력 바이트 스트림 포인터

public:
    uint8_t cache{0};       // 최근에 읽어온 1바이트
    uint8_t cached_bits{0}; // cache에서 아직 소비하지 않은 비트 수(0~8)
    size_t in_ptr{0};       // in_에서 다음에 읽을 바이트 인덱스

    InCacheString() = default;
    explicit InCacheString(const std::string* in) : in_(in) {}

    // 새로운 입력 스트림으로 초기화
    inline void reset(const std::string* in) {
        in_ = in;
        cache = 0;
        cached_bits = 0;
        in_ptr = 0;
    }

    // 입력에서 비트 1개를 읽어서 value에 shift-in
    inline void get(uint32_t& value) {
        if (cached_bits == 0) {
            if (!in_ || in_ptr == in_->size()) {
                value <<= 1; // pad with 0 if stream exhausted
                return;
            }
            // 아직 읽을 바이트가 있으면 cache = in_[in_ptr++], cached_bits=8
            cache = static_cast<uint8_t>((*in_)[in_ptr]);
            in_ptr++;
            cached_bits = 8;
        }
        value <<= 1;
        value |= (cache >> (cached_bits - 1)) & 1; // cache의 남은 비트 중 "가장 왼쪽(MSB 쪽)"" 비트를 읽어 value의 LSB로 넣음
        cached_bits--;
    }

    // 디코더 시작 시 value 레지스터에 초기 32비트 채우기
    inline void initialize(uint32_t& value) {
        for (int i = 0; i < 32; ++i) get(value);
    }
};

// ------------------------- Core helpers -------------------------

static inline cdf_t binsearch16(const cdf_t* cdf, cdf_t target, int64_t offset /* i*17 */) {
    cdf_t left = 0;
    cdf_t right = 16; // max_sym+1, cdf has length 17

    while (left + 1 < right) {
        const cdf_t m = static_cast<cdf_t>((left + right) / 2);
        const cdf_t v = cdf[offset + m];
        if (v < target) left = m;
        else if (v > target) right = m;
        else return m;
    }
    return left;
}


// ------------------------- Streaming Encoder -------------------------

class ArithmeticEncoder16 {
public:
    uint32_t low_{0};
    uint32_t high_{0xFFFFFFFFU};
    uint64_t pending_bits_{0};
    OutCacheString out_;

    ArithmeticEncoder16() = default;

    void reset(bool reset_output=true) {
        low_ = 0;
        high_ = 0xFFFFFFFFU;
        pending_bits_ = 0;
        if (reset_output) out_.clear();
    }

    // Appends bits for this chunk; does NOT finalize/flush.
    void encode_chunk(const uint16_t* cdf, const int16_t* sym, int64_t N) {
        
        
        

        
        const auto* cdf_ptr = cdf;
        auto sym_acc = sym;

        for (int64_t i = 0; i < N; ++i) {
            const int16_t s = sym_acc[i];
            

            const uint64_t span = static_cast<uint64_t>(high_) - static_cast<uint64_t>(low_) + 1;
            const int64_t offset = i * kLp;

            const uint32_t c_low  = static_cast<uint32_t>(cdf_ptr[offset + s]);
            const uint32_t c_high = (s == kMaxSym) ? kFull : static_cast<uint32_t>(cdf_ptr[offset + s + 1]);

            high_ = (low_ - 1) + ((span * static_cast<uint64_t>(c_high)) >> kPrecision);
            low_  = (low_)     + ((span * static_cast<uint64_t>(c_low )) >> kPrecision);

            while (true) {
                if (high_ < 0x80000000U) {
                    out_.append_bit_and_pending(0, pending_bits_);
                    low_ <<= 1;
                    high_ = (high_ << 1) | 1U;
                } else if (low_ >= 0x80000000U) {
                    out_.append_bit_and_pending(1, pending_bits_);
                    low_ <<= 1;
                    high_ = (high_ << 1) | 1U;
                } else if (low_ >= 0x40000000U && high_ < 0xC0000000U) {
                    pending_bits_++;
                    low_  = (low_ << 1) & 0x7FFFFFFFU;
                    high_ = (high_ << 1) | 0x80000001U;
                } else {
                    break;
                }
            }
        }
    }

    // Finalize + byte-align and return bytes.
    // By default resets state and clears output after returning.
    std::string finish(bool reset_after=true, bool clear_output_after=true) {
        uint64_t pending = pending_bits_ + 1;
        if (low_ < 0x40000000U) out_.append_bit_and_pending(0, pending);
        else                    out_.append_bit_and_pending(1, pending);

        out_.flush_to_byte_boundary();

        std::string res(out_.out);

        if (reset_after) reset(clear_output_after);
        return res;
    }

    // Optional: returns completed bytes so far (NOT decodable alone unless stream finalized).
    std::string get_buffer_bytes() const { return std::string(out_.out); }

    // Optional: bits buffered in last (incomplete) byte (0..7)
    int buffered_bitcount() const { return static_cast<int>(out_.count); }
};

// ------------------------- Streaming Decoder -------------------------

class ArithmeticDecoder16 {
public:
    uint32_t low_{0};
    uint32_t high_{0xFFFFFFFFU};
    uint32_t value_{0};

    std::string in_storage_{};
    InCacheString in_cache_{};

    bool initialized_{false};

    ArithmeticDecoder16() = default;

    // Load a new stream and reset decode state.
    void reset(const std::string& bitstream) {
        in_storage_ = std::string(bitstream); // copies bytes
        in_cache_.reset(&in_storage_);

        low_ = 0;
        high_ = 0xFFFFFFFFU;
        value_ = 0;
        initialized_ = false;
    }

    // Decode one chunk of length N = cdf.size(0), consuming bits continuously.
    std::vector<int16_t> decode_chunk(const uint16_t* cdf, int64_t N) {
        
        

        
        const auto* cdf_ptr = cdf;

        std::vector<int16_t> out(static_cast<size_t>(N));
        auto out_acc = out.data();

        if (!initialized_) {
            in_cache_.initialize(value_);
            initialized_ = true;
        }

        for (int64_t i = 0; i < N; ++i) {
            const uint64_t span = static_cast<uint64_t>(high_) - static_cast<uint64_t>(low_) + 1;
            const uint16_t count =
                static_cast<uint16_t>(((static_cast<uint64_t>(value_) - static_cast<uint64_t>(low_) + 1) * kFull - 1) / span);

            const int64_t offset = i * kLp;
            const cdf_t s = binsearch16(cdf_ptr, static_cast<cdf_t>(count), offset);

            out_acc[i] = static_cast<int16_t>(s);

            const uint32_t c_low  = static_cast<uint32_t>(cdf_ptr[offset + s]);
            const uint32_t c_high = (s == kMaxSym) ? kFull : static_cast<uint32_t>(cdf_ptr[offset + s + 1]);

            high_ = (low_ - 1) + ((span * static_cast<uint64_t>(c_high)) >> kPrecision);
            low_  = (low_)     + ((span * static_cast<uint64_t>(c_low )) >> kPrecision);

            while (true) {
                if (low_ >= 0x80000000U || high_ < 0x80000000U) {
                    low_ <<= 1;
                    high_ = (high_ << 1) | 1U;
                    in_cache_.get(value_);
                } else if (low_ >= 0x40000000U && high_ < 0xC0000000U) {
                    low_  = (low_ << 1) & 0x7FFFFFFFU;
                    high_ = (high_ << 1) | 0x80000001U;
                    value_ -= 0x40000000U;
                    in_cache_.get(value_);
                } else {
                    break;
                }
            }
        }

        return out;
    }

    // Optional: how many input bytes have been consumed
    int64_t bytes_consumed() const { return static_cast<int64_t>(in_cache_.in_ptr); }

    // Optional: remaining bytes in buffer
    int64_t bytes_remaining() const {
        // in_cache_는 항상 in_storage_를 가리키도록 reset()에서 설정됨
        const size_t consumed = in_cache_.in_ptr;
        const size_t total = in_storage_.size();
        if (consumed >= total) return 0;
        return static_cast<int64_t>(total - consumed);
    }
};


}
