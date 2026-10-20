// Research prototype: the existing integer algorithms fused into CPU loops.
// No architecture-specific instruction set or floating-point approximation.
#include <cstddef>
#include <cstdint>

#if defined(_WIN32)
#define ELIC_EXPORT extern "C" __declspec(dllexport)
#else
#define ELIC_EXPORT extern "C"
#endif

static inline uint64_t spread21(uint64_t value) {
    value &= 0x1fffffULL;
    value = (value | (value << 32)) & 0x1f00000000ffffULL;
    value = (value | (value << 16)) & 0x1f0000ff0000ffULL;
    value = (value | (value << 8)) & 0x100f00f00f00f00fULL;
    value = (value | (value << 4)) & 0x10c30c30c30c30c3ULL;
    return (value | (value << 2)) & 0x1249249249249249ULL;
}

template<class T>
static void encode(const T* xyz, int64_t rows, int64_t row_stride,
                   int64_t column_stride, int64_t* code) {
    for (int64_t i = 0; i < rows; ++i) {
        const T* row = xyz + i * row_stride;
        code[i] = static_cast<int64_t>(spread21(static_cast<uint64_t>(row[0])) |
                     (spread21(static_cast<uint64_t>(row[column_stride])) << 1) |
                     (spread21(static_cast<uint64_t>(row[2 * column_stride])) << 2));
    }
}

template<class T>
static int64_t count_parents(const T* bxyz, const int64_t* codes, int64_t rows,
                             int64_t row_stride, int64_t code_stride) {
    int64_t count = 0;
    for (int64_t i = 0; i < rows; ++i) {
        if (!i || (codes[i * code_stride] >> 3) != (codes[(i - 1) * code_stride] >> 3) ||
            bxyz[i * row_stride] != bxyz[(i - 1) * row_stride]) ++count;
    }
    return count;
}

template<class T>
static void reduce_parents(const T* bxyz, const int64_t* codes, int64_t rows,
                           int64_t row_stride, int64_t column_stride, int64_t code_stride,
                           int32_t* parents, int64_t* parent_codes, int32_t* occupancy) {
    int64_t parent = -1;
    for (int64_t i = 0; i < rows; ++i) {
        const T* row = bxyz + i * row_stride;
        const int64_t pc = codes[i * code_stride] >> 3;
        const int32_t bit = int32_t{1} << (codes[i * code_stride] & 7);
        if (!i || pc != (codes[(i - 1) * code_stride] >> 3) || row[0] != bxyz[(i - 1) * row_stride]) {
            ++parent;
            parents[4 * parent] = static_cast<int32_t>(row[0]);
            for (int axis = 1; axis < 4; ++axis)
                parents[4 * parent + axis] = static_cast<int32_t>(row[axis * column_stride]) >> 1;
            parent_codes[parent] = pc;
            occupancy[parent] = bit;
        } else {
            // Unique sorted children guarantee distinct octants, so OR equals
            // the production scatter_add of powers of two without overflow.
            occupancy[parent] |= bit;
        }
    }
}

#define ELIC_GEOMETRY_API(T, suffix) \
ELIC_EXPORT void elic_morton_##suffix(const T* xyz, int64_t n, int64_t rs, int64_t cs, int64_t* out) { \
    encode(xyz, n, rs, cs, out); \
} \
ELIC_EXPORT int64_t elic_count_##suffix(const T* xyz, const int64_t* codes, int64_t n, int64_t rs, int64_t ks) { \
    return count_parents(xyz, codes, n, rs, ks); \
} \
ELIC_EXPORT void elic_down_##suffix(const T* xyz, const int64_t* codes, int64_t n, int64_t rs, int64_t cs, int64_t ks, \
                                   int32_t* parents, int64_t* parent_codes, int32_t* occupancy) { \
    reduce_parents(xyz, codes, n, rs, cs, ks, parents, parent_codes, occupancy); \
}

ELIC_GEOMETRY_API(int32_t, i32)
ELIC_GEOMETRY_API(int64_t, i64)
