"""Create the isolated candidate from a current-core snapshot; production is untouched."""
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / "before_arithmetic_core.h"
if not BASE.exists():
    BASE.write_bytes((ROOT / "csrc/arithmetic_core.h").read_bytes())
source = BASE.read_text(encoding="utf-8")
start = source.index("    inline void append_bits(uint32_t bits, unsigned width) {")
end = source.index("    inline void append_repeated", start)
replacement = '''    inline void append_bits(uint32_t bits, unsigned width) {
        // The public buffer still contains every complete byte after each call.
        // At most 7 cached + 32 incoming bits fit into this local register.
        const uint64_t combined = (static_cast<uint64_t>(cache) << width) |
                                  (static_cast<uint64_t>(bits) & low_mask(width));
        const unsigned total = count + width;
        const unsigned bytes = total / 8;
        const unsigned remainder = total % 8;
        char complete[4];
        for (unsigned j = 0; j < bytes; ++j)
            complete[j] = static_cast<char>(combined >> (total - (j + 1) * 8));
        if (bytes) out.append(complete, bytes);
        count = remainder;
        cache = static_cast<uint8_t>(combined & low_mask(remainder));
    }
'''
(HERE / "arithmetic_core.h").write_text(source[:start] + replacement + source[end:], encoding="utf-8")
