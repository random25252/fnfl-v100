// draft_fmt: the format of the LM head rows the draft copy holds.
// q5_k  - default; IQ3_XXS and most GSQ-RCO quants
// q8_0  - UD-Q4_K_XL style output.weight
// q6_k  - IQ3_S / GSQ-RCO style output.weight (216 bytes per 256 rows -> see dense::row_bytes)
#pragma once
#include <cstdint>

namespace fn { namespace head {

enum class draft_fmt : int32_t { q5_k, q8_0, q6_k };

} }
