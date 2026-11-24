# Prefill and time-to-first-token headroom on the development box (2026-11-23)

Measured on the development machine: two V100 32 GB (PG500-216) at PCIe 3.0 x16 on separate root ports, Swift IQ3_S
with the shared Q8_0 MTP draft, the recommended serving flags, `-c 147456`. Swift does not fit at 198,720 cells: the
HC and PLE replicas' allocation runs out of memory. [`decode-headroom.md`](decode-headroom.md) has the platform and its
peer-write collapse. This page records:
- where prompt time goes;
- the hardware floor;
- the fix made so far (`--fnfl-prompt-graphs off`);
- the ranked list of what is left.

## How it was measured

- **`docs/iq3s-tools/ttft_bench.py PORT LABEL [LENGTHS] [REPS]`:** cache-busted fresh prompts.
  - Each request is a token-exact prompt (`/completion` with a token array) cut from a different offset of this
    repo's text, so no checkpoint can be reused.
  - `n_predict` 1, streamed. TTFT is the client's time to the first token.
  - `bench.py`'s long prompts share their filler, so later requests partly hit the cache; don't use them for TTFT.
- **`docs/iq3s-tools/append_bench.py PORT LABEL [DEPTH] [LENGTHS] [REPS]`:** short additions to a long resident
  context, the agent-turn pattern. The server restores the context's checkpoint and reads only the addition.
- **Nsight Systems on the running server:** `nsys profile -t cuda,nvtx --cuda-graph-trace=node`, exported to SQLite.
  Kernels, copies and idle gaps were summed per request and per card.
- **Telemetry:** `docs/iq3s-tools/p2p/gpumon.sh` during every run.

## Where prompt time stands

Fresh prompts, before the fix (server-side and client times agree within 1–45 ms; the tokenizer and HTTP cost
nothing that matters):

| Tokens | 16 | 28 | 128 | 256 | 512 | 640 | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 | 32,768 | 65,536 | 131,072 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| TTFT, s | 0.096 | 0.125 | 0.23 | 0.40 | 0.67 | 0.30 | 0.43 | 0.69 | 1.21 | 2.26 | 4.38 | 8.64 | 17.4 | 35.3 |
| tok/s | 166 | 223 | 559 | 600 | 760 | 2,110 | 2,400 | 2,960 | 3,380 | 3,620 | 3,740 | 3,790 | 3,770 | 3,710 |

The dip from 640 down to 65 tokens was the graph route (below). From 8K up, the rate is flat at ~3,600–3,800 tok/s.
The first 30K-token read right after a server start took 15 s instead of ~8 s, because of the background PLE warm
read; benchmark after `PLE warm: ... resident 100%` appears in the log.

## Fixed: the graph route for small prompt pieces (`--fnfl-prompt-graphs off`)

**What it was.** Prompt pieces of at most 512 rows run as CUDA unit graphs (`prompt_graph_rows`, APF2). That covers:
- short fresh prompts;
- **short appends to a conversation**, such as a tool result or a short user turn;
- a prompt's leftover rows past a multiple of 8,192.

In the graphs' replays, both cards sat idle ~4.5 ms before each of the 96 sublayer units, which was 440 ms of a
674 ms request at 512 tokens.

**The cause.** The unit graphs are captured once and replayed with every node's parameters set again
(`fn/model/graph_rec.cpp`). On this box, a card-to-card copy node that is set again **to the parameters it already
holds** falls off the copy engine onto a path about 14 times slower:
- 2.6 MB: 0.31 ms → 4.4 ms, the SM peer-store collapse rate;
- the same with `cudaGraphExecMemcpyNodeSetParams` and with `cudaGraphExecUpdate`.

An update that *changes* the parameters (another size or offset, or changed and then changed back) stays fast, and
eager `cudaMemcpyPeerAsync` and freshly instantiated graphs run at ~10 GB/s. The reproducer is
`docs/iq3s-tools/p2p/graph_copy_update.cu.txt`.

**Two switches** (server flags, wired through `fn/fnfl.h` into `fn/glue`):
- **`--fnfl-graph-copy-skip on`** (environment for `fnfl-drive`: `FNFL_GRAPH_COPY_SKIP=1`).
  - `graph_rec` keeps each copy node's current parameters and skips an update that would not change them.
  - It is the same copy, so outputs are bitwise the same, and the 4.5 ms gaps disappear.
- **`--fnfl-prompt-graphs off`**: direct launches instead of graphs.
  - It sets at startup the switch `POST /props {"prompt_graphs": false}` already changes live.
  - **The better setting here.** Even with the skip, graphs ran 2–5% slower than direct launches on this box: each
    unit graph joins both cards before the next one starts, which removes the overlap direct launches keep.
  - It also avoids the ~190 ms `cudaGraphInstantiate` the first time each piece size is seen.

Both default to upstream's behaviour (graphs on, no skip).

**Results** (medians; the old row is the production build at its defaults):

| | Fresh 65 | Fresh 128 | Fresh 256 | Fresh 512 | Append 128 @30K | Append 256 @30K | Append 512 @30K | ≥640 tokens |
|---|---|---|---|---|---|---|---|---|
| Graphs on (old) | 0.193 s | 0.252 | 0.398 | 0.700 | 0.241 | 0.414 | 0.702 | — |
| Graphs + copy skip | 0.135 | 0.155 | 0.191 | 0.270 | 0.168 | 0.222 | 0.294 | unchanged |
| **Graphs off** | **0.130** | **0.150** | **0.185** | **0.265** | **0.161** | **0.203** | **0.284** | unchanged |

**Correctness:** greedy 64-token continuations of 10 prompts (fresh 100–600 tokens, appends of 50–512 after 20K) were
bitwise identical across four arms: the old production build (graphs, no skip), graphs with the skip, graphs off
through `/props`, and the startup flag.

## Done: the HC up product's epilogue (`--fnfl-hc-up-staged on`)

**The problem** (Nsight Compute, full set, on `fnfl-drive hcp-test 0 bench`):
- The original `hct_up` (a 128 × 128 CTA of four 64 × 64 warps, 242 registers, 8 warps an SM) stalled mostly on
  global loads (long scoreboard, 6.6 a issue).
- Its epilogue reads the F32 streams, `rinv`, gamma and the scales per lane, scattered: 19.9M L1 sectors for 3.4M
  requests, ~7× the streams' bytes, at a 27% L1 hit rate.

**The change** (`hct_upe_kernel` in `fn/kernels/hc/hc_prompt.cu`; switch `set_hc_up_staged`, server
`--fnfl-hc-up-staged on`, `fnfl-drive`'s environment `FNFL_HC_UP_STAGED=1`):
- The product runs on a 64 × 128 CTA of eight 32 × 32 warps.
- Its raw accumulators are staged in shared memory as `[row][stream][feature]`, reusing the pipeline's buffers after
  the k loop's last barrier. A per-row rotation of the features keeps the eight-row stores free of bank conflicts.
- The original epilogue's arithmetic then runs with a warp a row and a lane a feature, so the streams, gamma and both
  outputs move in contiguous 128-byte runs.
- **Bitwise the same:** every output's K order is the tile's own k loop on any tile shape, and the epilogue performs
  the same F32 operations in the same order.

**Variants tried** (the same bench, the HC sublayer at 4,096 rows: 2.52 ms originally):

| Variant | 4,096 rows | Note |
|---|---|---|
| 128 × 128, eight 64 × 32 warps | 2.16 ms | the best plain tile |
| 64 × 128, eight 32 × 32 warps | 2.32 | |
| two smaller variants | 2.69–2.77 | slower |
| **64 × 128 + staged coalesced epilogue (kept)** | **2.06** | 4.75M shared-store bank conflicts before the rotation |
| 128 × 128 + epilogue staged in two 64-row phases | 2.21 | the serialized phases cost more than the halved weight reads save |

**Results:**
- **`hcp-test`:** all 83 checks pass with the switch off and on, against the FP64 bounds, the range stress and the
  bitwise reruns.
- **`hcp-test bench`:** the HC sublayer at 512, 2,048 and 4,096 rows is 26%, 19–22% and 18% faster, with every
  output bitwise the original's. `hct_up` alone goes from 1.57 to 1.17 ms at 4,096 rows.
- **Server** (`ttft_bench.py`, medians of 3, the page cache warm):

| Prompt | Original | Staged | Change |
|---|---|---|---|
| 512 tokens | 0.270 s | 0.255 s | −5.6% |
| 2,048 | 0.704 s | 0.677 s | −3.8% |
| 8,192 | 2.273 s | 2.204 s | −3.0% (3,605 → 3,718 tok/s) |
| 32,768 | 8.626 s | 8.413 s | −2.5% (3,799 → 3,895 tok/s) |

- **Correctness:** greedy 64-token continuations of 12 prompts (fresh 100–9,000 tokens, appends after 20K) were
  bitwise identical with the switch off and on.
- **Telemetry:** GPU 0 averaged 57–61 °C core (peaks 63–70) and 59–63 °C HBM (peaks 66–72). It hit the 250 W cap
  (`0x4`) in short bursts, somewhat more often with the faster kernel. GPU 1 had no throttling. There was no thermal
  throttling.

**Why only ~3%:** `hct_up` was ~13% of a long chunk, so a 26% faster kernel is worth ~3.4% at most, and GPU 0 meets
its power cap a little more often. That is less than the 7–9% estimated from the first profile, which counted
`hct_down` and a larger cut in `hct_up`.

**HC's share now, per 4,096-row half** (bench): `hct_split` 0.32 ms (memory-bound), `hct_down` 0.59 ms, `hct_mid`
0.04 ms, `hct_up` 1.17 ms.
- `hct_up`'s floor is ~0.3–0.45 ms: 27 GFLOP and ~230 MB.
- It now waits on its operand loads more than its epilogue. Each 64-row tile re-reads the 6.5 MB of up weights, and
  occupancy is 24 warps an SM.

**Trap:** after a run with `--fnfl-ple-warm off`, or anything else that evicts the page cache, the PLE table's warm read
takes ~135 s from disk. Requests during it are slowed by page faults (a 2,048-token prompt took 7.2 s). Wait for
`PLE warm: ... resident 100%` in the log before benchmarking.

## Where a long chunk's time goes

One 8,192-row chunk (fresh, depth 0–8K) takes 2.10 s. **Both cards are kernel-busy 94–96% of it.** GPU 0's kernel time:

| Family | ms | Share | Notes |
|---|---|---|---|
| Dense products (`pt_kernel`, `pd_convert`, `pt_bf16`) | ~570 | 28% | ~44 TFLOP/s |
| Routed experts (`ep_gate_up` 303, `ep_down` 180, sum and group 42) | ~525 | 26% | ~37 TFLOP/s; ~80 rows an expert in each half |
| **HC** (`hct_up` 274, `hct_split` 136, `hct_down` 116, mid and convert 13) | **~540** | **27%** | **~19 TFLOP/s, and replicated: each card computes all 2,560 features** |
| QSA (`attend` 138, `select` 20, scoring 8) | ~166 | 8% | |
| GDN (`wy`, `scan`, `conv_norm`, `gate`) | ~144 | 7% | |
| PLE and the rest (`pb_kernel` 58, …) | ~100 | 5% | |

What the timeline rules out:
- **Exchanges.**
  - The copy-engine exchange moves ~4.25 GB a card a chunk at 6.2 GB/s, ~0.69 s of copying, almost all hidden behind
    the other half's compute.
  - The SM peer-store collapse doesn't touch long prompts; it only hit the graph route through the copy-node update
    above.
- **Idle time:** 63–102 ms a chunk (3–5%).
  - Most of it is one card waiting 3–8 ms at each QSA layer while the other runs that layer's indexer (it alternates
    between the cards).
  - About 30 ms is the draft catch-up's exchanges at the chunk's end.
- **Host gaps and the first PLE gather:** the first chunk's gather is ~7 ms here (with the table mlocked), not the
  128 ms seen upstream.
- **Depth:** a 131K prompt averages ~2.2 s a chunk against 2.1 s at the start, partly GPU 0's clock (below). QSA
  scoring of dead blocks (Strata-V100's finding) can be worth at most a few percent here.

### The kernels under Nsight Compute (2026-11-23)

One fresh 8,192-token prompt; GPU 0; the first launches of each kernel (layers 0–2, a 4,096-row half). Only metrics
that one pass collects (see the method note below):

| Kernel | Time | DRAM | Tensor pipe busy | Active warps | Registers | Reading |
|---|---|---|---|---|---|---|
| `hct_split` | 0.34 ms | **823 GB/s** | — | 54% | 56 | memory-bound near peak; only fusion (fewer bytes) helps |
| `hct_down` | 0.63 ms | 204 GB/s | 47% | 18% | 159 | moderate; occupancy limited by registers |
| **`hct_up`** | **1.49 ms** | 176 GB/s | **20%** | **12%** | **242** | **bound by neither: 8 warps an SM can't hide the epilogue's latency** |
| `pt_kernel` (dense) | 1.3–3.6 ms | ~280 GB/s | 53–58% | 12% | 246 | fairly efficient already |
| `ep_gate_up` | 3.35–3.39 ms | ~150 GB/s | 48% | 31% | 96 | like dense |
| `ep_down` | 1.76–2.07 ms | ~260 GB/s | 39–46% | 23% | 122–126 | like dense |
| `ep_sum` | 0.32 ms | 845 GB/s | — | 71% | 36 | memory-bound near peak |
| `ep_group` | 0.15 ms | — | — | — | 63 | **a grid of one block**: ~14 ms a chunk for a trivial job |

What it changes:
- **`hct_up` is the outlier.** Its product could run at least at `hct_down`'s rate. 1.43 ms → ~0.6–0.7 ms a launch
  saves ~140–160 ms a chunk, and `hct_down` maybe ~30 ms more: **~7–9% on long prompts**. That is less than the 10–12%
  estimated before profiling, because `hct_split` is already at the memory roof.
- **Dense and expert products have less headroom than their TFLOP/s suggested.** The tensor pipe is already 40–58%
  busy; reaching ~70% would take serious tile work for ~15–25% of their time.

**Method note (this box):**
- Nsight Compute needs root here (`RmProfilingAdminOnly: 1`).
- A multi-pass collection backs up the GPU memory a kernel may touch into host RAM. With the engine filling a card,
  that was ~30 GB, plus the 27 GB locked PLE table, and the kernel's OOM killer stopped the server.
- What works:
  - `--fnfl-ple-warm off`;
  - one-pass metrics only, with `--replay-mode application`, so a second pass could never back up memory;
  - `--devices 0`;
  - the server under `sh -c "timeout -s TERM 120 …; exit 0"`. A root server can't be stopped from the user account,
    and Nsight Compute drops the report when the program exits non-zero.

**Short prompts** are dominated by work done once a chunk whatever its size. At 28 tokens, ~85 ms of kernels:
- `pd_convert` 12 ms and `hct_convert` 5.5 ms, the dense and HC weights converted to FP16 for every chunk;
- small-row products over the FP16 weights, 22 ms;
- reading the touched experts, ~16 ms.

Each prompt half re-reads every touched expert, so experts dominate small chunks.

**The kernel-size ladder is not the answer here.** The table (`fn/model/ladder_table.inc`) is UD-Q4_K_XL's, so Swift
runs the retained routes. Even there, the ladder's tiles beat the retained route only at 1,024 rows or fewer (1.2–4.6×
on small dense products) and tie it at 4,096. For Swift's dense formats the ladder's tile loader uses a slow quarter
decode (`fn/kernels/dense/dense_ladder.h`).

## The floor

From the GGUF's shapes, a card does about **7.1 GFLOP a token** in prompts:

| Part | GFLOP a token, a card |
|---|---|
| Dense (attention, GDN and shared-expert projections, router), split | 3.0 |
| Routed experts (10 of 512), split | 2.4 |
| HC (4 × 157M weights), replicated | 1.26 |
| QSA and GDN | ~0.5 |

An 8,192-row chunk is ~58 TFLOP a card against a peak of ~102 TFLOP/s (72 SMs × 1,380 MHz × 1,024 FLOP a clock):
**0.57 s, against 2.1 s measured, ~27% of peak.** A realistic ceiling, with every GEMM at 55–65% of peak, is
~1.0–1.2 s a chunk, ~7,000–8,000 tok/s. Nothing is broken; the losses are spread across moderately efficient kernels.

## Power and thermals

During back-to-back 65K and 131K prompts:
- **GPU 0** reached 81 °C core and 85 °C HBM. It hit the 250 W cap (`0x4`) in ~40% of busy samples and flagged a
  thermal slowdown (`0x20`) in 3 samples at 1,282–1,320 MHz. It also drifted to 1,327–1,357 MHz with no reason
  flagged.
- **GPU 1** held 1,380 MHz with no throttling, peaking at 71 °C and 68 °C HBM.

The cards wait on each other, so this costs ~2–4% on long prompts. In shorter runs (up to 32K), GPU 0 peaked at
70–74 °C core and 72–77 °C HBM, with only short `0x4` bursts. **Neither the power cap nor the copy engine is what
limits long prompts; kernel efficiency is.**

## Scoping: splitting HC across the cards (2026-11-23)

**Today.** Each card computes every HC sublayer in full for every prompt row (the replicated form, FC-PF3). That is
`hct_split`, `hct_down`, `hct_mid` and `hct_up` over all 2,560 features: ~2.1 ms a 4,096-row half (bench; more in
production, where `hct_split` also folds the previous exchange in). That is ~400 ms of a ~2.2 s chunk on each card, and
half of it is the other card's work done twice.

**What the sublayer body needs.** The full mixed row (FP16 `[rows][2560]`, `a.mixed16`) on both cards: the QKV, GDN
and router products take K = 2,560. So a split must exchange mixed halves either way. The draft's catch-up already
does this (the split form: `pkind::mom`, then `pkind::dmix`, the mixed half), so the protocol has a template.

**Two stages:**

| | A: split the up product only | B: the full split (decode's form) |
|---|---|---|
| Work | each card computes `hct_up` for its 1,280 features; the halves cross as FP16 (2.5 KB a row, 10.5 MB a 4,096-row half), like DMIX | streams held as halves; `hct_split`, `hct_down` over 1,280 features; F32 moments cross (5.2 KB a row) before `hct_mid`; mixed halves cross; the sublayer outputs reduce-scatter (half their bytes) |
| Saves a chunk | ~100 ms (`hct_up` ~1.07 → ~0.53 ms × 192 half-sublayers): **~4.5%** | ~+125 ms more (down, split and combine halved): ~6% more |
| Copy-engine load | +2 GB a chunk a card: ~0.7 → ~1.0 s of a ~2.1 s chunk, still below the compute | ~8 GB a chunk, ~1.3 s at 6.2 GB/s: close to the compute time; the moments sit on a tight critical path (down → copy → mid → up) |
| Numerics | **bitwise**: each mixed value is computed as now on the card that owns it, and copied exactly | sums in card order (as decode does), not bitwise; checked against the numerical contract |
| Memory a card | ~+60–85 MB (a mixed buffer and a receive slot a piece); 0.47 GiB is free at 147,456 cells | similar |
| Effort | ~1–2 days | 1–2 weeks |

**The catch for A: two halves cannot hide both exchanges.**
- Per piece and sublayer the chain becomes HC → mixed copy → body → output copy → the next HC.
- With two halves interleaved, every copy needs other work between its producer and its consumer. Only two of the four
  gaps a sublayer can hold a body; the other two hold only an HC step.
- The output copy (21 MB a 4,096-row half, ~3.1 ms at the measured ~6.2 GB/s) then sits behind an HC step (~1.9 ms after
  the split). That stalls ~1.2 ms a sublayer, ~115 ms a chunk, about what A saves.
- **Three pieces a chunk** (MUP2's `prompt_set_pieces`) give every gap enough work: H_A H_B H_C B_A B_B B_C.
- But each piece re-reads the touched experts' weights (+~25 ms a chunk for a third piece), so **A nets ~+3%** for ~2–3
  days of reordering `chunk_layers` (`fn/model/prompt.cpp`).
- The graph route and pieces of ≤ `hct_small_rows` would stay replicated.

**Instead, first: `hct_up` toward its floor on one card (tried 2026-11-23, little left).**
- Nsight Compute on the staged kernel (2,048 rows): the epilogue holds ~54% of the stall samples and the k loop ~31%.
  The issue rate is 42%, with 0.77 eligible warps a scheduler.
- Unrolling the epilogue's row loop (2, 4 or 8 rows a warp in flight) changed nothing at 4,096 rows (2.065–2.077 ms
  against 2.065). The warps already hide each other's latency.
- The kernel is near an instruction floor: ~179M warp instructions at 4,096 rows is ≥0.45 ms of issue alone.
  - 45% are in the k loop: the 32 × 32 warp tiles' address, load and store work for each tensor op.
  - ~31% are in the epilogue: four precise `expf` sigmoids and IEEE divisions an output, which the bitwise contract
    fixes.
- Going below ~1.1 ms needs a better tensor-core product core (larger warp tiles with the staged epilogue in less
  shared memory, or fewer instructions a tensor op). That is the same work the dense and expert products need (item 3
  below), not a quick HC fix.

## The tensor-core core: how far from the ceiling (2026-11-23)

**The ceiling, measured.** cuBLAS (`cublasGemmEx`, FP16 in, FP32 accumulate, best of the tensor-op algorithms) on the
deployed shapes at a 4,096-row half (`docs/iq3s-tools/tc/cublas_ceil.cu.txt`; FNFL's times from the 8K-chunk trace):

| Product (N × K) | cuBLAS | FNFL | FNFL slower by |
|---|---|---|---|
| GDN QKV + gate (8192 × 2560) | 2.24 ms (77 TFLOP/s) | 3.29 ms (52) | 1.47× |
| QSA Q + K + V (6656 × 2560) | 1.82 ms (77) | 2.83 ms (49) | 1.55× |
| Outputs (2560 × 3072) | 0.87 ms (74) | — | — |
| Shared gate + up (640 × 2560) | 0.29 ms (47) | 0.41 ms | 1.45× |
| HC up, the product only (10240 × 320) | 0.49 ms (55) | 1.17 ms with its epilogue (staged) | — |
| HC down, 4 streams (320 × 2560) | 0.50 ms (54) | 0.60 ms | 1.21× |
| Experts gate + up, 80 rows × 512 experts | 6.85 ms (20) | **3.16 ms** | FNFL faster |
| Experts down, 80 rows × 512 experts | 2.24 ms (30) | **1.87 ms** | FNFL faster |

The experts are faster in FNFL: cuBLAS has to read FP16 weights (1.7 GB a call), and FNFL decodes the 2–4-bit codes
in its tile loader. **The gap is in the dense products.**

**Why the dense gap: the same tile, a worse loop.** cuBLAS runs CUTLASS's `s884gemm_f16_128x128_tn`. That is the same
configuration as FNFL's `pt_kernel`: 128 × 128, four 64 × 64 warps, ~240 registers, 32 KB shared. Nsight Compute on
the GDN shape (`docs/iq3s-tools/tc/harness.cu.txt`, the engine's core in a standalone harness):

| | FNFL, launch order | FNFL, grouped order | cuBLAS |
|---|---|---|---|
| Time | 3.83 ms | 2.77 ms | 2.34 ms |
| Tensor pipe busy | 50% | 69% | 81% |
| L2 hit rate | 58% | 89% | 89% |
| Long-scoreboard stall a issue | 1.11 | — | 0.20 |
| Warp instructions | 515M | — | 428M |

- **The L2 part is fixed: `--fnfl-dense-grouped on`.**
  - The CTAs take their tiles in groups of 8 row tiles (CUTLASS's threadblock swizzle), so a wave shares the weights'
    column slices instead of streaming all 42 MB again every two row tiles.
  - Only the tile-to-CTA assignment changes: bitwise the same.
  - In the harness: GDN 4.28 → 3.18 ms, QSA 3.23 → 2.61 ms (F32 epilogue), small N unchanged.
  - Code: `pt_kernel` in `fn/kernels/dense/dense_prompt.cu`, `pt_kargs::group`; the switch is `set_dense_grouped`,
    environment `FNFL_DENSE_GROUPED=1`.
- **What is left is the loop.** FNFL's k loop issues ~20% more instructions. Its hottest stalls are address arithmetic.
  Unrolling it by two spills (the 64 × 64 warp tile is at 246 registers) and is slower. Closing the rest to CUTLASS's
  81% means rewriting the k loop's scheduling within the same register budget: weeks, uncertain. The alternatives
  would be cuBLAS or CUTLASS as a dependency (sums in another order, so not bitwise).

**In the server** (`ttft_bench.py`, medians of 3, `--fnfl-hc-up-staged on` in both arms):

| Prompt | Launch order | Grouped |
|---|---|---|
| 2,048 tokens | 0.671 s | 0.664 s |
| 8,192 | 2.205 s | 2.180 s (−1.1%) |
| 32,768 | 8.427 s | 8.341 s (−1.0%; 3,888 → 3,928 tok/s) |

- Greedy outputs of 12 frozen prompts were bitwise identical across three binaries: the production build (`10d5590`),
  and the new build with the switch off and on. `pd-test` passes all 173 checks either way.
- The test prompts are now frozen in `gencmp-toks.json`. An earlier "difference" came from tokenizing docs that had
  been edited between runs.

**Why only ~1% in the server, and the ceiling behind it: GPU 0's power.** During busy prompt time (samples at ≥90%
utilization):
- **GPU 0** draws 223–225 W on average, against **GPU 1**'s 195 W for the same work. It is at the 250 W cap 32–38% of
  that time and below 1,380 MHz 82% of it, averaging 1,357 MHz. GPU 1 never leaves 1,380 MHz.
- 250 W is the card's hard maximum (`nvidia-smi -q -d POWER`: limits 100–250 W), so it can't be raised.
- The cards wait on each other, so GPU 0 sets the pace. Every efficiency gain raises its power and loses part of the
  gain to clock. The cuBLAS bench peaked at 274 W and dropped to 1,207 MHz.
- **So kernel speedups are partly capped by GPU 0's power.** Why GPU 0 draws ~30 W more is **not established**. Two
  explanations fit:
  - leakage rising with temperature (it runs 4–7 °C hotter): this would show as power capping (`0x4`), not as thermal
    throttling, which barely occurs;
  - plain chip-to-chip variation (V100 dies differ in leakage and voltage).

  GPU 0 also often sits at 1,327–1,357 MHz with no reason flagged, GPU Boost's temperature-dependent clock steps. The
  test to settle it is in the list below.
- If it is leakage, cooler air on GPU 0 is the one hardware knob left. The fans already follow its wattage through
  `/home/dom/scripts/gpu_watt_temp.sh` and CoolerControl, so a steeper curve may buy back clock under the cap. Measure
  GPU 0's power and clock at the same load before and after.

**Revised outlook for this direction:**
- The grouped order (done, +1%).
- A CUTLASS-class k loop for the dense products: perhaps +3–4% more at most once GPU 0's power is counted. A rewrite of
  weeks, or a library dependency.
- The HC up product on such a core: ~+2%.
- **That is not the +10–20% estimated before measuring.** It overstated the dense gap: cuBLAS is ~1.5× FNFL, not 2×.
  It assumed the experts had headroom, which they don't against cuBLAS. And it ignored the power cap.

## Prefix caching beyond the slot: the sizing (2026-11-23, not built)

For everyday use this is probably the biggest TTFT lever left. Every fresh-prompt item above is worth a few percent;
a cache hit turns seconds into a fraction of one.

**Today.** One slot keeps its conversation and up to eight checkpoints: the recurrent state at multiples of 8,192 and
at each prompt's end. A follow-up turn in the same conversation already reads only its new tokens. What is not kept:
- a conversation the server has switched away from;
- a long context sent again after something else ran in between;
- an edited system prompt or early message, read again from the last checkpoint before the change, or from token 0.

At ~3,800 tok/s that costs ~8.6 s for 32K tokens and ~35 s for 131K.

**What a parked conversation weighs** (`engine.md` "Memory"):
- **Keys and values:** about 10 KB a context token a card with the default 8-bit KV. 32K tokens are ~0.65 GB over both
  cards, 131K ~2.6 GB.
- **Recurrent state** (what a checkpoint copies): the 36 GDN layers' states (24 heads a card × 128 × 128 F32, ~56 MB a
  card), the convolution and PLE histories and the draft's carry. Roughly 0.1–0.15 GB a conversation, whatever its
  length.

**Restoring vs recomputing:**
- From pinned host RAM at the measured ~10–11 GB/s a card, both cards in parallel, a 32K conversation comes back in
  ~0.04–0.1 s, against ~8.6 s to recompute. A 131K one takes ~0.15–0.3 s against ~35 s.
- Pageable memory is slower but still far under a second.

**Budget:** 62 GB of RAM, of which the PLE table locks 27 GB. ~20–25 GB is left for a host archive: about 25–30
conversations of 32K, or 6–8 of 131K. NVMe could be a second tier (ReVolta: a 32K document back from disk in 1.5 s).

**Ways in, cheapest first:**
1. **sFNX-style checkpoints at message boundaries** (and every 4,096 tokens of a long first message) instead of only
   at multiples of 8,192. An edited late message is then re-read from the nearest boundary. Small: the checkpoint
   machinery exists (`fn/runtime/prefill.h`); only where the checkpoints are taken changes. It helps edits within one
   conversation, not switching.
2. **Park a slot's KV and state in host RAM** when another conversation takes the slot, keyed on the token-prefix hash
   the checkpoints already use (`token_key`). Restore on a match. Medium–large:
   - the KV pool and slot ledger have to accept a restored prefix;
   - the copies have to be ordered with the copy streams;
   - eviction is least recently used first.
3. **A second tier on NVMe** for restarts and overflow. Large.

**Before building:**
- Log real usage to see how often requests switch conversations or re-send long prefixes. The server log's
  `FNFL prefill: from token … (N kept by the shell …)` lines show what was reused.
- Measure the actual KV and state bytes a conversation holds in this build.

### Measured (2026-11-24): what the slot already keeps, and message-boundary checkpoints

- **Switching conversations is the expensive case.** With `-np 1`, a 1,500-token side request between two turns of a
  30K conversation makes the next turn re-read everything: 7.7 s instead of ~0.2 s. Two slots (`-np 2`) don't fit with
  Swift (out of memory at 147K, 114K and 98K cells), so they are no workaround here. Host parking (way 2 above) is the
  fix; the engine already has `target::save_ready` and `restore_ready` (all non-per-cell state to host and back), so
  what is missing is the per-cell data (KV, index keys and summaries, the draft's KV and index) and the checkpoint book.
- **Message-boundary checkpoints (way 1) don't pay.** Built and measured on a 34,882-token chat of 22 messages: an
  edit inside the last message re-read 1,736 tokens instead of 2,120 (0.67 s against 0.80 s), but the extra chunk ends
  cost every fresh read ~0.9 s (11.3 s against 10.4 s). The prompt-end checkpoints and those at multiples of 8,192
  already bound a divergence to about one turn. Not kept. Cuts that weren't on a 64-row GDN block made reads ~1.6–1.9×
  slower: any future change that ends chunks elsewhere must keep them 64-aligned.
- **Decision (2026-11-24): host parking is skipped for now.** It only pays when a client interleaves requests (side
  requests for titles, summaries or subagents between turns, or switching chats), and there is no record yet of how
  often that happens in real use. One conversation at a time already reads only each turn's new tokens. To revisit:
  keep the server log through a normal day of use and count the `FNFL prefill: from token 0 at 0` lines that follow a
  long conversation (a full re-read after a switch). Frequent ones make parking worth its 1–2 weeks; the pieces it
  needs are listed in the first bullet above.

## A lossy lead: RidgeFill (2026-11-24, not built)

[RidgeFill](https://arxiv.org/abs/2611.09709) (preprint, Nov 2026) runs the early layers exactly for every prompt token
and, for older prompt rows, predicts the late layers' inputs from the split layer's state with a per-layer linear map
fitted once in closed form (no training). Those layers then only write the row's K/V or GDN state, skipping its MLP
and experts; the last ~2K tokens and some selected rows run exactly, and decode runs the full model. On
Qwen3.8-Flash-Next (vLLM, two R9700, split at layer 24): prefill 1.50–1.60× at 9K–32K, 1.74× at 64K; needle retrieval
99.8 against 100, perplexity on new tokens +2.65% (its interval includes zero). On a smaller dense model some
extraction tasks lost 10–14 points. For FNFL it would break the reference contract, so only as an opt-in mode; it is
still the largest prompt lever found, and needs no kernel rewrite (a calibration tool, a skip path in the prompt
layers, quality checks): roughly 1–3 weeks.

**Speed measured (2026-11-24): ~1.3× on this engine.** The authors' release
([ridgefill-data](https://github.com/Approximate-Science/ridgefill-data), Apache-2.0) has calibration code and
ready-made Flash-Next maps on Hugging Face (24 maps for layers 24–47, each 2,560 × 10,240 from the four streams entering
layer 24 to the layer's input, int8 with scales: 0.60 GiB, so ~0.3 GiB a card split, inside the ~0.47 GiB free at
147,456 cells); its vLLM engine is not public. A skip path in FNFL's prompt pass (`docs/iq3s-tools/rf/skip_path.patch`,
env `FNFL_RF_SPLIT=24`; garbage outputs, timing only) leaves out, in layers 24–47, both HC mixes, the GDN and QSA
output products, QSA attention and the whole FFN, and keeps the input products, the GDN state updates, the K/V and
index-key writes and every exchange's bookkeeping. `ttft_bench.py`, cache-busted, two reps:

| Prompt | Exact | Skip path, every row filled | + maps and the exact 2K tail (estimate) |
|---|---|---|---|
| 32K | 8.36 s (3,925 tok/s) | 6.12 s (5,366), 1.37× | ~6.4 s, **~1.30×** |
| 64K | 16.75 s (3,916) | 12.16 s (5,398), 1.38× | ~12.7 s, **~1.32×** |

- The maps' GEMM (4,096 × 10,240 × 1,280 a card a half, `docs/iq3s-tools/rf/mapgemm.cu.txt`) takes 1.35 ms with
  cuBLAS (79 TFLOP/s): ~65 ms a chunk, ~100 ms at FNFL's own kernels' rate.
- The quality mode (3–10% of rows exact) would land around ~1.2–1.25×.
- Telemetry: GPU 0 averaged 61 °C core (peak 70), HBM peak 72 °C, 218 W with the 250 W cap in 75 of 192 samples; GPU 1
  peak 61 °C; no thermal throttling.
- Lower than the paper's 1.5–1.6×: their engine streams two-thirds of the experts from host memory, so the skipped
  FFN is a bigger share of their prefill.
- Not done: quality on Swift (their maps were fitted on an MXFP4/FP8 build), which decides whether ~1.3× is worth a
  lossy mode. **Not pursued for now (2026-11-24):** ~1.3× for a lossy opt-in mode was judged not worth the quality
  work yet. To revisit: load the published maps, compare against exact prefill on retrieval tests and real agent
  prompts, refit on Swift if they transfer poorly, then turn the patch into a real path.

### The dense products' ceiling, measured in the real prefill (2026-11-24)

An experimental build (`docs/iq3s-tools/tc/cublas_standin.patch`, env `FNFL_TC_CUBLAS=1`; timing only, outputs and
error words ignored) replaced every prompt dense product except the router (kept so the routing stays real) with a
cuBLAS FP16 GEMM of the same rows × n × k on a non-zero dummy weight: the time a CUTLASS-class kernel with in-loader
decode could take, with the real power draw. `ttft_bench.py`, cache-busted, two reps:

| Prompt | FNFL today | Dense at cuBLAS speed | Gain |
|---|---|---|---|
| 32K | 8.32 s | 7.75 s (one outlier rep at 9.09) | ~1.07× |
| 64K | 16.74 s | 15.71 s | ~1.07× |

With all-zero dummy weights (less tensor-core power) it was ~1.10×; the non-zero run's GPU 0 touched the 250 W cap in
3 of 102 samples (61 °C average, 68 peak; HBM 68 peak; GPU 1 peak 59 °C; no thermal throttling). So the dense
products, a quarter of a chunk, are worth ~+7% at cuBLAS's speed, not more.

**The whole ceiling, with the rest bounded** (per 8,192-row chunk, ~2.09 s today):
- dense at cuBLAS speed: −~140 ms (measured above, power included);
- HC split across the cards: −~150–250 ms (it is computed twice today; the extra exchange must stay hidden);
- experts grouped over the whole chunk: −~200 ms (an estimate; FNFL already beats cuBLAS per launch);
- attention and GDN at flash-attention-class speed: −~120 ms;
- together ~1.4–1.45 s, **~1.45×** if every piece landed, ~1.3× realistic under GPU 0's cap. Each piece is its own
  project of one to three weeks, worth ~7–12% alone; the "shared tensor-core core" by itself is the ~7%.

## The realistic maximum with code changes, and the path (2026-11-24, estimates)

Per 8,192-token chunk (today ~2.08 s, ~3,930 tok/s with all the switches):

| Ceiling | Chunk time | tok/s | Assumes |
|---|---|---|---|
| FP16 peak (~58 TFLOP a card at ~102 TFLOP/s) | 0.57 s | ~14,000 | every unit busy all the time: not reachable |
| A well-built engine on these cards | ~1.0–1.2 s | ~7,000–8,000 | GEMMs at cuBLAS-class efficiency, HC split, the memory-bound kernels at bandwidth |
| The same under GPU 0's 250 W cap | ~1.3–1.5 s | **~5,500–6,500** | efficient kernels draw more power; part of each gain turns into clock |

Where the gap is, and what closes it, largest first:

| Area | Now | Could be | Approach | Effort |
|---|---|---|---|---|
| HC | ~400 ms | ~150 ms | split across the cards (three pieces), its four kernels fused into two, both products on a better core | 2–3 weeks |
| Experts | ~525 ms | ~320 ms | rows grouped by expert over the whole chunk rather than each half (twice the rows an expert, the weights read once); a better tile loop that keeps the in-loader decode | 2–3 weeks |
| Dense | ~520 ms | ~350 ms | a CUTLASS-class k loop (~80% tensor-busy against ~55–69%), or CUTLASS or cuBLAS as a dependency (not bitwise) | 2–4 weeks, shared with the two above |
| QSA attention | ~166 ms | ~100 ms | a flash-attention-style kernel (the V100 ports in the survey) | 1–2 weeks |
| GDN | ~144 ms | ~90 ms | the FlashQLA chunked kernels | 1–2 weeks |
| Idle and small kernels | ~150 ms | ~50 ms | QSA indexer balance, PLE products on tensor cores, `ep_group` as a real grid | days each |

The largest single lever is **one shared high-efficiency tensor-core core with pluggable weight loaders** (FP16, BF16
and the quantized formats) **and fused epilogues**, used by dense, HC and the experts alike. It is most of the gain and
the least certain part: the cheap attempt (unrolling the current loop) ran out of registers. Altogether this is
~1.5× today, for weeks to months of kernel work. Prefix caching (above) makes a re-read of cached context near-instant
for a fraction of that.

## What is left, ranked by gain against effort

Estimates, not measured. Long-prompt rate today ~3,720–3,900 tok/s with `--fnfl-hc-up-staged on`; the target is 4,000,
and 5,000+ would need the first three items.

| # | Idea | Expected gain | Effort |
|---|---|---|---|
| 1 | **HC prompt kernels, the rest.** `hct_up`'s staged epilogue is done (+3–4%, above); unrolling its epilogue gained nothing, and it is near an instruction floor without a new product core (see the scoping above). Left: `hct_down` (47% tensor-busy, register-limited) and fusing `hct_split` (memory-bound at ~820 GB/s) into its neighbours to cut bytes | maybe +1–3% | days each, uncertain |
| 1b | **Split HC's up product across the cards, in three pieces a chunk** (scoping above; two halves cannot hide the extra exchange) | ~+3% | ~2–3 days; bitwise |
| 1c | **`ep_group` as one block** (a grid of one CTA, ~14 ms a chunk) | ~+0.5% | small |
| 2 | **Split HC across the cards** instead of replicating it, its moments exchanged by the copy engines | another ~+5–10% | 1–2 weeks; sums in another order, so the numerical contract, not bitwise |
| 3 | **A CUTLASS-class k loop for the dense products** (measured: cuBLAS 1.5× FNFL on the big shapes; the L2 part done, `--fnfl-dense-grouped on`, +1%). The experts already beat cuBLAS; grouping them over the whole chunk (rows an expert ×2) is a separate idea | +3–5% at most, capped by GPU 0's power (section above) | weeks, or cuBLAS/CUTLASS as a dependency (not bitwise) |
| 4 | **QSA card balance** (the indexer's card alternates; the other waits 3–8 ms a QSA layer) | +2–4% | days |
| 5 | **Prefix caching beyond the slot** (*skipped for now, 2026-11-24: [measured below the sizing](#measured-2026-11-24-what-the-slot-already-keeps-and-message-boundary-checkpoints)*) (sizing in its own section above; [ReVolta](https://github.com/Leonccaa/ReVolta), [QEngine](https://github.com/bpx03/QEngine), [sFNX](https://github.com/johnfink8/sfnx)'s message-boundary checkpoints, then host parking) | repeated or switched-back context: seconds → well under a second; nothing for fresh prompts | medium–large; ~20–25 GB of host RAM is free beside the locked PLE table |
| 6 | **Tiny prompts' fixed costs** (FP16 weight conversions every chunk, ~18 ms; both halves re-reading the experts) | 20–30 ms of a ~0.1 s TTFT | small–medium |

**To eliminate soon: why GPU 0 draws ~30 W more than GPU 1.** Run the same fixed GEMM load on each card cold (~35 °C),
then warmed (~70 °C), and log power and clock (`gpumon.sh`; `docs/iq3s-tools/tc/cublas_ceil.cu.txt` makes a steady
load).
- GPU 0's power rising clearly with temperature at the same clock means leakage, and cooling GPU 0 would buy clock
  under the cap.
- GPU 0 drawing more even when both are cold means it's the chip, and cooling helps little.

~10 minutes.

Ruled out or small:
- the copy-engine exchange and the power cap as the long-prompt limit;
- bigger chunks (others measured +2–5%);
- the ladder for Swift's large chunks;
- INT8-activation GEMMs (V100's tensor cores take FP16 only);
- a pipeline (layer-split) prompt mode on two cards.

Kernel sources from the survey matter only for the smaller families: FlashQLA's GDN and the V100 flash-attention ports
cover ~15% of a chunk between them.
