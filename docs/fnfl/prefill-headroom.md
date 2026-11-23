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

## What is left, ranked by gain against effort

Estimates, not measured. Long-prompt rate today ~3,700 tok/s; the target is 4,000, and 5,000+ would need the first
three items.

| # | Idea | Expected gain | Effort |
|---|---|---|---|
| 1 | **HC prompt kernels.** `hct_up` (~19 TFLOP/s; also re-reads the F32 streams in its epilogue) and the memory-bound `hct_split` (~610 GB/s) | ~−250 ms a chunk, **+10–12%** from 2K tokens up | days; bitwise-equal is likely possible if only data movement changes |
| 2 | **Split HC across the cards** instead of replicating it, its moments exchanged by the copy engines | another ~+5–10% | 1–2 weeks; sums in another order, so the numerical contract, not bitwise |
| 3 | **Expert and dense GEMMs** (~37 and ~44 TFLOP/s): better Volta tiles, experts grouped over the whole chunk rather than each half, faster decode in the loader | +10–20% | weeks, uncertain; run Nsight Compute on `ep_gate_up`, `pt_kernel` and `hct_up` first (half a day) |
| 4 | **QSA card balance** (the indexer's card alternates; the other waits 3–8 ms a QSA layer) | +2–4% | days |
| 5 | **Prefix caching beyond the slot** ([ReVolta](https://github.com/Leonccaa/ReVolta), [QEngine](https://github.com/bpx03/QEngine), [sFNX](https://github.com/johnfink8/sfnx)'s message-boundary checkpoints, then host parking) | repeated or switched-back context: seconds → well under a second; nothing for fresh prompts | medium–large; ~20–25 GB of host RAM is free beside the locked PLE table |
| 6 | **Tiny prompts' fixed costs** (FP16 weight conversions every chunk, ~18 ms; both halves re-reading the experts) | 20–30 ms of a ~0.1 s TTFT | small–medium |

Ruled out or small:
- the copy-engine exchange and the power cap as the long-prompt limit;
- bigger chunks (others measured +2–5%);
- the ladder for Swift's large chunks;
- INT8-activation GEMMs (V100's tensor cores take FP16 only);
- a pipeline (layer-split) prompt mode on two cards.

Kernel sources from the survey matter only for the smaller families: FlashQLA's GDN and the V100 flash-attention ports
cover ~15% of a chunk between them.
