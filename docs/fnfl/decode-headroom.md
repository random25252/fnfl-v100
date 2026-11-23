# Decode headroom on the development box: the card-to-card link, what was fixed, what is left

Measured 2026-11-23 on the development machine: i9-7900X on an EVGA X299 FTW K, two V100 32 GB (PG500-216) at PCIe 3.0
x16, on separate CPU root ports (`nvidia-smi topo`: NODE), Swift IQ3_S with the shared Q8_0 MTP draft. This page is the
record of the decode-speed investigation: the cause, the two fixes, where a step's time goes now, and the ideas left for
squeezing more decode later, ranked.

## The cause: peer writes collapse on this platform

- Kernel stores into the other card's memory run at full speed for bursts up to about 64 KB in flight, then collapse
  to 0.6–1 GB/s. The cost tracks how *fragmented* the writes are more than the byte count: many small pieces from many
  blocks are the worst case.
- Peer loads (pull) stay at about 9 GB/s, kernel stores to pinned host memory about 11 GB/s, and `cudaMemcpyPeer`
  about 10 GB/s. Only SM-issued peer stores collapse.
- FNFL's exchanges (`fn/xchg`) are all push-style SM stores, so a multi-token verify step paid heavily. The HC sublayer
  kernel at 5 rows took 23 µs with its sends redirected to the card's own memory, against 160 µs with real exchanges.
- Platform changes had no effect: driver P2P mode (`RMForceP2PType=1`), C-states and governor, uncore clock pinned at
  2.4 GHz. Moving both cards to one root port isn't possible here because of cooling.

## What was done: two server switches (2026-11-23)

| Server flag (fnfl-drive: environment) | What it does | Code |
|---|---|---|
| `--fnfl-hmix-route host` (`FNFL_HMIX_HOST=1`) | X-HMIX (the mixed row, 6,720 B a row) goes through pinned host memory. The sender writes the receiver's host slot; a small landing kernel on the receiver polls each 16-byte group once, copies it into the device slot the consumers already poll, and resets the host group. | `fn/xchg/xchg.{h,cu}` (`rings::host_hmix`, `land_hmix`, `set_hmix_host`), `hc_phase` in `fn/model/layer.cpp` |
| `--fnfl-hc-packed on` (`FNFL_HC_PACKED=1`) | X-HC (the moments) in the joint layout (`xchg.h hcj_run_at`): each down block gathers its moments of every row in shared memory and one warp sends them as one contiguous run; the reader polls each 16-byte group as two 8-byte pairs at remapped addresses. Fused, non-head `hc::sub` only. | `down_phase`/`up_phase` in `fn/kernels/hc/hc.cu`, `poll_mixed`/`hcp_at` in `fn/xchg/xchg.cuh`, `hc_packed` in `fn/kernels/hc/hc.h`, `set_hc_packed` in `fn/xchg` |

Both default off, so the engine behaves as upstream unless asked. The server sets them from its flags before loading
(`fn/glue/fnfl_glue.cpp`, which prints both at startup). `fnfl-drive` and the benches read the environment variables
instead.

Tests:
- `xchg-test` runs every round over both routes. The host route checks every X-HMIX word in every consumer block, as
  the peer route does, and checks that every host slot is back at its sentinels afterwards.
- `hc2-test` runs a `fused-packed` case beside each `fused` one, against the same reference bits, and adds packed
  rows to the batch-invariance check.

**Important: having the consumers read host memory directly is 2–3× slower than today.** Every consumer block reads the
whole message, so the PCIe reads multiply about 72 times, and L2 doesn't absorb them. The landing step is what makes
the host route work.

Coverage: the target's HC sublayers in decode, which are the exchanges that mattered. These still use the old route
and layout:
- the draft head's single Q8_0 HC sublayer a draft step (`fn/model/draft.cpp`): one small exchange, not worth a
  second path;
- prompt processing (`pass.cpp`);
- the joint multi-user kernel (it has its own packed layout already).

The host route needs the two cards on separate devices and streams, since the landing waits on the peer's send. Rings
made on one device never take the host slots, so single-card mode stays on the peer route.

### Results

`docs/iq3s-tools/bench.py`, 24 requests a setting (4 prompt lengths to 25k tokens, 3 tasks, 2 reps), decode tok/s:

| Proposals | Before | Host route | Host route + packed moments |
|---|---|---|---|
| 2 | 161.9 | 166.3 | 166.1 |
| 3 | 151.4 | 167.0 | **183.3** |
| 4 | 136.9 | 159.1 | 177.6 |

- The best setting moves from 2 proposals at 161.9 to **3 proposals with both switches at 183.3 (+13%)**. The gain
  holds at every prompt length, and prompt processing is unchanged.
- Correctness:
  - Token counts were identical in all 72 bench requests.
  - `fnfl-drive spec run` gave byte-identical tokens with sampling (T 1, fixed seed) and greedy, with the switches
    off, host only, and host + packed.
  - `xchg-test`, `hc2-test`, and `model-dry` in graph, eager and spec modes all pass with the switches on.
- Matched `fnfl-drive spec run` (one prompt, 3 runs, median): 4 proposals went 145 → 169 → 191 tok/s, and 3 proposals
  163 → 178 → 197 tok/s (before → host → host + packed).

## Where a decode step's time goes now

From the wait counters (`FNFL_WAIT_STATS`, below): the average wait per poll that waited, times the share of polls that
waited, times the exchanges per step. That's 96 for X-HC and X-HMIX (two HC sublayers a layer) and 48 each for X-A and
X-F. The diagnostic build's own overhead distorts step times, so use it for shares only.

| | 3 proposals, host only | 3 proposals, both | 4 proposals, host only | 4 proposals, both |
|---|---|---|---|---|
| X-HC | 28 µs, ~2.3 ms a step | 12 µs, ~1.0 ms | 39 µs, ~3.1 ms | 15 µs, ~1.2 ms |
| X-HMIX | ~0 | ~0 | ~0 | ~0 |
| X-A + X-F | ~0.3 ms | ~0.3 ms | ~0.4 ms | ~0.4 ms |
| Exchange stall, share of step | ~2.6 ms (15%) | **~1.3 ms (8%)** | ~3.5 ms (18%) | **~1.6 ms (9%)** |

The rest, about 90% of a step, is the work itself:
- the draft step, about 1.5 ms;
- reading the routed experts' and dense weights from HBM, and the layers' compute.

Upstream's 5-token verify on a box without the collapse takes 14.2 ms, against about 16 ms here; the remaining
exchange stall accounts for most of that difference. Draft acceptance is about 1.84 drafts a step at 3 proposals and
2.13 at 4.

## What to try next for decode: the shortlist (2026-11-23)

Ranked by expected gain against effort. Today: ~183 tok/s on `bench.py` (3 proposals, both switches, 2× V100). The
gains are estimates from other engines' reports and this doc's measurements, not measured on FNFL. The details are in
the sections below.

| # | Idea | Expected gain | Effort | Where it comes from |
|---|---|---|---|---|
| 1 | **Lookup (n-gram) drafts** within the 4-proposal limit, used when the context has a match | +30–60% on edit-heavy turns (agents rewriting files), ~0 on new prose; maybe +10–25% over an agentic session | medium (draft source, acceptance, tests) | sFNX, QEngine ([other engines](#other-engines-surveyed-2026-11-23)) |
| 2 | **Adaptive draft count** from the draft head's confidence or recent acceptance (2–4 a step) | +3–8% | small–medium | QEngine (3–5 drafts from confidence), sFNX (acceptance-sized) |
| 3 | **Host path for every decode exchange** (X-HC, X-A, X-F as well as X-HMIX), chosen at startup by timing real decode steps on each path, cached | +3–5% here (exchanges are ~8–9% of a step); also makes the right choice on other platforms automatically | medium | QEngine; this doc's measurements |
| 4 | **The DFlash2 drafter** ([incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)): 8-token blocks instead of the MTP chain | ReVolta reports +30% on English code (4× V100 NVLink); with FNFL's 5-row verify limit maybe +10–20% on code, less on Chinese or prose | large (a new drafter architecture in the engine) | ReVolta |
| 5 | **Wider verify windows** (more than 5 rows) to use long lookup or DFlash2 drafts in full | lifts the ceiling of 1 and 4 on edits; on its own, nothing | large (`max_rows` = 5 is baked into the exchange layouts and kernels) | sFNX (16-row windows) |
| 6 | Exchange shaving ([ranked ideas below](#ideas-for-later-ranked-by-expected-payoff), items 1–3) | +3–6% in total | medium | this doc |
| 7 | Kernel profiling of the expert and dense products | unknown, 0–10% | medium–large | this doc |

**Expected overall:** on general text, ~190–200 tok/s (+5–10%) from items 2, 3 and 6. On agentic coding with file
edits, roughly 220–260 tok/s (+20–40%) once lookup drafts land, because those turns copy existing text. Measure on an
edit-heavy replay, not only on `bench.py`, which has no edits. Ruled out: fine-tuning the MTP head, and multi-user
batching on this box.

## The hardware floor and the path to a realistic maximum (2026-11-24, estimates)

**The floor.** Decode is bound by memory and latency, not compute, and draws well below GPU 0's 250 W cap. A verify
step at 3 proposals (4 rows) reads on each card:
- the dense weights, ~2.15 GB;
- the touched experts' halves, ~1.2–1.5 GB (4 rows × 10 experts × 48 layers);
- GDN states, KV and the rest, ~0.3–0.5 GB.

That is ~3.7–4 GB, **~4 ms at ~1,020 GB/s**, against ~15.5 ms measured (183 tok/s at 2.84 tokens a step). So decode
uses ~25–30% of the memory bandwidth. Prefill, by comparison, reaches ~50% of what a well-built engine could do
(`prefill-headroom.md`).

**A realistic floor for a well-built engine is ~8 ms a step, not 4:**
- small-batch kernels rarely sustain more than ~75–80% of HBM bandwidth;
- ~200 card-to-card synchronizations a step cost ≥0.5–1 ms of PCIe latency even when overlapped;
- the draft chain stays at ~1 ms even when fused.

**The realistic maximum:** ~300–350 tok/s on general text, and ~450–500+ on code and edit-heavy agent turns with
bigger drafts. The fastest published two-card numbers for this model are QEngine's 201–248 tok/s (2× RTX 3090) and
sFNX's ~140–160 on new code.

**The path, in order** (speed = tokens a step ÷ step time):

| Phase | What | Lever | Expected after it | Effort |
|---|---|---|---|---|
| 0. Measure | Nsight Systems on a decode step with node-level graph tracing, plus the wait counters: each kernel's time and bandwidth efficiency, idle gaps, draft against verify. Today there is only "~90% is the work" | — | a map of the ~11 ms above the floor | 1–2 days |
| 1. Cheap acceptance | Lookup drafts within the 4-proposal limit; adaptive draft depth (shortlist items 1–2) | tokens a step | ~190–200 general; ~230–260 on edits | 1–2 weeks |
| 2. Exchanges | The host route and packing for every exchange, chosen at startup by timing real steps; fewer, larger X-HC writes (shortlist item 3, ideas 1–2) | step −0.8–1 ms | ~200–215 general | 1–2 weeks |
| 3. Kernels to bandwidth | The 1–5-row expert and dense kernels toward ~70–80% of HBM bandwidth; each layer's small kernels fused; the draft step fused | step ~15 → ~10 ms | ~270–300 general; ~350+ on edits | 4–8 weeks; the core of it |
| 4. Fewer sync points | Merge or overlap the per-sublayer hand-offs; persistent per-layer kernels that keep both cards in lockstep without launch gaps | step ~10 → ~8 ms | ~300–350 general | 4–8 weeks, risky |
| 5. Bigger drafts | Verify windows past 5 rows (`max_rows` = 5 is baked into the exchange layouts) with DFlash2 or long lookup runs (shortlist items 4–5) | tokens a step on code 2.8 → 4–5 | ~450–500+ on code and edits | 1–2 months |

**Skip: the layer split** (idea 6 below) for one user. Each card would run half the layers one after the other, which
halves the usable memory bandwidth. It only pays if synchronization dominated, and phases 3–4 attack that more
directly.

**Start with phase 0**, which is cheap and turns phase 3's estimates into measurements. Then phase 1, the largest
visible gain for the least work, especially in agent use.

## Ideas for later, ranked by expected payoff

These rank the exchange and kernel work. For code-editing work, **n-gram drafting** (from sFNX, in the next section)
probably outweighs all of them. Try it before items 1–6.

1. **Shave X-HC further (about 1.0–1.2 ms a step left; maybe +3–5%).** Part of it can't be removed: each card waits for
   the other's down phase to finish, plus a couple of microseconds of PCIe latency. Options:
   - **Fewer, larger writes:** the fused kernel is cooperative and already has a grid barrier, so a few blocks could
     gather all 72 runs and send one or two large contiguous writes.
   - **Pull instead of push:** peer loads don't collapse. The receiver could read the sender's moments from the
     sender's memory after a single flag. That costs one extra latency hop; measure whether it beats the packed push.
   - **Overlap:** start the parts of the up phase that don't need the peer's moments before polling.
   - **Copy engine:** stage the moments locally and send them with `cudaMemcpyPeerAsync`. Launch latency probably
     makes this a loss at these sizes; worth a microbenchmark first.
2. **X-A and X-F (about 0.3–0.4 ms a step; maybe +1–2%).** Same treatment: host route with landing, or packed runs.
   Only worth it bundled with idea 1, since on their own they're small.
3. **X-DRAFT-IN (about 22 µs a wait, about half the polls wait).** The count per step is small; measure its share before
   touching it.
4. **The draft step (about 1.5 ms).** Its HC head still uses the old route and layout. Extending both switches to
   `fn/model/draft.cpp` would cover one small exchange a draft step; probably not measurable, so it was left out.
5. **Kernel time (about 90% of a step).** Profile the expert and dense products with Nsight Compute (`sudo -n
   /opt/cuda/nsight_compute/ncu` is allowed on this box). FNFL's kernels are already tuned, so the payoff is uncertain.
6. **Fewer sync points by design (large rewrite).** A layer split instead of a per-layer tensor split would replace
   about 200 card-to-card hand-offs a step with a few. Each hand-off costs at least a PCIe round trip, so that's a hard
   floor of about 0.3–0.5 ms a step today. Estimate the layer split's own costs (the pipeline bubble, the KV placement)
   before starting.
7. **Default depth.** With both switches, 3 proposals beats 2 and 4. Re-check after any of the above; cheaper deeper
   steps move the best depth up.

Ruled out:
- Fine-tuning the MTP head: Swift accepts drafts no worse than the base model (`docs/iq3s-swift.md`).
- Multi-user batching on this box: no throughput gain, and 3–4 users don't fit.
- Platform tweaks: listed above.

## Ideas from another engine: sFNX (checked 2026-11-23)

[sFNX](https://github.com/johnfink8/sfnx) (MIT, by johnfink8) runs the same model on V100s over PCIe Gen3. It also
splits every layer across the cards, with GPU threads writing each layer's sums into the next card around a ring (fp16
by default). **Most of its published numbers are for three cards. Check the card count before comparing anything with
this two-card box.** All figures are the author's own (its `docs/benchmarks.md`), not reproduced here.

| | sFNX, 3 cards (Unsloth Q4) | sFNX, 2 cards (Unsloth Q4, 36% of experts in RAM) | FNFL here, 2 cards (Swift IQ3_S, all in VRAM) |
|---|---|---|---|
| Code edit, greedy | 367–382 tok/s | 209–211 | not measured |
| New code, greedy | 200–229 | 137–163 | ~180–190 (bench.py mix, greedy) |
| Prose | 139 (T 1.0) | 109–119 (T 0.6) | ~180 (greedy; not comparable) |
| Prompt read | 3,373 tok/s at 114k | ~2,830 at 32k; 114k in 42.1 s (~2,710) | ~3,560 at 21k; upstream 3,558 at 130k |
| Warm turn, +2k on 30k, first token | 0.95 s median | not published | not measured |

Not like for like: different quants (Q4 against IQ3_S, which reads fewer bytes), sFNX's two-card build serves a third of
its experts from RAM, and the prompts and sampling differ. It wouldn't run usefully on this box anyway:
- it doesn't support Swift (it requires the embeddings, shared expert and draft projections at Q8_0);
- its two-card build keeps a pinned copy of every expert in RAM, likely more than 62 GB;
- its decode sums are GPU threads writing into the other card, which would hit this board's write collapse;
- it needs CUDA 12.9.

Ideas worth taking, best first:

1. **N-gram (prompt-lookup) drafting: probably the biggest decode lever for agentic coding.** Besides the MTP chain,
   sFNX finds the recent tokens elsewhere in the context and drafts the continuation found there, 3 to 15 tokens. Edits
   mostly copy existing code, so those drafts are accepted at a very high rate. On two cards it does ~210 tok/s on a code
   edit against ~150 on new code, a ~40% lift. That lift is partly the n-gram drafts and partly edits being more
   predictable. On three cards a verify window costs about 13 ms at 3 rows, 16 ms at 5 and 28 ms at 16.
   - **The catch:** FNFL is built for at most 5 rows a verify step. `fn::xchg::max_rows` = 5 is baked into the
     exchange slot layouts (`hc_own_at`, `hmix_layout`, `hcj_run_at`) and into the kernels' instantiations for B = 1..5.
     Windows of 16 rows would be a large change.
   - **Cheap first version:** when an n-gram match exists, draft its next tokens instead of running the MTP chain,
     within the existing 4-proposal limit. That skips the ~1.5 ms draft step and should accept more on edits. Measure
     on an edit-heavy replay (agentic sessions) as well as on `bench.py`, which has no edits.
2. **Adaptive draft depth.** sFNX sizes each request's drafts from its measured acceptance and the cost of the extra
   rows (`src/batch_plan.hpp`). FNFL uses a fixed `--fnfl-spec-d`. With the two switches on, 3 is best on average, but
   code and prose differ. A simple rule that picks 2–4 from recent acceptance could add a few percent.
3. **A link check at startup.** sFNX writes across every card pair before serving and warns when a link is under
   1 GB/s. FNFL could time a burst of SM-issued peer stores at load. When they collapse, as on this X299 board, it
   could say so and suggest `--fnfl-hmix-route host --fnfl-hc-packed on`. Small and cheap; useful to anyone else with
   this platform.
4. **For the prefill/TTFT work:** sFNX checkpoints at message boundaries (and every 4,096 tokens in a long first
   message), so a changed system prompt is read again only from the last checkpoint before the change. It parks idle
   conversations in host RAM (pageable, least recently used out first). FNFL keeps 8 checkpoints at multiples of 8,192
   tokens. Its 0.95 s warm-turn figure is from three cards, so measure FNFL's own warm turns before setting a target.
   Its prompt chunks are 4,096 rows (FNFL: 8,192, in two halves of 4,096).

Not worth taking here:
- **fp16 exchange payloads:** they halve the bytes but cost exactness (sFNX's own code and edit prompts go over its
  KL bar on two cards with them), and FNFL's remaining exchange stall is only ~2% of a step after packing.
- **The three-card layout:** this box has two cards.

## Other engines surveyed (2026-11-23)

A survey of engines that run Qwen3.8-Flash-Next, mostly on V100s. Every figure is the project's own claim, and the card
count and quant differ from row to row, so compare within a row's hardware only. Prefill leads from the same survey
are in [`prefill-headroom.md`](prefill-headroom.md).

| Project | Hardware | Decode, one request | Prompt reading | Notes |
|---|---|---|---|---|
| [QEngine](https://github.com/bpx03/QEngine) | **2× RTX 3090** / 3× 3090 | 201–248 / 310–452 tok/s | 2 cards: 3,760–4,265 (10K), 5,179–7,134 (90K) | INT4 experts; prefill GEMMs use INT8 activations on INT8 tensor cores, which V100 doesn't have |
| [ReVolta](https://github.com/Leonccaa/ReVolta) (built on [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM)) | **4× V100-32GB** | 201 (one PCIe switch), 234 (NVLink) | 3,382 (8K), 3,016 (131K) | AWQ 4-bit; MTP4, or the DFlash2 drafter |
| [llama.cpp-V100](https://github.com/1115714829/Llama.cpp-V100) | 2× / 6× V100-**16GB** | ~43 / 122.5 | 1,155–2,152 / 3,402 | experts spill to host RAM on 2 cards |
| [Strata-V100](https://github.com/jmnargi/Strata-V100) | 2× V100-**16GB** (one card at x4) | 62–81 | ~1,000–2,100 | Strata fork; streams experts |
| [qwen38-flash-next-2x3090](https://github.com/DominikBucko/qwen38-flash-next-2x3090) (vLLM) | 2× RTX 3090 + 128 GB RAM | 104–111 | up to 4,191 (131K) | experts in RAM |
| [sFNX](https://github.com/johnfink8/sfnx) | 3× (2×) V100-32GB | 200–229 (137–163), new code | 3,373 (~2,710) | the previous section |
| **FNFL here** | **2× V100-32GB** | **~183** (`bench.py` mix) | ~3,560 | Swift IQ3_S, all in VRAM |

Also seen, not useful here:
- [GooseLLM](https://github.com/haohervchb/GooseLLM) and [sglang-V100](https://github.com/haohervchb/sglang-V100):
  V100 builds of vLLM and SGLang, mostly for the 27B model;
- [Strata-AC922](https://github.com/eelgaev/Strata-AC922): IBM's POWER9 server with NVLink to the CPU;
- the archived ninfer backport.

On two V100s, FNFL's decode is the fastest found.

Decode ideas from the survey (ranked in the shortlist above):
- **QEngine's host path for every decode exchange.** It converged on the same fix as this doc. Each card writes its
  exchange words once into pinned host memory and the other card reads them there; the attention exchanges are copied
  into device memory in one step before use. It applies this to all decode exchanges, and at startup it runs a short
  dummy decode on each path (one sequence, then four, interleaved rounds) and keeps the faster one. It caches the
  result, keyed on the cards, links, driver and settings. Its finding is useful here: neither a microbenchmark nor an
  idle all-reduce predicted which path wins; only real decode steps did. Prefill transfers get their own probe (pull
  with SM loads when they're not slower than stores; else the copy engine), plus a data check that P2P writes really
  arrive.
- **Lookup drafts with confidence-sized draft counts.** QEngine drafts 3–5 tokens a step for one conversation,
  picked from the MTP head's confidence, plus lookup drafts when recent tokens repeat earlier text. On 2× 3090 it does
  226 tok/s on code against 201 on narrative; on 3× 3090, 452 against 310.
- **DFlash2.** [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) is a
  drafter with 5 sliding-window layers, an 8-token block and a 2,048-token window, reading the target's
  hyper-connection streams (`hc_mean`). ReVolta runs it on V100 with CUDA graphs: 234 tok/s against 180 with MTP4 on
  English code, one request, 4× V100 NVLink. MTP4 stays ReVolta's default because Chinese text and 8 concurrent
  requests are faster with it.
- **Kernel sources** (V100 ports, if profiling points at attention or GDN):
  - flash-attention for V100 ([ai-bond](https://github.com/ai-bond/flash-attention-v100),
    [zhinianqin](https://github.com/zhinianqin/flash-attention-v100)), and TileLang's FA-V100 (in GooseLLM);
  - 1Cat-vLLM's grouped verify-attention and small-batch quantized products (also ported into llama.cpp-V100).

## Tools

- **Wait counters:** configure a build with `-DCMAKE_CUDA_FLAGS=-DFNFL_WAIT_STATS` (this box uses `build-wait`). Run
  the server with `FNFL_WAIT_STATS_DIR=DIR`; touching `DIR/dump-request` writes `DIR/wait-stats.txt`, with polls,
  polls that waited, cycles waited and the maximum for each message and card. Take a snapshot before and after a set
  of requests and subtract. Bit 14 (`X-HC-own`) counts the fused kernel's wait on its own card's moments separately.
- **`docs/iq3s-tools/p2p/`:**
  - `p2pall.cu.txt`: the platform regression set (peer write size sweep; stores and loads to local, peer and host).
  - `hcpat6.cu.txt`: the X-HC/X-HMIX stand-in, which compares peer, host-direct, host-with-landing and no X-HMIX, with
    the moments shipped or contiguous.
  - Both are saved as `.txt` because the include check rejects CUDA sources outside `fn/`. Copy one out and build it
    with `nvcc -O3 -arch=sm_70 FILE.cu`.
  - `hc_bench_xhc.patch`: an `xhc` mode for `fnfl-bench hc2-bench 0 1 xhc`, timing the fused sublayer at 1–5 rows with
    both messages crossing against X-HC only. Apply it with `git apply`, then build the `fnfl-bench` target. Its
    `xhc_none` point still sends to sinks on the *peer*, so it isn't a no-exchange bound. For that bound, point the
    sinks at the card's own memory.
  - `gpumon.sh FILE`: the GPU telemetry logger (temperature, HBM temperature, power, clocks, throttle reasons, every
    500 ms). Run it during every benchmark.
- **Telemetry:** GPU 0 runs 4–7 °C hotter than GPU 1 (its cooling), peaking at about 71 °C core and 73 °C HBM on the
  full bench. During long prompts it touches its 250 W software power cap (`0x4`, not thermal) for about 20–40 s, at
  1372 MHz instead of 1380. There has been no thermal throttling.
