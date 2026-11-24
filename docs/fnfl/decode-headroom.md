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

## What was done: the exchange switches (2026-11-23, X-HC route 2026-11-24)

| Server flag (fnfl-drive: environment) | What it does | Code |
|---|---|---|
| `--fnfl-hmix-route host` (`FNFL_HMIX_HOST=1`) | X-HMIX (the mixed row, 6,720 B a row) goes through pinned host memory. The sender writes the receiver's host slot; a small landing kernel on the receiver polls each 16-byte group once, copies it into the device slot the consumers already poll, and resets the host group. | `fn/xchg/xchg.{h,cu}` (`rings::host_hmix`, `land_hmix`, `set_hmix_host`), `hc_phase` in `fn/model/layer.cpp` |
| `--fnfl-hc-route host` (`FNFL_HC_HOST=1`; `pull`: `FNFL_HC_PULL=1`), with packed | X-HC off peer writes (2026-11-24). host: the sender's down phase writes its moments into the receiver's pinned host slot; pull: nobody sends, and the receiver reads the sender's own copy over the peer mapping. Either way the 32 blocks of the fused HC launch that are idle in its up phase land the moments, one coalesced 8-byte pair a thread, into the device slot the up phase already polls (host also resets the host words). [Measured below](#hc-in-decode-measured-inside-the-kernel-2026-11-24): +4.4% (host), +2.6% (pull). | the landers in `hc_sub_kernel`, `sub` and `sub_params::land_src` in `fn/kernels/hc/hc.cu`; `rings::host_hc`, `set_hc_host` in `fn/xchg`; `set_hc_pull` in `fn/kernels/hc/hc.h`; `hc_phase` in `fn/model/layer.cpp` |
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

Ranked by expected gain against effort. Today: ~194 tok/s on `bench.py` (3 proposals, the exchange switches with `--fnfl-hc-route host`, 2× V100; 183 before the X-HC route of 2026-11-24). The
gains are estimates from other engines' reports and this doc's measurements, not measured on FNFL. The details are in
the sections below.

| # | Idea | Expected gain | Effort | Where it comes from |
|---|---|---|---|---|
| 1 | **Lookup (n-gram) drafts** within the 4-proposal limit, used when the context has a match | **Done 2026-11-24** (`--fnfl-lookup 6`): +13–14% on edit-heavy turns, flat on general text; [measured below](#lookup-drafts-measured-2026-11-24). Less than the +30–60% estimated: the MTP head already copies well (~3.9 of its 4 tokens a step on edits) | medium (draft source, acceptance, tests) | sFNX, QEngine ([other engines](#other-engines-surveyed-2026-11-23)) |
| 2 | **Adaptive draft count** from the draft head's confidence or recent acceptance (2–4 a step) | +3–8% | small–medium | QEngine (3–5 drafts from confidence), sFNX (acceptance-sized) |
| 3 | **Host path for every decode exchange** (X-HC, X-A, X-F as well as X-HMIX), chosen at startup by timing real decode steps on each path, cached | +3–5% here (exchanges are ~8–9% of a step); also makes the right choice on other platforms automatically. **X-HC done 2026-11-24** (`--fnfl-hc-route host`, +4.4%); X-A and X-F left, and the automatic choice | medium | QEngine; this doc's measurements |
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

**Phase 0, measured (2026-11-24).** Swift, 3 proposals, the recommended flags, three greedy 300-token requests: two
short prompts (one code continuation at 3.6 tokens a step, one at 1.9) and one at 30K tokens of context.
- **Untraced, from the server log:** a step takes 14.4–14.8 ms at short context and 15.8 ms at 30K: the draft chain
  ~1.15 ms, the verify pass ~13.3 ms, the catch-up ~0.4 ms.
- **The tools:** Nsight Systems with node-level graph tracing (`--cuda-graph-trace=node`), which adds ~9% to a step, so
  its times are proportions; the wait counters (`build-wait`, rebuilt from the current code); `fnfl-bench hc-bench`
  for HC on one card with no exchange.

Kernel time a step (GPU 0, traced ~16.6 ms; GPU 0 runs a kernel 96% of the step, GPU 1 90%; ~817 kernels a step a
card):

| Family | ms a step | Calls a step | Bandwidth floor | Reading |
|---|---|---|---|---|
| Dense products (`product_kernel`) | 4.19 | 165 | ~2.1 ms (~2.15 GB) | ~50% of HBM bandwidth |
| **HC sublayers (`hc_sub`)** | **4.18** (GPU 1: 3.29) | 97 | **~0.6 ms** (~0.63 GB of BF16 HC weights) | 43 µs a call (GPU 1: 34); see below |
| Experts (`gate_up_kernel_u`, `down_kernel`) | 3.51 | 99 | ~1.7 ms (≤ 40 distinct experts a layer at 4 rows) | ~50% of HBM bandwidth |
| Exchange kernels (`recv_add`, `land_hmix`, `send_half`, `recv`, `recv_select`) | ~1.9 | ~300 | — | hand-off work and waiting, 2–8 µs a call |
| QSA attention (`attend`) | 0.49 at short context, 0.73 at 30K | 15 | | |
| GDN core | 0.43 | 36 | | |
| The draft's HC, the head, the sampler, the rest | ~1.9 | | | |

**Waits** (wait counters, both cards alike):
- X-HC: ~1.0–1.1 ms a step (90% of polls wait, ~11–13 µs each).
- X-A and X-F: ~0.2–0.35 ms.
- X-HMIX and X-DRAFT-IN: negligible.

**HC is the outlier.** One card's half of a sublayer at 3 rows takes 17.5 µs on its own, as separate down and up
kernels with no exchange (`fnfl-bench hc-bench`: `ctx_card_B3`), against a ~7 µs bandwidth floor. In decode the fused
sublayer takes 34–43 µs. Roughly:
- ~12 µs is the X-HC wait;
- ~17 µs is the work itself, at ~2.5× its floor;
- the rest is the fused kernel's own overhead (grid barriers, polling, sends).

*Corrected by the in-kernel timeline the same day* ([below](#hc-in-decode-measured-inside-the-kernel-2026-11-24)): at
the verify's 4 rows the X-HC wait was ~13–17 µs a call and bimodal (about half the calls hit the write collapse). The
work was ~18 µs a call: ~9.7 µs down and ~8.3 µs up.

**What it says, against the ~4.5 ms floor of the bytes above:**
- **HC sublayers:** −1.5 to −2.5 ms a step. A leaner fused kernel near its 7 µs of reads, and less X-HC wait (pull
  instead of push, or overlap the up phase's independent part).
- **The ~300 small exchange kernels a step:** −~1 ms. Fold the landing and the receive-and-add into the kernels that
  consume them.
- **Dense and expert products at ~50% of bandwidth:** −~1.5–2 ms together, with better small-batch products and fewer,
  larger launches (165 dense launches a step).
- **The draft:** ~1.5 ms with its catch-up.

Together ~14.6 → ~10 ms a step, ~265 tok/s at today's acceptance: phases 2–3 of the path below, now with measured
targets. **HC first:** it is the largest gap and the best understood.

**The path, in order** (speed = tokens a step ÷ step time):

| Phase | What | Lever | Expected after it | Effort |
|---|---|---|---|---|
| 0. Measure | Nsight Systems on a decode step with node-level graph tracing, plus the wait counters: each kernel's time and bandwidth efficiency, idle gaps, draft against verify. Today there is only "~90% is the work" | — | a map of the ~11 ms above the floor | 1–2 days |
| 1. Cheap acceptance | Lookup drafts within the 4-proposal limit (**done**: edits ~4.8 tokens a step, +13–14%); adaptive draft depth (shortlist items 1–2) | tokens a step | ~190–200 general; ~230–260 on edits | 1–2 weeks |
| 2. Exchanges | The host route and packing for every exchange, chosen at startup by timing real steps; fewer, larger X-HC writes (shortlist item 3, ideas 1–2) | step −0.8–1 ms | ~200–215 general | 1–2 weeks |
| 3. Kernels to bandwidth | The 1–5-row expert and dense kernels toward ~70–80% of HBM bandwidth; each layer's small kernels fused; the draft step fused | step ~15 → ~10 ms | ~270–300 general; ~350+ on edits | 4–8 weeks; the core of it |
| 4. Fewer sync points | Merge or overlap the per-sublayer hand-offs; persistent per-layer kernels that keep both cards in lockstep without launch gaps | step ~10 → ~8 ms | ~300–350 general | 4–8 weeks, risky |
| 5. Bigger drafts | Verify windows past 5 rows (`max_rows` = 5 is baked into the exchange layouts) with DFlash2 or long lookup runs (shortlist items 4–5) | tokens a step on code 2.8 → 4–5 | ~450–500+ on code and edits | 1–2 months |

**Skip: the layer split** (idea 6 below) for one user. Each card would run half the layers one after the other, which
halves the usable memory bandwidth. It only pays if synchronization dominated, and phases 3–4 attack that more
directly.

**Start with phase 0**, which is cheap and turns phase 3's estimates into measurements. Then phase 1, the largest
visible gain for the least work, especially in agent use.

## HC in decode, measured inside the kernel (2026-11-24)

**The method.** A diagnostic build (`-DFNFL_HC_TL` in `CMAKE_CUDA_FLAGS`, as the wait counters) timestamps each block
of the fused HC launch (`hc_sub_kernel`) at its phase boundaries with `clock64`. Every launch adds to per-row-count sums
and 1 µs histograms, and the process prints them at exit (`hc-tl` lines on stderr). The run: the server, three greedy
300-token requests (`decreq.py`: two short prompts and one at 30K), ~31,500 calls at 4 rows. The numbers are µs from
each block's start, means over the 40 up blocks:

| Phase (4 rows) | push (before), GPU 0 / 1 | host route, GPU 0 / 1 |
|---|---|---|
| Down: staging (activations in, x out) | 5.7 / 5.6 | 5.7 / 5.6 |
| Down: products (3.3 MB of BF16 weights), moments sent | 9.6 / 9.5 | 9.7 / 9.7 |
| Up: both cards' moments polled (**X-HC**) | **26.7 / 23.4** | **21.0 / 17.9** |
| Up: low-rank formed | 28.9 / 25.6 | 23.1 / 19.8 |
| Up: products (3.3 MB) | 33.1 / 29.7 | 27.2 / 23.9 |
| Up: mixed row out, end | 35.0 / 31.6 | 29.1 / 25.8 |

**What it showed:**
- The moments wait was the largest piece, and the same on both cards, so it was the link's latency rather than one
  card lagging.
- With push, the poll's histogram is bimodal. About half the calls finish ~14 µs in, close to ideal (the slowest sender
  block finishes its moments at ~12 µs, p99 13). The other half wait until ~40 µs: the peer-write collapse.
- On one card with the moments pre-filled, the same poll ends ~4 µs after the up phase starts (`hc2-bench 0 1 tl`).

**What was tried:**

| Change | Result |
|---|---|
| **Host route** (`--fnfl-hc-route host`): sender writes host memory, the idle up-phase blocks land it | moments at ~17 µs, steady; **194.2 tok/s vs 186.0 on `bench.py` (+4.4%), faster on all 12 rows**; step medians −0.5 to −0.75 ms |
| **Pull** (`--fnfl-hc-route pull`): the receiver's idle blocks read the sender's own copy | moments at ~18–19 µs, steady; 190.9 tok/s (+2.6%) |
| L2 prefetch of both weight matrices at kernel start | no change (cold bench); dropped |
| Rolled weight-row loops (code −25%; Nsight Compute showed ~40% of stalls as instruction fetch, cold bench) | −1–2 µs a call in the cold bench, nothing measurable in decode; dropped |

All three routes gave bitwise-identical tokens (content hashes of the three requests), `hc2-test` passes on each (284
bitwise checks, plus a new check that every host X-HC word is back at its sentinel). `xchg-test`, and `model-dry` in
graph, eager and spec, pass with the host route.

**Why host beats pull:** read latency from device 0 (`docs/iq3s-tools/p2p/readlat.cu.txt`), coalesced 8-byte loads:

| Loads in flight | Peer card memory | Pinned host memory |
|---|---|---|
| 1 | 1.24 µs | 0.78 µs |
| 2,600 (one X-HC message at 4 rows) | 1.86 µs | 1.61 µs |

Uncoalesced 8-byte loads are much worse (~11.7 µs peer, ~9 µs host at 2,600 in flight), so the landers take
consecutive pairs.

**What is left in HC, a call at 4 rows with the host route (~26–29 µs against a ~7 µs read floor):**
- *Moments wait*, ~7–11 µs after the up phase starts, against ~4 µs with the moments already local. The rest is the
  cards' start offset, the host write and the landers' poll pass. Small further gains only.
- *Staging*, 5.6 µs before the first product. 18 blocks of each stream re-stage the same activations (~2.9 MB of L2
  reads at 4 rows), while the weight loads issued at the same time flood DRAM.
- *Up phase on 40 of 72 SMs*: its 3.3 MB run on 40 SMs at ~4 µs, the low-rank formation (~2 µs, precise `sqrtf`,
  `expf` and divisions fixed by bitwise) and the finish (~1.9 µs).
- Next HC steps, in order: stage the activations once a stream (or from fewer blocks) and start the products
  earlier; spread the up products over all 72 SMs (the run-aligned quantize then needs a cross-block finish, as HC3's
  72-block shape had); fold the landing into fewer, larger reads. Maybe another 5–8 µs a call, ~0.5–0.8 ms a step
  (+4–5%).

## Lookup drafts, measured (2026-11-24)

**What it is** (`--fnfl-lookup N`, env `FNFL_LOOKUP=N`; `FNFL_LOOKUP_MIN`, default 4). Before a speculative step, the
host searches the slot's context (prompt and output so far) for the most recent earlier occurrence of its last N tokens.
If at least 4 tokens followed it, the step proposes those 4 and skips the draft step (~1.15 ms); otherwise the draft
head proposes as before. Code: `lookup_propose` and `step_propose` in `fn/runtime/runtime.cpp`.
- **Correctness.** Greedy verify accepts a token only if it is the target's own choice, so greedy outputs are bitwise
  the same with lookup on (checked on every edit-bench request). At T > 0 the accept step compares the target's p(x)
  with the draft's retained q(x); for a lookup proposal the host writes the one-entry list {x: 1} into the row's
  `q_list` on both cards, so a proposal is taken with probability p(x), and a rejection resamples from p without x:
  the target's distribution, with no sampler change. The full sampling paths (top-k off) keep the draft head, and the
  joint multi-user rounds don't use lookup (their lanes share one draft depth).
- **The bench:** `docs/iq3s-tools/edit_bench.py`. Four agent-style edits (a repo file in the prompt, one small change,
  the whole file back) and one prose control, chat, thinking off, up to 1,600 tokens.

| | Lookup off | `--fnfl-lookup 6` |
|---|---|---|
| Edits, greedy, mean tok/s | 262.3 | **296.9 (+13%)** |
| Edits, T 0.7 | 263.6 | **299.6 (+14%)** |
| Edits, tokens a speculative step | ~3.8–3.96 | ~4.3–4.8 (max 5) |
| Prose control, greedy | 222.4 | 221.7 |
| `bench.py` (general, 24 requests), mean tok/s | 194.2 | 193.7 (lookup on ~3% of the steps) |

- **The trigger matters.** A 3-token match with at least 2 proposals fired on prose and cost 10% there, for +3% on
  edits. Matches of 4, 6 and 8 with 4 proposals gave edits +7%, +13% and +12%; 6 is the default to use.
- **Why less than the estimate:** the MTP head already gets ~3.9 of its 4 tokens a step (3 proposals) on copied
  text, so lookup only lifts that toward the verify window's 5. The rest of the gain on edits needs wider verify
  windows (phase 5, `max_rows` = 5).
- **Telemetry** (all lookup runs): GPU 0 up to 67 °C core and 68 °C HBM, `0x4` only during long prefills; GPU 1 up to
  59 °C; no thermal throttling.

## Follow-ups that didn't pay (2026-11-24)

Measured with the HC timeline build and three greedy decode requests; every change was bitwise the same and none
moved the step time, so none was kept:

| Change | What it showed |
|---|---|
| L2 prefetch of the up weights while the moments are awaited | the up products stay at ~4.1 µs: instruction-bound on their 40 SMs, not DRAM-bound |
| First down weight rows issued only after the activations land | the activations still take ~2.8 µs to arrive (cold reads and the launch), and the products end later |
| X-HMIX landing folded into the HC launch's idle blocks (no `land_hmix` launch) | the HC launch just ends later: `land_hmix` was waiting for the other card, not costing its own time |
| Verify at 5 rows against 4 (fixed depth 4 vs 3) | not a 5-row anomaly: 3 rows are slow (IQ2 gate/up at B = 3 takes ~65 µs, ~43 at 4); from 4 to 5 the experts grow with the distinct experts (~+8 µs a gate/up call), HC and the products steadily; only expert down shows a small extra (~0.1 ms a step). Depth 3 stays best |

**What this says.** The small exchange kernels' times in a trace (~1.9 ms a step) are mostly waiting for the other
card, so folding them saves little. What is left in a step is a chain of cross-card hand-offs (about four a layer)
and cold starts of many small launches. Removing that needs fewer synchronization points by design (phase 4 of the
path), not per-kernel tuning. Remaining kernel items (the up products on all 72 SMs, a leaner low-rank finish) are
worth ~1% each.

## Ideas for later, ranked by expected payoff

These rank the exchange and kernel work. For code-editing work, **n-gram drafting** (from sFNX, in the next section)
probably outweighs all of them. Try it before items 1–6.

1. **Shave X-HC further (about 1.0–1.2 ms a step left; maybe +3–5%).** *Done in part 2026-11-24: pull and the host
   route ([measured above](#hc-in-decode-measured-inside-the-kernel-2026-11-24)); host gave +4.4%. The notes below
   are from before.* Part of it can't be removed: each card waits for
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
| **FNFL here** | **2× V100-32GB** | **~194** (`bench.py` mix, 2026-11-24) | ~3,560 | Swift IQ3_S, all in VRAM |

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
  - `readlat.cu.txt`: read latency from one card to local, peer and pinned host memory, by loads in flight (the
    X-HC route choice, 2026-11-24).
  - `gpumon.sh FILE`: the GPU telemetry logger (temperature, HBM temperature, power, clocks, throttle reasons, every
    500 ms). Run it during every benchmark.
- **The HC timeline** (2026-11-24): a build with `-DCMAKE_CUDA_FLAGS=-DFNFL_HC_TL` prints per-phase means and
  histograms of the fused HC launch at exit (`hc-tl` lines; [above](#hc-in-decode-measured-inside-the-kernel-2026-11-24)).
  `fnfl-bench hc2-bench 0 1 tl` (same build) times card 0's pre-filled launch cold at 1, 3 and 4 rows.
- **HC benches** (`fnfl-bench hc2-bench 0 1 MODE`): `cold` now includes 4 rows (the verify size); `FNFL_BENCH_COLD=l2`
  keeps the instruction cache warm (the L2 flush only) and `=i` the reverse; `ncu-fused4` runs card 0's pre-filled
  fused launch at 4 rows after a cold flush, for `sudo -n /opt/cuda/nsight_compute/ncu --set full -k regex:hc_sub`.
  A cold launch at 4 rows takes ~33 µs alone, ~19 µs with the instruction cache warm.
- **Telemetry:** GPU 0 runs 4–7 °C hotter than GPU 1 (its cooling), peaking at about 71 °C core and 73 °C HBM on the
  full bench. During long prompts it touches its 250 W software power cap (`0x4`, not thermal) for about 20–40 s, at
  1372 MHz instead of 1380. There has been no thermal throttling.
