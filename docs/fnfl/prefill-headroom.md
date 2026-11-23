# Prefill and time-to-first-token headroom: leads (2026-11-23)

The starting point for the prefill/TTFT investigation on the development machine (2× V100-32GB, PCIe 3.0 x16 on
separate root ports; see [`decode-headroom.md`](decode-headroom.md) for the platform). **Nothing here is measured on
FNFL's prefill yet.** These are leads from a survey of other engines and a back-of-envelope floor, ranked by how
promising they look. The investigation should replace them with measurements and record what it finds in this file.

## Where prefill stands

- **Fresh prompts (Swift IQ3_S, `docs/iq3s-tools/bench.py`, cache-busted prompts still to be made):**
  - ~28 tokens: ~254 tok/s, TTFT ~0.11 s, almost all fixed cost;
  - ~1.7k tokens: ~2,930 tok/s (~0.58 s);
  - ~6.9k tokens: ~3,710 tok/s (~1.85 s);
  - ~21k tokens: ~3,560 tok/s (~6 s).
- **Upstream (`engine.md`):**
  - 130,865 tokens fresh at 3,558 tok/s;
  - a 3,391-token append after 196,608 at 2,647 tok/s, of which the first chunk's host PLE gather took 128 ms.
  - Targets: 4,000 tok/s fresh up to 192K, 3,000 for appends.
- **Caveat:** `bench.py`'s long prompts share the same filler across tasks, so later requests reuse cached prefixes
  (checkpoints at multiples of 8,192 tokens). Its long-prompt rates are partly cache hits; make cache-busted prompts
  for clean numbers.
- **Power:** GPU 0 hits its 250 W software power cap (throttle `0x4`, ~1,372 MHz) during long prompts; GPU 1 doesn't.

**Back-of-envelope floor.** About 6B active parameters a token is ~12 GFLOP a token. At 3,560 tok/s that is ~43
TFLOP/s, ~21 per card, about **19% of a V100's FP16 tensor peak** at 1,380 MHz (~113 TFLOP/s). Expert weights are
read once per 8,192-token chunk (160 rows an expert on average), so memory traffic is small. This says prefill is
neither compute-bound nor memory-bound in aggregate: the time goes to small GEMMs, non-GEMM work (GDN, attention, HC,
the router), exchanges, host gaps or the PLE gather, which is what profiling has to find.

## Leads, most promising first

1. **Profile a chunk and close the gap to the floor (expected: the biggest win; +15–40% on fresh prompts, uncertain
   until measured).**
   - **Evidence there is headroom:** [QEngine](https://github.com/bpx03/QEngine) reads a 90K prompt at 5,179–7,134
     tok/s on **2× RTX 3090**, about double FNFL's ~3,560, and its rate *rises* with prompt length, while FNFL's stays
     flat from 8K to 130K. Part of that is INT8 activations on Ampere's INT8 tensor cores, which V100 lacks, so the
     gap isn't all transferable. But V100's FP16 tensor rate is higher than the 3090's.
   - **Method:** an Nsight Systems timeline of one 8,192-token chunk on both cards; per-kernel share (dense,
     experts, GDN, QSA, HC, exchanges, PLE gather, draft catch-up, checkpoint copies); idle gaps; copy-engine overlap.
2. **Prefix caching beyond the slot (expected: TTFT on repeated context drops from seconds to well under a second on
   a hit; nothing for fresh prompts).**
   - [ReVolta](https://github.com/Leonccaa/ReVolta) (4× V100) brings a cached 32K-token document back from CPU memory
     in 0.6 s, or 1.5 s from NVMe after a restart, against 10.1 s to recompute.
   - [QEngine](https://github.com/bpx03/QEngine) keeps a host-RAM archive of conversations and parks waiting replies
     there.
   - [sFNX](https://github.com/johnfink8/sfnx) checkpoints at message boundaries and before late notes (and every
     4,096 tokens of a long first message), so an edited system prompt is re-read only from the last checkpoint
     before the change. It parks idle conversations in pageable host RAM (pinned memory costs VRAM).
   - FNFL keeps up to 8 checkpoints at multiples of 8,192 tokens and the prompt's end, within the slot.
   - **The budget here:** 62 GB RAM with the 27 GB PLE table locked, so a host archive has ~20–25 GB at most. NVMe
     is a second tier. Measure FNFL's KV and recurrent-state bytes per token before sizing it.
3. **Score only live QSA blocks in prompt chunks (expected: +5–15% at long prompts if FNFL scores unwritten blocks;
   nothing if it already doesn't).** [Strata-V100](https://github.com/jmnargi/Strata-V100) found its prompt path
   scoring blocks beyond the prompt's live range. Skipping them cut QSA selection time by 91%, and prefill rose from
   961 to 1,111 tok/s on a 15K prompt (2× V100-16GB). A quick read of FNFL's prompt QSA scoring settles it.
4. **The fixed cost of a short prompt or append (expected: tens to ~130 ms of TTFT).** The first chunk's host PLE
   gather can't overlap; it took 128 ms of a 3,391-token append upstream. Ideas: start the gather when the request
   arrives (before tokenizing finishes, if the server allows); a smaller first chunk; check graph launch and server
   overheads at ~28 tokens (~0.11 s).
5. **Prefill exchanges.** FNFL sends prompt partials through the copy engines as FP16, so this box's SM-store
   collapse shouldn't apply; verify it in the timeline. QEngine's prefill probe pulls with SM loads when they're not
   slower than stores, else uses the copy engine. It also measures how much each transfer slows a kernel running
   beside it (the co-run tax), which is worth measuring for the copy engines here.
6. **Kernel sources, if profiling points at them:**
   - the FlashQLA chunked GDN prefill kernels (TileLang; credited in
     [llama.cpp-V100](https://github.com/1115714829/Llama.cpp-V100));
   - TileLang's FA-V100 ([GooseLLM](https://github.com/haohervchb/GooseLLM)) and the V100 flash-attention ports
     ([ai-bond](https://github.com/ai-bond/flash-attention-v100),
     [zhinianqin](https://github.com/zhinianqin/flash-attention-v100));
   - sFNX's expert kernel on raw `mma.m8n8k4`, and its deferred attention-exchange merge (a 30K read went from 3,176
     to 3,720 tok/s on 3 cards with its prompt-path changes).

Not promising here:
- **Bigger chunks:** FNFL already uses 8,192 rows. QEngine measured +4–5% for 8,192 over 4,096 on 2 cards; sFNX
  measured 8,192 at only +2% over 4,096.
- **A pipeline (layer-split) prefill mode:** QEngine reads long prompts up to 1.8× faster that way on 3 cards, but on
  2 cards with P2P the two modes are about even, and FNFL would need a second weight layout to do it.
- **INT8-activation GEMMs:** V100's tensor cores take FP16 only.

## Expected overall

- **Fresh prompts:** ~3,560 → 4,000–5,000 tok/s (+12–40%) is a reasonable hope if profiling finds the usual
  suspects (small-GEMM efficiency, serialization, idle gaps). The 19% utilization suggests more is possible in
  principle; how much is reachable depends on what the timeline shows.
- **TTFT:** on repeated or switched-back context, a prefix cache turns seconds into well under a second on a hit. On
  short fresh prompts, at most ~0.1 s of fixed cost to save.
