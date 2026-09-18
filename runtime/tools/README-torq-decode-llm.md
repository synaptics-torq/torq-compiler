# torq-decode-llm

A dependency-free command-line driver that runs **autoregressive text
generation for decoder-only LLMs on the Torq NPU**.

Given a model directory (body vmfb + optional lm-head vmfb + `config.json`
+ embedding table), it loops:

```
embed -> transformer body on NPU (in-place KV feedback) -> lm-head -> argmax -> next token
```

printing per-token ids, per-phase timing, and tok/s.

Its three jobs:

1. **Measurement instrument** for decode performance — a per-phase timing
   breakdown (body invoke / lm-head / host transfers) that Python-level
   runners cannot give.
2. **Correctness harness** for bringing up decoder models on new toolchain
   builds.
3. **Minimal serving binary** for the board — no Python, no pip.

It is deliberately *not* a chat app: there is no tokenizer inside (prompt
input is raw token ids); the Python demo fills that role.

---

## Features

- **Tied-operand zero-copy KV feedback**: when the compiled body annotates
  outputs `iree.abi.tied` (the HAL writes each output into its tied input
  buffer), KV buffers are allocated once and never read back — no per-token
  feedback copies. Self-validates on the first step (pointer check) and
  falls back to copy-back feedback on non-tied vmfbs.
- **On-NPU lm-head in a second VM context**: avoids the cross-module XRAM
  thrash (overlapping weight arenas evicting each other every switch) that
  a single shared context exhibits; pays only a small hidden-state handoff.
- **Model-agnostic model-dir discovery**: body/head vmfbs, `config.json`,
  `token_embeddings.npy`, and optional `token_id_lut.npy` are picked up
  automatically; one binary serves any decoder-only LLM with KV-cache I/O.
- **Prefill with head skip**: prompt tokens run through the body without
  the lm-head (except after the last), matching reference-runner behavior.
- **Host fallbacks**: tiled NEON host lm-head (`--host-lmhead` or when no
  head vmfb exists) over the embedding table.

## Model directory layout

```
<model-dir>/
├── <body>.vmfb          # any *.vmfb not named lm_head*; N+2 inputs
│                        #   (token_embedding [1,1,H] bf16, position_ids
│                        #   [1,1] si32, N x past_kv) and N+1 outputs
│                        #   (hidden + N x present_kv), tied-annotated
├── lm_head*.vmfb        # optional on-NPU lm-head ([1,1,H] -> [1,1,V])
├── config.json          # num_hidden_layers, hidden_size, vocab_size,
│                        #   head_dim, num_key_value_heads, eos_token_id
├── token_embeddings.npy # [vocab, hidden] bf16 embedding table (mmap'd)
└── token_id_lut.npy     # optional trimmed-vocab remap (1-D i32/i64)
```

Producing the vmfbs (any Torq compiler with tied-operands support; no
other special flags beyond the recommended set):

```sh
torq-compile body.mlir -o transformer.vmfb \
  --torq-hw=SL2610 --torq-disable-slicing \
  --torq-enable-annotate-tied-operands \
  --torq-enable-split-constants-optimization \
  --iree-flow-inline-constants-max-byte-length=300000000
# same flags for lm_head.mlir -> lm_head.vmfb[.trim]
```

## Build and deploy

```sh
# host: cross-compile for the board (aarch64 SoC build tree)
ninja -C <iree-build-soc> runtime/tools/torq-decode-llm

# push (destination is any writable dir on the board; adb shell lands in
# the user's home by default)
adb -s <board> push <iree-build-soc>/runtime/tools/torq-decode-llm ./
```

## Usage

```sh
TORQ_DECODE_TIMING=1 torq-decode-llm --model-dir=<dir> \
    [--prompt-tokens=<csv>] [--max-new-tokens=N] [--past-len=N] \
    [--first-token=T] [--start-position=P]
```

Output:

```
step 14: token 7837 (130.4 ms)
decode: 16 tokens in 2.121s = 7.54 tok/s (132.5 ms/token)
TORQ_DECODE_TIMING: total=2.122s | invoke=1.221 lmhead_npu=0.867 d2h=0.016 ...
```

A/B and debug switches:

| flag | effect |
|---|---|
| `--no-tied-kv` | force copy-back KV feedback (correctness A/B) |
| `--host-lmhead` | force the host NEON lm-head even with a head vmfb |
| `--one-context` | single VM context for body+head (shows the XRAM thrash) |
| `--lmhead-weight=<bin>` | raw `[hidden, vocab]` bf16 table for the host path |
| `--past-len=N` | KV window; must match the compiled model (default 256) |
| `--vocab-size/-hidden-dim/--n-layer/--head-dim/--kv-heads` | overrides; normally read from `config.json` |

Text to token ids (host side, once, since the driver takes raw ids):

```sh
python3 -c "from tokenizers import Tokenizer; \
  print(Tokenizer.from_file('tokenizer.json').encode('your prompt').ids)"
```

## Reference measurement

sl2619 dev board (gemma3-270m-it, split-LM-head, bf16, KV window 256):
**132.5 ms/token, 7.54 tok/s** steady state (body invoke ~76 ms,
on-NPU trimmed lm-head ~54 ms, host ≈ 0). Full-vocabulary head is
weight-bandwidth-bound (~254 ms/token); the trimmed head is what fits
body + head in XRAM together (403 MB < 512 MB).

## Notes and gotchas

- **Tied operands matter.** Without `--torq-enable-annotate-tied-operands`
  the driver still works (copy-back fallback) but pays per-token feedback
  copies.
- **XRAM capacity.** Body + head arenas must each fit; a full-vocab head
  beside a 195 MB body exceeds 512 MB and forces evictions.
- **Greedy collapse.** With a bare seed token and no prompt, greedy decode
  can repeat a token forever — that is the model, not the driver. Use
  `--prompt-tokens` for context.
- The lm-head vmfb name must start with `lm_head` to be discovered.
