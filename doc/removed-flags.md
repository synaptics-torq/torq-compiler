# Removed flags — review reference

> It documents 13 flag removed in PR https://github.com/synaptics-torq/torq-compiler-dev/pull/2511, described *as if we had
> decided to keep it* (what it does, its default) plus the evidence for
> removing it, so each deletion can be judged on its merits. The flags the
> review asked us to **keep** are documented in
> `doc/dev-manual/debug_tips.md` ("Torq Compiler Flags and Options").
>

## Summary

| flag | default | status |
|---|---|---|
| `--torq-act-based-add` | off | experimental alternative lowering; open question on ACT vs ALU speed |
| `--torq-act-based-sub` | off | same |
| `--torq-convert-table-to-gather` | off | workaround for a TableOp bug; open question whether the bug is fixed |
| `--torq-convert-mul-i32-to-i16` | off | force-override of the i16 range check |
| `--torq-ews-relax-input-stride-check` | off | reviewer: "OK to go now" |
| `--torq-enable-deqw-verify` | **on** | default-on check, now unconditional (no behavior change) |
| `--torq-dw-ndl` | off | reviewer: "ok to remove now" (EK is the only live DW path) |
| `--torq-dump-descriptors-dir` | "" (off) | reviewer: "can be removed (and the whole dump_path logic too)" |
| `--torq-failure-driven-recovery` | **on** | default-on recovery, now unconditional (no behavior change) |
| `--torq-tile-and-fuse-distance-limit` | 0 (no limit) | tiling-split debug knob; open question |
| `--torq-disable-fit-shrink-reorder` | off | legacy-shrink fallback switch; open question |
| `--torq-disable-eliminate-redundant-lram-loads` | off | pass disable switch, referenced nowhere |
| `--torq-css-host` | off | build CSS programs as host binaries; sibling `-create-css-symbols` kept per review, this one not mentioned |

## Never-enabled experimental strategy flags

**`--torq-act-based-add`** (default off)
If kept: lower i32 elementwise add through the ACT unit
(`torq_hl ElementWiseBinaryOp::ADD`) instead of the ALU-based
`torq_hl AddOp` (the ALU pattern declines to match when the flag is on).
Removed: default-off, referenced by no test; the `ElementwiseOpEnum::ADD`
value and its HW branches (legacy NDL chain and EK `opTypeToAluOpMode`)
went with it. *Open question (@Meet-synaptics): ACT may be faster for some
shapes — there is no captured evidence either way in the tree.*

**`--torq-act-based-sub`** (default off)
If kept: same for i32 elementwise subtract (`ElementWiseBinaryOp::SUB`
vs `torq_hl SubOp`).
Removed: same as above, with `ElementwiseOpEnum::SUB`. Same open question.

**`--torq-convert-table-to-gather`** (default off)
If kept: alternative i8 table (LUT) lowering that emits a rotated gather
instead of the normal TOSA table path — a workaround for a bug in
`TableOp` (`rewriteTableAsGather` in ExtractPattern).
Removed: default-off, referenced by no test. *Open question
(@mdenna-synaptics / @xshang-synaptics): if the underlying TableOp bug is
fixed the workaround has no reason to exist; if a model still needs this
path it should be restored.*

**`--torq-convert-mul-i32-to-i16`** (default off)
If kept: force-convert a rescale-i32 + mul-i32 pair to the i16 path even
when the `isRescaleSafeForI16` range check rejects it (CastI32MulPattern).
Removed: the override only mattered while the range check was too
conservative. Reviewer: "might be OK to remove now". Restore it if
a model still trips the range check.

**`--torq-ews-relax-input-stride-check`** (default off)
If kept: relax the ElementWiseShift input-stride check in the HW lowering
(ElementWiseShiftPattern skips stride validation).
Removed: reviewer: "I think this is OK to go now".

**`--torq-enable-deqw-verify`** (default **on**)
If kept: toggle the DEQW bound checking in `SliceTaskOp::verify` — that
each DEQW NDL's reads/writes stay inside its corresponding Q buffer.
Removed: on by default, so removal makes the check unconditional; no
behavior change.

**`--torq-dw-ndl`** (default off)
If kept: force depthwise conv2d onto the legacy NDL kernel instead of the
EK kernel (`hasEkLoweringConv` returns false, printing "Warning: lowering
DW using NDL kernel!").
Removed: reviewer: "ok to remove now" — the EK lowering is the only live
depthwise path.

## Descriptor dump

**`--torq-dump-descriptors-dir=<DIR>`** (default "" = off)
If kept: dump the Torq API segment descriptors under
`DIR/job<N>/` while serializing, and append the list of extra segments to
load into `DIR/job0/tv.init.mem.lst`.
Removed: reviewer: "this can be removed (and the whole dump_path logic
too)". The now-dead `_dump_path` plumbing (Serializer member + ctor arg,
`dumpSegmentDescriptor` and its 7 call sites, the mem-lst writer, the
dump-id counters) was removed too — it was only reachable through
this flag.

## Failure-driven LRAM recovery

**`--torq-failure-driven-recovery`** (default **on**)
If kept: in a function whose LRAM allocation would fail, the
`RecoverLramResidencyPass` reverts load-CSE reuses and narrows read-only
weight loads to reduce peak residency. Functions that already fit are left
untouched (both transforms would split one shared load into more
concurrent loads and can make allocation worse).
Removed: on by default and referenced by no test, script or doc; removal
makes the recovery unconditional. No behavior change.

## Codegen debug flags (not restored by review)

**`--torq-tile-and-fuse-distance-limit=<N>`** (default 0 = no limit)
If kept: cap how far apart (in the op list) a producer and consumer may
be and still be fused/tiled together in TileAndFuse — a debug knob to
force a particular tiling split.
Removed: referenced by no test, script or doc. *Open question
(@fshaked): may still be useful for debugging — if so it can be
re-added trivially.*

**`--torq-disable-fit-shrink-reorder`** (default off)
If kept: disable the memory-aware domain ordering in
`fitTileToMemory`'s shrink pass, falling back to the legacy
`tilingOrder`-based shrinking.
Removed: same situation as the distance limit — a debug switch, never
referenced. Same open question.

**`--torq-disable-eliminate-redundant-lram-loads`** (default off)
If kept: disable the `EliminateRedundantLramLoads` pass.
Removed: pass-disable switch referenced by no test, script or doc.

**`--torq-css-host`** (default off)
If kept: build CSS programs as host binaries — force `hostBuild` for CSS
variant targets so they compile/link for the host CPU (triple `native` or
`--torq-target-host-*`) instead of the CSS target — i.e. the CSS kernel
would run on the local machine, e.g. for debugging.
Removed: its sibling `--torq-create-css-symbols` (dump each CSS program as
a standalone `.elf`) was explicitly asked to keep in the review, but
`--torq-css-host` was not mentioned, so it stays removed.
