# Phase 4 Write-Up: Long-Horizon Latent Communication

## Scope

This phase tested a narrow systems question for code-executing data-analysis agents: can natural-language agent-to-agent coordination be replaced by hidden-state / KV-cache communication while preserving accuracy and reducing decoded-token compute?

The benchmark remains a **multi-stage planning-coordination horizon** with one code execution at the end. It is not a scientific-discovery benchmark, not a per-stage `execute -> observe -> continue` benchmark, and not a trained latent-module experiment.

## Claim 1: The 7-Stage Collapse Reproduced And Was Attributable

The old long-horizon latent path was rerun exactly as `C1_phase3_exact`, with diagnostics added around the generation path rather than reimplementing it. The collapse reproduced:

| Variant | Runs | Final pass | Wilson CI | Median cache_len_at_decode | Readout |
|---|---:|---:|---|---:|---|
| `C1_phase3_exact` | 15 | 3/15 = 0.200 | [0.070, 0.452] | 2828 | Collapse reproduced |
| `C2_dedup` | 15 | 11/15 = 0.733 | [0.480, 0.891] | 532 | Clean-cache latent recovered |
| `B_textmas` | 15 | 12/15 = 0.800 | [0.548, 0.930] | 0 | Text baseline held |

`C2_dedup` beat `C1_phase3_exact` by +0.533 final pass rate; Fisher exact p=0.0092. This supports the attribution that the original 7-stage collapse was largely an implementation artifact: the latent path repeatedly re-encoded chat-templated copies of the full task into the KV cache before coder decode.

At 7 stages, clean-cache latent (`C2_dedup`) was statistically indistinguishable from the text baseline while using zero decoded coordination tokens.

### Repair Fixes Crashes, Not Contract Misses

The Version #8 xlong artifacts also exposed a repair-rescue asymmetry. First-attempt failures in `C1_phase3_exact` were mostly runtime crashes; first-attempt failures in `C2_dedup` were mostly valid Python that missed strict scorer contracts.

| Variant | First attempt passed | Runtime first-attempt failures | Semantic/scorer first-attempt failures | Repaired to final pass |
|---|---:|---:|---:|---:|
| `C1_phase3_exact` xlong | 1/15 | 8/15 | 6/15 | 5 runtime failures rescued |
| `C2_dedup` xlong | 1/15 | 4/15 | 10/15 | 0 failures rescued |

The frozen `text_reset` repair path is good at fixing concrete tracebacks, such as missing imports or serialization crashes. It did not fix valid-but-wrong code that had already chosen the wrong output contract or formula. That matters for interpreting xlong: repair cannot compensate for ambiguous task/spec alignment.

## Claim 2: Latent-Step Contribution Was Directional, Not Confirmed

Session 3 pooled `C2_dedup`, `C3_no_latent`, and `C5_anchor` to n=30 each under identical generation-path hashes.

| Variant | Runs | Final pass | Wilson CI | First-attempt pass | Readout |
|---|---:|---:|---|---:|---|
| `C2_dedup` | 30 | 25/30 = 0.833 | [0.664, 0.927] | 0.733 | Best observed clean latent path |
| `C3_no_latent` | 30 | 19/30 = 0.633 | [0.455, 0.781] | not significantly lower | No confirmed latent-step contribution |
| `C5_anchor` | 30 | 17/30 = 0.567 | [0.392, 0.726] | lower than C2 | Greedy anchors harmed |

`C2_dedup` remained directionally better than `C3_no_latent`, but the pooled test did not confirm the latent-step contribution at this sample size: Fisher p=0.1432. The correct wording is “no detectable latent-step contribution at this sample size,” not equivalence.

`C5_anchor` significantly harmed performance versus `C2_dedup` (Fisher p=0.0470). This was not a general grounding-channel result: the greedy <=24-token anchors were deterministic within each family, often parroted duplicated/truncated stage text, and in the orders family injected the wrong formula `net_revenue = units * (unit_price - discount_amount)`. All 5 orders C5 runs failed.

## Claim 3: Cache Pollution Shows A Dose-Response, But Composition Matters

Among variants that used latent steps, final pass rate fell as duplicated decoded/chat-templated cache text increased:

| Variant | Latent steps? | Median cache_len_at_decode | Final pass |
|---|---:|---:|---:|
| `C2_dedup` | yes | 532 | 0.833 |
| `C5_anchor` | yes | 1258 | 0.567 |
| `C1_phase3_exact` | yes | 2828 | 0.200 |
| `C3_no_latent` | no | 504 | 0.633 |

<svg xmlns="http://www.w3.org/2000/svg" width="620" height="310" viewBox="0 0 620 310" role="img" aria-label="Final pass rate versus median cache length for Phase 4 latent variants">
  <rect x="0" y="0" width="620" height="310" fill="white"/>
  <line x1="70" y1="250" x2="570" y2="250" stroke="#444" stroke-width="1.5"/>
  <line x1="70" y1="40" x2="70" y2="250" stroke="#444" stroke-width="1.5"/>
  <text x="300" y="295" text-anchor="middle" font-family="Arial, sans-serif" font-size="13">Median cache_len_at_decode</text>
  <text x="18" y="150" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" transform="rotate(-90 18 150)">Final pass rate</text>
  <text x="70" y="270" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">500</text>
  <text x="254" y="270" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">1250</text>
  <text x="438" y="270" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">2000</text>
  <text x="561" y="270" text-anchor="middle" font-family="Arial, sans-serif" font-size="11">3000</text>
  <text x="50" y="250" text-anchor="end" font-family="Arial, sans-serif" font-size="11">0.0</text>
  <text x="50" y="145" text-anchor="end" font-family="Arial, sans-serif" font-size="11">0.5</text>
  <text x="50" y="40" text-anchor="end" font-family="Arial, sans-serif" font-size="11">1.0</text>
  <polyline points="78,75 256,131 563,208" fill="none" stroke="#1f77b4" stroke-width="2.5"/>
  <circle cx="78" cy="75" r="6" fill="#1f77b4"/><text x="90" y="70" font-family="Arial, sans-serif" font-size="12">C2 0.833</text>
  <circle cx="256" cy="131" r="6" fill="#1f77b4"/><text x="268" y="126" font-family="Arial, sans-serif" font-size="12">C5 0.567</text>
  <circle cx="563" cy="208" r="6" fill="#1f77b4"/><text x="470" y="203" font-family="Arial, sans-serif" font-size="12">C1 0.200</text>
  <circle cx="70" cy="117" r="6" fill="#d62728"/><text x="84" y="118" font-family="Arial, sans-serif" font-size="12">C3 off-curve 0.633</text>
</svg>

The dose-response is strongest among latent-step variants. `C3_no_latent` sits off the curve: it has a short cache but lower accuracy than `C2_dedup`, so duplicated-text dose drives much of the harm, but cache composition and latent updates also matter.

## Phase 4C Horizon Ceiling

Phase 4C attempted to extend the clean-cache question from 7 stages to 9 stages (`xlong`). The pre-registered gate required the single-agent control to pass at least 13/15 before interpreting B-vs-C gaps. That gate did not clear.

| Run | Commit | Rows | A result | B result | Decision |
|---|---|---:|---:|---:|---|
| Full xlong attempt | `a3e4457` | 60 | 4/15 = 0.267, CI [0.109, 0.520] | 6/15 = 0.400 | A gate failed; no mode-gap interpretation |
| Smoke after contract clarification | `0d266b5` | 6 | 2/3 | 2/3 | Sensor still failed |
| Smoke after formula clarification | `9a169a9` | 6 | 2/3 | 3/3 | Sensor still failed for A |
| Sensor-only smoke | `9fb180b` | 2 | 0/1 | 0/1 | Formula wording induced invalid pandas code |
| Sensor-only smoke | `40d6c46` | 2 | 0/1 | 1/1 | A still wrote site-hour output to the site-summary file |
| P100 guard check | `db55d7d` | 0 | not run | not run | Kaggle assigned P100; GPU guard rejected before model/rows |
| Bounded re-authoring A gate | `db55d7d` | 15 | 9/15 = 0.600, CI [0.357, 0.802] | not run | A gate failed after the one allowed re-authoring pass; xlong closed |

Every imported zip was audited for caches, weights, and token-like secrets. The hidden-signal smoke and latent tool-roundtrip passed in the smoke sessions. The generation-path hash for the xlong attempts was `6ce3d3c4492384d2`.

The final bounded re-authoring pass used mechanical scorer-spec alignment rules copied from the A-qualified 7-stage tasks, kept exactly 9 dependent stages, left scorers unchanged, and passed reference-script checks. It still failed the pre-registered A gate: `A_single` reached 9/15 final pass and 7/15 first-attempt pass. By family, campaign passed 5/5, orders passed 2/5, and sensor passed 2/5. Therefore the hard stop fired: no B/C xlong matrix, no xxlong branch, and no further xlong prompt-tuning loop.

The measured ceiling readout is therefore guarded: 9-stage xlong is not measurable for Qwen3-8B 4-bit under this protocol because the single-agent control does not qualify after the single allowed re-authoring pass. Phase 4C does **not** support a claim that clean-cache latent tracks text at 9 stages, nor that it collapses against text at 9 stages. The binding constraint is benchmark construction/model capability at 9 stages, not the coordination channel.

### Appendix: Non-Citable Xlong Diagnostics

Versions #8-#12 are useful for task-forensics, but they are **non-citable as channel comparisons**. Version #8 failed the A-gate, and Versions #9-#12 were tiny diagnostic smokes after contract edits, not pre-registered full matrices.

| Version | Scope | Result | Why it is non-citable |
|---|---|---|---|
| #8 | 60-row xlong matrix | A_single 4/15, B_textmas 6/15, C1 6/15, C2 1/15 | A-gate failed before channel interpretation. |
| #9 | A/B xlong smoke, seed 17 | A 2/3, B 2/3 | Diagnostic after contract clarification. |
| #10 | A/B xlong smoke, seed 17 | A 2/3, B 3/3 | Diagnostic; B>A on same model indicated remaining task ambiguity. |
| #11 | Sensor-only smoke, seed 17 | A 0/1, B 0/1 | Code-like formula wording induced invalid pandas code. |
| #12 | Sensor-only smoke, seed 17 | A 0/1, B 1/1 | B>A on same model again indicated task ambiguity, not channel quality. |

The methodological lesson is that long-horizon benchmark tasks need scorer-spec alignment discipline before any latent-vs-text claim. A baseline ordering where B beats A on the same model is a warning sign: the task contract is probably being interpreted differently across prompts, so the benchmark is measuring ambiguity rather than coordination channel quality.

## Failure Classes

Long-horizon attribution failures were mostly valid Python with wrong task semantics or repair failures, not empty code:

| Variant | Passed | Runtime bugs | Semantic/scorer slips | Notes |
|---|---:|---:|---:|---|
| `C1_phase3_exact`, n=15 | 3 | 8 | 4 | Heavy duplicate prompt/cache pollution |
| `C2_dedup`, n=15 | 11 | 2 | 2 | Clean cache recovered most failures |
| `C2_dedup`, n=30 | 25 | 3 | 2 | Pooled confirmation |
| `C5_anchor`, n=30 | 17 | not primary | not primary | Harm linked to polluted/wrong anchor text |
| xlong A gate, Version #8 | 4 | mixed | high-partial semantic slips | Not interpretable as B-vs-C |
| xlong reauth A gate, Version #14 | 9 | 1 | 5 | Still failed A gate; B/C not run |

The repeated pattern is that exact output contracts matter sharply at small model scale. That is a benchmark-design constraint, not a latent-channel result.

## Plain-English Readout

The prototype found a real systems failure and a real systems fix. The apparent 7-stage collapse of training-free latent agent communication was not intrinsic: it was caused by polluted KV-cache construction, and deduplicating the latent cache recovered performance from 3/15 to 11/15 while preserving zero decoded coordination tokens. However, the bounded 9-stage ceiling retry still did not produce a valid latent-vs-text comparison because the single-agent control reached only 9/15 after the one allowed re-authoring pass. The honest current claim is: clean-cache latent communication is competitive with text through the measured 7-stage planning horizon, cache construction is a first-order design variable, and the 9-stage question is closed for this project phase because benchmark construction/model capability, not the coordination channel, is the binding constraint.

## Future Work

1. **Per-stage execute-observe-continue benchmark.** The current tasks execute once at the end. A stronger benchmark should execute after each stage and force latent memory to survive many `execute -> observe -> continue` boundaries. This is the better test of whether latent state carries operational information over long tool-using trajectories.

2. **Trained RecursiveMAS-style latent module.** Training is not part of Phase 4. It becomes reasonable only after the per-stage benchmark identifies what latent vectors need to carry. Candidate targets are replacing short decoded anchors, preserving tool observations, or repairing semantic drift.

Both follow-ups are gated on stronger GPU access. No 1.7B reruns, per-stage benchmark, or trained latent module were attempted in this phase.
