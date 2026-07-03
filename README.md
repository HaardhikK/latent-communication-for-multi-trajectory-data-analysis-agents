# Latent Communication for Multi-Trajectory Data Analysis Agents

This repository tests a narrow systems question: when several code-executing
data-analysis agents collaborate, do they need to exchange natural-language
messages, or can they coordinate through shared hidden-state / KV-cache memory
while preserving accuracy and reducing decoded-token compute?

The motivation is a Robin-style closed-loop discovery setting where multiple
analysis trajectories run independently and then reconcile their findings. Text
reports are dependable, but they require each trajectory to decode and re-read
large coordination messages. A latent channel could keep the cross-checking step
in representation space while still decoding code, tool calls, and final answers
when those are actually needed.

## What This Adds

This is not just a re-run of LatentMAS. LatentMAS demonstrates training-free
latent collaboration for reasoning tasks that decode once at the end. This
prototype adds:

1. A tool-using loop: agents write Python, execute it on CSV data, and feed the
   execution result back into the workflow.
2. A hybrid repair path: latent coordination on the happy path, with explicit
   text grounding for traceback/scorer feedback when generated code fails.
3. A horizon-scaling study over 3, 5, and 7 dependent data-analysis stages,
   measuring where latent coordination holds and where it breaks.

## Experimental Setup

- Local plumbing tier: `Qwen/Qwen3-1.7B` on a 6 GB laptop GPU.
- Free-cloud accuracy tier: `Qwen/Qwen3-8B` in 4-bit on a single T4 GPU.
- Tasks: deterministic CSV analysis pipelines across three families: marketing
  ROI, order KPIs, and sensor quality.
- Modes, always using the same model/backend within a comparison:
  - `A_single`: one agent writes the full script.
  - `B_textmas`: planner -> coder -> critic with text coordination.
  - `C_latentmas`: the same roles, with inter-role handoffs through latent
    hidden-state memory instead of decoded coordination text.

## Results

### Accuracy

Pass rate over the 8B free-cloud horizon sweep and the Phase 4A attribution run:

| Mode / variant | Short (3 stages) | Medium (5 stages) | Long (7 stages) |
|---|---:|---:|---:|
| A, single agent | 1.00 | 1.00 | 1.00 |
| B, text multi-agent | 0.93 | 0.87 | 0.80 |
| C, prior latent pipeline | 1.00 | 1.00 | 0.20 |
| C2, deduplicated latent cache | not rerun | not rerun | 0.73 |

### Efficiency

Median text-vs-latent coordination cost:

| Horizon | B coordination tokens | C coordination tokens | B model time | C model time |
|---|---:|---:|---:|---:|
| Short | about 979 | 0 | about 59 s | about 33 s |
| Medium | about 1768 | 0 | about 74 s | about 35 s |
| Long | about 2677 | 0 | about 104 s | about 103 s |

Short and medium tasks show the useful regime: latent coordination matches the
single-agent and text multi-agent baselines while eliminating decoded
coordination tokens and reducing model latency. The prior 7-stage latent result
looked like a sharp training-free latent collapse, but Phase 4A traced that
failure to an implementation artifact: the latent pipeline repeatedly appended
chat-templated copies of the full task prompt into the KV cache before code
decoding. The exact old path (`C1_phase3_exact`) reproduced the 3/15 long-horizon
result, while a deduplicated cache path (`C2_dedup`) reached 11/15, statistically
indistinguishable from the text baseline's 12/15 while still using zero decoded
coordination tokens.

The current evidence is therefore: cache construction matters strongly; latent
steps remain directionally helpful but were not statistically confirmed after
pooling to n=30. The tested greedy <=24-token anchor implementation is not a
general grounding result: it parroted duplicated/truncated stage text back into
the cache, acted as a medium pollution dose, and significantly harmed C5 versus
C2. In the orders task it even asserted the wrong net-revenue formula, and all
five orders C5 runs failed. See
[`reports/phase4a_findings.md`](reports/phase4a_findings.md) for the Phase 4A
tables, confidence intervals, and Fisher exact tests.

An initial 9-stage xlong attempt failed the pre-registered A_single gate, so it
is treated as a task-contract diagnostic rather than a latent-vs-text result.
The ambiguous xlong formulas were clarified while keeping the 9-stage structure.

## Setup

Use Python 3.10+ and install PyTorch for your CUDA environment first. For a
CUDA 12.4 cloud notebook, the tested stack used:

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
python -m pip install -r requirements.txt
```

The latent backend adapts mechanisms from the upstream
[LatentMAS](https://github.com/Gen-Verse/LatentMAS) repository (Apache-2.0).
Clone it outside this repository when running the upstream smoke test:

```bash
git clone https://github.com/Gen-Verse/LatentMAS third_party/LatentMAS
```

Runtime paths are environment-driven. Set `LATENT_AGENT_RUNTIME`, `HF_HOME`,
`HF_HUB_CACHE`, and `HF_DATASETS_CACHE` to keep model caches and run artifacts
outside the source checkout. If a Hugging Face token is needed, provide it as an
environment variable named `HF_TOKEN`; do not store it in files.

## Usage

Local environment and model smoke tests:

```bash
python scripts/check_env.py
python scripts/smoke_generate.py --model Qwen/Qwen3-1.7B
```

Text baselines and latent-mode development runs:

```bash
python scripts/run_phase1.py --mode both --repeat 1
python scripts/latent_hidden_smoke.py --model Qwen/Qwen3-1.7B
python scripts/latent_tool_roundtrip.py --model Qwen/Qwen3-1.7B
python scripts/run_phase3.py --mode all --horizons short,medium,long --repeat 5
python scripts/run_phase4.py --variants C1_current,C2_dedup,C3_no_latent --horizons long --repeat 2
```

Free-cloud 8B runs use 4-bit quantization and a single visible T4-class GPU:

```bash
python scripts/run_tier2_gate.py --model Qwen/Qwen3-8B --quantization 4bit
python scripts/run_tier2_full_sweep.py --model Qwen/Qwen3-8B --quantization 4bit
python scripts/run_tier2_phase4.py --model Qwen/Qwen3-8B --quantization 4bit
```

The full sweep runner supports checkpoint/resume with `--max-new-rows` and
`--resume-zip` for notebook sessions with time limits.

## Limitations and Future Work

This benchmark is a multi-stage planning-coordination horizon with one code
execution at the end. A stronger follow-up is a per-stage
`execute -> observe -> continue` horizon where latent memory must survive many
tool-call boundaries.

The immediate next step is to find the real horizon ceiling after fixing cache
construction: test 9- and 11-stage variants with `C2_dedup` against the text
baseline. If deduplicated latent coordination then collapses while text holds,
that becomes the evidence-backed target for a small RecursiveMAS-style latent
module or adapter. A larger coder model, such as an approximately 30B model in
4-bit, would also test whether the ceiling is partly model-scale dependent.
