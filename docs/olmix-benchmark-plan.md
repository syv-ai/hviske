# Olmix benchmark and mixture-optimisation plan

## Goal

Find a reproducible static mixture for Hviske's existing offline dataset sampler. The
method follows OlmixBase: train small proxy models on sampled domain mixtures, fit a
per-task log-linear performance surface, and optimise that surface under a natural-mix
prior and finite-data constraints.

This plan starts only after the private segmented P1 dataset described in
[`p1-segmentation-plan.md`](p1-segmentation-plan.md) is published, validated, and
pinned. No benchmark may silently remove or replace P1.

The implemented three-anchor, two-model matrix is a calibration experiment. It chooses
one proxy architecture and the shortest reliable training budget. It is not the full
Olmix experiment.

## Existing calibration matrix

The repository contains these anchors:

- `olmix_baseline`: the current production mixture;
- `olmix_read_speech_heavy`: more read and parliamentary speech;
- `olmix_spontaneous_heavy`: more broadcast, conversational, and meeting speech.

Each anchor preserves 60% Danish and 40% English. It is trained with both:

- `openai/whisper-tiny`;
- `syvai/hviske-v5-tiny` at
  `361051e8ed732798d68fcd5d5ec64fd4e39da40b`.

`src/scripts/run_olmix_benchmark.py` runs two-step smokes and then the serial six-job
matrix. It evaluates at global steps 250, 500, 1,000, 2,000, and 3,000 on deterministic
500-example validation subsets. Intermediate evaluations are retained as JSONL
metrics; only the final model checkpoint is retained temporarily.

## Six optimisation domains

Optimise these six groups rather than 16 independent sources:

| Group | Sources | Current baseline mass |
| --- | --- | ---: |
| Danish read or prepared | P1, CoRal read-aloud, FTSpeech, Nota, NST | 0.27 |
| Danish broadcast or conversation | DRTV, Danish YouTube, CoRal conversation | 0.27 |
| Danish parliament | VoxPopuli Danish | 0.06 |
| English conversation or meeting | People's Speech, AMI SDM, AMI IHM | 0.23 |
| English parliament | VoxPopuli English | 0.09 |
| English read speech | Three LibriSpeech splits | 0.08 |

The current production baseline is:

```text
p_base = [0.27, 0.27, 0.06, 0.23, 0.09, 0.08]
```

This grouping follows the style taxonomy encoded in the existing calibration anchors,
including their treatment of P1 as prepared speech. All three anchors preserve the
baseline proportions inside each of these six groups.

This is not automatically Olmix's natural prior. After P1 publication, derive the
natural vector from usable post-filter examples in each group:

```text
p0_j = N_j / sum(N_l for l in groups)
```

The paper defines its natural vector and finite-data limits in tokens. Hviske samples
examples, so example counts are the primary operational analogue. Also retain audio
seconds and duration distributions to quantify the effect of variable clip lengths.

For a group weight `p_j`, expand its source weights using the fixed conditional
proportions from `config/sparkie_bilingual.yaml`:

```text
w_s = p_j * w_base_s / sum(w_base_t for t in group j)
```

This keeps source allocation within a domain auditable and reduces the fitted surface
to six dimensions. Changing the group membership or within-group proportions creates
a new experiment rather than a continuation of an old one.

Do not constrain every sampled mixture to 60% Danish and 40% English. That would leave
only five effective dimensions and prevent the fit from measuring language-allocation
trade-offs. The measured natural vector and KL penalty keep optimisation grounded,
while explicit lower bounds ensure both languages remain trainable.

## Evaluation objective

Use the five pinned validation tasks already configured in
`config/sparkie_bilingual.yaml`:

- CoRal read-aloud, Danish;
- CoRal conversation, Danish;
- LibriSpeech clean, English;
- LibriSpeech other, English;
- FLEURS `en_us`, English.

Retain WER for every task. Define the optimisation objective before viewing swarm
results:

- Danish contributes 50% of the objective, split equally between its two tasks.
- English contributes 50%, split equally between its three tasks.
- Each task value is its predicted WER, not a pooled transcript-level WER.

Thus the task weights are `0.25, 0.25, 1/6, 1/6, 1/6`. Also report unweighted macro
WER, per-language macro WER, and every individual task. The fixed language-balanced
objective prevents the larger number of English tasks from deciding the result.

Validation examples, order, preprocessing, decoding parameters, and revisions must be
identical across mixtures and models. Store the stable example identifiers or a digest
of the deterministic selection, not only the seed.

## Phase 0: P1 and environment gate

Before any smoke:

- pin the final private `syvai/p1-segments` revision;
- verify the repository is private and accessible inside the Sparkie container;
- stream, decode, filter, and collate at least one real P1 segment;
- verify all 16 source streams yield post-filter training examples;
- record the exact Hviske commit and `uv.lock` digest;
- check free disk, GPU memory, existing GPU processes, container health, and output
  permissions;
- confirm no unrelated workload will be interrupted;
- confirm the output directory contains no colliding run IDs.

Use Hugging Face's stored authentication. Never print tokens, `.env`, request headers,
or credential-bearing URLs.

**Gate:** the strengthened preflight and all 33 benchmark-focused tests pass inside the
same container used for training.

## Phase 1: serial two-step smokes

Run one two-step job for each proxy model. A smoke must exercise:

- all 16 source configurations, including segmented P1;
- weighted interleaving;
- per-example Danish and English prompt construction;
- variable-length feature padding;
- forward and backward passes;
- one evaluation event;
- metrics-file creation;
- final-checkpoint retention and cleanup.

Run smokes serially in a visible tmux session named `hviske-olmix-smoke`. Use separate
run directories and capture metadata-only logs.

**Gate:** both smokes finish with finite loss and metrics, and at least one genuine P1
example reaches a training batch. A zero-P1 fallback is a hard failure.

## Phase 2: anchor calibration

Run the six calibration jobs serially in a visible tmux session named
`hviske-olmix-calibration`. Fix seed `4242` and preserve the existing evaluation
checkpoints.

For each model, anchor, and checkpoint, retain:

- per-task WER and language-balanced objective;
- stable evaluation-example IDs and per-example `S`, `D`, `I`, and `H` edit counts;
- global step, examples exposed, and audio seconds consumed when available;
- train and evaluation loss;
- wall-clock train and evaluation time;
- peak GPU memory and GPU utilisation samples;
- exact model, dataset, code, configuration, and validation-selection revisions;
- process exit status and the final checkpoint path.

Do not retain reference or hypothesis text in bootstrap artefacts. The IDs and edit
counts are sufficient to recompute aggregate WER and paired differences.

### Choose the earliest reliable budget

After all six seed-`4242` jobs finish, find a preliminary step `t` where:

1. each model's three-anchor ranking at `t` matches its own ranking at step 3,000;
2. the two models agree on the best anchor at `t`;
3. every claimed pairwise ordering has a 95% confidence interval excluding zero;
4. the same result holds at the next scheduled checkpoint;
5. no individual task has a severe late reversal hidden by the aggregate objective.

Compute 10,000 deterministic paired-bootstrap resamples within each task from stable
example IDs and `S`, `D`, `I`, and `H` counts. Recompute each task WER as
`sum(S + D + I) / sum(S + D + H)`, then recompute the language-balanced objective.
Treat a pair as tied when the confidence interval for its difference includes zero.
Rankings match only when their significant pairwise relations and tie relations match.
A severe late reversal moves an anchor from best to worst on a task by more than `0.01`
absolute WER.

Run seeds `4243` and `4244` for all three anchors and both models through preliminary
step `t`. The final budget is eligible only when all three seeds preserve the same best
anchor and the pooled training-seed standard deviation is below half the smallest
significant anchor margin. Validation bootstrapping and training-seed replication
measure different uncertainty and must both be reported.

Choose the earliest eligible checkpoint. If none is eligible, extend calibration to a
predeclared later checkpoint and repeat the seed gate rather than selecting step 3,000
by convenience.

The current exposure is:

| Steps | Training examples |
| ---: | ---: |
| 500 | 128,000 |
| 1,000 | 256,000 |
| 2,000 | 512,000 |
| 3,000 | 768,000 |

### Choose the swarm proxy

Among models meeting the three-seed stability gate, choose the one with the lowest
median seconds per training example across seeds. Prefer `syvai/hviske-v5-tiny` when
runtime is close because it is architecturally nearer the 2.06B target. Prefer Whisper
Tiny when it is materially faster and produces the same stable ordering.

Do not launch the swarm when the proxies disagree on the best anchor or show materially
different per-task trade-offs. In that case, add diagnostic anchors or a later
checkpoint until the disagreement is understood.

**Gate:** a checked-in calibration report fixes the chosen proxy, training budget,
objective, and measured runtime estimate for the full swarm.

## Phase 3: deterministic mixture swarm

Generate 21 fitting mixtures and 7 independent held-out mixtures. The fitting count is
Olmix's recommendation of at least `3 * (m + 1)` for `m = 6` domains.

Build the fitting set with seed `4242`:

1. include the three existing anchor vectors and measured natural vector `p0`;
2. remove an exact duplicate if `p0` equals an anchor;
3. sample symmetric `Dirichlet(alpha = 1)` vectors until the set contains 21 mixtures;
4. set sampled components below `0.05` to zero and renormalise;
5. reject vectors with Danish or English mass below `0.15`;
6. reject a duplicate after rounding to eight decimal places;
7. redraw until `[1, p_1, ..., p_5]` has full column rank and a 2-norm condition
   number no greater than `100`.

This is a sparse swarm under Olmix's clipping definition. The paper does not specify
its Dirichlet concentration, so symmetric `alpha = 1` is a local, predeclared design
choice rather than a reproduced paper parameter. Store the raw and clipped vectors.

Generate the seven held-out vectors independently with seed `4245` and the same rules.
Lock their manifests before training, but exclude their metrics from fitting and model
selection. The paper used 30 held-out mixtures and repeated its methodology analysis
with three seeds. This smaller hold-out is a compute-saving adaptation; failed
validation adds another predeclared batch of seven rather than weakening the gate.

Reuse the chosen proxy's three calibrated anchor results at the selected checkpoint.
Average their three seed results for the initial fit. Train every other fitting and
held-out vector once with training seed `4242`; mixture-generation seeds remain
separate metadata. Phase 5 defines the preliminary fit, variance estimate, deterministic
seed-escalation decision, and final refit.

Expand each group vector to the 16 source probabilities with the fixed conditional
mapping. Validate non-negative values, sum-to-one tolerance, source order, language
mass, and a positive post-filter sample from every source with non-zero weight.

Write one immutable mixture manifest before training. It includes:

- experiment ID and seed;
- six-domain order and source membership;
- baseline, natural, raw, clipped, and expanded vectors;
- fitting or held-out role;
- selected proxy and step budget;
- source and evaluation revisions;
- source-size estimates after filtering;
- Hviske commit and lock digest;
- objective and constraint definitions.

Review this manifest without performance results. Once training starts, do not replace
failed or unattractive vectors. Retry the same vector and seed; record permanent
failures explicitly.

## Phase 4: serial swarm execution

Reuse the three selected-checkpoint anchor results and run the remaining 25 fitting and
held-out jobs serially in a visible tmux session named `hviske-olmix-swarm`. The
launcher must stop after a failed job unless that exact run can be resumed safely. Each
output path includes the experiment ID, mixture ID, model, training seed, and fitting
or held-out role.

Retain for every run:

- mixture manifest and fully resolved Hydra configuration;
- configured and realised examples and audio seconds per domain;
- per-task JSONL metrics at global step;
- runtime, resource, and failure metadata;
- final aggregate report.

Delete proxy checkpoints after their metric and metadata artefacts pass schema and
completeness checks. Do not delete the checkpoint of an incomplete or unparsed run.
No proxy model is uploaded to the Hub.

Estimate total wall time from the measured calibration throughput plus 20% operational
headroom. Before launch, check that Sparkie can remain available for that interval.

**Gate:** every successful mixture has all five per-task metrics and matching exposure.
Realised domain shares must be within one percentage point of their requested shares.
A larger difference blocks fitting until the sampler is corrected or its actuation is
modelled explicitly. The fit must not treat a failed run as a poor score.

## Phase 5: fit Olmix response surfaces

For each evaluation task `i`, test the OlmixBase form:

```text
f_i(p) = c_i + exp(A_i^T p)
```

Olmix validated this form for language-model BPB. Its suitability for ASR WER is a
hypothesis, not an established transfer. Express WER as a non-negative ratio rather
than a percentage, and never clip values above `1`; insertions can produce WER above
100%. Fit with bounded nonlinear least squares and multiple deterministic
initialisations. Weight observations by bootstrap uncertainty when available.

Fit `c_i` and all six entries of `A_i`, without adding another intercept inside the
exponential. An extra intercept would be redundant because the simplex components sum
to one. Bound `c_i` below the smallest observed WER with a small fixed epsilon. Permit
a slightly negative lower bound when an observed WER is zero, then reject any fit that
predicts negative WER in the feasible region.

Fit only the 21 designated fitting mixtures. First produce a preliminary fit from the
initial runs. For each task and the language-balanced objective, estimate pooled
within-anchor training variance from the three anchors and their three seeds:

```text
sigma_train^2 = sum_a sum_s (y_as - mean_s(y_as))^2 / sum_a (n_a - 1)
```

Optimise the preliminary surface once and define `delta_pre` as its best predicted WER
improvement over `p_base`. Train every single-seed fitting and held-out vector again
with seed `4244` when `delta_pre <= 0` or the objective's `sigma_train` exceeds
`delta_pre / 2`. Then discard the preliminary decision surface and refit from all
designated fitting results. Record both paths even when escalation is not triggered.

Assess the final surface on both leave-one-mixture-out predictions and the seven
untouched held-out mixtures:

- mean and maximum absolute prediction error per task;
- rank correlation between predicted and measured WER;
- residuals against every domain weight;
- residuals against fitted values and run order;
- parameter stability across bootstrap refits;
- comparison with constant, linear, and alternative nonlinear baselines;
- sensitivity to each fitting mixture.

Use a hierarchical bootstrap for every fit, diagnostic, and optimised vector. In each
of 10,000 replicates, paired-resample shared evaluation IDs within each task. Resample
available training seeds within each mixture. For a single-seed mixture, add a
zero-centred residual drawn from the pooled within-anchor training residuals. Clamp a
negative perturbed WER to zero but never cap its upper value. This propagates validation
and training-run uncertainty separately.

Predeclare acceptance targets in the implementation report. At minimum, the Olmix fit
must beat the constant and linear baselines on held-out error and preserve useful
ranking information. Target held-out Spearman correlation of at least `0.8` per task
and investigate any error above `0.02` absolute WER.

If diagnostics fail, add the next seed-defined batch of seven fitting mixtures and run
a new independent hold-out batch. Do not optimise a visibly misspecified surface or
move held-out observations into the fit without replacing them.

## Phase 6: exact constrained optimisation

Minimise the language-balanced predicted objective over a predeclared KL grid:

```text
sum_i task_weight_i * f_i(p) + lambda * KL(p || p0)
```

Use WER fractions and test `lambda` values `0`, `0.01`, `0.025`, `0.05`, and `0.10`.
Olmix found `0.05` best for transfer in its BPB experiments, but that numerical scale
is not automatically portable to WER. Treat it as one candidate, not the default
answer.

Subject to:

```text
p_j >= 0
sum_j p_j = 1
Danish mass >= 0.15
English mass >= 0.15
p_j <= k * N_j / R
```

Olmix defines `N_j` and `R` in tokens. This benchmark adapts them to usable examples
because Hviske's configured probabilities select examples. Here `N_j` is the number of
usable post-filter examples in group `j`, and `R` is the target number of example
draws. Derive both from pinned revisions and the exact training filter. Also calculate
unique and requested audio seconds. Enforce the same bound for each expanded source
and reject a solution when requested audio seconds divided by unique audio seconds
exceeds `1.1 * k` for any source.

Choose the smallest integer `k` that makes `p_base` feasible for the target exposure.
This preserves the repetition already implied by the production baseline while
preventing the optimiser from exploiting a small domain much further. Record `k`,
`N_j`, `R`, audio-second exposure, and every active bound.

`interleave_datasets(..., stopping_strategy="all_exhausted")` can make realised shares
differ from requested probabilities. Instrument actual domain draws and fit on realised
shares. If the one-percentage-point gate fails, correct the sampler before applying the
constraint or selected configuration.

Solve the constrained problem directly from multiple starts, including `p0`, every
observed swarm vector, and deterministic random starts. Report objective values,
convergence status, gradients or residual optimality, active constraints, and agreement
between starts.

Bootstrap the complete fit-and-optimise procedure. Report the selected vector's
uncertainty, not only one point estimate. Also optimise these sensitivity variants:

- every predeclared KL coefficient;
- each leave-one-mixture-out fit;
- task weights with one task omitted;
- plausible source-count uncertainty.

Reject a brittle solution whose dominant domain allocation changes radically under
small perturbations. A KL candidate is eligible only when its 95% bootstrap lower bound
for WER improvement over `p_base` is positive, neither language regresses by more than
`0.01`, and every group's bootstrap interquartile range is at most `0.10`.

Select the eligible candidate with the greatest lower-bound WER improvement. If
candidates differ by less than `0.001` absolute WER, choose the larger `lambda`, which
stays closer to the natural prior. Held-out mixtures may validate the response model
but may not choose `lambda`. If no candidate is eligible, stop without a mixture.

Expand the selected group vector back to all 16 sources and commit a named Hydra
configuration. Keep full decimal precision in the manifest; round only the displayed
report.

## Phase 7: measured confirmation

Train the chosen proxy on four vectors:

- the current production baseline `p_base`;
- the measured natural vector `p0`;
- the exact selected mixture;
- one stable alternative from the sensitivity analysis.

Run each vector with fresh training seeds `4246` and `4247`. Compare measured WER with
predictions, paired evaluation bootstrap intervals, and between-seed variation.

**Gate:** the selected mixture improves the language-balanced objective over `p_base`
without a material regression on either language. A material language regression is
more than `0.01` absolute macro WER unless product requirements set a stricter limit.
Improvement over `p0` is reported but is not a substitute for beating production.

A failed confirmation invalidates the selected surface. Add the confirmation points to
the design, refit, and repeat rather than hand-adjusting weights.

## Phase 8: transfer to the full Cohere model

Proxy ranking is evidence, not proof of transfer to
`CohereLabs/cohere-transcribe-03-2026`. Run a matched target-model comparison between
`p_base` and the selected mixture with identical initial weights, seed, exposure,
evaluation subsets, and decoding.

Use the shortest target-model budget that includes the proxy-selected stable exposure.
If the production run is substantially longer, confirm the decision again at a later
production checkpoint before committing the remaining compute.

Adopt the mixture only when the target model improves the predeclared objective and
does not cross the language-regression guardrail. Otherwise retain `p_base` and report
the proxy-to-target transfer failure.

## Required implementation

After P1 integration, add:

- a checked-in six-domain source map;
- a deterministic swarm-manifest generator;
- a serial tmux-friendly swarm launcher;
- per-domain example and audio-second counters in training;
- per-example evaluation IDs and `S`, `D`, `I`, and `H` counts without transcript text;
- metric-schema and run-completeness validation;
- surface fitting with held-out and cross-validation diagnostics;
- exact constrained optimisation and bootstrap sensitivity analysis;
- a report generator that never depends on retained checkpoints;
- tests for group expansion, clipping, rank, objective weights, repetition bounds,
  fitting recovery on synthetic data, and optimiser feasibility.

Keep training, fitting, and optimisation as separate commands. A completed training
matrix must be reusable without GPU access.

## Reproducibility artefacts

Retain:

- final P1 and all other dataset revisions;
- model and remote-code revisions;
- Hviske commit and `uv.lock` digest;
- container image identity and GPU metadata;
- calibration report and selection rule output;
- raw and expanded swarm manifests;
- deterministic validation-selection digests;
- per-task metrics and bootstrap summaries;
- fit parameters, diagnostics, constraints, and optimiser traces;
- selected and sensitivity mixture configurations;
- measured proxy and target confirmation reports.

Discard proxy checkpoints after metrics are validated. Generated caches, credentials,
and temporary audio must never enter the repository.

## Primary reference

- [Olmix paper](https://arxiv.org/abs/2602.12237)
