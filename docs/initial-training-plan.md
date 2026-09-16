# Hviske v6.0 initial Danish ASR release training plan

## Status

The production consumer now loads `syvai/p1-segments` directly as a normal Hub dataset
with `audio` and `text`. Training remains gated until the manually gated dataset's final
immutable revision is supplied as `P1_SEGMENTS_REVISION` and the complete data preflight
passes; the revision must not be replaced with a moving branch or a provisional snapshot.

## Decision summary

Train and release `syvai/hviske-v6.0` as a Danish-specialised continuation of
`CohereLabs/cohere-transcribe-03-2026`. This is a proper release run, not a dummy or
throwaway baseline. The major version reflects the substantially larger and broader
training corpus compared with previous Hviske releases.

Use one architecture and one fixed data mixture. Do not spend the first training window
on an architecture bake-off or manual mixture tuning. Hviske v6.0 should establish the
released full-model baseline for the subsequent architecture and data-mixture
experiments.

The critical path is:

1. finish and pin the data;
2. remove leaderboard leakage and materialise a training-ready quality manifest;
3. run bounded data and model smokes on Sparkie;
4. launch the direct full run at the measured 200,000-step horizon;
5. release the earliest checkpoint that clears the quality gates;
6. run the official leaderboard harness once after checkpoint selection; and
7. publish the model, provenance, and raw benchmark outputs.

## Goal and success criteria

As of 14 September 2026, the published Danish ASR leaderboard contains 34 models. Its
current leader is the proprietary `syv-transcribe` model at 10.97% mean WER. The best
published open model is `thorh/whisper-large-v3-turbo-danish` at 11.45% mean WER.

The leaderboard target is moving. Let `W*` be the lowest published, independently
verified mean WER at the final benchmark freeze. Define the significant-win threshold
as:

```text
T = min(9.75, W* - 1.00, 0.90 * W*)
```

The initial training has three success levels:

- **Minimum useful release:** at least 10% relative mean-WER improvement over
  `syvai/hviske-v5` on the frozen five-domain development suite, with no material domain
  regression.
- **Leaderboard win:** official mean WER below `W*` when the candidate is submitted.
- **Significant win:** official mean WER at or below `T`, meaning at least 1.00 absolute
  WER point and 10% relative below the strongest verified model, while retaining the
  original 9.75% ceiling.

The moving threshold prevents a nominal win over the stale 10.97% result from satisfying
the plan after a stronger competitor appears.

Model selection must not use leaderboard test scores. Select the checkpoint on a frozen,
leakage-free development suite. Run the official test harness only after the checkpoint,
decoding settings, and model card draft are frozen.

If the model materially improves the earlier Hviske v5 releases but misses the current
leader, it may still be released honestly as v6.0. Do not claim state of the art. Record
the miss and use the frozen result as the full-model baseline for later architecture and
data-mixture experiments rather than tuning against the leaderboard test sets.

## Architecture decision

Use the 2.06B-parameter Conformer encoder-decoder model
`CohereLabs/cohere-transcribe-03-2026`, pinned to revision
`b1eacc2686a3d08ceaae5f24a88b1d519620bc09`.

### Why Conformer

- The current 10.97% leader is in the same 2.06B parameter class. A Danish continuation
  of a strong general ASR checkpoint is the most direct path to surpassing it.
- Cohere Transcribe is a purpose-built ASR model, rather than a general sequence model.
  Its card reports strong multilingual accuracy and efficient inference.
- The checkpoint is Apache-2.0, supported natively in Transformers and vLLM, and already
  pinned and implemented in Hviske.
- `config/sparkie_bilingual.yaml`, the Sparkie runbook, model loading, and training
  already target this exact checkpoint.
- Reusing that path avoids delaying the release for a second training stack and makes
  the released v6.0 checkpoint the natural full-model baseline for the experiments that
  follow.

Cohere's public card does not list Danish among its 14 pretrained languages. This makes
the Danish pilot a real gate: the architecture is selected because the repository and
full experiment path are ready for it, not because Danish quality is assumed.

### Why not Whisper for the primary run

`thorhojhus/whisper-large-v3-turbo-danish`, pinned to
`8cc640f726b08708193c38bd474ac1b31f44bd38`, is the strongest fallback. It is an
809M-parameter MIT-licensed model at 11.45% mean WER and can be trained cheaply.

It is not the primary choice because changing to Whisper would fork the current Sparkie
path, while v6.0 is meant to establish a Cohere full-model baseline before later
architecture and data-mixture experiments. Whisper also needs explicit short-form
hallucination checks: the
leaderboard documents repetition and subtitle-credit failures on very short clips for
members of this model family.

Switch to this Whisper checkpoint only if the Cohere path fails one of these gates:

- the model cannot complete save, reload, resume, and Danish decoding in the two-step
  smoke;
- it cannot fit safely on Sparkie after reducing the per-device batch and preserving the
  effective batch through accumulation;
- neither learning-rate pilot improves the frozen Danish development objective; or
- Cohere training is blocked by access or an unresolved technical incompatibility.

Before activating the fallback for a leaderboard release, obtain immutable revisions,
train-only split proof, and overlap exclusions for every listed source: CoRal, Common
Voice, FLEURS, FTSpeech, NST, and all human- or machine-labelled inputs behind its
pseudo-labelled stream. Its card currently names datasets but not exact splits,
revisions, or pseudo-label provenance. Without complete evidence, it may be used only
for an internal engineering run, not as the basis of a leakage-free leaderboard claim.

Do not start both full-size architectures in parallel. Reserve the time and compute
for systematic mixture experiments instead.

### Why not Parakeet for the primary run

Parakeet is the strongest later speed/serving candidate, not the fastest route to this
accuracy release. The current open `svale-600M` result is 12.64% mean WER. The strongest
multilingual Parakeet uses FastConformer-TDT, while Hviske currently supports
Transformers-native Parakeet CTC and RNNT but not TDT. Using TDT would therefore add a
new NeMo or Transformers TDT training path before the first model can run.

## Freeze the benchmark first

Before changing data or starting a pilot:

1. Record the exact commit of the leaderboard evaluation repository and the exact
   revision of `RyeAI/danish-asr-leaderboard`.
2. Archive the published leader and best-open scores, including all five domain scores.
3. Set `W*` and `T` from the strongest verified result before the first full-model
   continuation gate, then freeze them for this run.
4. Save the official decoding configuration and normaliser version.
5. Add a native Cohere backend to the upstream harness. The current `cohere-asr` backend
   calls `model.transcribe()`, which a saved native `CohereAsrForConditionalGeneration`
   checkpoint does not provide.
6. Upstream or otherwise agree the backend with the leaderboard maintainers, then pin
   that accepted harness revision.
7. Confirm that the pinned harness reproduces one published reference result within
   rounding tolerance and can evaluate the untouched Cohere base and a saved two-step
   Hviske checkpoint.
8. Make no further harness changes. Do not copy only its WER normaliser into Hviske and
   call the result a leaderboard score.

The current benchmark macro-averages corpus-level WER across these five equally weighted
test sets:

- CoRal conversation `test`;
- CoRal read-aloud `test`;
- Common Voice 17 Danish `test`;
- FLEURS `da_dk/test`; and
- FTSpeech `test_balanced`.

The live benchmark currently applies NFKC, Danish numeral canonicalisation, lowercasing,
punctuation removal, filler-word removal, and whitespace collapse. Its Whisper backend
uses normal short-form decoding below roughly 30 seconds and timestamped long-form
decoding above that threshold; Cohere must follow its own native chunking contract. Pin
the code rather than relying on this prose because the methodology has changed several
times.

## Data contract

No source may enter training from a moving branch. For every source, record the Hub
revision or a local manifest digest, row count, decoded hours, accepted hours, duration
distribution, text distribution, and rejection counts.

### P1 segments

Use the final immutable revision of `syvai/p1-segments` directly with its `audio` and
`text` columns. Do not use the old runtime join between `syvai/p1` and
`syvai/p1-transcripts`.

The authoritative final publication manifest at commit `ee6ab0...` records 1,044
manifests covering 16,634 shards/programmes, 4,767,938 published rows, and 230,963
duration rejections. Use these immutable publication totals when checking the P1
coverage of the production campaign.

Before training:

- verify the publication finalisation report against these immutable totals;
- pin the final 40-character revision;
- resolve the current mismatch between the plan's private-repository requirement and the
  live manually gated public repository;
- verify the expected schema and required provenance fields;
- stream, decode, filter, process, and collate genuine post-filter rows; and
- reconcile remote row and duration totals with the finalisation report.

P1 word timestamps and speaker attribution are best-effort annotations. Keep
structurally valid examples rather than rejecting them only because those annotations
are uncertain.

### DRTV and Danish YouTube

Use the local Sparkie WAV/VTT corpora through immutable JSONL manifests. A metadata-only
scan on Sparkie measured:

- **DRTV:** 5,195,010 cues across 20,406 files and 6,442.55 overlap-corrected hours;
  5,178,843 cues and 6,390.27 overlap-corrected hours survive the strict 1--10 second
  filter.
- **Danish YouTube:** 3,558,613 cues across 11,646 files and 2,094.05 overlap-corrected
  hours; 3,208,785 cues and 2,001.97 overlap-corrected hours survive the filter.

The manifests were last modified on 9 September 2026. Neither contains malformed rows,
empty text, or duration mismatches. Overlapping cues account for less than 0.1 hours in
either corpus. These are manifest coverage hours, not independently decoded audio-file
hours.

The following are still required before the first pilot:

- rebuild or verify each manifest;
- hash the manifest and all referenced file identities;
- reject missing audio, malformed captions, empty text, invalid offsets, and decode
  failures;
- report accepted rows and hours after the shared 1--10 second filter;
- remove caption credits, repeated intros/outros, and rolling-caption duplication;
- estimate Danish-language and speech presence on a stratified sample; and
- document the right to train and release model weights from each corpus.

Do not substitute statistics from a similarly named public YouTube dataset for the local
Sparkie corpus.

### Unified v5-tiny quality manifest

Treat `syvai/danish-asr-unified-hviske-v5-tiny` as a quality and relabelling manifest,
not as an audio dataset. It has no audio column. The overlay main branch is moving, so
v6.0 does not carry a stale overlay SHA. Set `HVISKE_OVERLAY_REVISION` to the completed,
immutable 40-character overlay commit before preflight or training; the five active unified
Danish sources all resolve that same environment value. Keep the matching unified-audio
revision (`5a3a49ee981baab6e1e37ddd2c45f9943c27d08f`) in the source configurations until
the overlay completion is pinned.

After the job finishes:

1. pin its final revision and the matching revision of `syvai/danish-asr-unified`;
2. build a compact manifest containing only stable join coordinates, source, action,
   selected text, and audit fields;
3. join every row to the matching base-audio row and fail on missing, duplicate, or
   inconsistent coordinates;
4. keep only `action in {keep, relabel, strip}`;
5. use non-empty `new_text` for `relabel` and `strip`, otherwise use `reference_text`;
   and
6. exclude `flag`, `quarantine`, and `drop` from this clean first run.

At the current moving revision, the manifest contains 3,414,589 rows and marks about
96.3% as clean by that rule. VoxPopuli is not an active v6.0 source: at the pinned
unified audio revision
`5a3a49ee981baab6e1e37ddd2c45f9943c27d08f` and positional overlay shard range `0--354`,
its first 100 joined clips are OGG mono 16 kHz (duration min 16.15, median 30, max 30),
so none survive the strict `1 < duration < 10` filter. Keeping it active would scan
1.745 million rows without yielding an example. Its known subtitle-credit hallucinations
are therefore a release-blocking regression category, not harmless label noise.

Use the quality manifest for the five active Danish unified sources. Create
source-filtered streams for CoRal read-aloud, CoRal conversation, FTSpeech, Nota, and
NST so the existing per-source sampling weights remain explicit. The unified repository
and its overlay remain part of training provenance through those sources. Each stream
uses the same source filter on the base and overlay, strict positional row matching,
source and reference-text equality checks, and the ordered
`new_text`-then-`reference_text` fallback for relabel/strip actions. The preparation
command applies the reusable strict overlay join one mirrored physical shard at a time
and writes only embedded compressed audio bytes, final text, and source to checksummed
local Parquet shards. Training and preflight require the deterministic
manifest and complete marker through `HVISKE_MATERIALISED_OVERLAYS_ROOT`; they fail
closed rather than returning to remote positional joins. This safe local graph supports
four spawned DataLoader workers while keeping dataset preprocessing serial. Avoid
loading the same source both directly and through the unified repository.

Do not use the manifest's FLEURS rows. Common Voice rows may be used only when their
split membership is proved and all leaderboard test rows are excluded; otherwise omit
Common Voice from the first run.

### Leakage and duplication gate

Build the exclusion set from the exact five leaderboard test revisions before
materialising training manifests.

Exclude matches by, at minimum:

- original dataset identity and split membership;
- stable source ID;
- canonical decoded-audio hash after channel and sample-rate normalisation; and
- duplicate or near-duplicate duration plus normalised transcript signatures.

Apply the exclusion set to P1, DRTV, YouTube, and every unified source. Also deduplicate
across the training sources so repeated broadcasts or captions do not receive accidental
extra weight.

Keep the exclusion report, source counts before and after filtering, and the resulting
manifest digests. Any unresolved exact benchmark overlap blocks training. Approximate
matches can be quarantined for review without blocking the rest of the corpus.

## Initial mixture

Start from the current Sparkie source weights, but treat them as provisional until P1's
final segment count and mean duration are known. P1 is a radio programme and belongs in
the broadcast/conversation domain, not prepared/read speech.

- **Danish prepared/read, 0.06:** CoRal read 0.017143, Nota 0.021428, and NST
  0.021429.
- **Danish broadcast/conversation, 0.45:** P1 0.102857, DRTV 0.128571, YouTube 0.09,
  and CoRal conversation 0.128572.
- **Danish parliament, 0.09:** FTSpeech 0.09.
- **English mixed/conversation/meeting, 0.23:** People's Speech mixed 0.16, AMI SDM
  0.04, and AMI IHM 0.03.
- **English parliament, 0.09:** VoxPopuli English.
- **English read speech, 0.08:** LibriSpeech 100h 0.01, 360h 0.025, and 500h 0.045.

This provisional mix remains 60% Danish and 40% English. The English mass preserves
broad acoustic and decoder behaviour and keeps the first run comparable with the
existing baseline.

Conversation is the primary use case, but People's Speech is heterogeneous and must not
be counted wholly as conversation. The preset now allocates 75% of the Danish sampling
mass to broadcast/conversation: 0.45 of the overall mixture. It retains 60% Danish and
40% English and leaves the English group masses unchanged for this initial release. This
is an explicit product-use prior, not a claim derived from corpus size. The previous and
current matrices are documented in
[`training-data-inventory.md`](training-data-inventory.md).

The provisional source values preserve the old within-group proportions. Recalculate
P1, DRTV, YouTube, and CoRal conversation within their combined 0.45 mass when P1's mean
accepted duration is available. Probabilities select examples, not hours. For a
desired audio-exposure share `a_s` and mean accepted segment duration `d_s`, use:

```text
p_s proportional to a_s / d_s
```

Do not set probabilities from repository bytes or total hours alone. DRTV averages 4.44
seconds per accepted example, while YouTube averages 2.25 seconds. Their current
0.128571 to 0.09 probability ratio therefore already produces an audio-exposure ratio
close to their 6,391 to 2,002 hour corpus ratio. The P1 share is frozen at 0.102857 for
this production preset; measure its final row count and mean duration rather than
silently changing the configured source order or probability.

Freeze the revised probabilities before either learning-rate pilot. Record realised
examples and audio seconds per source during every pilot and the full run. Missing
streams must fail explicitly rather than silently renormalising the mixture.

## Development suite and checkpoint objective

Create one frozen development suite before the pilots. It must be disjoint by programme,
video/uploader, speaker where reliable, and canonical audio hash.

The primary objective is the equal-weight macro WER over five Danish domain proxies:

1. CoRal conversation validation;
2. CoRal read-aloud validation;
3. Common Voice Danish validation;
4. FLEURS Danish validation; and
5. FTSpeech validation or a frozen non-test balanced split.

Add diagnostic, non-primary held-outs from P1, DRTV, and YouTube. They should expose
programme-level and web/broadcast failure modes but must not change the declared primary
objective after training starts.

Also report:

- per-domain WER and CER;
- insertion, deletion, and substitution counts;
- short-clip buckets, especially clips below two seconds;
- duration buckets through the 10-second training cutoff and a separate long-form set;
- subtitle-credit and common hallucination phrase rates;
- empty-reference and no-speech behaviour;
- Danish characters, casing, punctuation, and numeral behaviour; and
- English retention on the existing AMI and LibriSpeech validation subsets.

The current training evaluator is useful for progress tracking but does not produce all
of these artefacts. Add a deterministic release-evaluation path that writes stable
per-example IDs, references, hypotheses, and edit counts.

### Predeclared selection and regression gates

Calculate all deltas against the untouched Cohere base on the same frozen examples and
decoding settings. Freeze these thresholds before either pilot:

- **Pilot improvement:** at least 0.20 absolute mean-WER improvement on the five-domain
  deterministic subset at step 2,000.
- **Severe Danish reversal:** any primary domain more than 1.0 absolute WER worse than
  the base during a pilot. Such a pilot is ineligible even if its macro WER wins.
- **Release domain guardrail:** no primary Danish domain more than 0.5 absolute WER
  worse than the base on the full development suite.
- **English retention:** no more than 5% relative macro-WER degradation across the
  frozen AMI and LibriSpeech diagnostics.
- **Repetition event:** a hypothesis with at least three consecutive copies of the same
  four-token sequence, or more than the greater of 50 words and three times the
  reference word count.
- **Hallucination guardrail:** repetition-event and known subtitle-credit phrase rates
  must each remain at or below 0.5%, and may not increase by more than 0.05 percentage
  points over the base.
- **Patience reset:** track the lowest macro WER among all full-suite checkpoints that
  pass every guardrail. Reset patience only when a new eligible checkpoint improves that
  value by at least 0.10 absolute WER; otherwise increment it. Stop at five consecutive
  evaluations without a reset.
- **Pilot tie-break:** if pilot macro WERs differ by at most 0.05 absolute, select the
  lower learning rate. Otherwise select the lower eligible macro WER.
- **Final checkpoint order:** discard every checkpoint that fails a guardrail. Find the
  minimum macro WER among the remainder, then select the earliest checkpoint no more
  than 0.05 absolute WER above that minimum.

The minimum useful-release and leaderboard targets remain the higher-level release
criteria. These thresholds define previously ambiguous pilot and continuation terms.

## Training recipe

Start from the pinned Cohere checkpoint and preserve the existing tested Sparkie
defaults unless a pilot shows a concrete failure:

- **Audio:** 16 kHz, 1--10 second examples.
- **Prompt:** Danish or per-source language, punctuation enabled.
- **Parameters:** full encoder and decoder fine-tuning.
- **Precision:** BF16 where supported.
- **Memory:** gradient checkpointing; reduce per-device batch before changing the
  effective batch.
- **Streaming shuffle:** use the global one-row buffer for already-sharded Hub sources,
  128 rows for the one-shard local DRTV and YouTube manifests, and 16 rows for the five
  positional unified sources. Shuffle joined metadata after the positional overlay and
  before audio decoding; the source probabilities continue to interleave every stream.
  The old global 128-row smoke took 127 minutes, read 90 GB, and reached 19 GB worker
  RSS before its first batch.
- **Effective batch:** 256 examples.
- **Optimiser:** AdamW, betas 0.9 and 0.98, max gradient norm 1.0.
- **Augmentation:** existing peak normalisation, gain, background/coloured noise, and
  filtering.
- **Seed:** 4242 for selection; 4243 for confirmation.
- **Scheduler horizon:** 200,000 steps for the direct full run.
- **Pilot stop:** 2,000 steps, controlled separately from the optional pilot horizon.
- **Checkpoints:** best, latest resumable, and one rollback checkpoint.

Training uses `max_steps` for the cosine-scheduler horizon and the tested
`stop_after_steps` callback for bounded stopping. The current direct full run sets
`max_steps=200000` and leaves `stop_after_steps` unset. The owner-waived 2,000-step
learning-rate pilots, if requested as optional diagnostics, retain their explicit
`max_steps=100000` horizon; they do not gate or redefine the full run. The direct run
starts from the pinned Cohere checkpoint and continues to 200,000 steps.

At batch 256 and the frozen source probabilities, the campaign coverage is:

| Source | Published/accepted rows | Minimum steps | Expected rows at 200,000 steps | Coverage |
| --- | ---: | ---: | ---: | ---: |
| P1 | 4,767,938 | 181,075 | 5,266,278 | 110.5% |
| DRTV | 5,178,843 | 157,344 | 6,582,835 | 127.1% |
| YouTube | 3,208,785 | 139,271 | 4,608,000 | 143.6% |

Do not add a broad hyperparameter sweep. If the owner requests the waived diagnostics,
compare only learning rates `5e-6` and `1e-5`. All other settings, source ordering, data
revisions, development examples, and seed remain fixed.

## Execution plan

### Workstream A: finish the data path

This can start before the final datasets are available.

- Add a source-filtered join from the compact v5-tiny manifest to the pinned unified
  audio repository.
- Update P1 configuration to consume `syvai/p1-segments` directly.
- Strengthen preflight so every source reaches a real processed training batch.
- Add manifest statistics, benchmark exclusion, and cross-source deduplication.
- Update the Sparkie preset and tests to the predeclared v6.0 group-mass matrix.

The final pinning and complete audit happen as soon as P1 and v5-tiny stop moving.

### Workstream B: freeze evaluation and release artefacts

Run in parallel with workstream A.

- Implement and upstream the native saved-checkpoint Cohere backend.
- Pin and reproduce the accepted leaderboard harness.
- Smoke the exact backend with the base model and a locally saved two-step checkpoint.
- Materialise the development suite and stable IDs.
- Add per-example hypotheses and edit counts to release evaluation.
- Draft the model card with placeholders for final revisions, counts, hours, and scores.
- Define the Hub repository as private by default.

### Gate 1: data-only Sparkie preflight

Set `P1_SEGMENTS_REVISION`, `HVISKE_OVERLAY_REVISION`, and
`HVISKE_MATERIALISED_OVERLAYS_ROOT` before resolving the preset. First run
`src/scripts/materialise_finetuning_overlays.py` against the pinned revisions. It
establishes strict extra-row, missing-row, action/text, source, and equality gates while
writing one paired physical shard at a time. Preflight rejects mutable revisions and
validates the same manifest, complete marker, source/file provenance, schemas, counts,
receipts, and checksums before any local shard is opened.

Keep the existing GPU service running during this gate. In the same container and mounts
that training will use:

- verify Hub access without printing credentials;
- verify every immutable revision and local manifest digest;
- check all 15 active streams produce accepted examples;
- decode and process multiple examples from every source, including every action type
  retained from v5-tiny;
- collate one representative mixed batch;
- record the Hviske commit, `uv.lock` digest, container image, mounts, and output path;
- check free disk, checkpoint headroom, GPU memory, processes, and temperatures; and
- run focused tests plus Ruff and Ty.

Stop `qwen38-ar` only after this gate passes and immediately before the GPU smoke.

### Gate 2: two-step model smoke

The smoke must cover forward pass, backward pass, evaluation, save, clean reload, and
resume from checkpoint. Confirm Danish and English examples both decode sensibly.

Failure here triggers a focused fix. If Cohere remains incompatible or unsafe on
Sparkie, activate the pinned Whisper fallback rather than opening an architecture sweep.

### Gate 3: owner-waived learning-rate pilots

The owner has waived the learning-rate pilots for the current campaign. They are not a
prerequisite for the direct full run. If a diagnostic pilot is requested, run the pilots
serially with seed 4242, using their explicit 100,000-step scheduler horizon and separate
2,000-step run limit:

1. 2,000 steps at `5e-6`;
2. 2,000 steps at `1e-5`; and
3. evaluate at 250, 500, 1,000, and 2,000 steps on deterministic development subsets.

Choose by the predeclared five-domain macro WER and tie-break. Require finite losses,
stable memory, successful resume, the minimum pilot improvement, all Danish domain
guardrails, and the English-retention threshold. Do not delay or block the direct full
run on these optional diagnostics.

### Full release run

Launch directly from the pinned Cohere checkpoint with its fresh optimiser and scheduler
state. Evaluate the full frozen development suite every 2,000 steps and make
continuation decisions at 10k, 25k, 50k, and 75k steps.

At each gate:

- compare against the untouched Cohere base, Hviske v5, and any optional pilot
  checkpoints;
- inspect all five Danish domains and the P1/DRTV/YouTube diagnostics;
- inspect subtitle-credit, repetition, silence, and short-clip failures;
- verify realised examples and audio seconds per source;
- verify throughput, memory, temperature, disk, and checkpoint reload; and
- apply the predeclared patience-reset rule.

The 200,000-step value is a ceiling. At stopping or at the ceiling, apply the final
checkpoint order exactly. The minimum useful-release gate determines whether the result
is releasable; the leaderboard and headline targets determine the claims that may be
made. Do not continue merely to consume the budget.

## Blind leaderboard and release

After checkpoint selection:

1. freeze the checkpoint, decoding settings, model card draft, code commit, lockfile,
   source revisions, manifest digests, and development report;
2. rerun the pinned native Cohere backend smoke on the exact frozen checkpoint;
3. run the pinned official leaderboard harness once on all five test sets;
4. retain its raw per-example output and identify the model that defines frozen `W*`;
5. compare the candidate against that exact model with the predeclared significance
   procedure below;
6. refresh the live leaderboard snapshot, identify the model defining refreshed `W*`,
   recompute `T`, and repeat the compatible comparison against that model; and
7. do not retrain or select another checkpoint from any test result.

Use this significance procedure for both the frozen and refreshed comparisons:

- require matching example IDs and raw candidate and comparator hypotheses;
- score both outputs under one common pinned test-set and normaliser revision;
- use 10,000 paired bootstrap draws with seed 4242;
- resample examples with replacement independently inside each of the five dataset
  strata;
- for each draw, compute corpus WER in each stratum, macro-average the five WERs, and
  record candidate minus comparator;
- use the 2.5th and 97.5th percentiles as the 95% confidence interval; and
- require both candidate mean WER at or below `T` and the interval's upper bound below
  zero for a significant-win claim.

If the live benchmark changes only its normaliser, rescore both sets of raw hypotheses
under the new pinned normaliser. If decoding, examples, or splits change, rerun the
frozen candidate once under the refreshed harness without changing the checkpoint. If
the `W*` model lacks compatible per-example outputs, report the numerical result but
withhold statistical and “significantly better” claims.

Upload the candidate first to private `syvai/hviske-v6.0`. From a clean environment:

- download the pinned revision;
- run short, long, silence, and Danish-character inference smokes;
- verify Transformers and the intended serving path;
- verify that no local manifests, audio, credentials, or private metadata were uploaded;
  and
- make the repository public only after data-rights and privacy sign-off.

The model card must include:

- base model and exact revision;
- architecture, parameter count, intended use, and VAD recommendation;
- every training source, revision or manifest digest, accepted rows, unique hours, and
  sampled hours;
- filtering, relabelling, deduplication, and benchmark-exclusion rules;
- full resolved training recipe, code and lockfile revisions, seeds, hardware, wall
  time, and checkpoint-selection rule;
- development and official benchmark scores, raw output links, and decoding settings;
- known failure modes and domain limitations; and
- third-party data notices and the approved model licence.

Submit `results/<model>.json` and the complete raw output directory to the leaderboard
as required by its current process. Claim the best Danish ASR model only after the
leaderboard maintainers independently reproduce and publish the result.

## Earliest practical sequence

### Before P1 and v5-tiny finish

- implement the final P1 and quality-manifest data paths;
- freeze the benchmark and development suite;
- audit the local DRTV and YouTube manifests;
- prepare the exclusion and deduplication reports; and
- prepare the private model repository and model-card skeleton.

### As soon as P1 finishes

- validate its finalisation report;
- pin the immutable revision;
- resolve repository visibility;
- run P1-specific processing and collation tests; and
- start the full leakage/deduplication audit.

### As soon as v5-tiny finishes

- pin it and the matching unified audio revision;
- materialise the compact clean manifest;
- reconcile all action and source counts; and
- run the complete data-only Sparkie preflight.

### First free GPU window

- stop the conflicting GPU service only after preflight;
- run the two-step smoke; and
- launch the direct full run at the 200,000-step horizon.

This ordering keeps data and evaluation work off the critical GPU path. The campaign
horizon and source coverage are based on the immutable publication totals and measured
batch probabilities, not another architecture's published training time.

## Stop conditions and non-goals

Stop or pause the run for any of these:

- unresolved exact leaderboard leakage;
- moving or unpinned data revisions;
- a missing stream that causes silent probability renormalisation;
- non-finite loss, repeatable streaming corruption, or failed checkpoint reload;
- unsafe GPU temperature, memory pressure, or insufficient checkpoint disk headroom;
- regression of known subtitle-credit or repetition failures;
- no meaningful development improvement across five full evaluations; or
- unresolved rights/privacy questions for a public release.

The following are explicitly not part of this first release:

- manual source-weight optimisation;
- a three-architecture bake-off;
- training on leaderboard test splits;
- repeated leaderboard submissions used as a tuning loop;
- long-form or diarisation feature development; and
- the follow-on mixture-optimisation programme.

After release, freeze the exact v6.0 checkpoint, fixed mixture, realised exposures,
development outputs, and official benchmark outputs as the full-model baseline for the
post-v6.0 experiments.

## Release lineage after v6.0

Keep the post-v6.0 programme separate from this initial release:

1. compare alternative ASR architectures, including full-size and compact candidates;
2. run data-mixture experiments for the viable architecture candidates;
3. confirm that mixture conclusions transfer to the selected full-size and compact
   models; and
4. train and release the selected models with complete evaluation and provenance.

Use subsequent `v6.x` versions for these architecture and data-mixture improvements. The
large increase in training-data scale establishes v6.0 as the major release boundary;
later architecture changes within this programme do not create another immediate major
version.

## Primary references

- [Danish ASR leaderboard](https://hf.co/spaces/RyeAI/danish-asr-leaderboard)
- [Leaderboard results dataset](https://hf.co/datasets/RyeAI/danish-asr-leaderboard)
- [Leaderboard evaluation code](https://github.com/Rye-A1/danish-asr-leaderboard)
- [Cohere Transcribe](https://hf.co/CohereLabs/cohere-transcribe-03-2026)
- [Whisper fallback](https://hf.co/thorhojhus/whisper-large-v3-turbo-danish)
- [Parakeet TDT 0.6B v3](https://hf.co/nvidia/parakeet-tdt-0.6b-v3)
- [v5-tiny manifest](https://hf.co/datasets/syvai/danish-asr-unified-hviske-v5-tiny)
- [`docs/p1-segmentation-plan.md`](p1-segmentation-plan.md)
- [`SPARKIE.md`](../SPARKIE.md)
