# Hviske v6.0 initial training plan

## Status

The production consumer loads `syvai/p1-segments` directly as a normal Hub dataset with
`audio` and `text`. Training is gated until its manually gated immutable revision is
supplied through `P1_SEGMENTS_REVISION` and the data checks pass.

The retired P1 producer is preserved in this exact source archive:

- source: `syvai/p1-segments`
- path: `archives/hviske-p1-pipeline/f3dcf16/`
- Hub commit: `44284e5849b6b1d96b874891c579654a644e0e2f`

The published dataset remains manually gated. This repository consumes it; it no longer
contains producer, segmentation, publication-layout, or validation modules.

## Model and objective

Train and release a Danish-specialised continuation of
`nvidia/parakeet-tdt-0.6b-v3` at revision
`541d1f99c6b0c3cd0b11a95167540bb8edefd82b`. Keep one architecture and one fixed data
mixture for the initial release. Select a checkpoint on a frozen development suite,
then run the official benchmark once after selection. Do not tune against leaderboard
test data.

The minimum useful release is a 10% relative mean-WER improvement over `syvai/hviske-v5`
on the frozen five-domain development suite without a material domain regression. A
leaderboard win or significant win may be reported only when the pinned external harness
verifies it.

## Data contract

Every source must have a pinned Hub revision, schema, row count, duration summary, and
rejection report. The shared runtime filter accepts 1--8-second examples and retains
best-effort annotations when the audio and transcript are structurally valid.

### P1 segments

Use `syvai/p1-segments` directly. Before training:

1. supply a full 40-character `P1_SEGMENTS_REVISION`;
2. verify Hub access and the expected `audio` and `text` columns;
3. stream and decode genuine post-filter examples;
4. process and collate a representative batch; and
5. reconcile the resolved revision with the publication archive above.

Do not reintroduce the retired producer or a runtime transcript join.

### Public training sources

The production preset combines P1, CoRal read-aloud and conversation, FTSpeech, Nota,
NST, People's Speech, AMI, VoxPopuli, and LibriSpeech. Their dataset configuration
files pin the reusable public sources and define the columns, splits, language, and
trust policy. The bilingual preset keeps a 60/40 Danish/English sampling split with
source-specific probabilities.

## Training recipe

- Parakeet TDT with native Transformers loss and generation.
- Exact unpadded `[blank_token_id, *labels]` decoder inputs.
- 16 kHz audio and 1--8-second examples.
- BF16 where supported, gradient checkpointing, and effective batch size 60.
- AdamW with the configured learning rate and 200,000-step scheduler horizon.
- Seed 4242 for selection and 4243 for confirmation.
- Two DataLoader workers and serial dataset preprocessing.
- Evaluation every 2,000 steps; optional pilots stop at 2,000 steps.

The Parakeet contract is checked both while examples are processed and when batches are
collated. The model configuration's blank ID takes precedence over processor metadata;
malformed or mixed transducer batches fail closed rather than silently changing targets.

## Gates

### Data gate

Resolve the bilingual configuration with `P1_SEGMENTS_REVISION`, verify every active
stream yields examples, and record the Hviske commit, lockfile digest, revisions, and
hardware. Run focused data, configuration, Parakeet, and publication tests.

### Model smoke

The smoke must cover forward and backward passes, evaluation, save, clean reload, and
checkpoint resume for both Danish and English examples. Preserve the last complete
checkpoint if a run is interrupted.

### Full run and release

Run the production campaign only after the data and model gates pass. Inspect loss,
WER/CER, source exposure, memory, temperature, disk use, and checkpoint reload. Reject
checkpoints with non-finite loss, severe Danish regression, repeated output, or
unresolved data and privacy issues.

After checkpoint selection, freeze the model, code, lockfile, revisions, development
report, and decoding settings. Publish first to a private Hub repository with
`src/scripts/publish_model.py`; verify that no local data, credentials, caches, or
training artefacts are uploaded.

## Non-goals

This release does not add dataset producers, local caption ingestion, architecture
sweeps, or leaderboard-driven tuning. Those concerns are outside the reusable training
and evaluation package.
