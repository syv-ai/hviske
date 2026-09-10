# P1 segmentation and private publication plan

## Goal

Build a private, immutable Danish ASR training dataset from the P1 programme audio
and full-programme transcripts. The derived examples must be short, accurately
aligned, and directly streamable by Hviske.

The pipeline must never materialise the complete source or derived corpus on local
disk or Sparkie. It processes bounded batches, uploads completed Parquet shards,
verifies the remote bytes, and deletes the local copies.

This dataset is a hard prerequisite for the Olmix benchmark. The current runtime join
between `syvai/p1` and `syvai/p1-transcripts` is not a training solution because each
usable transcript covers a complete radio programme while Hviske rejects audio at or
above 10 seconds.

## Fixed source coordinates

The first implementation must pin these source revisions:

- Audio: `syvai/p1` at
  `449b9c2294026df6d0d37538f279fdec03f565ff`.
- Transcripts: `syvai/p1-transcripts` at
  `41132579816d86e889635f84f30511279f026359`.
- Join key on both sides: `file_id`.
- Transcript text: `transcript_text`.

The measured transcript side contains 16,640 unique programme rows. Two rows have no
usable text and must be recorded as rejected rather than treated as fatal. The usable
rows include word timestamps and speaker identifiers. Programme durations range from
120 to 9,060 seconds, with a median of 1,500 seconds.  The source row's duration
metadata is only a planning hint: for embedded compressed audio, libsndfile decodes
bytes first and the frame count plus sampling rate supplies the authoritative
programme duration. Transcript bounds are checked against that decoded duration.

The source coordinates, alignment backend, model revision, normalisation rules, and
pipeline version form the reproducibility identity. A change to any of them requires a
new derived dataset revision and new deterministic segment identifiers.

## Output contract

The proposed target is a private dataset repository named `syvai/p1-segments`. The
repository must remain private throughout creation, upload, validation, and use.

Each `train` row has these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `audio` | `Audio(16000)` | Audio feature backed by a mono FLAC payload. |
| `audio_sha256` | string | Digest of the encoded FLAC payload. |
| `text` | string | Verbatim segment text before model normalisation. |
| `alignment_text` | string | Canonical text supplied to the aligner. |
| `alignment_word_map` | list[string] | Mapping from alignment units to exact owned source-text chunks. |
| `language` | string | Always `da`. |
| `segment_id` | string | Deterministic content and provenance identifier. |
| `source_file_id` | string | P1 programme join key. |
| `source_start_ms` | int64 | Final clip start in the source programme. |
| `source_end_ms` | int64 | Final clip end in the source programme. |
| `source_duration_ms` | int64 | Authoritative decoded source-programme duration. |
| `duration_ms` | int32 | Exact decoded clip duration. |
| `speaker_ids` | list[string] | Speakers represented in the clip. |
| `proposal_start_ms` | int64 | Start from the supplied word timestamps. |
| `proposal_end_ms` | int64 | End from the supplied word timestamps. |
| `alignment_score` | float32 | Backend-specific segment confidence. |
| `alignment_score_type` | string | Pinned score algorithm and scale. |
| `start_drift_ms` | int32 | Final start minus proposal start. |
| `end_drift_ms` | int32 | Final end minus proposal end. |
| `vad_speech_ratio` | float32 | Fraction of the final clip classified as speech. |
| `alignment_backend` | string | Pinned aligner and model identity. |
| `pipeline_version` | string | Human-readable segmentation-contract version. |
| `pipeline_config_sha256` | string | Digest of the complete identity manifest. |

Do not include credentials, cache paths, machine names, or transient job identifiers.
Preserve enough provenance to reproduce or audit every segment.

First build a canonical identity manifest containing source and code revisions, VAD and
aligner code and model revisions, text normalisation, segmentation and quality
thresholds, output encoding, and schema version. Serialise it with UTF-8, Unicode NFC,
sorted keys, no insignificant whitespace, JSON escaping, and exact numeric types. Store
its SHA-256 digest as `pipeline_config_sha256`.

Build `segment_id` from a second canonical JSON object containing that digest,
`source_file_id`, final integer millisecond boundaries, and exact published `text`.
Apply the same serialisation and take its SHA-256 digest. Canonical JSON provides field
framing and prevents concatenation collisions. A changed identity component produces a
new configuration digest and new segment IDs.

Write shards under `data/train/part-NNNNN.parquet`. Target approximately 500 MB per
file, matching existing repository conventions. Store lossless FLAC payloads and write
Hugging Face feature metadata where the Parquet publication path supports it. Validate
the exact uploaded schema. If streaming load does not reconstruct the feature, cast the
column explicitly with `Audio(sampling_rate=16000)` in the Hviske loader.

## Alignment design

### Use supplied timestamps as proposals

Do not run unconstrained speech recognition over each full programme. The transcript
already provides word timestamps and speaker information. Use them to define local
text and audio windows, detect obvious source defects, and construct initial segment
boundaries.

Normalise timestamps into one monotonic millisecond timebase. Reject a programme
before alignment when timestamps are missing, non-finite, outside the audio duration,
or substantially non-monotonic. Record the exact rejection reason in the ledger.

Create a canonical `alignment_text` separately from the verbatim training text. Record
a reversible mapping from every retained alignment unit to its exact owned source-text
chunk. The mapping
must make case folding, punctuation removal, number expansion, unsupported characters,
and optional romanisation auditable. The pinned Roest tokenizer is lowercase-only, so
P1 normalisation version 5 requires case folding for canonical CTC text. Published
`text`, source spans, and the word map remain verbatim and case-preserving; never infer
published text by reversing aligner normalisation. Untimed and zero-duration lexical
records use bounded, per-transcript ownership lookahead: leading and interior text
belongs to the following positive-duration word, while terminal text belongs to the
final previous word. Ownership is accepted only when its speaker evidence is
consistent with that anchor; a missing token speaker is accepted between two
same-speaker anchors. Zero-duration points at either closed gap boundary are valid.
Owned lexical text remains in both the exact candidate text and canonical CTC text.

### Form candidate segments

Build candidates from consecutive words with these rules:

- Target speech-bearing clips between 2 and 8 seconds.
- Enforce a final duration below 10 seconds, never equal to 10 seconds.
- Prefer punctuation, speaker changes, and VAD silence as boundaries.
- Do not cross a speaker change unless the pilot proves that the supplied speaker
  labels are too noisy to use safely.
- Do not split a word or duplicate a word across adjacent candidates.
- Add bounded context around each proposal for alignment, but remove it from the
  published clip.
- Preserve the original text. Apply Hviske's model normalisation only during training.

The exact target duration, context, and silence thresholds are pilot parameters, not
hard-coded assumptions. Store them in a versioned configuration file.

### Refine with VAD and CTC alignment

Run a pinned, offline VAD model over the programme once. Use VAD to snap proposal
edges to nearby silence and reject low-speech-coverage clips. VAD is a boundary and
speech-coverage signal, not a transcript, music, or noise classifier. Apply any music
or noise rejection through a separate pinned heuristic or classifier validated in the
pilot.

Refine and verify each local candidate with a Danish-capable CTC forced aligner. The
implementation uses the pinned `ctc-segmentation` source and the
[`CoRal-project/roest-v3-wav2vec2-315m`](https://huggingface.co/CoRal-project/roest-v3-wav2vec2-315m)
checkpoint at `beb3e790246d6b9dec1df596b0b21d5c42f4d99c`. Its
`Wav2Vec2ForCTC`/`wav2vec2` configuration has a 16 kHz processor and a 320-sample
convolution stride (20 ms); its pinned `config.json` does not declare a sampling
rate. The processor/preprocessor metadata is authoritative for the input clock, and
the pipeline derives and validates frame duration from it. A model sampling-rate
field, when present, is only an additional consistency check.

The model card at the pinned model revision designates the model `openrail`, and its
prose describes a custom OpenRAIL-M licence. The pipeline pins both that model-card
revision and digest, plus the immutable revision and digest of the underlying
referenced licence. P1 uses the checkpoint for ASR alignment only; Roest weights are
internal and are not distributed. It is not labelled Apache-2.0. Its 46-token
vocabulary includes digits, Danish letters, blank/pad token 45, and word delimiter
`|` at token 36. A model is eligible only when:

- its exact Hub revision can be pinned;
- its declared licence permits this private commercial data-preparation use;
- its processor sampling rate and convolution stride match the P1 16 kHz/20 ms
  alignment clock;
- its tokenizer covers Danish letters and the normalised transcript sufficiently;
- it produces stable word scores and boundaries on representative P1 programmes;
- it is independent enough from the target training model to provide useful checks.

WhisperX supports the locality rationale: VAD and transcription proposals are followed
by a language-specific phoneme model and dynamic time warping. It is not evidence for
the CTC implementation. The CTC method uses the pinned `ctc-segmentation`
implementation only. `ctc-forced-aligner` remains excluded because its licence
metadata is contradictory. NeMo Forced Aligner is a fallback only if a suitable
licensed Danish CTC checkpoint is available. Montreal Forced Aligner is not the first
choice
because it adds lexicon and acoustic-model maintenance without using the supplied word
timestamps.

### Correct drift once

Compare CTC word boundaries with the supplied proposals inside each programme window.
If high-confidence anchor words show a consistent offset or linear drift, fit a robust
piecewise-affine correction and rerun alignment once with recentered windows.

Reject the affected region instead of repeatedly correcting it when residuals are
nonlinear, discontinuous, or concentrated around missing transcript spans. Report
start drift, end drift, residual quantiles, and drift against programme position.
This catches clock drift and transcript insertions or omissions separately.

### Score and filter

Keep both raw backend scores and derived quality signals. A segment is publishable
only when all of these gates pass:

- decoded duration is above Hviske's minimum and below 10 seconds;
- text is non-empty and contains at least one trainable character;
- timestamps are ordered and within the decoded source duration;
- CTC alignment confidence exceeds the pilot threshold;
- proposal-to-final drift is within the pilot limit;
- VAD speech ratio exceeds the pilot threshold;
- boundary speech clipping is below the pilot threshold;
- no word is duplicated or dropped within an accepted contiguous transcript region;
- speaker-overlap and music heuristics pass;
- FLAC encoding and a fresh 16 kHz mono decode succeed;
- `segment_id` is unique.

The `ctc-segmentation` reference implementation scores an utterance from minima over
chunk-level means of aligned frame probabilities. Store the exact backend, formula, revision,
and raw inputs needed to interpret a score. These scales are not interchangeable;
choose thresholds from the P1 pilot rather than copying a value across backends.

Do not silently repair unsupported or mismatched text. Record rejection categories
such as `empty_text`, `invalid_timestamps`, `low_alignment_score`, `excessive_drift`,
`low_speech_ratio`, `speaker_overlap`, `boundary_clipping`, and `decode_error`.

## Bounded processing and publication

### Scratch-space contract

Use a dedicated scratch root for source downloads, decoded programme audio, open
Parquet files, and pending upload batches. Do not use an unbounded default Hugging Face
cache.

Before starting, calculate the required free space as:

1. the maximum source and decoded bytes for every active worker and queue slot;
2. every open shard and configured pending shard in an unverified batch;
3. temporary encoder, Parquet, upload, and checksum files;
4. one remote-verification stream buffer per concurrent verification;
5. the SQLite ledger and metadata manifests;
6. a fixed safety margin.

Abort before downloading source audio when the free-space or quota check fails. Expose
`--max-scratch-bytes`, `--max-source-bytes`, and `--shards-per-commit`. Also check GPU
memory, the selected CUDA device, Hub access, target repository privacy, and source
revision availability before processing.

One worker owns one programme at a time. A bounded queue may overlap CPU FLAC encoding
with GPU alignment, but every slot must be part of the scratch calculation. Once all
rows from a programme are durably appended and fsynced to a recoverable local shard,
record that state and delete its source, decoded, and alignment temporary bytes. A lost
or corrupt uncommitted shard is regenerated by deterministic redownload from the pinned
source. Never retain source programmes until a multi-shard upload batch completes, and
never let Datasets materialise the complete audio split or derived dataset.

### Durable ledger

Maintain a local SQLite ledger containing metadata only. It must survive restarts but
must not contain audio bytes, credentials, or the complete transcript corpus.

Track these state transitions:

`discovered -> processing -> sharded -> committed -> verified -> purged`

A failed item moves to `rejected` or `retryable`. A state transition and its evidence
must commit atomically. Record source ID, source revisions, pipeline version, attempts,
counts, durations, rejection counts, shard paths, byte sizes, SHA-256 digests, Hub
commit IDs, verification time, and purge time.

On restart:

- bind and validate the pipeline digest before constructing models or selecting new
  programmes; an incompatible populated ledger is rejected without mutation;
- recover sharded and committed publication states before constructing VAD/CTC or
  retrieving new audio;
- reset abandoned `processing` rows to `retryable`;
- reuse complete local shards only when their digests match the ledger;
- query the remote commit before re-uploading a `committed` shard;
- never regenerate or overwrite a `verified` path with different bytes;
- resume from the first state lacking durable evidence.

Selection retains the exact audio row pointer discovered by the metadata scan and
joins it to the disk-backed transcript pointer. Processing first performs structural
transcript checks: empty, untimed, ambiguous, malformed, and invalid-timestamp
records are terminal rejections before audio retrieval and model construction. A
qualified transcript then permits bounded audio decoding. Transcript-versus-audio
duration validation happens after that decode and before VAD or CTC work; the decoded
audio is the duration authority, not a declared metadata duration. Unexpected failures
before sharding are categorised as retryable; publication failures leave durable
sharded/committed evidence unchanged so a restart can resume the exact bytes and
commit.

### Private repository setup

Create the target with `HfApi.create_repo(..., repo_type="dataset", private=True)`.
Immediately query repository metadata and abort unless `private` is true. Commit a
dataset card and `.gitattributes` before data upload. Recheck authenticated repository
metadata immediately before and after every data commit and every metadata or card
update. Abort before sending bytes when `private is not True`; treat a post-commit
privacy failure as an incident and stop all further processing.

The dataset card must use the custom `other` licence metadata, link to the committed
`LICENSE`, and describe source provenance, permitted use, private-access terms,
alignment method, field schema, known limitations, rejection policy, and immutable
source, model, and target-licence provenance. The dataset-licence template is the
[pinned CoRal-v3 licence][p1-dataset-licence] at revision
`01f7c93c21fc9dec87fe9f7149c79569cc433f08`, never a mutable `main` URL. Access, use,
and distribution are subject to `LICENSE`.
Confirm the organisation has enough private storage for the pilot estimate before the
full run.

Use the existing authenticated Hub session. Do not place a token in a command, tmux
history, environment dump, log, repository file, or dataset metadata.

### Incremental upload

Close each Parquet shard atomically, decode and validate it, then add it to the pending
batch. Upload a bounded batch with explicit `HfApi.create_commit` operations and
fewer than 100 files per commit. Each operation must name one allow-listed shard or
manifest path.
If `upload_folder` is retained as a fallback, populate a new isolated batch directory
with only allow-listed regular files, reject symlinks and unexpected entries, and remove
the directory after verification. Start with 8 to 16 shards per commit, subject to the
scratch budget.

For every batch:

1. write a metadata-only batch manifest with paths, sizes, row counts, and SHA-256
   digests;
2. upload the Parquet shards and manifest in one commit;
3. record the returned immutable Hub commit ID;
4. query every path at that commit with `get_paths_info`;
5. compare path and size, plus the LFS or Xet digest when exposed;
6. stream each remote object through SHA-256 without retaining it when the API does not
   expose a comparable content digest;
7. open the committed revision with `load_dataset(..., streaming=True)` and decode a
   deterministic sample from every shard;
8. mark the batch `verified` only after all checks pass;
9. delete the verified local shards and batch staging files;
10. fsync the ledger state to `purged`.

If any check fails, retain the complete pending batch and retry. Never delete a local
object based only on a successful HTTP status or process exit code.

Hugging Face recommends `upload_folder` or `hf upload` for resumable large uploads,
fewer than 100 files per commit, fewer than 100,000 files per repository, fewer than
10,000 entries per folder, and files below 200 GB. The proposed layout and shard size
stay well inside those limits. Parquet is explicitly recommended for large datasets
and avoids a custom loading script.

### Finalise an immutable release

After the final batch:

- compare discovered, processed, rejected, accepted, committed, and purged programme
  counts;
- recompute total rows, audio hours, bytes, and rejection counts from remote shards;
- verify no duplicate `segment_id`, source interval, or remote path exists;
- update the dataset card with final statistics and the full quality report;
- record the final 40-character commit SHA;
- pin that exact SHA in `config/datasets/p1.yaml`;
- optionally create a human-readable tag, but never use it as the immutable pin;
- remove the transcript join fields from the training configuration;
- keep only the metadata ledger and reports locally.

The old partial-transcript join remains useful defensive code for other datasets, but
P1 training must load the derived segmented dataset directly.

## Phased implementation

### Phase 0: source, legal, and storage gate

- Reconfirm both immutable source revisions and schemas.
- Measure source object-size and duration maxima without downloading the full split.
- Confirm target repository ownership, privacy, licence wording, and storage quota.
- Select candidate VAD and CTC models whose licences permit the intended use.
- Define the scratch budget and failure policy.

**Gate:** no audio processing starts until privacy, licensing, storage, and bounded
cache behaviour are proven.

### Phase 1: repository implementation

Add these components:

- `src/hviske/p1_segments.py` for proposal parsing, segmentation, alignment scoring,
  drift correction, filtering, deterministic IDs, and shard writing;
- `src/hviske/p1_ledger.py` for atomic state transitions and restart logic;
- `src/hviske/p1_publish.py` for private repository checks, commits, and verification;
- `src/scripts/build_p1_segments.py` as the non-interactive Hydra entry point;
- `config/p1_segments.yaml` for pinned revisions and tunable pilot thresholds;
- focused tests using synthetic waveforms and in-memory Hub fakes.

The script must support `mode=plan`, `programme_limit=...`, `source_file_id=...`,
and `resume=true`. Plan mode performs every precondition and size calculation without
retrieving audio.

**Gate:** tests prove deterministic IDs, exact duration filtering, no duplicate or
missing candidate words, bounded shard rotation, crash-safe resume, private-only
upload, and checksum verification. They separately prove source-temporary deletion
after recoverable local sharding and publication-artefact deletion only after remote
verification.

### Phase 2: representative pilot

Process a small stratified sample, not the first rows in the stream. Include short and
long programmes, several show types, multiple speaker counts, music-heavy material,
low language probability, early and late programme positions, and the tails of the
proposal-confidence and drift distributions.

Manually audit at least 200 accepted clips and 100 rejected or borderline clips. The
audit records clipped speech, wrong text, missing words, inserted words, speaker
mixing, music dominance, and boundary quality. Reviewers must listen without seeing
whether a clip passed or failed.

Use an independent ASR model as an anomaly detector and report normalised WER, but do
not treat model agreement as ground truth. Stratify the audit across confidence and
drift deciles so aggregate random sampling cannot hide bad tails.

Tune thresholds once, version the resulting configuration, and rerun the pilot from
scratch.

**Gate:** at least 98% of audited accepted clips have no material text mismatch or
speech clipping, no systematic defect appears by show or programme position, and the
rejection report explains every excluded clip. If this gate fails, change the method
or thresholds and repeat the pilot.

### Phase 3: bounded production run

Run one visible, named tmux session on Sparkie. Check existing GPU processes and free
disk before launch. Do not stop or alter unrelated services without separate
permission.

Publish one bounded batch at a time. Capture progress from the SQLite ledger and Hub
commit history, not from tmux scrollback alone. Emit metadata-only JSONL operational
logs with counts, timings, scratch use, and rejection categories.

**Gate:** every remote batch is committed, content-verified, and stream-decodable
before its local Parquet shards and batch staging are purged. Programme source and
alignment temporaries were already purged after recoverable local sharding. Scratch use
never exceeds the configured cap.

### Phase 4: corpus validation

Produce a remote-derived report covering:

- accepted and rejected programmes and segments;
- total and per-show audio hours;
- duration, word-count, score, speech-ratio, and drift distributions;
- coverage against usable transcript duration;
- rejection reasons and rates;
- duplicate IDs and overlapping source intervals;
- speaker-count and programme-position breakdowns;
- independent-ASR anomaly rates;
- a fresh stratified manual audit from the final revision.

**Gate:** all structural checks pass, the final manual audit meets the Phase 2 target,
and unexplained show-level or position-level quality regressions are resolved.

### Phase 5: Hviske integration

Update `config/datasets/p1.yaml` to read `syvai/p1-segments`, `train`, `audio`, and
`text` at the final immutable revision. Remove all runtime transcript-join environment
variables from the P1 path.

Strengthen `src/scripts/preflight_finetuning_data.py` so P1 preflight retrieves and
processes at least one genuine post-filter segment. It must decode the audio, verify
text, duration, language, and required metadata, and prove that
`load_data_for_finetuning` yields a training example.

Update `SPARKIE.md`, tests, and the Olmix launcher documentation with the pinned derived
revision.

**Gate:** focused tests, Ruff, Ty, and both two-step model smokes pass using the private
segmented P1 dataset.

## Evidence to retain

Retain only:

- source, model, and code revisions, including the CTC licence designation and URL;
- the pinned CoRal-v3 dataset-licence template revision and digest, the exact
  licensor-identity-only adaptation, and the target `LICENSE` digest. The byte/text
  replacement preserves the template's wrapping exactly except for the target identity:

  ```text
  The Licensed Material (as defined below) is made available to You by Alexandra
  Instituttet A/S, Åbogade 34, 8200 Aarhus N, Denmark
  ```

  becomes

  ```text
  The Licensed Material (as defined below) is made available to You by syv.ai ApS,
  Rosenvængets Allé 11, 1. tv, 2100 København Ø, Denmark
  ```

- versioned configuration and normalisation rules, including the CTC architecture,
  sampling rate, convolution stride, vocabulary size, blank, and delimiter IDs; the
  pipeline digest covers the lowercase-only Roest compatibility invariant,
  `p1-text-normalisation-5` case-folding, and
  `speaker-consistent-following-word-with-terminal-suffix-v5` source-text ownership;
- metadata-only SQLite ledger and batch manifests;
- shard paths, sizes, row counts, SHA-256 digests, and Hub commit IDs;
- aggregate quality reports and manual-audit decisions;
- operational timings and peak scratch usage.

Delete programme audio, extracted clips, alignment tensors, and temporary transcripts
after their rows are recoverably sharded and fsynced. Delete local Parquet shards,
batch staging, and remote-verification downloads only after the remote commit and
content verification are durable in the ledger.

[p1-dataset-licence]: https://huggingface.co/datasets/CoRal-project/coral-v3/resolve/01f7c93c21fc9dec87fe9f7149c79569cc433f08/LICENSE

## Primary references

- [WhisperX paper](https://arxiv.org/abs/2303.00747)
- [WhisperX implementation](https://github.com/m-bain/whisperX)
- [CTC segmentation paper](https://arxiv.org/abs/2007.09127)
- [CTC segmentation implementation](https://github.com/lumaku/ctc-segmentation)
- [CTC forced aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner)
- [NeMo NFA](https://github.com/NVIDIA/NeMo/tree/main/tools/nemo_forced_aligner)
- [Hugging Face upload guide](https://huggingface.co/docs/huggingface_hub/guides/upload)
- [Hub guidance](https://huggingface.co/docs/hub/repositories-recommendations)
- [Datasets Parquet loading](https://huggingface.co/docs/datasets/en/loading#parquet)
- [Datasets audio loading](https://huggingface.co/docs/datasets/en/audio_load)
