# Changelog

## Unreleased

### Fixed

- Replace the P1 CTC acoustic-model pin with the immutable CoRal Røst-v3 Wav2Vec2
  checkpoint, record its openrail/OpenRAIL-M licence metadata, and derive its 20 ms
  alignment frame duration from the validated model configuration.
- Normalise accepted P1 audio to the deterministic millisecond-duration sample
  boundary, enforce the configured decoded-audio cap for compressed and injected
  arrays, and classify transcript endpoints beyond decoded audio separately from
  structural timestamp defects.
- Bound generic compressed-audio expansion by inspecting libsndfile headers and
  enforcing a configurable 2 GiB decoded float32 PCM cap before allocation; derive
  source duration from decoded frames, and validate transcript bounds against that
  duration rather than unreliable metadata declarations.
- Recreate validated P1 transcript-index database files and SQLite sidecars between
  builds, reclaiming high-water space without following unsafe filesystem entries.
- Enforce the P1 scratch quota during pointer-index and selection-dedup growth, and
  recreate selection state between passes so deleted SQLite high-water space is reclaimed.
- Make native P1 selection deduplicate through crash-safe scratch SQLite state and
  preserve exact scalar audio and transcript pointer metadata through pilot sampling.
- Reduce deterministic P1 pilot selection to one audio metadata pass by retaining only
  bounded scalar row locators and reconstructing selected audio pointers from the plan.
- Increase P1 projected metadata batches to 1,024 rows while retaining the 64 MiB
  hard byte bound and audio-column projection, add targeted source selection, aggregate
  missing-transcript evidence, and expose safe periodic pipeline progress.
- Preserve exact P1 transcript characters, including zero-duration and untimed
  records, by assigning each source span to a deterministic neighbouring timed word;
  reject ambiguous ownership and classify transcripts without timed words explicitly.
- Harden P1 transport logging by suppressing verbose dependency records and
  redacting signed URLs from root and file handlers.
- Parse Hugging Face string-compatible CommitInfo objects using their immutable
  object identifiers during P1 publication.
- Harden P1 VAD framing, corpus-wide audit sampling, rejection completion,
  local shard validation, source playback locators, and crash-safe recovery.
- Make terminal programme rejection resumable, persist audit candidates across
  publication recovery, and emit manifests consumable by the validation CLI.
- Commit accepted audit local locators with sharded ledger transitions so crash
  recovery can publish and resolve the audit manifest.
- Bind P1 ledgers to one pipeline digest and make sharded and committed publication
  states restart-safe without retryable downgrades or premature local purging.
- Bump the P1 pipeline identity and reject incompatible populated ledgers before work;
  discover immutable audio pointers during bounded selection, qualify transcripts before
  audio or model work, and classify deterministic source defects without leaking IDs,
  payloads, URLs, or exception details in operational output.
- Make incompatible ledger migration fully rollback-safe, keep selected source IDs out of
  reports and CLI logs, classify parser timestamp and empty-timeline defects correctly,
  emit aggregate audio-scan progress at every crossed threshold, and count selected
  programmes correctly for bounded and streaming builds.

### Added

- Add Phase 1A P1 segmentation contracts, pinned configuration, and deterministic
  identity helpers.
- Add the restartable Phase 1C P1 pipeline with bounded planning, processing,
  metadata-only indexing, scratch quotas, ledger resume, and verified private
  publication.
- Add pinned offline Silero and Danish CTC alignment adapters, source-audio
  resampling, and bounded Audio-compatible Parquet sharding.
- Reassemble P1 execution behind a reusable, metadata-only planning and
  restart-safe pipeline with private-target initialisation and bounded audit metadata.
- Finalise native P1 integration: pinned model preflight, atomic ledger allocation,
  verified restart recovery, source FLAC decoding/resampling, and committed audit
  locators.
