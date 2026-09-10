# Changelog

## Unreleased

### Fixed

- Increase P1 projected metadata batches to 1,024 rows while retaining the 64 MiB
  hard byte bound and audio-column projection, add targeted source selection, aggregate
  missing-transcript evidence, and expose safe periodic pipeline progress.
- Preserve zero-duration P1 transcript text while omitting only non-spacing timed
  tokens, classify invalid source timestamps as terminal rejections, and abort
  unexpectedly failed programmes with safe retryable ledger evidence.
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
