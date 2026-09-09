# Changelog

## [Unreleased]

### Added

- Added per-dataset Cohere language prompts, streaming audio/transcript joins, and
  zero-copy local WAV/VTT manifest loading.
- Added the private Sparkie bilingual Cohere preset, streamable English sources, and a
  verified separate model publication command.

### Fixed

- Made dataset probability validation tolerant of floating-point rounding while rejecting
  invalid values, and made validation caching and sample caps split-aware.
- Made local VTT manifests and finetuning data fully lazy, with anti-aliased cue resampling
  and validation before loading the full model.
