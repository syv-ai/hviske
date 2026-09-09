# Changelog

## [Unreleased]

### Added

- Added per-dataset Cohere language prompts, streaming audio/transcript joins, and
  zero-copy local WAV/VTT manifest loading.
- Added the private Sparkie bilingual Cohere preset, streamable English sources, and a
  verified separate model publication command.
- Added a bounded Sparkie data preflight that checks Hub/model access, pinned schemas,
  and local manifests without loading the ASR model.

### Changed

- Pinned the production Sparkie dataset revisions and fixed its ordered 60% Danish / 40%
  English source mix; FLEURS remains evaluation-only and English Common Voice was
  removed.
- Disabled dataset remote-code trust by default in training and evaluation paths.
- Hardened private model publication to stage an explicit top-level artefact allowlist,
  include a bilingual internal-use model card, and verify private Hub visibility around
  one upload commit. Trainer automatic pushes remain disabled.

### Fixed

- Made dataset probability validation tolerant of floating-point rounding while rejecting
  invalid values, and made validation caching and sample caps split-aware.
- Made local VTT manifests and finetuning data fully lazy, with anti-aliased cue resampling
  and validation before loading the full model.
