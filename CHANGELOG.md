# Changelog

## [Unreleased]

### Added

- Added Transformers-native NVIDIA Parakeet CTC and RNNT fine-tuning.
- Added Danish vocabulary adaptation for native Parakeet tokenizers, preserving
  existing vocabulary and blank-token IDs while resizing CTC and RNNT heads.
- Added Parakeet CTC and RNNT model presets.

### Changed

- Removed unsupported Parakeet TDT fine-tuning and the NeMo-only Danish RNNT preset.

### Fixed

- Preserved repeated reference tokens in Parakeet RNNT metrics.
