# Changelog

## [Unreleased]

### Added

- Added optional NeMo inference for NeMo-only Parakeet RNNT, TDT, and hybrid TDT-CTC
  checkpoints, including the Danish 110M RNNT checkpoint.
- Added Transformers-native NVIDIA Parakeet CTC and RNNT fine-tuning.
- Added Danish vocabulary adaptation for native Parakeet tokenizers, preserving
  existing vocabulary and blank-token IDs while resizing CTC and RNNT heads.
- Added Parakeet CTC and RNNT model presets.

### Changed

- Removed unsupported Parakeet TDT fine-tuning and the NeMo-only Danish RNNT preset.

### Fixed

- Restored the codebase MIT license and removed the P1 dataset license files.
- Preserved repeated reference tokens in Parakeet RNNT metrics.
