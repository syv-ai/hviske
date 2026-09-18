# Changelog

## [Unreleased]

### Changed

- Reconciled the retained script inventory and documented P1 as a consumed, manually
  gated published dataset with its exact archival source and Hub commit.
- Kept Parakeet RNNT/TDT blank-prefix validation at both preprocessing and collation
  boundaries, giving model configuration blank IDs precedence over processor metadata.

### Removed

- Removed the unreferenced comparison-plot and dataset-download one-off scripts,
  the inactive Olmix plan, and the obsolete plotting extra and direct dependencies.
- Removed stale local-caption, overlay, and P1-producer instructions from the active
  documentation and removed the obsolete vulture and Ruff script entries.

### Added

- Added privacy-safe private P1 Hub diagnostics and bounded transient retries, deterministic
  bucketed shard publication with crash-safe ledger migration, and completed-corpus
  finalisation with metadata-only scanning and optional card CAS updates.
- Added resumable, shard-bounded local materialisation for Sparkie's five positional
  overlays, with durable compressed audio, strict join parity, atomic receipts,
  checksummed provenance manifests, free-disk reserves, and shared preflight validation.
- Added reproducible PyTorch 2.10 CUDA resolution for Sparkie, with a fail-fast check
  that prevents the production preset from starting without a usable CUDA device.
- Added robust online W&B tracking for the production Sparkie v6.0 workflow, including
  credential-safe preflight, resumable run IDs, and local-only model checkpoints.
- Added reusable keyed and strict positional ASR dataset overlays with bounded SQLite
  indexing, shared preflight validation, and safe overlay provenance; configured the
  v6.0 Danish unified sources to use an environment-supplied immutable overlay revision.
- Added Transformers-native NVIDIA Parakeet CTC, RNNT, and TDT fine-tuning.
- Added Danish vocabulary adaptation for native Parakeet tokenizers, preserving
  existing vocabulary, blank-token IDs, and TDT duration heads while resizing
  vocabulary-dependent CTC and transducer heads.
- Added Parakeet CTC, RNNT, and revision-pinned TDT model presets.
- Added a publication-only command with strict Cohere and Parakeet TDT package
  validation, including package-family/provenance matching and an explicit preserved
  Cohere publication preset, while keeping reviewed uploads separate from training.

### Changed

- Made the local positional-overlay artefact mandatory for production bilingual training
  and enabled two spawned DataLoader workers while retaining serial dataset preprocessing
  and generic remote keyed/non-materialised overlay support; four workers exceeded the
  thermal limit and three caused excessive long-run swap pressure.
- Tuned the production Sparkie shuffle buffers after the wf2t12vq smoke: a one-row
  global buffer for already-sharded Hub sources, 128 rows for local DRTV and YouTube,
  and 16 rows for the five positional unified sources. The old global 128-row graph took
  127 minutes, read 90 GB, and reached 19 GB worker RSS before its first batch; the
  200,000-step training horizon remains unchanged.
- Switched production Sparkie P1 training to direct `syvai/p1-segments` loading with a
  required immutable `P1_SEGMENTS_REVISION`; generic transcript joins remain supported.
- Renamed the host-specific bilingual preset to `config/bilingual.yaml` and made the
  validated Parakeet TDT production parameters its defaults: 8-second clips, learning
  rate `5e-6`, two loader workers, batch 6/effective batch 60, 200,000 steps, and
  2,000-step evaluation cadence. Run paths and tracking identity remain launch-time
  concerns.

### Removed

- Removed the inactive calibration launcher, anchor configurations, model preset, tests,
  and runbook instructions; retained the design plan for future reference.

### Fixed

- Enabled private P1 finalisation to verify non-empty batch ancestry through the
  production Hub adapter and to CAS-update the actual three-section dataset card.
- Retried Hugging Face HTTP 499 client-closed responses during remote streaming,
  preventing transient proxy cancellations from terminating long training runs.
- Ensured a process-local shutdown watcher lets spawned DataLoader workers that are idle
  or prefetched exit cleanly after the terminal sentinel without needing to enter Hub
  retry code.
- Filtered processor-backed training and validation examples whose cleaned text produces
  empty token labels, preventing zero-target Parakeet TDT loss division while preserving
  lazy streaming, and rejected malformed transducer decoder/label contracts.
- Fixed cooperative shutdown for spawned DataLoader workers: terminal Transformers
  callbacks now signal the inherited per-run sentinel before the training iterator is
  destroyed, and terminal Hub retries exit disposable workers directly on Linux rather
  than entering unsafe native finalisation. The parent still re-raises transient errors,
  retains bounded join grace, and leaves local/materialised reads unchanged.
- Deferred bounded runs' terminal scheduled evaluation until training workers have shut
  down when early stopping is disabled, preventing spawned DataLoader worker aborts while
  preserving intermediate evaluations, terminal checkpoints, W&B reporting, and
  step-tagged metrics; early-stopping runs retain in-loop terminal evaluation.
- Hardened evaluation transport logging so temporary signed Hugging Face dataset URLs
  are not emitted at INFO by HTTP, Hub, or filesystem transport loggers.
- Made Parakeet transducer generation safe under distributed wrappers, enabled
  revision-pinned Hydra-composable TDT evaluation, and documented bounded checkpoint
  and metric persistence for isolated TDT smoke and pilot runs.
- Added a tiny Parakeet TDT checkpoint resume lifecycle regression test.
- Fixed the Parakeet TDT runbook smoke and resume retention limit so both the tracked
  best/source checkpoint and newest resume checkpoint survive rotation.
- Added bounded, jittered retries for transient Hugging Face Hub streaming reads,
  including process-local client recreation for closed-client failures; preserved
  upstream 416-as-EOF handling and policy reconfiguration; local and materialised
  file reads remain unchanged.
- Partitioned local VTT manifests into configurable metadata-only shards so Sparkie's
  spawned DataLoader workers can consume DRTV and YouTube examples in parallel.
- Replaced Trainer padding sentinels in generated token IDs before ASR metric decoding
  without mutating aggregated evaluation predictions.
- Matched floating-point Cohere generation features to the model inference dtype while
  preserving decoder IDs, attention masks, labels, and training inputs, including when
  evaluation uses standard distributed model wrappers.
- Removed the zero-yield Danish VoxPopuli stream from the active Sparkie mix and
  reassigned its 0.049091 probability to FTSpeech, preserving the 15-source 60/40
  Danish/English mixture; the reusable dataset configuration and unified provenance
  remain available for other sources.
- Restricted mirrored base and overlay shard validation to effective positional joins,
  preserving independent keyed-overlay shard selection in training and preflight.
- Hardened finetuning-data preflight transport logging before Hub access, preventing
  signed URLs and credentials from entering its logs.
- Bounded Sparkie startup by selecting only each Danish source's pinned base and
  positional-overlay shards, validating mirrored shard order, and shuffling joined
  metadata before audio decoding.
- Restored background-noise augmentation with torchaudio 2.10 by using a project-owned
  soundfile decoder while retaining torch-audiomentations resampling semantics.
- Recognised refreshed Danish overlay `review` actions while continuing to exclude
  them from training alongside the other disallowed actions.
- Removed package-import logging configuration so Hydra's policy controls both main
  and spawned DataLoader worker logs, keeping temporary signed dataset URLs out of
  training logs; CLI fallback logging no longer runs during spawned worker imports.
- Made local VTT and standardised streaming training graphs pickleable for PyTorch
  `spawn` workers without changing lazy audio loading or restartability.
- Normalised Sparkie `dataset_num_workers=1` to in-process filtering and mapping, and
  stopped requesting multiprocessing when materialising iterable validation datasets;
  the production positional-overlay graph now uses one training audio loader worker
  because multi-shard bases are incompatible with worker-local positional offsets.
  Generic keyed and non-positional graphs remain unchanged.
- Used the configured Hugging Face datasets cache when materialising validation
  streaming datasets without an explicit cache directory.
- Corrected the production NST source discriminator to `nst_da` across the base and
  overlay filters, restoring accepted NST rows.
- Restricted shared positional-overlay preflight to string source discriminators with
  unambiguous equality checks and non-colliding output columns.
- Projected overlay metadata before applying row filters, avoiding large unused model
  columns and audio requests during Sparkie v6 preflight while validating the original
  audio schema.
- Resolved features for untyped streaming datasets before filtering and overlaying,
  and made repeated streaming filters safe without materialising or decoding audio.
- Made W&B preflight non-interactive, scrubbed inherited Sparkie W&B identity, and
  redacted checkpoint paths from online configuration payloads.
- Separated bounded pilot stopping from the 100,000-step scheduler horizon, added
  resumable tracking finalisation with success and failure statuses, and hardened W&B
  preflight, IDs, environment policy, and configuration redaction.
- Restored the codebase MIT license and removed the P1 dataset license files.
- Preserved repeated reference tokens in Parakeet RNNT and TDT metrics.
