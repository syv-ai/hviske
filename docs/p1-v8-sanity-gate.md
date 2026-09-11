# P1 v8 structural sanity gate

Run `uv run python src/scripts/run_p1_v8_sanity_gate.py` after publication. The
command reads `scratch/audit-candidates.jsonl`, requires at least twelve accepted
candidates, and retrieves the selected rows one at a time from the private
`syvai/p1-segments` repository at one immutable commit.

Sampling uses the supplied seed and round-robins deterministic hash-ranked candidates
within programme and stratum groups. The default seed is `p1-v8-dozen`; `--pilot-head`
is required and must identify the complete final publication commit.

The model-free gate verifies private immutable retrieval, Parquet, metadata and audio
hashes, the exact `p1-segments-v2` schema, the `p1-segmentation-8` identity, and the
pipeline configuration digest recorded in the audit evidence (or supplied with
`--pipeline-config-sha256`). It also verifies PCM16 FLAC mono 16 kHz audio, exact
duration and source bounds,
non-empty trainable text, and one timed-anchor speaker. It does not load or invoke
ASR, CTC, VAD, Whisper, Transformers, CUDA, or model caches. Transcription
agreement is not a P1 acceptance criterion.

The report contains only the immutable publication head, contract versions, an
aggregate sample-set digest, checks, counts, and the pass boolean. It never contains
transcripts, audio, source identifiers, paths, locators, URLs, or cache paths. A
non-zero exit status means that the structural gate failed or could not be run.
