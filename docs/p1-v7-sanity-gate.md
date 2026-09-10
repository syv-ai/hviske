# P1 v7 dozen-clip sanity gate

Run `uv run python src/scripts/run_p1_v7_sanity_gate.py` only after the pilot
has been published. This is a separate validation command and is not called by
segmentation. It reads `scratch/audit-candidates.jsonl`, requires at least twelve
accepted candidates, and retrieves the selected rows one at a time from the private
`syvai/p1-segments` repository at one immutable pilot commit. Before retrieval it
verifies the dataset namespace, private visibility, and exact resolved commit SHA.

Sampling uses the supplied seed and round-robins deterministic hash-ranked candidates
within programme/stratum groups. The default seed is `p1-v7-dozen`; `--pilot-head`
is required and must identify the complete final pilot commit, because candidates
from several publication commits are rebound to that immutable head in memory.

The gate checks the exact `p1-segments-v2` schema and `p1-segmentation-7` contract,
verified metadata/audio/Parquet hashes, in-memory PCM_16 FLAC decoding, finite mono
16 kHz audio, exact millisecond/sample duration, 1--10 second duration, trainable text,
and one speaker. An independently loaded `openai/whisper-small` model is pinned to
revision `973afd24965f72e36ca33b3055d56a652f456b4d`, loaded once, and run in Danish.
GPU 0 is used when available; `--device -1` forces CPU.

The conservative pass assumptions are: every one of the twelve clips passes the
structural checks, no clip produces empty/non-speech ASR, median normalised WER is at
most 0.5, and no more than three clips have WER above 0.8. The JSON report contains
only the pilot head, contract versions, an aggregate sample-set digest, counts, WER
distribution, thresholds, and the pass boolean. It never contains transcripts,
hypotheses, audio, source identifiers, paths, locators, URLs, or cache paths. A
non-zero exit status means that the gate failed or could not be run.
