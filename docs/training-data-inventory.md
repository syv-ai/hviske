# Hviske v6.0 training-data inventory

## Scope

This inventory covers the 15 streams in `config/bilingual.yaml`. The figures are
nominal source or dataset-card estimates, not a substitute for the immutable revision
and post-filter run report required before training.

P1 is consumed as a published, manually gated dataset. Its retired producer is archived
at `syvai/p1-segments`, path `archives/hviske-p1-pipeline/f3dcf16/`, Hub commit
`44284e5849b6b1d96b874891c579654a644e0e2f`.

## Dataset inventory

| Dataset / stream | Language | Category |
| --- | --- | --- |
| P1 segments | da | Broadcast/conversation |
| DRTV local VTT | da | Broadcast/captions |
| YouTube local VTT | da | Video/captions |
| CoRal v3 conversation | da | Broadcast/conversation |
| CoRal v3 read-aloud | da | Read/prepared |
| FTSpeech | da | Parliament |
| Nota | da | Read/prepared |
| NST-da | da | Read/prepared |
| People's Speech clean | en | Mixed/heterogeneous |
| AMI SDM | en | Conversation/meeting |
| AMI IHM | en | Conversation/meeting |
| VoxPopuli | en | Parliament |
| LibriSpeech train-clean-100 | en | Read/prepared |
| LibriSpeech train-clean-360 | en | Read/prepared |
| LibriSpeech train-other-500 | en | Read/prepared |

All sources use the shared 1--8-second runtime filter unless their dataset
configuration states otherwise. Source revisions, columns, and splits—including the
frozen P1 revision—are pinned in `config/datasets/`.

## Sampling implications

The bilingual preset allocates 60% of examples to Danish sources and 40% to English
sources. Source probabilities are an explicit acoustic and linguistic balance rather
than a claim about corpus size or unique hours. Record accepted rows, audio seconds,
and rejection reasons for every source during the data gate and training run.

People's Speech is mixed rather than wholly conversational. AMI microphone views are
separate configured streams but overlap in meeting content. FTSpeech is parliamentary
meeting speech, while Nota and NST remain read/prepared sources.

## Evidence and archival boundary

Use the dataset cards and pinned Hub revisions named by the configuration as evidence for
public sources. The exact P1 producer archive is listed above. This repository contains
only reusable loading, processing, training, evaluation, and publication code. DRTV and
YouTube are consumed from the established local caption
manifests under `$HOME/drtv-asr-dataset`; the manifests and source-specific producers
remain outside this repository.
