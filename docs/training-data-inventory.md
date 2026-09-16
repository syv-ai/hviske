# Hviske v6.0 training-data inventory

## Scope

This inventory covers the 15 active streams in `config/sparkie_bilingual.yaml`. The
reusable Danish VoxPopuli configuration remains available but is not selected. Hours are
for the configured training split where a split-level figure is available.

The figures are not yet a uniform post-filter measurement:

- DRTV and YouTube were measured directly from their Sparkie manifests.
- P1 is the current reported estimate for the initial segmented release.
- Most public-dataset figures come from dataset cards or existing model provenance.
- People's Speech clean/train is an estimate from its row count and partial duration
  statistics because its card publishes only the 30,000+ hour all-configuration total.

Exact post-quality, post-deduplication hours under the strict `1 < duration < 10`
runtime filter remain a required pre-training artefact.

## Dataset inventory

| Dataset / stream            | Lang. | Category               | Nominal hours |
| --------------------------- | ----- | ---------------------- | ------------: |
| P1 segments                 | da    | Broadcast/conversation |        ~7,500 |
| DRTV local                  | da    | Broadcast/conversation |      6,442.55 |
| Danish YouTube local        | da    | Broadcast/conversation |      2,094.05 |
| CoRal v3 conversation       | da    | Broadcast/conversation |        156.27 |
| CoRal v3 read-aloud         | da    | Read/prepared          |        521.46 |
| Nota                        | da    | Read/prepared          |          ~584 |
| NST-da                      | da    | Read/prepared          |          ~250 |
| FTSpeech                    | da    | Parliament             |      1,816.29 |
| People's Speech clean       | en    | Mixed/heterogeneous    |        ~5,790 |
| AMI SDM                     | en    | Conversation/meeting   |           ~80 |
| AMI IHM                     | en    | Conversation/meeting   |           ~80 |
| VoxPopuli English           | en    | Parliament             |          ~543 |
| LibriSpeech train-clean-100 | en    | Read/prepared          |          ~100 |
| LibriSpeech train-clean-360 | en    | Read/prepared          |          ~360 |
| LibriSpeech train-other-500 | en    | Read/prepared          |          ~500 |

Only the local manifests currently have exact post-filter measurements:

| Dataset              |         Accepted rows |   Strict 1--10s hours |
| -------------------- | --------------------: | --------------------: |
| DRTV local           |             5,178,843 |              6,390.27 |
| Danish YouTube local |             3,208,785 |              2,001.97 |
| P1 segments          | Pending final release | Pending final release |

### Interpretation notes

- The strictly read-aloud Danish stream, CoRal read-aloud, contains **521.46 hours**.
- The broader Danish read/prepared category contains approximately **1,355 hours**
  before the shared duration filter.
- FTSpeech is parliamentary meeting speech, not read-aloud speech.
- Nota consists of professionally prepared audiomagazine readings. It remains in
  read/prepared even though an earlier model card called it broadcast media.
- AMI SDM and IHM contain different microphone views of substantially the same meetings.
  They are separate training streams but not 160 unique meeting hours.
- Danish VoxPopuli is excluded from the active mix because, at unified audio revision
  `5a3a49ee981baab6e1e37ddd2c45f9943c27d08f` and positional overlay shard range
  `0--354`, its first 100 joined clips are OGG mono 16 kHz (duration min 16.15,
  median 30, max 30); none survive the strict `1 < duration < 10` filter. Keeping it
  active would scan 1.745 million rows without yielding an example. Its reusable dataset
  YAML is retained for future use.
- NST's approximately 250 hours come from the existing Hviske v5 provenance rather than
  a full-split duration scan.
- People's Speech has 1,501,271 clean/train rows. The approximately 5,790-hour estimate
  uses the dataset server's partial mean duration of 13.89 seconds and is not precise
  enough for final weighting. The 10-second filter may reject a large share.
- People's Speech mixes interviews, government recordings, radio, lectures, sermons, and
  other sources. Do not count the whole corpus as confirmed conversation without a
  source-level stratification.

## Language by category matrix

This matrix sums the nominal stream hours above. It is suitable for understanding corpus
scale, not for freezing sampling probabilities.

| Lang.     |       Read | Conversation |  Parliament |      Mixed |       Total |
| --------- | ---------: | -----------: | ----------: | ---------: | ----------: |
| Danish    |     ~1,355 |      ~16,193 |      ~1,816 |          0 |     ~19,364 |
| English   |       ~960 |         ~160 |        ~543 |     ~5,790 |      ~7,453 |
| **Total** | **~2,315** |  **~16,353** |  **~2,359** | **~5,790** | **~26,817** |

The English conversation total counts both AMI microphone streams. Subtract roughly 80
hours for a unique-meeting estimate. The matrix also includes estimated People's Speech
hours, so its totals must not be presented as final usable hours.

## Sampling implications

The previous probability matrix was:

| Lang.     |     Read | Conversation | Parliament |    Mixed |    Total |
| --------- | -------: | -----------: | ---------: | -------: | -------: |
| Danish    |     0.14 |         0.35 |       0.11 |        0 |     0.60 |
| English   |     0.08 |         0.07 |       0.09 |     0.16 |     0.40 |
| **Total** | **0.22** |     **0.42** |   **0.20** | **0.16** | **1.00** |

The earlier 0.58 conversation figure incorrectly counted all 0.16 People's Speech mass
as conversation. Confirmed conversational sources received only 0.42.

The v6.0 configuration now allocates 75% of the Danish sampling mass to
broadcast/conversation. This is a product-policy prior, not a conclusion inferred from
corpus hours. It preserves the 60/40 Danish/English split and leaves the English groups
unchanged for the initial release:

| Lang.     |     Read | Conversation | Parliament |    Mixed |    Total |
| --------- | -------: | -----------: | ---------: | -------: | -------: |
| Danish    |     0.06 |         0.45 |       0.09 |        0 |     0.60 |
| English   |     0.08 |         0.07 |       0.09 |     0.16 |     0.40 |
| **Total** | **0.14** |     **0.52** |   **0.18** | **0.16** | **1.00** |

Stratify People's Speech by source type when practical, but continue to report it as
mixed until then. Do not inflate the conversation total by repeatedly oversampling AMI's
roughly 80 unique meeting hours.

The provisional source probabilities preserve the previous within-group proportions.
Recalculate them after every source has accepted row counts and mean accepted durations.
For desired audio share `a_s` and mean segment duration `d_s`:

```text
p_s proportional to a_s / d_s
```

This prevents a short-segment corpus from receiving much less audio exposure than its
example probability suggests. Freeze the source probabilities before the learning-rate
pilots and record realised examples and audio seconds during training.

## Evidence

- [CoRal v3](https://hf.co/datasets/CoRal-project/coral-v3)
- [FTSpeech documentation](https://ftspeech.github.io/about.html)
- [Nota](https://hf.co/datasets/alexandrainst/nota)
- [NST-da](https://hf.co/datasets/alexandrainst/nst-da)
- [People's Speech](https://hf.co/datasets/MLCommons/peoples_speech)
- [VoxPopuli](https://hf.co/datasets/facebook/voxpopuli)
- [LibriSpeech](https://www.openslr.org/12/)
- [Hviske v5 provenance](https://hf.co/syvai/hviske-v5)
