"""Focused tests for Transformers-native Parakeet support."""

import typing as t
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from hydra import compose
from omegaconf import OmegaConf
from transformers import (
    EvalPrediction,
    ParakeetProcessor,
    ParakeetTokenizer,
    TrainingArguments,
)
from transformers.modeling_outputs import CausalLMOutput
from transformers.models.parakeet import (
    ParakeetCTCConfig,
    ParakeetEncoderConfig,
    ParakeetForCTC,
    ParakeetForRNNT,
    ParakeetForTDT,
    ParakeetRNNTConfig,
    ParakeetTDTConfig,
)
from transformers.models.parakeet.feature_extraction_parakeet import (
    ParakeetFeatureExtractor,
)
from transformers.pipelines.automatic_speech_recognition import (
    AutomaticSpeechRecognitionPipeline,
)

from hviske.cohere import get_asr_call_kwargs, load_asr_transcriber
from hviske.compute_metrics import compute_error_rate_metrics
from hviske.data import process_example
from hviske.data_collators import DataCollatorParakeetWithPadding
from hviske.data_models import Processor
from hviske.model_setup import load_model_setup
from hviske.parakeet import (
    ParakeetGenerationTrainer,
    ParakeetModelSetup,
    parakeet_family,
)


def test_parakeet_collator_pads_frames_masks_and_labels() -> None:
    """Feature width is inferred by the feature extractor rather than hard-coded."""
    processor = SimpleNamespace(
        feature_extractor=ParakeetFeatureExtractor(feature_size=3),
        tokenizer=MagicMock(),
    )
    processor.tokenizer.pad.return_value = {"input_ids": torch.tensor([[4, 5], [6, 0]])}
    collator = DataCollatorParakeetWithPadding(
        processor=t.cast(Processor, processor), sample_rate=16_000, padding="longest"
    )

    batch = collator(
        [
            {
                "input_features": torch.zeros(4, 3),
                "attention_mask": torch.ones(4),
                "labels": [4, 5],
            },
            {
                "input_features": torch.zeros(6, 3),
                "attention_mask": torch.ones(6),
                "labels": [6],
            },
        ]
    )

    assert batch["input_features"].shape == (2, 6, 3)
    assert batch["attention_mask"].shape == (2, 6)
    assert batch["attention_mask"][0].tolist() == [1, 1, 1, 1, 0, 0]
    assert batch["labels"].tolist() == [[4, 5], [6, 0]]


def test_parakeet_collator_pads_rnnt_decoder_inputs() -> None:
    """RNNT decoder inputs are padded independently from transcript labels."""
    processor = SimpleNamespace(
        feature_extractor=ParakeetFeatureExtractor(feature_size=3),
        tokenizer=MagicMock(),
        blank_token_id=0,
    )
    processor.tokenizer.pad.side_effect = [
        {"input_ids": torch.tensor([[1, 2], [3, 0]])},
        {"input_ids": torch.tensor([[0, 1, 2], [0, 3, 0]])},
    ]
    collator = DataCollatorParakeetWithPadding(
        processor=t.cast(Processor, processor), sample_rate=16_000, padding="longest"
    )

    batch = collator(
        [
            {
                "input_features": torch.zeros(4, 3),
                "attention_mask": torch.ones(4),
                "labels": [1, 2],
                "decoder_input_ids": [0, 1, 2],
            },
            {
                "input_features": torch.zeros(6, 3),
                "attention_mask": torch.ones(6),
                "labels": [3],
                "decoder_input_ids": [0, 3],
            },
        ]
    )

    assert batch["labels"].tolist() == [[1, 2], [3, 0]]
    assert batch["decoder_input_ids"].tolist() == [[0, 1, 2], [0, 3, 0]]


def test_parakeet_collator_uses_blank_distinct_from_pad() -> None:
    """TDT/RNNT decoder padding retains the processor's separate blank ID."""
    processor = _tiny_parakeet_processor()
    collator = DataCollatorParakeetWithPadding(
        processor=processor, sample_rate=16_000, padding="longest"
    )

    batch = collator(
        [
            {
                "input_features": torch.zeros(2, 1),
                "labels": [1],
                "decoder_input_ids": [5, 1],
            },
            {
                "input_features": torch.zeros(2, 1),
                "labels": [1, 2],
                "decoder_input_ids": [5, 1, 2],
            },
        ]
    )

    assert batch["decoder_input_ids"].tolist() == [[5, 1, 0], [5, 1, 2]]


def _tiny_parakeet_processor() -> ParakeetProcessor:
    tokenizer = ParakeetTokenizer(
        vocab={"<pad>": 0, "a": 1, "<unk>": 2, "<s>": 3, "</s>": 4, "<blank>": 5},
        pad_token="<pad>",
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        blank_token="<blank>",
    )
    return ParakeetProcessor(
        feature_extractor=ParakeetFeatureExtractor(feature_size=1),
        tokenizer=tokenizer,
        blank_token="<blank>",
        decoder_type="rnnt",
    )


def test_parakeet_danish_tokens_resize_ctc_and_save_processor(tmp_path: Path) -> None:
    """Danish additions preserve CTC rows, IDs, and processor persistence."""
    setup = _tiny_parakeet_setup()
    processor = setup._adapt_processor(processor=_tiny_parakeet_processor())
    tokenizer = processor.tokenizer
    assert tokenizer("æøå", add_special_tokens=False)["input_ids"] == [6, 7, 8]
    assert tokenizer("a", add_special_tokens=False)["input_ids"] == [1]
    assert tokenizer.pad_token_id == 0
    assert tokenizer.convert_tokens_to_ids("<blank>") == 5

    processor.save_pretrained(save_directory=tmp_path)
    reloaded = ParakeetProcessor.from_pretrained(tmp_path)
    assert reloaded.tokenizer("æøå", add_special_tokens=False)["input_ids"] == [6, 7, 8]

    model = ParakeetForCTC(
        ParakeetCTCConfig(
            encoder_config=_tiny_parakeet_encoder_config(), vocab_size=6, pad_token_id=0
        )
    )
    old_weight = model.ctc_head.weight.detach().clone()
    assert model.ctc_head.bias is not None
    old_bias = model.ctc_head.bias.detach().clone()
    setup.processor = processor
    setup._resize_model_vocabulary(model=model, family="ctc")
    assert model.ctc_head.out_channels == 9
    assert torch.equal(model.ctc_head.weight[:6], old_weight)
    assert model.ctc_head.bias is not None
    assert torch.equal(model.ctc_head.bias[:6], old_bias)
    assert model.config.vocab_size == 9

    outputs = model(
        input_features=torch.randn(2, 64, 16),
        attention_mask=torch.ones(2, 64),
        labels=torch.tensor([[6, 7, 8], [6, 7, 0]]),
    )
    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)


def _tiny_parakeet_encoder_config() -> ParakeetEncoderConfig:
    return ParakeetEncoderConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        conv_kernel_size=3,
        subsampling_conv_channels=4,
        num_mel_bins=16,
        max_position_embeddings=100,
        dropout=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
    )


def _tiny_parakeet_setup() -> ParakeetModelSetup:
    return ParakeetModelSetup(
        config=OmegaConf.create(
            {
                "model": {
                    "type": "parakeet",
                    "characters_to_keep": "æøå",
                    "pretrained_model_id": "checkpoint",
                }
            }
        )
    )


def test_parakeet_danish_tokens_resize_rnnt_and_keep_blank() -> None:
    """RNNT heads retain rows while the blank remains the generation start."""
    setup = _tiny_parakeet_setup()
    processor = setup._adapt_processor(processor=_tiny_parakeet_processor())
    model = ParakeetForRNNT(
        ParakeetRNNTConfig(
            encoder_config=_tiny_parakeet_encoder_config(),
            vocab_size=6,
            decoder_hidden_size=4,
            num_decoder_layers=1,
            pad_token_id=0,
            blank_token_id=5,
        )
    )
    old_embedding = model.decoder.embedding.weight.detach().clone()
    old_head_weight = model.joint.head.weight.detach().clone()
    old_head_bias = model.joint.head.bias.detach().clone()
    setup.processor = processor
    setup._resize_model_vocabulary(model=model, family="rnnt")

    assert model.decoder.embedding.num_embeddings == 9
    assert model.joint.head.out_features == 9
    assert torch.equal(model.decoder.embedding.weight[:6], old_embedding)
    assert torch.equal(model.joint.head.weight[:6], old_head_weight)
    assert torch.equal(model.joint.head.bias[:6], old_head_bias)
    assert model.config.vocab_size == 9
    assert model.config.blank_token_id == 5
    assert model.generation_config.decoder_start_token_id == 5

    outputs = model(
        input_features=torch.randn(2, 64, 16),
        attention_mask=torch.ones(2, 64),
        decoder_input_ids=torch.tensor([[5, 6, 7, 8], [5, 6, 7, 0]]),
        labels=torch.tensor([[6, 7, 8], [6, 7, 5]]),
    )
    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)


def test_parakeet_danish_tokens_resize_tdt_preserves_duration_head() -> None:
    """TDT vocabulary growth preserves duration rows and the blank start ID."""
    setup = _tiny_parakeet_setup()
    processor = setup._adapt_processor(processor=_tiny_parakeet_processor())
    model = ParakeetForTDT(
        ParakeetTDTConfig(
            encoder_config=_tiny_parakeet_encoder_config(),
            vocab_size=6,
            decoder_hidden_size=4,
            num_decoder_layers=1,
            pad_token_id=0,
            blank_token_id=5,
            durations=(0, 1, 2),
        )
    )
    old_head = model.joint.head.weight.detach().clone()
    old_duration_head = old_head[-3:].clone()
    setup.processor = processor
    setup._resize_model_vocabulary(model=model, family="tdt")

    assert model.decoder.embedding.num_embeddings == 9
    assert model.joint.head.out_features == 12
    assert torch.equal(model.joint.head.weight[-3:], old_duration_head)
    assert model.config.blank_token_id == 5
    assert model.generation_config.decoder_start_token_id == 5


def test_parakeet_dispatch() -> None:
    """The model factory selects the Parakeet setup."""
    config = OmegaConf.create({"model": {"type": "parakeet"}})
    assert isinstance(load_model_setup(config), ParakeetModelSetup)


def test_parakeet_evaluation_revision_is_hydra_composable() -> None:
    """The documented zero-shot revision is accepted by the evaluation config."""
    config = compose(
        config_name="evaluation",
        overrides=[
            "model_id=nvidia/parakeet-tdt-0.6b-v3",
            "model_revision=541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
        ],
    )

    assert config.model_revision == "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"


def test_parakeet_family_rejects_collection_checkpoint() -> None:
    """NeMo collection configs fail before an auto model is selected."""
    with pytest.raises(ValueError, match="Unsupported Parakeet checkpoint"):
        parakeet_family(SimpleNamespace(model_type="nemo"))


def test_parakeet_family_supports_tdt_architecture() -> None:
    """TDT checkpoints use the native Transformers family."""
    assert parakeet_family(SimpleNamespace(model_type="parakeet_tdt")) == "tdt"


def test_parakeet_family_uses_checkpoint_architecture() -> None:
    """RNNT is identified from native Transformers config data."""
    assert (
        parakeet_family(
            SimpleNamespace(model_type="unknown", architectures=["ParakeetForRNNT"])
        )
        == "rnnt"
    )


@pytest.mark.parametrize("wrapper_kind", ["data_parallel", "distributed_data_parallel"])
def test_parakeet_generation_trainer_unwraps_parallel_model(
    wrapper_kind: str, tmp_path: Path
) -> None:
    """Generation uses the underlying model for parallel wrappers."""
    process_group_initialised = False
    try:
        if wrapper_kind == "distributed_data_parallel":
            torch.distributed.init_process_group(
                backend="gloo",
                init_method=f"file://{tmp_path / 'ddp-init'}",
                rank=0,
                world_size=1,
            )
            process_group_initialised = True

        underlying_model = _TinyGenerationModel()
        if wrapper_kind == "data_parallel":
            data_parallel_model = torch.nn.DataParallel(underlying_model)
            # Exercise the real wrapper type without coupling this CPU Trainer test to
            # whichever CUDA devices happen to be visible on the host.
            data_parallel_model.device_ids = []
            data_parallel_model.module.to(device="cpu")
            wrapped_model: torch.nn.Module = data_parallel_model
        else:
            wrapped_model = torch.nn.parallel.DistributedDataParallel(underlying_model)
        trainer = ParakeetGenerationTrainer(
            model=underlying_model,
            args=TrainingArguments(
                output_dir=str(tmp_path), use_cpu=True, report_to=[]
            ),
        )

        loss, predictions, labels = trainer.prediction_step(
            model=wrapped_model,
            inputs={
                "input_features": torch.zeros(1, 2, 3),
                "attention_mask": torch.ones(1, 2),
                "labels": torch.tensor([[1, 2]]),
            },
            prediction_loss_only=False,
        )

        assert loss is not None
        assert torch.equal(t.cast(torch.Tensor, predictions), torch.tensor([[1, 2]]))
        assert torch.equal(t.cast(torch.Tensor, labels), torch.tensor([[1, 2]]))
    finally:
        if process_group_initialised:
            torch.distributed.destroy_process_group()


class _TinyGenerationModel(torch.nn.Module):
    config = SimpleNamespace(model_type="parakeet_rnnt", keys_to_ignore_at_inference=[])

    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> CausalLMOutput:
        return CausalLMOutput(
            loss=t.cast(torch.FloatTensor, self.scale * 0.5),
            logits=t.cast(torch.FloatTensor, input_features),
        )

    def generate(
        self, input_features: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> SimpleNamespace:
        return SimpleNamespace(sequences=torch.tensor([[1, 2]]))


def test_parakeet_generation_trainer_unwraps_transducer_output(tmp_path: Path) -> None:
    """Trainer metrics receive sequences, not the RNNT generation wrapper."""
    trainer = ParakeetGenerationTrainer(
        model=_TinyGenerationModel(),
        args=TrainingArguments(output_dir=str(tmp_path), use_cpu=True, report_to=[]),
    )
    loss, predictions, labels = trainer.prediction_step(
        model=t.cast(torch.nn.Module, trainer.model),
        inputs={
            "input_features": torch.zeros(1, 2, 3),
            "attention_mask": torch.ones(1, 2),
            "labels": torch.tensor([[1, 2]]),
        },
        prediction_loss_only=False,
    )

    assert loss is not None
    assert torch.equal(t.cast(torch.Tensor, predictions), torch.tensor([[1, 2]]))
    assert torch.equal(t.cast(torch.Tensor, labels), torch.tensor([[1, 2]]))


def test_parakeet_inference_omits_whisper_generation_kwargs() -> None:
    """Shared pipelines must not send language/task kwargs to Parakeet."""
    transcriber = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(model_type="parakeet_rnnt"))
    )
    assert (
        get_asr_call_kwargs(t.cast(AutomaticSpeechRecognitionPipeline, transcriber))
        == {}
    )


def test_parakeet_load_model_selects_native_auto_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CTC uses AutoModelForCTC while transducers use AutoModel."""
    config = OmegaConf.create(
        {
            "model": {
                "type": "parakeet",
                "pretrained_model_id": "checkpoint",
                "freeze_feature_encoder": False,
            },
            "gradient_checkpointing": False,
        }
    )
    setup = ParakeetModelSetup(config=config)
    ctc_model = SimpleNamespace(config=SimpleNamespace())
    ctc_loader = MagicMock(return_value=ctc_model)
    general_loader = MagicMock()
    monkeypatch.setattr(
        "hviske.parakeet.AutoConfig.from_pretrained",
        MagicMock(return_value=SimpleNamespace(model_type="parakeet_ctc")),
    )
    monkeypatch.setattr("hviske.parakeet.AutoModelForCTC.from_pretrained", ctc_loader)
    monkeypatch.setattr("hviske.parakeet.AutoModel.from_pretrained", general_loader)

    assert setup.load_model() is ctc_model
    ctc_loader.assert_called_once()
    general_loader.assert_not_called()


def test_parakeet_load_model_selects_native_tdt_auto_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TDT uses AutoModelForTDT rather than the generic auto model."""
    config = OmegaConf.create(
        {
            "model": {
                "type": "parakeet",
                "pretrained_model_id": "checkpoint",
                "freeze_feature_encoder": False,
            },
            "gradient_checkpointing": False,
        }
    )
    setup = ParakeetModelSetup(config=config)
    tdt_model = SimpleNamespace(config=SimpleNamespace())
    tdt_loader = MagicMock(return_value=tdt_model)
    monkeypatch.setattr(
        "hviske.parakeet.AutoConfig.from_pretrained",
        MagicMock(return_value=SimpleNamespace(model_type="parakeet_tdt")),
    )
    monkeypatch.setattr("hviske.parakeet.AutoModelForTDT.from_pretrained", tdt_loader)

    assert setup.load_model() is tdt_model
    tdt_loader.assert_called_once()


def test_parakeet_local_paths_omit_base_revision(tmp_path: Path) -> None:
    """Saved checkpoints are not resolved against the pinned base revision."""
    setup = _tiny_parakeet_setup()
    setup.config.model.revision = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
    assert setup._hub_kwargs(model_id=str(tmp_path)) == {}
    assert setup._hub_kwargs(model_id="nvidia/parakeet-tdt-0.6b-v3")["revision"] == (
        "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
    )


def test_parakeet_local_pipeline_omits_revision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Local evaluation paths are not given a Hub-only revision argument."""
    pipeline = MagicMock(return_value=object())
    monkeypatch.setattr("hviske.cohere.pipeline", pipeline)
    monkeypatch.setattr(
        "hviske.cohere.AutoConfig.from_pretrained",
        MagicMock(return_value=SimpleNamespace(model_type="wav2vec2")),
    )

    load_asr_transcriber(
        model_id=str(tmp_path), no_lm=False, device=torch.device("cpu")
    )

    assert "revision" not in pipeline.call_args.kwargs


def test_parakeet_metric_decodes_generated_sequences() -> None:
    """RNNT generated IDs use the same tokenizer path as CTC labels."""
    tokenizer = MagicMock(pad_token_id=0)
    tokenizer.batch_decode.side_effect = [["hej"], ["hej"]]
    processor = SimpleNamespace(
        tokenizer=tokenizer, batch_decode=tokenizer.batch_decode
    )
    prediction = EvalPrediction(
        predictions=torch.tensor([[1, 2]]).numpy(),
        label_ids=torch.tensor([[1, 2]]).numpy(),
    )

    metrics = compute_error_rate_metrics(
        pred=prediction, processor=t.cast(Processor, processor), log_examples=False
    )

    assert metrics == {"cer": 0.0, "wer": 0.0}
    assert tokenizer.batch_decode.call_count == 2


def test_parakeet_metrics_replace_trainer_padding() -> None:
    """Variable-length generated IDs never pass Trainer's -100 to decoding."""
    tokenizer = MagicMock(pad_token_id=0)
    tokenizer.batch_decode.side_effect = [["hej", "du"], ["hej", "du"]]
    processor = SimpleNamespace(
        tokenizer=tokenizer, batch_decode=tokenizer.batch_decode
    )
    setup = ParakeetModelSetup(
        config=OmegaConf.create(
            {"model": {"type": "parakeet", "pretrained_model_id": "checkpoint"}}
        )
    )
    setup.processor = t.cast(Processor, processor)

    metrics = setup.load_compute_metrics()(
        EvalPrediction(
            predictions=np.array([[1, 2], [3, -100]]),
            label_ids=np.array([[1, 2], [3, 0]]),
        )
    )

    assert metrics == {"cer": 0.0, "wer": 0.0}
    decoded_predictions = tokenizer.batch_decode.call_args_list[0].args[0]
    assert -100 not in decoded_predictions
    assert decoded_predictions.tolist() == [[1, 2], [3, 0]]


def test_parakeet_processing_keeps_joint_decoder_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RNNT processing retains decoder IDs returned by the real processor contract."""
    monkeypatch.setattr("hviske.data.download_background_noises", lambda: None)

    class FakeParakeetProcessor:
        blank_token = "<blank>"
        blank_token_id = 4
        decoder_type = "rnnt"

        def __call__(self, audio: object, text: str, sampling_rate: int) -> dict:
            assert len(t.cast(np.ndarray, audio)) == 32_000
            assert text == "hej"
            assert sampling_rate == 16_000
            return {
                "input_features": [[[0.0]]],
                "attention_mask": [[1]],
                "decoder_input_ids": [[4, 7]],
                "labels": [[7]],
            }

    processed = process_example(
        example={
            "text": "hej",
            "audio": {
                "array": np.zeros(32_000, dtype=np.float32),
                "sampling_rate": 16_000,
            },
        },
        characters_to_keep=None,
        conversion_dict={},
        text_column="text",
        audio_column="audio",
        lower_case=False,
        convert_numerals=False,
        processor=t.cast(t.Callable, FakeParakeetProcessor()),
        normalise_audio=False,
        augment_audio=False,
    )

    assert processed["decoder_input_ids"] == [4, 7]
    assert processed["num_seconds"] == 2.0


@pytest.mark.parametrize(
    ("decoder_input_ids", "labels"),
    [([5, 7], [7]), ([4, 8], [7]), ([4], [7])],
    ids=["wrong blank", "wrong label", "wrong length"],
)
def test_parakeet_processing_rejects_invalid_transducer_contract(
    decoder_input_ids: list[int], labels: list[int]
) -> None:
    """Transducer decoder inputs must be the blank-prefixed label sequence."""

    class MismatchedParakeetProcessor:
        blank_token = "<blank>"
        blank_token_id = 4
        decoder_type = "tdt"

        def __call__(self, audio: object, text: str, sampling_rate: int) -> dict:
            del audio, text, sampling_rate
            return {
                "input_features": [[[0.0]]],
                "attention_mask": [[1]],
                "decoder_input_ids": [decoder_input_ids],
                "labels": [labels],
            }

    with pytest.raises(ValueError, match="exactly one more token"):
        process_example(
            example={
                "text": "hej",
                "audio": {
                    "array": np.zeros(16_000, dtype=np.float32),
                    "sampling_rate": 16_000,
                },
            },
            characters_to_keep=None,
            conversion_dict={},
            text_column="text",
            audio_column="audio",
            lower_case=False,
            convert_numerals=False,
            processor=t.cast(t.Callable, MismatchedParakeetProcessor()),
            normalise_audio=False,
            augment_audio=False,
        )


def test_parakeet_rejects_max_length_padding() -> None:
    """Frame max length cannot be inferred from the shared training config."""
    processor = SimpleNamespace(
        feature_extractor=ParakeetFeatureExtractor(feature_size=3),
        tokenizer=MagicMock(),
    )
    with pytest.raises(ValueError, match="padding='max_length'"):
        DataCollatorParakeetWithPadding(
            processor=t.cast(Processor, processor),
            sample_rate=16_000,
            padding="max_length",
        )


def test_parakeet_rnnt_forward_computes_native_loss() -> None:
    """A real tiny Transformers RNNT model computes its native loss."""
    encoder_config = ParakeetEncoderConfig(
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        conv_kernel_size=3,
        subsampling_conv_channels=4,
        num_mel_bins=16,
        max_position_embeddings=100,
        dropout=0.0,
        layerdrop=0.0,
        activation_dropout=0.0,
        attention_dropout=0.0,
    )
    model = ParakeetForRNNT(
        ParakeetRNNTConfig(
            encoder_config=encoder_config,
            vocab_size=5,
            decoder_hidden_size=4,
            num_decoder_layers=1,
            pad_token_id=4,
            blank_token_id=4,
        )
    )

    outputs = model(
        input_features=torch.randn(2, 64, 16),
        attention_mask=torch.ones(2, 64),
        decoder_input_ids=torch.tensor([[4, 1, 2], [4, 1, 4]]),
        labels=torch.tensor([[1, 2], [1, 4]]),
    )

    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)


def test_parakeet_tdt_forward_has_finite_loss_and_gradients() -> None:
    """A native tiny TDT model computes a trainable finite loss."""
    model = ParakeetForTDT(
        ParakeetTDTConfig(
            encoder_config=_tiny_parakeet_encoder_config(),
            vocab_size=6,
            decoder_hidden_size=4,
            num_decoder_layers=1,
            pad_token_id=0,
            blank_token_id=5,
            durations=(0, 1, 2),
        )
    )
    outputs = model(
        input_features=torch.randn(2, 64, 16),
        attention_mask=torch.ones(2, 64),
        decoder_input_ids=torch.tensor([[5, 1, 2], [5, 1, 0]]),
        labels=torch.tensor([[1, 2], [1, 0]]),
    )
    assert outputs.loss is not None
    assert torch.isfinite(outputs.loss)
    outputs.loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert gradients
    assert all(
        gradient is not None and torch.isfinite(gradient).all()
        for gradient in gradients
    )


def test_parakeet_tdt_runbook_overrides_compose() -> None:
    """Runbook checkpoint and metric overrides compose in the training config."""
    smoke = compose(
        config_name="asr_finetuning",
        overrides=[
            "model=parakeet-tdt",
            "model_dir=/tmp/parakeet-tdt-smoke",
            "stop_after_steps=2",
            "max_steps=2",
            "save_steps=2",
            "save_total_limit=2",
            "evaluation_steps=[2]",
            "evaluation_metrics_path=/tmp/parakeet-tdt-smoke/metrics.jsonl",
            "enable_experiment_tracking=false",
        ],
    )
    assert smoke.save_steps == 2
    assert smoke.save_total_limit == 2
    assert smoke.evaluation_steps == [2]
    assert smoke.evaluation_metrics_path.endswith("metrics.jsonl")
    assert smoke.enable_experiment_tracking is False

    pilot = compose(
        config_name="asr_finetuning",
        overrides=[
            "model=parakeet-tdt",
            "save_steps=250",
            "save_total_limit=8",
            "evaluation_steps=[250,500,1000,2000]",
            "stop_after_steps=2000",
            "max_steps=100000",
            "enable_experiment_tracking=false",
        ],
    )
    assert pilot.save_steps == 250
    assert pilot.save_total_limit == 8
    assert pilot.evaluation_steps == [250, 500, 1000, 2000]
    assert pilot.stop_after_steps == 2000
    assert pilot.max_steps == 100000


def test_parakeet_tdt_runbook_saves_requested_checkpoints_and_metrics() -> None:
    """Runbook commands retain every requested checkpoint and metric record."""
    runbook = (
        Path(__file__).parents[1] / "docs" / "parakeet-tdt-runbook.md"
    ).read_text(encoding="utf-8")

    smoke = runbook[runbook.index("smoke_dir=") : runbook.index("After it exits")]
    assert "save_steps=2 save_total_limit=2" in smoke
    assert "save_total_limit=1" not in smoke
    assert 'evaluation_metrics_path="$smoke_dir/evaluation-metrics.jsonl"' in smoke
    assert "enable_experiment_tracking=false" in smoke

    resume_start = runbook.index("resume_from_checkpoint")
    resume = runbook[resume_start : runbook.index("**Gate:", resume_start)]
    assert "checkpoint-2" in resume
    assert "save_steps=2 save_total_limit=2" in resume
    assert "save_total_limit=1" not in resume
    assert 'evaluation_metrics_path="$smoke_dir/evaluation-metrics.jsonl"' in resume

    assert "best_model_checkpoint" in runbook
    assert "both the best/source checkpoint and newest resume checkpoint" in runbook

    pilots = runbook[runbook.index("for lr") :]
    assert "evaluation_steps='[250,500,1000,2000]'" in pilots
    assert "save_steps=250 save_total_limit=8" in pilots
    assert 'evaluation_metrics_path="$run_dir/evaluation-metrics.jsonl"' in pilots


def test_parakeet_tdt_save_and_clean_reload(tmp_path: Path) -> None:
    """A tiny TDT model and processor reload from local files."""
    processor = _tiny_parakeet_processor()
    processor.decoder_type = "tdt"
    model = ParakeetForTDT(
        ParakeetTDTConfig(
            encoder_config=_tiny_parakeet_encoder_config(),
            vocab_size=6,
            decoder_hidden_size=4,
            num_decoder_layers=1,
            pad_token_id=0,
            blank_token_id=5,
            durations=(0, 1, 2),
        )
    )
    processor.save_pretrained(tmp_path)
    model.save_pretrained(tmp_path)

    reloaded_processor = ParakeetProcessor.from_pretrained(tmp_path)
    reloaded_model = ParakeetForTDT.from_pretrained(tmp_path)
    assert reloaded_processor.decoder_type == "tdt"
    assert reloaded_model.config.blank_token_id == 5
    assert reloaded_model.joint.head.out_features == 9


def test_parakeet_tdt_trainer_resumes_saved_checkpoint(tmp_path: Path) -> None:
    """A tiny TDT trainer writes step 2 and resumes it through step 4."""
    processor = _tiny_parakeet_processor()
    processor.decoder_type = "tdt"
    model = ParakeetForTDT(
        ParakeetTDTConfig(
            encoder_config=_tiny_parakeet_encoder_config(),
            vocab_size=6,
            decoder_hidden_size=4,
            num_decoder_layers=1,
            pad_token_id=0,
            blank_token_id=5,
            durations=(0, 1, 2),
        )
    )
    processor.save_pretrained(tmp_path)
    examples = [
        {
            "input_features": torch.zeros(64, 16),
            "attention_mask": torch.ones(64, dtype=torch.long),
            "decoder_input_ids": [5, 1, 2],
            "labels": [1, 2],
        }
        for _ in range(4)
    ]
    data_collator = DataCollatorParakeetWithPadding(
        processor=processor, sample_rate=16_000, padding="longest"
    )

    def training_arguments(max_steps: int) -> TrainingArguments:
        return TrainingArguments(
            output_dir=str(tmp_path),
            max_steps=max_steps,
            save_strategy="steps",
            save_steps=2,
            save_total_limit=2,
            logging_steps=2,
            report_to=[],
            use_cpu=True,
            per_device_train_batch_size=2,
            remove_unused_columns=False,
        )

    train_dataset = t.cast(torch.utils.data.Dataset[object], examples)
    trainer = ParakeetGenerationTrainer(
        model=model,
        args=training_arguments(max_steps=2),
        data_collator=data_collator,
        train_dataset=train_dataset,
        processing_class=processor.tokenizer,
    )
    trainer.train()

    checkpoint = tmp_path / "checkpoint-2"
    assert trainer.state.global_step == 2
    assert checkpoint.is_dir()
    model.save_pretrained(tmp_path)

    setup = ParakeetModelSetup(
        config=OmegaConf.create(
            {
                "model": {
                    "type": "parakeet",
                    "characters_to_keep": None,
                    "pretrained_model_id": "unused",
                    "sampling_rate": 16_000,
                },
                "model_dir": str(tmp_path),
                "model_id": "unused",
                "hub_organisation": "unused",
                "padding": "longest",
            }
        )
    )
    saved = setup.load_saved()
    resumed = ParakeetGenerationTrainer(
        model=saved.model,
        args=training_arguments(max_steps=4),
        data_collator=saved.data_collator,
        train_dataset=train_dataset,
        processing_class=saved.processor.tokenizer,
    )
    resumed.train(resume_from_checkpoint=str(checkpoint))

    assert resumed.state.global_step == 4
    assert (tmp_path / "checkpoint-4").is_dir()


@pytest.mark.parametrize("decoder_type", ["rnnt", "tdt"])
def test_parakeet_transducer_metrics_keep_repeated_reference_tokens(
    decoder_type: str,
) -> None:
    """Transducer metrics retain consecutive identical reference tokens."""
    tokenizer = ParakeetTokenizer(
        vocab={"<pad>": 0, "a": 1, "<blank>": 2, "<unk>": 3},
        pad_token="<pad>",
        unk_token="<unk>",
        blank_token="<blank>",
    )
    processor = ParakeetProcessor(
        feature_extractor=ParakeetFeatureExtractor(feature_size=1),
        tokenizer=tokenizer,
        blank_token="<blank>",
        decoder_type=decoder_type,
    )
    setup = ParakeetModelSetup(
        config=OmegaConf.create(
            {"model": {"type": "parakeet", "pretrained_model_id": "checkpoint"}}
        )
    )
    setup.processor = processor

    metrics = setup.load_compute_metrics()(
        EvalPrediction(predictions=np.array([[1, 1]]), label_ids=np.array([[1, 1]]))
    )

    assert metrics == {"cer": 0.0, "wer": 0.0}


def test_parakeet_zero_shot_pipeline_uses_pinned_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generic Parakeet evaluation passes its immutable Hub revision to pipeline."""
    pipeline = MagicMock(return_value=object())
    monkeypatch.setattr("hviske.cohere.pipeline", pipeline)
    monkeypatch.setattr(
        "hviske.cohere.AutoConfig.from_pretrained",
        MagicMock(return_value=SimpleNamespace(model_type="parakeet_tdt")),
    )

    load_asr_transcriber(
        model_id="nvidia/parakeet-tdt-0.6b-v3",
        no_lm=False,
        device=torch.device("cpu"),
        revision="541d1f99c6b0c3cd0b11a95167540bb8edefd82b",
    )

    assert pipeline.call_args.kwargs["revision"] == (
        "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
    )
