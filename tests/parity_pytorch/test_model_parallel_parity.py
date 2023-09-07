# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import numpy as np
import pytest
import torch
from omegaconf import DictConfig
from pytorch_lightning import seed_everything, Trainer

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from .measure import measure_loops


def assert_parity_relative(mp_values, v_values, norm_by: float = 1, max_diff: float = 0.1):
    # assert speeds
    diffs = np.asarray(mp_values) - np.mean(v_values)
    # norm by vanilla time
    diffs = diffs / norm_by
    # relative to mean reference value
    diffs = diffs / np.mean(v_values)
    assert np.mean(diffs) < max_diff, f"Model Parallel diff {diffs} was worse than vanilla GPT (threshold {max_diff})"


def assert_parity_absolute(mp_values, v_values, norm_by: float = 1, max_diff: float = 0.55):
    # assert speeds
    diffs = np.asarray(mp_values) - np.mean(v_values)
    # norm by event count
    diffs = diffs / norm_by
    assert np.mean(diffs) < max_diff, f"Model Parallel {diffs} was worse than vanilla GPT (threshold {max_diff})"


# ParityModuleMNIST runs with num_workers=1
@pytest.mark.parametrize(
    ("cls_model", "tp", "sequence_parallel", "max_diff_speed", "max_diff_memory", "num_epochs", "num_runs"),
    [
        (MegatronGPTModel, 2, True, 0.05, 0.001, 4, 3),
    ],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="test requires GPU machine")
@pytest.mark.unit
def test_pytorch_parity(
    cls_model, model_cfg, tp, sequence_parallel, max_diff_speed, max_diff_memory, num_epochs, num_runs
):
    """Verify that the same model with or without model parallel can achieve the same results."""

    vanilla = measure_loops(cls_model, DictConfig(model_cfg), kind="Vanilla GPT", loop=vanilla_loop, num_epochs=num_epochs, num_runs=num_runs)
    model_cfg["tensor_model_parallel_size"] = tp
    model_cfg["sequence_parallel"] = sequence_parallel
    mp = measure_loops(
        cls_model, DictConfig(model_cfg), kind="Model Parallel GPT", loop=model_parallel_loop, num_epochs=num_epochs, num_runs=num_runs
    )

    # make sure the losses match exactly  to 5 decimal places
    print(f"Losses are for... \n vanilla: {vanilla['losses']} \n model-parallel: {mp['losses']}")
    for pl_out, pt_out in zip(mp["losses"], vanilla["losses"]):
        np.testing.assert_almost_equal(pl_out, pt_out, 5)

    # drop the first run for initialize dataset (download & filter)
    assert_parity_absolute(
        mp["durations"][1:], vanilla["durations"][1:], norm_by=num_epochs, max_diff=max_diff_speed
    )

    assert_parity_relative(mp["memory"], vanilla["memory"], max_diff=max_diff_memory)


def _hook_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        used_memory = torch.cuda.max_memory_allocated()
    else:
        used_memory = np.nan
    return used_memory


def vanilla_loop(cls_model, cfg, idx, device_type: str = "cuda", num_epochs=1):
    seed_everything(idx)

    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,
        gradient_as_bucket_view=cfg.gradient_as_bucket_view,
        find_unused_parameters=False,
    )

    # init model parts
    trainer = Trainer(
        # as the first run is skipped, no need to run it long
        max_epochs=num_epochs if idx > 0 else 1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        accelerator="gpu" if device_type == "cuda" else "cpu",
        devices=1,
        logger=False,
        limit_val_batches=1,
        use_distributed_sampler=False,
        benchmark=False,
        strategy=strategy,
    )
    model = cls_model(cfg, trainer)
    trainer.fit(model)

    return model._loss[-1], _hook_memory()


def model_parallel_loop(cls_model, cfg, idx, device_type: str = "cuda", num_epochs=1):
    seed_everything(idx)

    strategy = NLPDDPStrategy(
        no_ddp_communication_hook=True,
        gradient_as_bucket_view=cfg.gradient_as_bucket_view,
        find_unused_parameters=False,
    )

    # init model parts
    trainer = Trainer(
        # as the first run is skipped, no need to run it long
        max_epochs=num_epochs if idx > 0 else 1,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        accelerator="gpu" if device_type == "cuda" else "cpu",
        devices=cfg.tensor_model_parallel_size*cfg.pipeline_model_parallel_size,
        logger=False,
        limit_val_batches=1,
        use_distributed_sampler=False,
        benchmark=False,
        strategy=strategy,
    )
    model = cls_model(cfg, trainer)
    trainer.fit(model)

    return model._loss[-1], _hook_memory()


@pytest.fixture()
def model_cfg(test_data_dir):

    model_cfg = {
        'precision': 32,
        'micro_batch_size': 1,
        'global_batch_size': 2,
        'tensor_model_parallel_size': 1,
        'pipeline_model_parallel_size': 1,
        'resume_from_checkpoint': None,
        'encoder_seq_length': 512,
        'max_position_embeddings': 512,
        'num_layers': 1,
        'hidden_size': 128,
        'ffn_hidden_size': 512,
        'num_attention_heads': 2,
        'init_method_std': 0.02,
        'hidden_dropout': 0.1,
        'kv_channels': None,
        'apply_query_key_layer_scaling': True,
        'layernorm_epsilon': 1e-5,
        'make_vocab_size_divisible_by': 128,
        'pre_process': True,
        'post_process': True,
        'persist_layer_norm': True,
        'gradient_as_bucket_view': True,
        'tokenizer': {
            'library': 'megatron',
            'type': 'GPT2BPETokenizer',
            'model': None,
            'vocab_file': os.path.join(test_data_dir, 'nlp/gpt_vocab_merges/vocab.json'),
            'merge_file': os.path.join(test_data_dir, 'nlp/gpt_vocab_merges/merges.txt'),
            'delimiter': None,
        },
        'native_amp_init_scale': 4294967296,
        'native_amp_growth_interval': 1000,
        'hysteresis': 2,
        'fp32_residual_connection': False,
        'fp16_lm_cross_entropy': False,
        'megatron_amp_O2': False,
        'seed': 1234,
        'use_cpu_initialization': False,
        'onnx_safe': False,
        'apex_transformer_log_level': 30,
        'activations_checkpoint_method': None,
        'activations_checkpoint_num_layers': 1,
        'data': {
            'data_impl': 'hf_wikitext wikitext-2-v1',
            "data_prefix": [],
            'splits_string': '900,50,50',
            'seq_length': 512,
            'skip_warmup': True,
            'num_workers': 2,
            'dataloader_type': 'single',
            'reset_position_ids': False,
            'reset_attention_mask': False,
            'eod_mask_loss': False,
        },
        'optim': {
            'name': 'fused_adam',
            'lr': 2e-4,
            'weight_decay': 0.01,
            'betas': [0.9, 0.98],
            'sched': {'name': 'CosineAnnealing', 'warmup_steps': 500, 'constant_steps': 50000, 'min_lr': 2e-5},
        },
    }
    return model_cfg
