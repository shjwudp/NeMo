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

from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

from lightning.pytorch import seed_everything, Trainer
from parity_pytorch.measure import measure_loops
from tests_pytorch.helpers.advanced_models import ParityModuleMNIST, ParityModuleRNN

_EXTEND_BENCHMARKS = os.getenv("PL_RUNNING_BENCHMARKS", "0") == "1"
_SHORT_BENCHMARKS = not _EXTEND_BENCHMARKS


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
def test_pytorch_parity(
    cls_model, tp: int, sequence_parallel: bool, max_diff_speed: float, max_diff_memory: float, num_epochs: int, num_runs: int
):
    """Verify that the same  pytorch and lightning models achieve the same results."""
    mp = measure_loops(
        cls_model, kind="Model Parallel GPT", loop=model_parallel_loop, num_epochs=num_epochs, num_runs=num_runs
    )
    vanilla = measure_loops(cls_model, kind="Vanilla GPT", loop=vanilla_loop, num_epochs=num_epochs, num_runs=num_runs)

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


def vanilla_loop(cls_model, idx, device_type: str = "cuda", num_epochs=10):
    device = torch.device(device_type)
    # set seed
    seed_everything(idx)

    # init model parts
    model = cls_model()
    dl = model.train_dataloader()
    optimizer = model.configure_optimizers()

    # model to GPU
    model = model.to(device)

    epoch_losses = []
    # as the first run is skipped, no need to run it long
    for epoch in range(num_epochs if idx > 0 else 1):
        # run through full training set
        for j, batch in enumerate(dl):
            batch = [x.to(device) for x in batch]
            loss_dict = model.training_step(batch, j)
            loss = loss_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # track last epoch loss
        epoch_losses.append(loss.item())

    return epoch_losses[-1], _hook_memory()


def model_parallel_loop(cls_model, idx, device_type: str = "cuda", num_epochs=10):
    seed_everything(idx)

    model = cls_model()
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
        use_distributed_sampler=False,
        benchmark=False,
    )
    trainer.fit(model)

    return model._loss[-1], _hook_memory()
