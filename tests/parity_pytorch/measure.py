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
import gc
import time
from typing import Callable

import torch
from tqdm import tqdm


def measure_loops(cls_model, cfg, kind: str, loop: Callable, num_runs: int = 10, num_epochs: int = 10):
    """Returns an array with the last loss from each epoch for each run."""
    hist_losses = []
    hist_durations = []
    hist_memory = []

    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.deterministic = True
    for i in tqdm(range(num_runs), desc=f"{kind} with {cls_model.__name__}"):
        gc.collect()
        if device_type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_accumulated_memory_stats()
            torch.cuda.reset_peak_memory_stats()
        time.sleep(1)

        time_start = time.perf_counter()

        final_loss, used_memory = loop(cls_model, cfg, idx=i, device_type=device_type, num_epochs=num_epochs)

        time_end = time.perf_counter()

        hist_losses.append(final_loss)
        hist_durations.append(time_end - time_start)
        hist_memory.append(used_memory)

    return {"losses": hist_losses, "durations": hist_durations, "memory": hist_memory}
