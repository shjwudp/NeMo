import os
import itertools

import torch
from datasets import load_dataset

from nemo.core.classes import Dataset


def fixed_seq_length_of_datasets(
    datasets,
    fixed_seq_length,
    tokenizer,
    load_from_cache_file=False,
):
    block_size = fixed_seq_length

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(itertools.chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        # Padding in front of tokens to align it with the group size.
        if total_length % block_size != 0:
            count_pad_ids = block_size - (total_length % block_size)
            concatenated_examples[list(examples.keys())[0]] = count_pad_ids*[tokenizer.pad_id] + concatenated_examples[list(examples.keys())[0]]

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }

        return result

    lm_datasets = datasets.map(
        group_texts,
        batched=True,
        num_proc=os.cpu_count(),
        load_from_cache_file=load_from_cache_file,
        desc=f"Grouping texts in chunks of {block_size}",
    )

    return lm_datasets


class WikitextDataset(Dataset):

    def __init__(
        self,
        tokenizer,
        seq_length,
        split="test",
        name=None,
        load_from_cache_file=True,
    ):
        """HuggingFace's WikiText dataset.
        link: https://huggingface.co/datasets/wikitext
        """
        super().__init__()
        self.seq_length = seq_length
        self.name = name
        raw_dataset = load_dataset("wikitext", name, split=split)
        column_names = raw_dataset.column_names
        tokenized_dataset = raw_dataset.map(
            lambda examples: {"tokens": tokenizer.text_to_ids(examples["text"])},
            batched=True,
            remove_columns=column_names,
            load_from_cache_file=load_from_cache_file,
            desc="Running tokenizer on dataset", 
        )

        lm_dataset = fixed_seq_length_of_datasets(
            tokenized_dataset,
            seq_length,
            tokenizer,
            load_from_cache_file=load_from_cache_file,
        )
        self.dataset = lm_dataset
        self.attention_mask = torch.tril(torch.ones((self.seq_length, self.seq_length))).unsqueeze(0)
        self.attention_mask = self.attention_mask < 0.5
        self.loss_mask = torch.ones(self.seq_length, dtype=torch.float)
        self.position_ids = torch.arange(self.seq_length, dtype=torch.int64)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        item.update(dict(
            tokens=torch.tensor(item["tokens"]),
            labels=torch.tensor(item["tokens"]),
            attention_mask=self.attention_mask,
            loss_mask=self.loss_mask,
            position_ids=self.position_ids,
        ))
        return item
