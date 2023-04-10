import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding
import numpy as np
import numbers
from typing import List, Tuple
from omegaconf import DictConfig, OmegaConf


from src.utils.text_processing import distribute_word_label_to_token


def encode_and_pad_batch(
    batch: List[Tuple[str, List[numbers.Number]]],
    tokenizer,
    model_name: str,
    label_pad_value=-999,
):
    """
    Encode a batch of sentences and their corresponding token labels, and pad both.
    :param batch: list of tuples (sentence, labels)
    :param tokenizer: tokenizer object
    :param max_length: maximum length for padding (optional)
    :return: input_ids, attention_masks, padded_labels
    """
    sentences, labels_batch = zip(*batch)

    # print(sentences)
    # print(labels_batch)

    if model_name == "gpt2":
        encoded_batch = tokenizer.batch_encode_plus(
            sentences,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
        )
    elif model_name == "bert-base-uncased":
        encoded_batch = tokenizer.batch_encode_plus(
            sentences,
            padding="longest",
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
            add_special_tokens=True,
        )
    else:
        raise ValueError("Model not supported")

    input_ids = encoded_batch["input_ids"]
    attention_masks = encoded_batch["attention_mask"]

    # print(f"Batch Encoding:\n", encoded_batch)

    # print(f"Batch Encoding:")
    # print(
    #     f"Shapes of input_ids, attention_masks: {input_ids.shape}, {attention_masks.shape}"
    # )
    # Pad the labels
    max_length = input_ids.size(1)
    padded_labels_batch = []
    for label_sequence in labels_batch:
        padded_label_sequence = label_sequence + [label_pad_value] * (
            max_length - len(label_sequence)
        )
        padded_labels_batch.append(padded_label_sequence)
        # print(len(padded_label_sequence))

    padded_labels_tensor = torch.tensor(padded_labels_batch)
    # print(
    #     f"Shapes of input_ids, attention_masks, padded_labels_tensor: {input_ids.shape}, {attention_masks.shape}, {padded_labels_tensor.shape}"
    # )
    # return dict with keys "input_ids", "attention_mask", "labels"
    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": padded_labels_tensor,
    }


class TokenTaggingDataset(Dataset):
    def __init__(self, inputs, targets, tokenizer, model_name: str):
        self.inputs = inputs
        self.targets = targets
        self.tokenizer = tokenizer
        self.model_name = model_name

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        """
        Returns:
        tokens, token_labels, mask
        shapes: (seq_len, 1)
        """
        text = self.inputs[idx]
        labels_per_word = self.targets[idx]
        tokens, token_labels = distribute_word_label_to_token(
            text, labels_per_word, self.tokenizer, self.model_name
        )
        assert len(tokens) == len(
            token_labels
        ), f"tokens and labels have different length: {len(tokens)} != {len(token_labels)}"
        return text, token_labels
