import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding
import numpy as np
import numbers
from typing import List, Tuple
from omegaconf import DictConfig, OmegaConf


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


def collate_fn(batch, eos_token_id, invalid_label=-1):
    max_len = max([len(item["input_ids"]) for item in batch])

    padded_token_ids = torch.full((len(batch), max_len), eos_token_id, dtype=torch.long)
    padded_tokenized_labels = torch.full(
        (len(batch), max_len), invalid_label, dtype=torch.float32
    )
    padded_loss_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, item in enumerate(batch):
        token_ids = torch.tensor(item["input_ids"])
        tokenized_labels = torch.tensor(item["tokenized_labels"])
        loss_mask = torch.tensor(item["loss_mask"])

        padded_token_ids[i, : len(token_ids)] = token_ids
        padded_tokenized_labels[i, : len(tokenized_labels)] = tokenized_labels
        padded_loss_mask[i, : len(loss_mask)] = loss_mask
        attention_mask[i, : len(token_ids)] = 1

    return {
        "input_text": [item["input_text"] for item in batch],
        "tokenized_text": [item["tokenized_text"] for item in batch],
        "original_labels": [item["original_labels"] for item in batch],
        "tokenized_labels": padded_tokenized_labels,
        "input_ids": padded_token_ids,
        "loss_mask": padded_loss_mask,
        "attention_mask": attention_mask,
        "word_to_tokens": [item["word_to_tokens"] for item in batch],
    }
