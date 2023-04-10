import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import BatchEncoding
import numpy as np
import numbers
from typing import List, Tuple
from omegaconf import DictConfig, OmegaConf
from transformers import BertTokenizer, GPT2Tokenizer


from src.utils.text_processing import distribute_word_label_to_token


def tokenize_text_with_labels(
    text,
    labels,
    tokenizer,
    model_type,
    invalid_label=-1,
    score_first_token=True,
    relative_to_prev=False,
    n_prev=1,
):
    """
    Tokenize the input text and associate labels with each token.

    Args:
        text (str): The input text to tokenize.
        labels (list): A list of labels corresponding to each word in the text.
        model_type (str): The type of the language model (e.g., 'gpt2', 'bert-uncased', 'bert-cased').
        invalid_label (int, optional): The label to assign to invalid tokens (e.g., punctuation and whitespace). Defaults to -1.
        score_first_token (bool, optional): If True, only the first token of a multi-token word will have a mask value of 1. Defaults to True.
        relative_to_prev (bool, optional): If True, adjust the labels to be relative to the average of the previous n_prev words. Defaults to False.
        n_prev (int, optional): The number of previous words to consider when adjusting labels. Only used if relative_to_prev is True. Defaults to 1.

    Returns:
        tuple: A tuple containing the following elements:
            - input_text (str): The input text.
            - tokenized_text (list): The tokenized text as a list of tokens.
            - tokenized_labels (list): The list of labels corresponding to each token in the tokenized_text.
            - token_ids (list): The list of token IDs corresponding to each token in the tokenized_text.
            - mask (list): A binary mask indicating whether a token should be scored (1) or not (0).
    """
    # TODO: create tokenizer object with hydra and pass it in here

    original_labels = labels
    words = text.split()
    assert len(words) == len(labels), "The number of words and labels should be equal"

    if relative_to_prev:
        new_labels = []
        for i, label in enumerate(labels):
            if i < n_prev:
                new_labels.append(label)
            else:
                avg_prev = sum(labels[i - n_prev : i]) / n_prev
                new_labels.append(label - avg_prev)
        labels = new_labels

    tokenized_text, tokenized_labels, token_ids, mask, word_to_tokens = (
        [],
        [],
        [],
        [],
        [],
    )

    if model_type.lower().startswith("bert"):
        tokenized_text.append(tokenizer.cls_token)
        tokenized_labels.append(invalid_label)
        token_ids.append(tokenizer.cls_token_id)
        mask.append(0)

    for word, label in zip(words, labels):
        tokens = tokenizer.tokenize(word)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        tokenized_text.extend(tokens)
        token_ids.extend(ids)
        word_to_tokens.extend((word, ids))

        if score_first_token:
            mask.extend([1] + [0] * (len(tokens) - 1))
            tokenized_labels.extend([label] + [invalid_label] * (len(tokens) - 1))
        else:
            mask.extend([1] * len(tokens))
            tokenized_labels.extend([label] * len(tokens))

    if model_type.lower().startswith("bert"):
        tokenized_text.append(tokenizer.sep_token)
        tokenized_labels.append(invalid_label)
        token_ids.append(tokenizer.sep_token_id)
        mask.append(0)

    return (
        text,
        tokenized_text,
        original_labels,
        tokenized_labels,
        token_ids,
        mask,
        word_to_tokens,
    )


# class TokenTaggingDataset(Dataset):
#     def __init__(self, inputs, targets, tokenizer, model_name: str):
#         self.inputs = inputs
#         self.targets = targets
#         self.tokenizer = tokenizer
#         self.model_name = model_name

#     def __len__(self):
#         return len(self.inputs)

#     def __getitem__(self, idx):
#         """
#         Returns:
#         tokens, token_labels, mask
#         shapes: (seq_len, 1)
#         """
#         text = self.inputs[idx]
#         labels_per_word = self.targets[idx]
#         tokens, token_labels, word_to_token = distribute_word_label_to_token(
#             text, labels_per_word, self.tokenizer, self.model_name
#         )
#         assert len(tokens) == len(
#             token_labels
#         ), f"tokens and labels have different length: {len(tokens)} != {len(token_labels)}"
#         return {}


class TokenTaggingDataset(Dataset):
    def __init__(
        self,
        input_texts,
        targets,
        tokenizer,
        model_name: str,
        score_first_token: bool = True,
        relative_to_prev: bool = False,
        n_prev: int = 1,
    ):
        """
        ::param inputs: list of strings
        ::param targets: list of lists of labels
        ::param model_name: name of the model to use
        ::param tokenizer: tokenizer object
        ::param score_first_token: whether to score only the first token of a word
        ::param relative_to_prev: whether to score relative to the previous token
        ::param n_prev: number of previous tokens to consider
        """
        self.inputs = input_texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.score_first_token = score_first_token
        self.relative_to_prev = relative_to_prev
        self.n_prev = n_prev

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        text = self.inputs[idx]
        labels_per_word = self.targets[idx]

        (
            input_text,
            tokenized_text,
            original_labels,
            tokenized_labels,
            token_ids,
            mask,
            word_to_tokens,
        ) = tokenize_text_with_labels(
            text=text,
            labels=labels_per_word,
            tokenizer=self.tokenizer,
            model_type=self.model_name,
            score_first_token=self.score_first_token,
            relative_to_prev=self.relative_to_prev,
            n_prev=self.n_prev,
        )

        assert len(tokenized_labels) == len(
            token_ids
        ), f"tokens and labels have different length: {len(tokenized_labels)} != {len(token_ids)}"

        return {
            "input_text": input_text,
            "tokenized_text": tokenized_text,
            "original_labels": original_labels,
            "tokenized_labels": tokenized_labels,
            "input_ids": token_ids,
            "loss_mask": mask,
            "word_to_tokens": word_to_tokens,
        }
