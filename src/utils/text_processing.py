from transformers import GPT2Tokenizer, GPT2Model
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf


def distribute_word_label_to_token(text, label, tokenizer, model_name):
    """
    Tokenizes the text and distributes the corresponding labels to each of it's tokens
    ::param text: string of text
    ::param label: list of labels
    ::param tokenizer: tokenizer object
    ::return tokens: list of tokens
    ::return token_labels: list of labels of same length as tokens
    """

    # encode each word into its tokens
    if model_name == "gpt2":
        word_encodings = [
            tokenizer.encode(x, add_prefix_space=True) for x in text.split()
        ]
    elif model_name == "bert-base-uncased":
        word_encodings = [
            tokenizer.encode(x, add_special_tokens=True) for x in text.split()
        ]
    else:
        raise ValueError("Model not supported")
    # print(f"word encodings \n", word_encodings)

    # add the labels to the tokens
    word_to_token = []
    idx = 0
    for word_tokens in word_encodings:
        tokenoutput = []
        for token in word_tokens:
            tokenoutput.append(idx)
            idx += 1
        word_to_token.append(tokenoutput)

    # print("word_to_token\n", word_to_token)

    # create the labels for each token
    token_labels = []
    for i, label in enumerate(label):
        tokens_of_word = word_to_token[i]
        token_labels += [label] * len(tokens_of_word)

    tokens = [item for sublist in word_encodings for item in sublist]
    return tokens, token_labels
