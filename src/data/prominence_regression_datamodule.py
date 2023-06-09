from argparse import ArgumentError
from typing import Optional, Tuple

import os
import torch

# import datasets
from pathlib import Path
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers import DataCollatorWithPadding
from omegaconf import DictConfig, OmegaConf
from transformers import GPT2Tokenizer, BertTokenizer, AutoTokenizer, AutoModel


from src.data.components.helsinki import HelsinkiProminenceExtractor
from src.data.components.datasets import TokenTaggingDataset
from src.data.components.collators import collate_fn, encode_and_pad_batch


class ProminenceRegressionDataModule(LightningDataModule):
    """
    LightningDataModule for HF Datasets.
    Requires a pre-processed (tokenized, cleaned...) dataset provided within the `data` folder.
    Might require adjustments if your dataset doesn't follow the structure of SNLI or MNLI.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
        self,
        data_dir: str,
        train_file: str,
        val_file: str,
        test_file: str,
        dataset_name: str,
        use_fast_tokenizer: bool = False,
        batch_size: int = 64,
        max_length: int = 128,
        num_workers: int = 0,
        pin_memory: bool = False,
        train_val_test_split: Tuple[int, int, int] = (0.8, 0.1, 0.1),
        model_name: str = None,
        score_first_token: bool = False,
        relative_to_prev: bool = False,
        n_prev: int = 1,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.dataset = None
        self.tokenizer = None
        self.collator_fn = None

        self.keep_columns = [
            "idx",
            "input_ids",
            "attention_mask",
            "token_type_ids",
            "labels",
            "bias",
            "teacher_probs",
        ]

    def prepare_data(self):
        """
        We should not assign anything here, so this function simply ensures
        that the pre-processed data is available.
        """
        pass

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.
        This method is called by lightning twice for `trainer.fit()` and `trainer.test()`, so be careful if you do a random split!
        The `stage` can be used to differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        if not self.tokenizer:
            if self.hparams.model_name == "gpt2":
                print("Using GPT2 tokenizer")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    "gpt2", add_prefix_space=True
                )
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            elif self.hparams.model_name.lower().startswith("bert"):
                case = (
                    "uncased"
                    if self.hparams.model_name.lower() == "bert-uncased"
                    else "cased"
                )
                print(f"Using BERT tokenizer, case: {case}")
                self.tokenizer = BertTokenizer.from_pretrained(f"bert-base-{case}")
                self.tokenizer.pad_token_id = self.tokenizer.sep_token_id
            else:
                raise ValueError("Model name not recognized.")
        self.pad_token_id = self.tokenizer.pad_token_id
        print(f"Dataloader: padding with token id: {self.pad_token_id}")

        self.dataset_path = Path(self.hparams.data_dir)
        if not os.path.exists(self.dataset_path):
            raise ValueError("The provided folder does not exist.")

        train_extractor = HelsinkiProminenceExtractor(
            self.dataset_path, self.hparams.train_file
        )
        self.train_texts = train_extractor.get_all_texts()
        self.train_prominences = train_extractor.get_all_real_prominence()
        self.train_dataset = TokenTaggingDataset(
            input_texts=self.train_texts,
            targets=self.train_prominences,
            tokenizer=self.tokenizer,
            model_name=self.hparams.model_name,
            score_first_token=self.hparams.score_first_token,
            relative_to_prev=self.hparams.relative_to_prev,
            n_prev=self.hparams.n_prev,
        )

        val_extractor = HelsinkiProminenceExtractor(
            self.dataset_path, self.hparams.val_file
        )
        self.val_texts = val_extractor.get_all_texts()
        self.val_prominences = val_extractor.get_all_real_prominence()
        self.val_dataset = TokenTaggingDataset(
            input_texts=self.val_texts,
            targets=self.val_prominences,
            tokenizer=self.tokenizer,
            model_name=self.hparams.model_name,
            score_first_token=self.hparams.score_first_token,
            relative_to_prev=self.hparams.relative_to_prev,
            n_prev=self.hparams.n_prev,
        )

        test_extractor = HelsinkiProminenceExtractor(
            self.dataset_path, self.hparams.test_file
        )
        self.test_texts = test_extractor.get_all_texts()
        self.test_prominences = test_extractor.get_all_real_prominence()
        self.test_dataset = TokenTaggingDataset(
            input_texts=self.test_texts,
            targets=self.test_prominences,
            tokenizer=self.tokenizer,
            model_name=self.hparams.model_name,
            score_first_token=self.hparams.score_first_token,
            relative_to_prev=self.hparams.relative_to_prev,
            n_prev=self.hparams.n_prev,
        )

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Validation dataset size: {len(self.val_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def encode_collate(self, batch):
        return encode_and_pad_batch(batch, self.tokenizer, self.hparams.model_name)

    def collate(self, batch):
        return collate_fn(batch, self.tokenizer.pad_token_id)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate,
            shuffle=False,
        )
