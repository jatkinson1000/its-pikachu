"""Module to load the pokemon types dataset."""

import os
from pathlib import Path
from typing import Optional, Dict, Tuple, Any
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import pandas as pd
from pandas import DataFrame
from PIL import Image


class TypesDataset(Dataset):
    """
    Class setting up a pokemon types and images Dataset.
    """

    def __init__(
        self,
        data_loc: Path,
        input_tfms: Optional[Compose] = None,
        target_tfms: Optional[Compose] = None,
        train: bool = False,
        typelist=None,
    ):
        self.dataset: pd.DataFrame = pd.read_csv(data_loc.joinpath("pokemon.csv"))
        if typelist is not None:
            self.dataset = self.dataset[self.dataset["Type1"].isin(typelist)]

        self.images_dir: Path = data_loc.joinpath("images", "images")
        img_file_names = os.listdir(self.images_dir)
        images = []
        for name in self.dataset.Name:
            images.append([x for x in img_file_names if Path(x).stem == name][0])
        self.dataset["image_file"] = images

        self.split = _split_data(self.dataset)["train" if train is True else "valid"]

        self.input_tfms = input_tfms
        self.target_tfms = target_tfms

    def __len__(self):
        return len(self.split)

    def __getitem__(self, idx: int, img_size=(120, 120)) -> Tuple[Any, Any]:
        """Return an input-target pair.

        Parameters
        ----------
        idx : int
            Index of the input-target pair to return.

        Returns
        -------
        feat : Any
            Inputs.
        target : Any
            Targets.

        """
        feat_img_file = f"{self.images_dir}/{self.split.iloc[idx]['image_file']}"
        tgt = self.split.iloc[idx]["Type1"]

        # read image from file and set to default size before returning
        feat = Image.open(feat_img_file).convert("RGBA")
        # feat = Compose([Resize(img_size, antialias=True)])(feat)

        if self.input_tfms is not None:
            feat = self.input_tfms(feat)

        if self.target_tfms is not None:
            tgt = self.target_tfms(tgt)

        return feat, tgt


def _split_data(types_dataset: DataFrame) -> Dict[str, DataFrame]:
    """Split the ``types_df`` into a training and validation set.

    Parameters
    ----------
    types_dataset : DataFrame
        The full types data set.

    Returns
    -------
    Dict[str, DataFrame]
        Dictionary holding the ``"train"`` and ``"valid"`` splits. The valid
        split has 1 pokemon of each type, and the training
        split contains the rest of the dataset.

    """
    # valid_df = types_dataset.groupby(by=["Type1"]).sample(
    valid_df = types_dataset.sample(
        n=int(np.floor(0.2 * types_dataset.shape[0])),
        random_state=321,
    )

    # The training items are simply the items *not* in the valid split
    train_df = types_dataset.loc[~types_dataset.index.isin(valid_df.index)]

    return {"train": train_df, "valid": valid_df}
