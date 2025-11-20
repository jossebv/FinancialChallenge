import math
import sys
from typing import List, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.append(".")

from options.options import Opt


class SequenceWindowDataset(Dataset):
    """
    Dataset for sequential windows over an ordered time series matrix.
    Returns X windows (and optional y aligned with the last timestep of each window).
    """

    def __init__(self, opt: Opt):
        self.window = opt.window
        self.stride = opt.window_stride or self.window if opt.phase == "train" else 1
        self.drop_tail = opt.drop_tail
        self.reg_feature = opt.reg_feature
        self.cls_feature = opt.cls_feature
        self.features = opt.features
        self.masked_features = [
            masked_feature
            for masked_feature in opt.masked_features
            if masked_feature in self.features
        ]
        self.data = self.load_data(
            opt.dataroot, opt.features, opt.split_ratio, opt.phase
        )
        self.return_window = opt.return_window

        # number of start indices that produce a full window
        max_start = len(self.data) - self.window
        if max_start < 0:
            # no full window available
            self._n = 0
        else:
            if self.drop_tail:
                self._n = 1 + max_start // self.stride
            else:
                # include one more sample if there is a remainder (pad the end with the last row)
                self._n = 1 + math.ceil(max_start / self.stride)

    def _standardize(self, data: pd.DataFrame):
        num_data = data.select_dtypes(include="number")
        mean = num_data.mean()
        std = num_data.std()
        num_data = (num_data - mean) / std
        data = num_data.merge(
            data.select_dtypes(exclude="number"), left_index=True, right_index=True
        )
        return data, mean, std

    def load_data(
        self,
        dataroot: str,
        features: List[str],
        train_ratio: float = 0.85,
        phase: Literal["train", "test"] = "train",
        standardize: bool = True,
        class_eps: float = 0.0025,
    ) -> pd.DataFrame:
        data = pd.read_csv(dataroot, index_col=0)
        data.fillna(0, inplace=True)
        N = int(len(data) * train_ratio)
        data = data[:N] if phase == "train" else data[N:]
        if phase == "train":
            data = data[features]
        # Apply standardization before adding the class
        # This is done because the method standardizes every numeric value
        if standardize:
            data, _, _ = self._standardize(data)
        data[self.cls_feature] = pd.cut(
            data["log_return"],
            bins=[-np.inf, -class_eps, class_eps, np.inf],
            labels=False,
        )
        self.cls_labels = {"Bearish": 0, "Neutral": 1, "Bullish": 2}
        return pd.DataFrame(data)

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, idx: int) -> dict:
        """
        Returns a tensor of shape [window, F].
        If drop_tail=False and the final window overruns, it is padded by repeating the last row.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        start = idx * self.stride
        end = start + self.window
        T = len(self.data)

        if end <= T:
            window = self.data[start:end]  # [window, F]
            if not isinstance(window, pd.DataFrame):
                raise RuntimeError("Window slice should be a DataFrame")

            X = window[self.features].copy()
            if "dow" in self.features:
                X = X.drop(columns=["dow"])
                dow = torch.Tensor(window["dow"].to_numpy()).to(torch.int32)
            else:
                dow = None

            X.loc[-1:, self.masked_features] = 0
            X = torch.Tensor(X.to_numpy())
            y_reg = torch.Tensor([window[-1:][self.reg_feature].item()])
            y_cls = torch.Tensor([window[-1:][self.cls_feature].item()])

        else:
            # TODO: complete this section
            # this is accessed when the last row is not completed
            #
            # Need to pad the tail by repeating the final row
            return {}
            needed = end - T
            base = self.X[start:T]
            pad = self.X[T - 1 : T].repeat(needed, 1)
            window = torch.cat([base, pad], dim=0)

        batch = {
            "features": X,
            "dow": dow,
            "y_reg": y_reg,
            "y_cls": y_cls,
        }
        if self.return_window:
            batch["window"] = window
        return batch


class FinanceDataLoader:
    """
    DataLoader wrapper for the SequenceWindowDataset.
    Receives an opt object with dataloader settings.
    Allows iteration: for data in loader: ...
    """

    def __init__(self, opt: Opt):
        self.opt = opt
        self.dataset = SequenceWindowDataset(opt)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=opt.shuffle,
            pin_memory=True,
            num_workers=opt.num_workers,
            collate_fn=self.collate_fn,
            # drop_last=getattr(opt, "drop_last", False),
        )

    def collate_fn(self, batch):
        elem = batch[0]
        collated = {}
        for key in elem:
            values = [d[key] for d in batch]
            if isinstance(values[0], torch.Tensor):
                collated[key] = torch.stack(values)
            elif "pandas" in str(type(values[0])) or str(type(values[0])).endswith(
                "DataFrame'>"
            ):
                # If it's a DataFrame, just collect as a list
                collated[key] = values
            elif values[0] is None:
                collated[key] = None
            else:
                # Fallback: try to stack, else keep as list
                try:
                    collated[key] = torch.stack(values)
                except Exception:
                    collated[key] = values
        return collated

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def get_dataset(self):
        return self.dataset


if __name__ == "__main__":
    opt = Opt.from_yaml("options/train_model.yaml")
    loader = FinanceDataLoader(opt)
    for data in loader:
        for key, val in data.items():
            print(f"{key}: {val.shape}")
        break  # Remove or modify as needed to process more batches
