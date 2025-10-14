import math
import numpy as np
import neptune
from tqdm import tqdm
from data.dataloader import FinanceDataLoader
from models.FiT import FiT
from options.options import Opt


def init_neptune(opt: Opt):
    neptune_run = neptune.init_run(
        project=opt.neptune_project_name,
        name=opt.name,
    )
    neptune_run["params"] = opt.to_dict()
    if opt.neptune_group_tags != "":
        neptune_run["sys/group_tags"].add(opt.neptune_group_tags)

    return neptune_run


def init_training(options_path: str):
    opt = Opt.from_yaml(options_path)
    neptune_run = init_neptune(opt)

    model = FiT(opt).load_from_checkpoint(opt.load_path)
    model.eval()
    loader = FinanceDataLoader(opt)

    gains_ls = []
    mse_close = []
    for data in tqdm(loader):
        pred = model.pred(data).item()
        window = data["window"][0]
        open = window.iloc[-1]["open"]
        close = window.iloc[-1]["close"]
        pred_close = open * math.exp(pred)
        gains = 0
        if pred_close > open:
            gains = close - open

        return_error = pred - data["y_reg"]
        neptune_run["return_error"].append(return_error)

        mse_close.append((pred_close - close) ** 2)
        neptune_run["close_error"].append((pred_close - close))

        gains_ls.append(gains)
        neptune_run["gains"].append(gains)


if __name__ == "__main__":
    TRAINING_OPTIONS = "options/test_model.yaml"
    init_training(TRAINING_OPTIONS)
