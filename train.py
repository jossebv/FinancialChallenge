import os
import shutil
import sys

import neptune
import numpy as np
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
    if len(opt.neptune_group_tags) != 0:
        neptune_run["sys/group_tags"].add(opt.neptune_group_tags)

    return neptune_run


def init_training(options_path: str):
    opt = Opt.from_yaml(options_path)
    checkpoints_dir = os.path.join(opt.checkpoints_path, opt.name)
    stored_config_path = os.path.join(checkpoints_dir, "config.yaml")
    if os.path.abspath(options_path) != os.path.abspath(stored_config_path):
        shutil.copyfile(options_path, stored_config_path)
    neptune_run = init_neptune(opt)

    model = FiT(opt)
    loader = FinanceDataLoader(opt)

    # Save training stats for testing
    stats_path = os.path.join(checkpoints_dir, "stats.pt")
    torch.save(
        {
            "mean": loader.dataset.mean,
            "std": loader.dataset.std,
            "cls_labels": loader.dataset.cls_labels,
        },
        stats_path,
    )

    test_opt = opt.to_dict()
    test_opt["phase"] = "test"
    test_opt["batch_size"] = 1
    test_opt["num_workers"] = 1
    test_opt["shuffle"] = False # Ensure shuffle is False for testing
    test_opt = opt.from_dict(test_opt)

    # Pass training stats to test loader
    test_opt = test_opt.updated(mean=loader.dataset.mean, std=loader.dataset.std)
    
    test_loader = FinanceDataLoader(test_opt)

    total_steps = 0
    for epoch in range(opt.epochs):
        model.train()
        for data in tqdm(loader, desc=f"Epoch {epoch}"):
            total_steps += 1
            model.backward(data, "reg")

            if total_steps % opt.log_step == 0:
                loss = model.loss_score.item()
                print(
                    f"\033[1;36m🟦 Epoch {epoch:03d} | 🟩 Step {total_steps:04d} | 🟨 Loss: {loss:.6f}\033[0m"
                )
                neptune_run["loss"].append(loss)

        if epoch % opt.save_step == 0:
            save_path = os.path.join(
                opt.checkpoints_path, opt.name, f"checkpoint_{epoch}.pt"
            )
            latest_save_path = os.path.join(
                opt.checkpoints_path, opt.name, "checkpoint_latest.pt"
            )
            model.save_checkpoint(save_path)
            model.save_checkpoint(latest_save_path)

        model.eval()
        losses = np.zeros(len(test_loader))
        for i, data in enumerate(tqdm(test_loader)):
            pred = model.pred(data).item()
            losses[i] = (pred - data["y_reg"]) ** 2
        mse = losses.mean()
        neptune_run["test/loss"].append(mse)


if __name__ == "__main__":
    TRAINING_OPTIONS = "options/train_model.yaml"
    init_training(TRAINING_OPTIONS)
