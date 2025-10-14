import os
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
    if len(opt.neptune_group_tags) != 0:
        neptune_run["sys/group_tags"].add(opt.neptune_group_tags)

    return neptune_run


def init_training(options_path: str):
    opt = Opt.from_yaml(options_path)
    neptune_run = init_neptune(opt)

    model = FiT(opt)
    loader = FinanceDataLoader(opt)

    total_steps = 0
    for epoch in range(opt.epochs):
        for data in tqdm(loader):
            total_steps += 1
            model.backward(data)

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


if __name__ == "__main__":
    TRAINING_OPTIONS = "options/train_model.yaml"
    init_training(TRAINING_OPTIONS)
