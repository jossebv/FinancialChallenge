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
    if opt.neptune_group_tags:
        neptune_run["sys/group_tags"].add(opt.neptune_group_tags)

    return neptune_run


def init_training(options_path: str):
    opt = Opt.from_yaml(options_path)
    neptune_run = init_neptune(opt)

    model = FiT(opt).load_from_checkpoint(opt.load_path)
    model.eval()

    # Load stats
    checkpoints_dir = os.path.dirname(opt.load_path)
    stats_path = os.path.join(checkpoints_dir, "stats.pt")
    if os.path.exists(stats_path):
        stats = torch.load(stats_path)
        opt = opt.updated(mean=stats["mean"], std=stats["std"])
        print(f"✅ Loaded stats from {stats_path}")
    else:
        print(f"⚠️ Stats file not found at {stats_path}.Using self-derived stats (LEAKAGE RISK/BAD SCALE).")

    loader = FinanceDataLoader(opt)

    gains_ls = []
    mse_close = []
    for data in tqdm(loader):
        pred = model.pred(data).item()
        
        # Un-standardize prediction
        reg_mean = opt.mean[opt.reg_feature]
        reg_std = opt.std[opt.reg_feature]
        
        pred_raw = pred * reg_std + reg_mean
        
        window = data["window"][0]
        open = window.iloc[-1]["open"]
        close = window.iloc[-1]["close"]
        
        # Calculate predicted close price based on RAW predicted log return
        # Log return formula: r = ln(P_t / P_{t-1}) -> P_t = P_{t-1} * exp(r)
        # Here we assume prediction is the log return for the NEXT step? 
        # Or is it reconstruction?
        # Based on dataloader: y_reg = window[-1:][self.reg_feature].item() 
        # So y_reg is the log return of the LAST step in the window.
        # Wait, if y_reg is the target, we should compare pred_raw to the true raw log_return.
        
        # Re-calc close from open using predicted return
        # NOTE: If Feature "log_return" is close/open, then close = open * exp(log_return).
        pred_close = open * math.exp(pred_raw)
        
        gains = 0
        if pred_close > open:
            gains = close - open

        # Calculate error on RAW scale (or standardized? kept standardized for now for return_error)
        # Actually, let's log the raw error too or keep consistent with training
        return_error = pred - data["y_reg"] # This is still in standardized space if y_reg is standardized
        neptune_run["return_error"].append(return_error)

        mse_close.append((pred_close - close) ** 2)
        neptune_run["close_error"].append((pred_close - close))

        gains_ls.append(gains)
        neptune_run["gains"].append(gains)


if __name__ == "__main__":
    TRAINING_OPTIONS = "options/test_model.yaml"
    init_training(TRAINING_OPTIONS)
