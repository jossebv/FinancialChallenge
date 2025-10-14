import torch
from torch import nn
from typing import Optional, Literal


class FiTokenizer(nn.Module):
    def __init__(self, opt):
        super().__init__()
        k = opt.k
        s = opt.s
        s = k if s is None else s

        self.d_model = opt.d_model
        self.dow_emb = nn.Embedding(
            num_embeddings=5, embedding_dim=opt.dow_embedding_dim
        )
        self.conv = nn.Sequential(
            nn.Conv1d(opt.in_channels, opt.d_model, kernel_size=k, stride=s, padding=0),
            nn.GELU(),
            nn.BatchNorm1d(opt.d_model),
        )

    def get_sinusoidal_positional_encoding(self, seq_len, d_model, device):
        position = torch.arange(seq_len, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device).float()
            * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe = torch.zeros(seq_len, d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (seq_len, d_model)

    def forward(self, batch: dict):
        """
        x: (B,T, C)
        dow: (B,T)
        """
        dow = batch["dow"]
        features = batch["features"]
        if dow is not None:
            dow_emb = self.dow_emb(dow)  # (B, T, e_dim)
            x = torch.cat([features, dow_emb], dim=-1).permute(0, 2, 1)  # (B, C, T)
        else:
            x = features.permute(0, 2, 1)
        z = self.conv(x)
        z = z.transpose(1, 2)  # (B, T, D)

        # Add positional encoding
        _, T, D = z.size()
        pe = self.get_sinusoidal_positional_encoding(T, D, z.device)  # (T, D)
        z = z + pe.unsqueeze(0)  # (B, T, D)

        return z


class FiTCore(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.tokenizer = FiTokenizer(opt)
        enc = nn.TransformerEncoderLayer(
            opt.d_model,
            opt.nhead,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc, opt.t_num_layers)

    def forward(self, x: dict):
        z = self.tokenizer(x)
        z_hat = self.transformer(z)
        return z_hat[:, -1, :]


class RegressionHead(nn.Module):
    def __init__(self, opt):
        super().__init__()
        hidden = opt.head_hidden
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(opt.d_model, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, 1),
            )
        else:
            self.net = nn.Linear(opt.d_model, 1)

    def forward(self, z):
        return self.net(z)  # (B,)


class BinaryClsHead(nn.Module):
    def __init__(self, opt):
        super().__init__()
        hidden = opt.head_hidden
        if hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(opt.d_model, hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, 1),  # logit
            )
        else:
            self.net = nn.Linear(opt.d_model, 1)

    def forward(self, z):
        return self.net(z).squeeze(-1)  # (B,) logits (use BCEWithLogitsLoss)


class FiT(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.backbone = FiTCore(opt)
        self.task_type = opt.task_type
        self.device = opt.device
        self.optimizer = torch.optim.AdamW(self.parameters(), lr=opt.lr)
        self._loss_score: Optional[torch.Tensor] = None

        self.margin_eps = opt.margin_eps

        if self.task_type in ("reg", "multi"):
            self.reg_head = RegressionHead(opt)
            self.reg_loss = nn.MSELoss()
        if self.task_type in ("cls", "multi"):
            self.cls_head = BinaryClsHead(opt)
            self.cls_loss = nn.BCEWithLogitsLoss()

        self.to(self.device)

    @property
    def loss_score(self) -> torch.Tensor:
        if self._loss_score is None:
            raise RuntimeError("loss_score tensor has not been set yet.")
        return self._loss_score

    def _input_to_device(self, batch: dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        return batch

    def loss(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        y_raw: Optional[torch.Tensor] = None,
        task: Literal["reg", "cls"] = "reg",
    ) -> torch.Tensor:
        if task == "reg":
            return self.reg_loss(prediction, target)

        elif task == "cls":
            if self.margin_eps is not None and y_raw is not None:
                mask = y_raw > self.margin_eps
            else:
                mask = None

            if mask is not None and mask.any():
                loss = self.cls_loss(
                    prediction[mask],
                    target[mask],
                )
            else:
                loss = self.cls_loss(prediction, target)

            return loss

    def forward(self, x: dict, task: Optional[Literal["reg", "cls"]] = None):
        z = self.backbone(x)  # (B, D)
        if self.task_type == "reg":
            return self.reg_head(z)  # (B,)
        elif self.task_type == "cls":
            return self.cls_head(z)  # (B,) logits
        else:  # "multi"
            return self.reg_head(z) if task == "reg" else self.cls_head(z)

    def pred(self, batch: dict, task: Optional[Literal["reg", "cls"]] = None):
        batch = self._input_to_device(batch)
        pred = self(batch)
        return pred

    def backward(self, batch: dict, task: Literal["reg", "cls"] = "reg"):
        """
        Run one training step: forward pass, loss computation, backpropagation,
        and optimizer update.

        Args:
            batch (dict): Training data with inputs and targets
                ("y_reg" for regression, "y_cls" for classification).
        """
        batch = self._input_to_device(batch)
        self.optimizer.zero_grad(set_to_none=True)

        y_hat = self(batch)
        target = batch["y_reg"] if self.task_type == "reg" else batch["y_cls"]
        y_raw = batch["y_reg"] if self.task_type == "reg" else batch["y_cls"]
        task = self.task_type if self.task_type in ["reg", "cls"] else task

        self._loss_score = self.loss(y_hat, target, y_raw, task)
        self.loss_score.backward()
        self.optimizer.step()

    # ---- save/load methods ----
    def save_checkpoint(self, path: str):
        """
        Save model weights only (state_dict).
        """
        torch.save({"model_state": self.state_dict()}, path)
        print(f"✅ Saved weights to {path}")

    def load_from_checkpoint(
        self, path: str, what: str = "full", map_location="cpu", strict: bool = True
    ):
        """
        Load weights from checkpoint.

        Args:
            path: checkpoint path created by save_checkpoint
            what: "full" -> load entire model (backbone + current head)
                "backbone" -> load only backbone.* params
            map_location: device mapping for torch.load
            strict: passed to load_state_dict for shape/key checking (used for "full" or backbone submodule)
        """
        ckpt = torch.load(path, map_location=map_location)
        state = ckpt["model_state"]

        if what == "full":
            missing, unexpected = self.load_state_dict(state, strict=strict)
            if not strict:
                if missing:
                    print(f"ℹ️ Missing keys: {len(missing)}")
                if unexpected:
                    print(f"ℹ️ Unexpected keys: {len(unexpected)}")
            print(f"✅ Loaded FULL model from {path}")
        elif what == "backbone":
            # extract only backbone.* keys and strip the prefix for submodule load
            bb = {
                k.replace("backbone.", ""): v
                for k, v in state.items()
                if k.startswith("backbone.")
            }
            missing, unexpected = self.backbone.load_state_dict(bb, strict=strict)
            if not strict:
                if missing:
                    print(f"ℹ️ Missing backbone keys: {len(missing)}")
                if unexpected:
                    print(f"ℹ️ Unexpected backbone keys: {len(unexpected)}")
            print(f"✅ Loaded BACKBONE only from {path}")
        else:
            raise ValueError("what must be either 'full' or 'backbone'")

        return self

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
        print("🧊 Backbone frozen.")

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        print("🔥 Backbone unfrozen.")
