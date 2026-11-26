"""Optional deep sequence models (LSTM, Temporal CNN, Transformer) on the last L spike-free tick returns.

Only used when PyTorch is installed AND the training block is large enough; otherwise they raise
`SkipModel` with the reason, which is recorded in the model rankings instead of silently ignored.
Inputs are built causally (window ends at the evaluation tick) by `sequence_windows`.
"""
from __future__ import annotations

import numpy as np

SEQ_LEN = 128
TST_LEN = 64
MAX_DEEP_TRAIN_ROWS = 15000
MIN_ROWS = 5000
MIN_POS = 100


class SkipModel(Exception):
    pass


def torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


def sequence_windows(rc: np.ndarray, spk: np.ndarray, seg_start: np.ndarray, idx: np.ndarray,
                     sigma: float, L: int = SEQ_LEN) -> np.ndarray:
    """(n_rows, L, 2): normalized spike-free returns and spike flags for ticks [i-L+1 .. i] within segment."""
    n = len(idx)
    out = np.zeros((n, L, 2), dtype=np.float32)
    offs = np.arange(-L + 1, 1)
    j = idx[:, None] + offs[None, :]
    ok = j >= seg_start[idx][:, None]
    jc = np.clip(j, 0, len(rc) - 1)
    out[:, :, 0] = np.where(ok, rc[jc] / (sigma or 1.0), 0.0)
    out[:, :, 1] = np.where(ok, spk[jc], 0.0)
    return np.clip(out, -20, 20)


def _build(kind: str):
    import torch
    from torch import nn

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(2, 32, batch_first=True)
            self.head = nn.Linear(32, 1)

        def forward(self, x):  # noqa: ANN001
            o, _ = self.rnn(x)
            return self.head(o[:, -1]).squeeze(-1)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            layers, ch = [], 2
            for d in (1, 2, 4, 8, 16):
                layers += [nn.ConstantPad1d((2 * d, 0), 0.0), nn.Conv1d(ch, 24, 3, dilation=d), nn.ReLU()]
                ch = 24
            self.net = nn.Sequential(*layers)
            self.head = nn.Linear(24, 1)

        def forward(self, x):  # noqa: ANN001
            h = self.net(x.transpose(1, 2))  # causal convolutions (left padding only)
            return self.head(h[:, :, -1]).squeeze(-1)

    class TST(nn.Module):
        """Small causal-input Transformer: mean-pools 4-tick patches (64 -> 16 tokens) to keep CPU cost low."""

        def __init__(self):
            super().__init__()
            self.patch = 4
            self.inp = nn.Linear(2 * self.patch, 16)
            self.pos = nn.Parameter(torch.zeros(1, TST_LEN // self.patch, 16))
            enc = nn.TransformerEncoderLayer(16, 2, 32, dropout=0.1, batch_first=True)
            self.enc = nn.TransformerEncoder(enc, 1, enable_nested_tensor=False)
            self.head = nn.Linear(16, 1)

        def forward(self, x):  # noqa: ANN001
            x = x[:, -TST_LEN:]
            b, t, c = x.shape
            x = x.reshape(b, t // self.patch, self.patch * c)
            h = self.enc(self.inp(x) + self.pos)
            return self.head(h[:, -1]).squeeze(-1)

    return {"lstm": LSTMNet, "temporal_cnn": TCN, "transformer": TST}[kind]()


class DeepSeq:
    kind = "model"
    complexity = 5

    def __init__(self, arch: str, epochs: int = 3):
        self.arch, self.name, self.epochs = arch, arch, epochs

    def fit(self, X, y, ctx):  # noqa: ANN001
        if not torch_available():
            raise SkipModel("PyTorch not installed")
        seq = ctx.extra.get("seq")
        if seq is None:
            raise SkipModel("sequence windows unavailable")
        if len(y) < MIN_ROWS or y.sum() < MIN_POS:
            raise SkipModel(f"insufficient data for deep model (rows={len(y)}, positives={int(y.sum())}; "
                            f"need >= {MIN_ROWS} rows and >= {MIN_POS} positives)")
        import torch
        torch.manual_seed(0)
        torch.set_num_threads(4)
        self.net = _build(self.arch)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        pos = float(y.sum()) if len(y) <= MAX_DEEP_TRAIN_ROWS else float(y[-MAX_DEEP_TRAIN_ROWS:].sum())
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(min((len(y) - pos) / pos, 50.0)))
        if len(y) > MAX_DEEP_TRAIN_ROWS:  # most recent rows only (bounded CPU cost)
            seq, y = seq[-MAX_DEEP_TRAIN_ROWS:], y[-MAX_DEEP_TRAIN_ROWS:]
        Xs, ys = torch.from_numpy(seq), torch.from_numpy(y.astype(np.float32))
        self.net.train()
        g = torch.Generator().manual_seed(0)
        for _ in range(self.epochs):
            perm = torch.randperm(len(ys), generator=g)
            for b in range(0, len(ys), 512):
                ix = perm[b:b + 512]
                opt.zero_grad()
                loss = lossf(self.net(Xs[ix]), ys[ix])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                opt.step()
        return self

    def score(self, X, ctx):  # noqa: ANN001
        import torch
        self.net.eval()
        seq = torch.from_numpy(ctx.extra["seq"])
        out = []
        with torch.no_grad():
            for b in range(0, len(seq), 4096):
                out.append(torch.sigmoid(self.net(seq[b:b + 4096])).numpy())
        return np.concatenate(out) if out else np.empty(0)


def deep_factories():
    return {"lstm": lambda: DeepSeq("lstm"), "temporal_cnn": lambda: DeepSeq("temporal_cnn"),
            "transformer": lambda: DeepSeq("transformer")}
