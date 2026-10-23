"""Evaluate trained GRU run(s) on V1/V2/V3 with the torch model (GPU if available), via src/evaluate.py.

  python train/eval_torch.py runs/a [runs/b ...] --name NAME [--weights best.pt]

Several runs are averaged. Writes results/NAME.npz/json in the same format as the ONNX evaluator, so
`python src/evaluate.py compare` works across both. Torch-vs-ONNX prediction differences are ~1e-6.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'train')]
from evaluate import evaluate, print_summary  # noqa: E402
from gpu_features import torch_features  # noqa: E402
from train_gru import TRANSFORMS, GRUModel, decode_dist, future_features, make_atoms, make_class_atoms  # noqa: E402
from shape_lookup import N_FEAT as SHAPE_N, features as shape_features, load_table  # noqa: E402
import i1_lead  # noqa: E402


def quantize(w, mode):
    """Round a weight matrix the way a reduced-precision kernel would store it (computation stays fp32)."""
    if mode == 'bf16':
        return w.to(torch.bfloat16).float()
    scale = w.abs().amax(1, keepdim=True).clamp_min(1e-12) / 127.0          # int8, symmetric, per output row
    return (w / scale).round().clamp(-127, 127) * scale


@torch.no_grad()
def quantize_model(model, mode):
    """Quantize the matrices src/fast_gru.pack(bf16=True) stores in reduced precision: the projection (after folding
    in the input standardisation) and the GRU input / hidden weights. Heads and biases stay fp32."""
    inv = 1.0 / model.std
    model.proj.weight.copy_(quantize(model.proj.weight * inv, mode) / inv)
    for name, p in model.named_parameters():
        if name.startswith('blocks.') and p.dim() == 2:
            p.copy_(quantize(p, mode))


def load(run: Path, weights: str, device, quant='none'):
    cfg = json.loads((run / 'config.json').read_text())
    atoms = len(make_atoms()) if cfg.get('dist_weight') else 0
    conv = (cfg['conv'][0], cfg['conv'][1], cfg['conv'][2:]) if cfg.get('conv') else None
    future = cfg.get('future_inputs') or []
    table = load_table(ROOT / cfg['shape_table']) if cfg.get('shape_table') else None
    i1 = i1_lead.load_tables(ROOT / cfg['i1_table'])[i1_lead.N_FOLDS] if cfg.get('i1_table') else None
    pair_atoms = make_class_atoms() if cfg.get('pair_weight') else None
    model = GRUModel(TRANSFORMS[cfg['features']][0] + SHAPE_N * (table is not None) + i1_lead.N_FEAT * (i1 is not None) + 112 * len(future), cfg['hidden'], cfg['layers'], cfg['skip'], cfg.get('proj', 0), atoms, conv,
                     cfg.get('cell', 'gru'), cfg.get('drop_blocks') or (), pair_atoms=pair_atoms)
    model.load_state_dict(torch.load(run / weights, map_location=device))
    model.future = future  # privileged teacher inputs (non-causal), see train_gru --future-inputs
    model.table = table    # exact-shape lookup inputs, see train_gru --shape-table
    model.i1 = i1          # full i1 lead-lag table, see train_gru --i1-table
    if quant != 'none':
        quantize_model(model, quant)
    return model.to(device).eval(), cfg['features']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+', type=Path)
    ap.add_argument('--name', required=True)
    ap.add_argument('--weights', default='best.pt')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--max-seqs', type=int, help='quick check on the first N sequences of each view')
    ap.add_argument('--quant', choices=['none', 'bf16', 'int8'], default='none',
                    help='simulate reduced-precision weight storage (fp32 compute)')
    ap.add_argument('--output', choices=['reg', 'dist', 'blend', 'pair'], default='reg',
                    help='regression head, decoded marginal distribution, their average, or decoded joint pair')
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = [load(r, args.weights, device, args.quant) for r in args.runs]
    atoms = torch.tensor(make_atoms(), device=device)

    def output(m, x):
        reg, _, top = m(x, None, return_top=True)
        if args.output == 'reg':
            return reg
        if args.output == 'pair':
            if m.pair_heads is None:
                raise ValueError('--output pair requires a run trained with --pair-weight')
            cls = m.class_head(top[:, :99]).mean(1).argmax(-1)
            pair_atoms = make_class_atoms()
            out = torch.zeros_like(reg)
            for ci, support in enumerate(pair_atoms):
                ix = torch.where(cls == ci)[0]
                if not ix.numel():
                    continue
                n = len(support)
                logits = m.pair_heads[ci](top[ix])
                joint = torch.softmax(logits, -1).view(len(ix), top.shape[1], n, n)
                av = torch.as_tensor(support, device=x.device, dtype=joint.dtype)
                p0, p1 = joint.sum(-1), joint.sum(-2)
                aw = av.abs()
                out[ix, :, 0] = (p0 * (aw * av)).sum(-1) / (p0 * aw).sum(-1).clamp_min(1e-8)
                out[ix, :, 1] = (p1 * (aw * av)).sum(-1) / (p1 * aw).sum(-1).clamp_min(1e-8)
            return out
        dist = decode_dist(m.dist_head(top).unflatten(-1, (2, len(atoms))), atoms)
        return dist if args.output == 'dist' else (reg + dist) / 2

    @torch.no_grad()
    def predict(Xb):
        X = torch.from_numpy(Xb).to(device)
        F = torch_features(X, clock=any(feature == 'engclock' for _, feature in models)) \
            if any(feature != 'raw' for _, feature in models) else None
        preds = []
        for m, feature in models:
            inp = F if feature == 'engclock' else (F[..., :TRANSFORMS['eng'][0]] if feature == 'eng' else X)
            if m.table is not None:
                L = np.stack([shape_features(x, *m.table)[0] for x in Xb])
                inp = torch.cat([inp, torch.from_numpy(L).to(device)], -1)
            if m.i1 is not None:
                L = np.stack([i1_lead.features(x, *m.i1) for x in Xb])
                inp = torch.cat([inp, torch.from_numpy(L).to(device)], -1)
            preds.append(output(m, torch.cat([inp, future_features(X, m.future)], -1) if m.future else inp))
        return torch.stack(preds).mean(0).float().cpu().numpy()

    print_summary(args.name, evaluate(predict, args.name, max_seqs=args.max_seqs, batch=args.batch))
