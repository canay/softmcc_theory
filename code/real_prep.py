"""real_prep.py - Build compact, imbalance-preserving .npz caches from the raw
creditcard / IoTID20 CSVs so the heavy harness can run fast and within limits.
Keeps ALL minority (positive) samples; subsamples the majority to a target
prevalence. This preserves the extreme-imbalance REGIME (unlike the prototype's
random 10k subsample, which destroyed it).
Call: python3 real_prep.py creditcard | iotid20
"""
import os, sys, numpy as np, pandas as pd

PLAN = os.environ.get("SOFTMCC_RAW_DIR", "./raw")  # dir holding creditcard.csv and IoTID20/ ; raw CSVs not in this tree (npz caches already in 02_data)
OUT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "02_data"))
os.makedirs(OUT, exist_ok=True)
SEED = 42


def subsample_to_prevalence(X, y, target_pi, seed=SEED):
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    n_neg = int(round(len(pos) * (1 - target_pi) / target_pi))
    n_neg = min(n_neg, len(neg))
    keep_neg = rng.choice(neg, size=n_neg, replace=False)
    idx = np.concatenate([pos, keep_neg]); rng.shuffle(idx)
    return X[idx], y[idx]


def prep_creditcard():
    df = pd.read_csv(f"{PLAN}/data/creditcard.csv")
    y = df["Class"].astype(int).values
    X = df.drop(columns=["Class", "Time"]).apply(pd.to_numeric, errors="coerce").fillna(0).values
    print(f"creditcard raw: {X.shape}, pos={int(y.sum())}, prev={y.mean():.5f}")
    for pi in (0.01, 0.005):
        Xs, ys = subsample_to_prevalence(X, y, pi)
        p = os.path.join(OUT, f"creditcard_pi{int(pi*1000)}.npz")
        np.savez_compressed(p, X=Xs.astype(np.float32), y=ys.astype(np.int8))
        print(f"  saved {p}: {Xs.shape}, pos={int(ys.sum())}, prev={ys.mean():.5f}")


def prep_iotid20():
    # read in chunks to bound memory; keep numeric cols + Label
    path = f"{PLAN}/IoTID20/data/IoTID20.csv"
    usecols = None
    chunks = []
    for ch in pd.read_csv(path, chunksize=100000, low_memory=False):
        if "Label" in ch.columns:
            lab = ch["Label"].astype(str)
        elif "Cat" in ch.columns:
            lab = ch["Cat"].astype(str)
        else:
            lab = ch.iloc[:, -1].astype(str)
        ynum = (~lab.str.lower().str.startswith("normal")).astype(int).values
        num = ch.select_dtypes(include=[np.number]).apply(pd.to_numeric, errors="coerce")
        num = num.replace([np.inf, -np.inf], np.nan).fillna(0)
        chunks.append((num.values.astype(np.float32), ynum.astype(np.int8)))
    X = np.vstack([c[0] for c in chunks]); y = np.concatenate([c[1] for c in chunks])
    print(f"iotid20 raw: {X.shape}, pos(attack)={int(y.sum())}, prev={y.mean():.4f}")
    # IoTID20 is attack-heavy; treat NORMAL as the rare/minority positive for an
    # imbalanced eval by flipping if normal is the minority.
    if y.mean() > 0.5:
        y = 1 - y
        print(f"  flipped: minority(normal)=positive, prev={y.mean():.4f}")
    # cap size: keep all positives + majority sample to ~40k
    rng = np.random.RandomState(SEED)
    pos = np.where(y == 1)[0]; neg = np.where(y == 0)[0]
    n_neg = min(len(neg), max(40000 - len(pos), len(pos) * 20))
    keep = np.concatenate([pos, rng.choice(neg, size=n_neg, replace=False)])
    rng.shuffle(keep)
    Xs, ys = X[keep], y[keep]
    p = os.path.join(OUT, "iotid20_compact.npz")
    np.savez_compressed(p, X=Xs, y=ys)
    print(f"  saved {p}: {Xs.shape}, pos={int(ys.sum())}, prev={ys.mean():.4f}")


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "creditcard"
    if which == "creditcard":
        prep_creditcard()
    elif which == "iotid20":
        prep_iotid20()
