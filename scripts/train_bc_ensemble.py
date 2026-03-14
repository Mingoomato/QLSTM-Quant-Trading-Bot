"""
train_bc_ensemble.py — BC Ensemble Training (5x 다른 시드)
═══════════════════════════════════════════════════════════════════════════════

원리:
  BC 모델 5개를 서로 다른 랜덤 시드로 학습
  → 각 모델이 독립적으로 피처 조합 발견
  → 앙상블 voting: 3/5 이상 동의 시만 진입
  → 오신호(한 모델의 실수)를 나머지 4개가 차단

기대 효과:
  WR +5~8%p (오신호 필터링)
  거래수 -30~50% (약한 신호 차단)

Usage:
  python scripts/train_bc_ensemble.py --symbol BTCUSDT --timeframe 15m \
    --days 1095 --epochs 20 --device cuda

Output:
  checkpoints/quantum_v2/ensemble/seed_42.pt
  checkpoints/quantum_v2/ensemble/seed_123.pt
  checkpoints/quantum_v2/ensemble/seed_456.pt
  checkpoints/quantum_v2/ensemble/seed_789.pt
  checkpoints/quantum_v2/ensemble/seed_1337.pt
  checkpoints/quantum_v2/ensemble/bc_scaler.pkl
  checkpoints/quantum_v2/ensemble/config.json
═══════════════════════════════════════════════════════════════════════════════
"""

import argparse
import os
import sys
import time
import json
import pickle
import random
import copy
import dataclasses

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import torch
import pandas as pd
from datetime import datetime, timezone, timedelta
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.integrated_agent import build_quantum_agent, AgentConfig
from src.models.loss import QuantumDivineLossV2, RegretWeightedBCLoss
from src.data.data_client import DataClient
from src.data.binance_client import fetch_binance_taker_history
from src.models.features_v4 import generate_and_cache_features_v4
from src.models.labeling import compute_clean_barrier_labels, standardize_1m_ohlcv
from src.models.qng_optimizer import DiagonalQNGOptimizer

# ─────────────────────────────────────────────────────────────────────────────
# 상수
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN      = 20
WARMUP       = 120
LABEL_SMOOTH = 0.1
ORTH_WEIGHT  = 0.05
REGRET_W     = 0.2
DEFAULT_SEEDS = [42, 123, 456, 789, 1337]


# ─────────────────────────────────────────────────────────────────────────────
# 헬퍼
# ─────────────────────────────────────────────────────────────────────────────
def raw_to_action(raw_label: int) -> int:
    if raw_label == 1:  return 1   # LONG
    if raw_label == -1: return 2   # SHORT
    return 0                        # HOLD


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


class BalancedBatchSampler:
    def __init__(self, labels: np.ndarray, batch_size: int = 192):
        self.batch_size = batch_size
        self.n_hold  = batch_size // 3
        self.n_long  = batch_size // 3
        self.n_short = batch_size - self.n_hold - self.n_long
        self.idx_hold  = np.where(labels == 0)[0]
        self.idx_long  = np.where(labels == 1)[0]
        self.idx_short = np.where(labels == 2)[0]

    def __iter__(self):
        batch = np.concatenate([
            np.random.choice(self.idx_hold,  self.n_hold,  replace=True),
            np.random.choice(self.idx_long,  self.n_long,  replace=True),
            np.random.choice(self.idx_short, self.n_short, replace=True),
        ])
        np.random.shuffle(batch)
        return iter(batch)

    def n_batches(self) -> int:
        total = len(self.idx_hold) + len(self.idx_long) + len(self.idx_short)
        return max(1, total // self.batch_size)


def build_batch(indices, features, actions, device):
    x_list, y_list = [], []
    n_feat = features.shape[1]
    for idx in indices:
        start = idx - SEQ_LEN + 1
        if start < 0:
            pad = np.zeros((-start, n_feat), dtype=np.float32)
            win = np.vstack([pad, features[:idx + 1]])
        else:
            win = features[start:idx + 1]
        x_list.append(win)
        y_list.append(int(actions[idx]))
    x = torch.from_numpy(np.array(x_list, dtype=np.float32)).to(device)
    y = torch.tensor(y_list, dtype=torch.long).to(device)
    return x, y


@torch.no_grad()
def evaluate_bc(agent, features, actions, device, batch_size=256):
    agent.eval()
    n = len(features)
    valid_start = WARMUP + SEQ_LEN
    all_preds, all_labels = [], []
    for i in range(valid_start, n, batch_size):
        batch_idx = np.arange(i, min(i + batch_size, n))
        x, y = build_batch(batch_idx, features, actions, device)
        logits, _, _, _ = agent.forward(x, last_step_only=True)
        if logits.dim() == 3:
            logits = logits.squeeze(1)
        preds = logits.argmax(dim=-1)
        all_preds.append(preds.cpu().numpy())
        all_labels.append(y.cpu().numpy())
    preds_arr  = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    per_class = {}
    for cls, name in [(0,"HOLD"),(1,"LONG"),(2,"SHORT")]:
        mask = (labels_arr == cls)
        per_class[name] = (preds_arr[mask] == labels_arr[mask]).mean() if mask.sum() > 0 else 0.0
    bal_acc = (per_class["HOLD"] + per_class["LONG"] + per_class["SHORT"]) / 3.0
    return {"bal_acc": bal_acc, **per_class}


def train_one_model(
    seed: int,
    train_feat, train_act, val_feat, val_act,
    raw_labels_train,
    feature_dim: int,
    args,
    device: torch.device,
    ensemble_dir: str,
) -> str:
    """단일 BC 모델 학습 (1 seed). 체크포인트 경로 반환."""
    print(f"\n{'═'*60}")
    print(f"  🌱 Seed {seed} 학습 시작")
    print(f"{'═'*60}")
    set_seed(seed)

    # 에이전트 생성
    config = AgentConfig(
        lr=args.lr,
        grad_clip=1.0,
        entropy_reg=0.0,
        dir_sym_coef=0.0,
        feature_dim=feature_dim,
        checkpoint_dir=ensemble_dir,
        confidence_threshold=0.40,
    )
    agent = build_quantum_agent(config=config, device=device)

    # LDA 피팅 (동일 데이터 → 동일 결과지만 agent별 decomposer에 저장)
    _decomposer = agent.encoder.decomposer
    if _decomposer.use_lda:
        _decomposer.fit_lda(train_feat, raw_labels_train)
        print(f"  [LDA] Fitted (seed={seed})")

    sampler = BalancedBatchSampler(train_act, batch_size=args.batch_size)

    # LR 스케줄러
    _sched_target = (
        agent.optimizer.classical_optimizer
        if isinstance(agent.optimizer, DiagonalQNGOptimizer)
        else agent.optimizer
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        _sched_target, T_max=args.epochs, eta_min=1e-5
    )

    _loss_fn = RegretWeightedBCLoss(
        tp_mult=args.tp_mult, sl_mult=args.sl_mult,
        fee=0.075, smoothing=LABEL_SMOOTH,
        orth_w=ORTH_WEIGHT, parity_w=0.005, regret_w=REGRET_W,
    )

    best_bal_acc = 0.0
    best_state   = None
    no_improve   = 0
    ckpt_path    = os.path.join(ensemble_dir, f"seed_{seed}.pt")

    for epoch in range(1, args.epochs + 1):
        agent.train()
        n_batches = sampler.n_batches()
        total_loss = 0.0

        for _ in range(n_batches):
            indices = np.fromiter(sampler, dtype=np.int64, count=sampler.batch_size)
            valid_start = WARMUP + SEQ_LEN
            indices = indices[indices >= valid_start]
            if len(indices) == 0:
                continue

            x, y = build_batch(indices, train_feat, train_act, device)
            agent.optimizer.zero_grad()
            logits, expvals, _, _ = agent.forward(x, last_step_only=True)
            if logits.dim() == 3:
                logits  = logits.squeeze(1)
                expvals = expvals.squeeze(1)
            loss, _ = _loss_fn(logits, y, expvals)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in agent.parameters() if p.requires_grad], max_norm=1.0
            )
            if isinstance(agent.optimizer, DiagonalQNGOptimizer):
                agent.optimizer.step(encoder=agent.encoder.quantum_layer)
            else:
                agent.optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        val_info = evaluate_bc(agent, val_feat, val_act, device)
        bal_acc  = val_info["bal_acc"]

        print(f"  Epoch {epoch:2d}/{args.epochs} | Loss={total_loss/max(n_batches,1):.4f} | "
              f"BalAcc={bal_acc:.3f} | "
              f"H={val_info['HOLD']:.2f} L={val_info['LONG']:.2f} S={val_info['SHORT']:.2f}")

        # 체크포인트 저장
        if bal_acc > best_bal_acc + 0.001:
            best_bal_acc = bal_acc
            best_state   = copy.deepcopy(agent.state_dict())
            no_improve   = 0
        else:
            no_improve += 1

        if no_improve >= args.patience:
            print(f"  [EarlyStop] Epoch {epoch}, BalAcc={best_bal_acc:.3f}")
            break

    # 최적 가중치 저장
    if best_state is not None:
        agent.load_state_dict(best_state)

    # LDA 가중치 포함 저장
    lda_W = None
    if _decomposer.use_lda and _decomposer._lda_fitted:
        lda_W = _decomposer._lda_W

    torch.save({
        "state_dict":  agent.state_dict(),
        "feature_dim": feature_dim,
        "seed":        seed,
        "best_bal_acc": best_bal_acc,
        "lda_W":       lda_W,
    }, ckpt_path)

    print(f"  ✅ Saved: {ckpt_path}  (BalAcc={best_bal_acc:.3f})")
    return ckpt_path


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BC Ensemble Training")
    parser.add_argument("--symbol",      default="BTCUSDT")
    parser.add_argument("--timeframe",   default="15m")
    parser.add_argument("--days",        type=int,   default=None,
                        help="학습 기간 (일수). --start-date 지정 시 무시됨")
    parser.add_argument("--start-date",  default=None,
                        help="학습 시작일 YYYY-MM-DD (권장: 2019-01-01)")
    parser.add_argument("--end-date",    default=None,
                        help="학습 종료일 YYYY-MM-DD (권장: 2025-09-01, 백테스트와 분리)")
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch-size",  type=int,   default=192)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--val-ratio",   type=float, default=0.2)
    parser.add_argument("--patience",    type=int,   default=10)
    parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--n-models",    type=int,   default=5)
    parser.add_argument("--seeds",       type=int,   nargs="+", default=DEFAULT_SEEDS)
    parser.add_argument("--vote-threshold", type=int, default=3,
                        help="앙상블 진입 조건: N개 이상 동의 (기본=3, 5모델 기준 과반)")
    parser.add_argument("--tp-mult",     type=float, default=2.0,
                        help="Regret 행렬 TP 배수 (pretrain_bc.py 기본값=2.0, 실제 거래 4.0과 다름 주의)")
    parser.add_argument("--sl-mult",     type=float, default=1.0,
                        help="Regret 행렬 SL 배수 (pretrain_bc.py 기본값=1.0)")
    parser.add_argument("--ensemble-dir", default="checkpoints/quantum_v2/ensemble")
    args = parser.parse_args()

    seeds = args.seeds[:args.n_models]
    os.makedirs(args.ensemble_dir, exist_ok=True)
    device = torch.device(args.device)

    print("═" * 60)
    print("  BC Ensemble Training")
    print(f"  Seeds: {seeds}  Vote threshold: {args.vote_threshold}/{len(seeds)}")
    print("═" * 60)

    # ── 1. 데이터 로드 (1회) ─────────────────────────────────────────────────
    dc = DataClient(mode="por", bybit_env="mainnet", bybit_category="linear")

    # 날짜 계산: --start-date/--end-date 우선, 없으면 --days 사용
    if args.end_date:
        _end_dt = datetime.strptime(args.end_date, "%Y-%m-%d").replace(
            tzinfo=timezone.utc, hour=23, minute=59)
    else:
        _end_dt = datetime.now(timezone.utc)

    if args.start_date:
        _start_str = args.start_date
    elif args.days:
        _start_str = (_end_dt - timedelta(days=args.days)).strftime("%Y-%m-%d")
    else:
        _start_str = "2019-01-01"   # 기본: 최대한 오래된 데이터

    _start_ms  = int(pd.Timestamp(_start_str).timestamp() * 1000)
    _end_ms    = int(_end_dt.timestamp() * 1000)

    print(f"\n  [Data] {args.symbol} {args.timeframe} {_start_str} ~ {_end_dt.strftime('%Y-%m-%d')}")
    print(f"  ⚠ 백테스트 기간과 분리 확인: 학습 종료({_end_dt.strftime('%Y-%m-%d')}) 이후 데이터로 백테스트 할 것")
    df_raw = dc.fetch_training_history(
        symbol=args.symbol, timeframe=args.timeframe,
        start_date=_start_str, end_ms=_end_ms, cache_dir="data"
    )
    if df_raw.empty:
        print("  [Error] Empty dataframe"); return

    df_clean = standardize_1m_ohlcv(df_raw) if args.timeframe == "1m" else df_raw

    # Funding
    df_funding = dc.fetch_funding_history(args.symbol, _start_ms, _end_ms, cache_dir="data")
    if not df_funding.empty:
        df_funding["ts"] = (pd.to_datetime(df_funding["ts_ms"], unit="ms", utc=True)
                            .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
        df_clean = df_clean.merge(df_funding[["ts","funding_rate"]], on="ts", how="left")
        df_clean["funding_rate"] = df_clean["funding_rate"].ffill().fillna(0.0)

    # OI
    df_oi = dc.fetch_open_interest_history(args.symbol, _start_ms, _end_ms, interval="1h", cache_dir="data")
    if not df_oi.empty:
        df_oi["ts"] = (pd.to_datetime(df_oi["ts_ms"], unit="ms", utc=True)
                       .dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S"))
        df_clean = df_clean.merge(df_oi[["ts","open_interest"]], on="ts", how="left")
        df_clean["open_interest"] = df_clean["open_interest"].ffill().fillna(0.0)

    # CVD
    _cache_start = _start_str.replace("-","")
    _cache_end   = _end_dt.strftime("%Y%m%d")
    taker_cache  = f"data/binance_taker_{args.symbol}_{args.timeframe}_{_cache_start}_{_cache_end}.csv"
    try:
        df_taker = fetch_binance_taker_history(
            symbol=args.symbol, interval=args.timeframe,
            start_date=_start_str, end_date=_end_dt.strftime("%Y-%m-%d"),
            cache_path=taker_cache, verbose=True,
        )
        if not df_taker.empty:
            df_clean = df_clean.merge(df_taker[["ts","taker_buy_volume"]], on="ts", how="left")
            df_clean["taker_buy_volume"] = df_clean["taker_buy_volume"].fillna(0.0)
            print(f"  [CVD] Merged: {df_clean['taker_buy_volume'].gt(0).sum():,} bars")
    except Exception as e:
        print(f"  [CVD] Failed ({e}) — fallback")

    # ── 2. 레이블 ────────────────────────────────────────────────────────────
    bars_h = 96 if args.timeframe == "15m" else 60
    df_labeled_raw = compute_clean_barrier_labels(
        df_clean, alpha=4.0, beta=1.5, hold_band=1.5, hold_h=20, h=bars_h
    )
    df_labeled = df_labeled_raw[df_labeled_raw["label"] != 2].reset_index(drop=True)
    keep_mask  = df_labeled_raw["label"].values != 2
    raw_labels = df_labeled_raw["label"].values[keep_mask]
    _lv = df_labeled["label"].value_counts().to_dict()
    print(f"  [Labels] LONG={_lv.get(1,0):,} SHORT={_lv.get(-1,0):,} HOLD={_lv.get(0,0):,} "
          f"DISCARD={(df_labeled_raw['label']==2).sum():,}")

    # ── 3. 피처 ──────────────────────────────────────────────────────────────
    cache_file = f"data/feat_cache_{args.symbol}_{args.timeframe}_{_cache_start}_{_cache_end}_v4cvd.npy"
    all_features_raw = generate_and_cache_features_v4(df_clean, cache_file)
    all_features = all_features_raw[keep_mask]
    actions      = np.array([raw_to_action(r) for r in raw_labels], dtype=np.int64)
    N = len(all_features)

    # ── 4. Train/Val 분리 ────────────────────────────────────────────────────
    split_idx  = int(N * (1 - args.val_ratio))
    train_feat = all_features[:split_idx]
    train_act  = actions[:split_idx]
    val_feat   = all_features[split_idx:]
    val_act    = actions[split_idx:]
    raw_labels_train = raw_labels[:split_idx]
    print(f"  [Split] Train={len(train_feat):,}  Val={len(val_feat):,}")

    # ── 5. 스케일러 (1회 피팅, 공유) ────────────────────────────────────────
    scaler = StandardScaler()
    train_feat = scaler.fit_transform(train_feat).astype(np.float32)
    val_feat   = scaler.transform(val_feat).astype(np.float32)
    scaler_path = os.path.join(args.ensemble_dir, "bc_scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  [Scaler] Saved → {scaler_path}")

    # ── 6. 5개 모델 순차 학습 ────────────────────────────────────────────────
    ckpt_paths = []
    bal_accs   = []
    t_total    = time.time()

    for seed in seeds:
        ckpt = train_one_model(
            seed=seed,
            train_feat=train_feat, train_act=train_act,
            val_feat=val_feat,     val_act=val_act,
            raw_labels_train=raw_labels_train,
            feature_dim=all_features.shape[1],
            args=args, device=device,
            ensemble_dir=args.ensemble_dir,
        )
        # BalAcc 읽기
        saved = torch.load(ckpt, map_location="cpu", weights_only=False)
        ckpt_paths.append(ckpt)
        bal_accs.append(saved["best_bal_acc"])

    # ── 7. 앙상블 config 저장 ────────────────────────────────────────────────
    config_data = {
        "symbol":         args.symbol,
        "timeframe":      args.timeframe,
        "feature_dim":    int(all_features.shape[1]),
        "seeds":          seeds,
        "n_models":       len(seeds),
        "vote_threshold": args.vote_threshold,
        "scaler_path":    scaler_path,
        "checkpoints":    ckpt_paths,
        "bal_accs":       [float(b) for b in bal_accs],
        "avg_bal_acc":    float(np.mean(bal_accs)),
    }
    config_path = os.path.join(args.ensemble_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)

    # ── 8. 결과 요약 ─────────────────────────────────────────────────────────
    elapsed = time.time() - t_total
    print(f"\n{'═'*60}")
    print(f"  ✅ BC Ensemble 학습 완료  ({elapsed/60:.1f}분)")
    print(f"{'═'*60}")
    for seed, acc, ckpt in zip(seeds, bal_accs, ckpt_paths):
        print(f"  Seed {seed:5d}: BalAcc={acc:.3f}  → {os.path.basename(ckpt)}")
    print(f"  평균 BalAcc: {np.mean(bal_accs):.3f}")
    print(f"  Vote threshold: {args.vote_threshold}/{len(seeds)}")
    print(f"\n  백테스트 명령어:")
    print(f"  python scripts/backtest_model_v2.py \\")
    print(f"    --ensemble-dir {args.ensemble_dir} \\")
    print(f"    --days 187 --confidence 0.45 \\")
    print(f"    --leverage 10 --tp-mult 4.0 --sl-mult 1.5")


if __name__ == "__main__":
    main()
