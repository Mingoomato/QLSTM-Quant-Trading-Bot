"""
optimize_backtest.py
--------------------------------------------------------------------
Optuna로 백테스트 파라미터 자동 최적화

탐색 대상:
  - confidence  : 0.30 ~ 0.70
  - tp_mult     : 2.0  ~ 8.0
  - sl_mult     : 0.5  ~ 2.0

고정 파라미터:
  - leverage    : 10
  - capital     : 100
  - timeframe   : 15m
  - pos_frac    : 0.5

목적 함수 (최대화):
  Sharpe × sign(ROI)  — ROI 양수 조건에서 Sharpe 최대화
  단, n_trades < 5 이면 -999 (통계 불충분)

사용법:
  python scripts/optimize_backtest.py \
    --model-path checkpoints/quantum_v2/agent_best.pt \
    --start-date 2025-09-01 --end-date 2026-03-01 \
    --n-trials 60 --study-name my_study

결과:
  - 최적 파라미터 출력
  - reports/optuna_<study>.db (재개 가능)
  - reports/optuna_<study>.png (importance plot)
--------------------------------------------------------------------
"""

import argparse
import os
import sys
import math
import warnings
import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import optuna
from optuna.samplers import TPESampler

# ── 백테스트 핵심 로직 임포트 ─────────────────────────────────────────
from src.models.integrated_agent import build_quantum_agent, AgentConfig
from src.data.data_client import DataClient
from src.models.features_v4 import build_features_v4

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── 공유 캐시 (데이터/피처 한 번만 로드) ─────────────────────────────────
_cache: dict = {}


def _load_data(args) -> tuple:
    """데이터 + 피처를 한 번만 로드해 캐시."""
    key = (args.start_date, args.end_date, args.timeframe)
    if key in _cache:
        return _cache[key]

    print(f"[data] Loading {args.symbol} {args.timeframe} "
          f"{args.start_date} ~ {args.end_date} ...")
    client = DataClient()
    df = client.get_ohlcv(args.symbol, args.timeframe,
                          start=args.start_date, end=args.end_date)
    print(f"[data] {len(df)} bars")

    feats = build_features_v4(df)
    _cache[key] = (df, feats)
    return df, feats


def _run_single(model_path: str, df, feats, confidence: float,
                tp_mult: float, sl_mult: float,
                leverage: float, capital: float,
                pos_frac: float, timeframe: str,
                device: str) -> dict:
    """단일 파라미터 조합 백테스트. dict 반환."""

    # ── 에이전트 로드 ─────────────────────────────────────────────────
    cfg = AgentConfig(confidence_threshold=0.0)
    agent = build_quantum_agent(cfg)
    ckpt = torch.load(model_path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    agent.load_state_dict(state, strict=False)
    agent.eval()

    # ── 수수료 ───────────────────────────────────────────────────────
    eta_maker  = 0.0002
    eta_taker  = 0.00055
    round_trip = eta_maker + eta_taker

    # ── 시뮬레이션 ────────────────────────────────────────────────────
    prices  = df["close"].values.astype(np.float64)
    highs   = df["high"].values.astype(np.float64)
    lows    = df["low"].values.astype(np.float64)
    n_bars  = len(prices)

    # ATR (Wilder, span=14)
    tr = np.maximum(highs[1:] - lows[1:],
         np.maximum(np.abs(highs[1:] - prices[:-1]),
                    np.abs(lows[1:]  - prices[:-1])))
    atr_arr = np.zeros(n_bars)
    atr_arr[1] = tr[0]
    alpha = 1.0 / 14.0
    for i in range(2, n_bars):
        atr_arr[i] = alpha * tr[i-1] + (1 - alpha) * atr_arr[i-1]

    cash       = float(capital)
    position   = None   # dict or None
    n_trades   = 0
    n_wins     = 0
    eq_curve   = [cash]

    MIN_FEAT   = 60     # spectral decomposer warm-up

    with torch.no_grad():
        for i in range(MIN_FEAT, n_bars):
            price = prices[i]
            atr   = atr_arr[i]
            if atr <= 0 or price <= 0:
                eq_curve.append(cash)
                continue

            atr_frac = atr / price
            tp_pct   = tp_mult * atr_frac
            sl_pct   = sl_mult * atr_frac

            # ── 포지션 청산 ───────────────────────────────────────
            if position is not None:
                ret = ((price - position["entry"]) / position["entry"]
                       if position["side"] == "long"
                       else (position["entry"] - price) / position["entry"])

                hit_tp = ret >=  position["tp"]
                hit_sl = ret <= -position["sl"]

                if hit_tp or hit_sl:
                    pnl_pct = ret if hit_tp else -position["sl"]
                    fee     = round_trip * position["notional"]
                    pnl_usd = pnl_pct * position["notional"] - fee
                    cash   += pnl_usd
                    if cash < 0:
                        cash = 0.0
                    n_trades += 1
                    if pnl_usd > 0:
                        n_wins += 1
                    position = None

            eq_curve.append(cash)

            if cash <= 0:
                break

            # ── 신규 진입 ─────────────────────────────────────────
            if position is None:
                x = torch.tensor(feats[i:i+1], dtype=torch.float32)
                action, conf, _ = agent.select_action(
                    x, atr_norm=float(atr_frac))

                if action in (1, 2) and float(conf) >= confidence:
                    if tp_pct / (sl_pct + 1e-10) >= 1.5:   # 최소 R:R 1.5
                        notional = cash * pos_frac * leverage
                        position = {
                            "side":     "long" if action == 1 else "short",
                            "entry":    price,
                            "tp":       tp_pct,
                            "sl":       sl_pct,
                            "notional": notional,
                        }

    eq_arr = np.array(eq_curve, dtype=np.float64)
    roi    = (cash - capital) / capital * 100.0

    # Sharpe
    if len(eq_arr) > 1:
        rets  = np.diff(eq_arr) / np.where(eq_arr[:-1] > 0, eq_arr[:-1], 1.0)
        mu, sig = np.mean(rets), np.std(rets)
        bars_py = {"1m": 525960, "5m": 105192, "15m": 35040, "1h": 8766}.get(
            timeframe, 35040)
        sharpe = float(mu / sig * math.sqrt(bars_py)) if sig > 1e-10 else 0.0
    else:
        sharpe = 0.0

    # MDD
    peak = np.maximum.accumulate(eq_arr)
    dd   = (peak - eq_arr) / np.where(peak > 0, peak, 1.0)
    mdd  = float(np.max(dd)) * 100.0

    wr = n_wins / n_trades * 100.0 if n_trades > 0 else 0.0

    return dict(roi=roi, sharpe=sharpe, mdd=mdd,
                n_trades=n_trades, wr=wr, final_cash=cash)


# ── Optuna Objective ──────────────────────────────────────────────────
def make_objective(args, df, feats):
    def objective(trial: optuna.Trial) -> float:
        confidence = trial.suggest_float("confidence", 0.30, 0.70)
        tp_mult    = trial.suggest_float("tp_mult",    2.0,  8.0)
        sl_mult    = trial.suggest_float("sl_mult",    0.5,  2.0)

        res = _run_single(
            model_path  = args.model_path,
            df          = df,
            feats       = feats,
            confidence  = confidence,
            tp_mult     = tp_mult,
            sl_mult     = sl_mult,
            leverage    = args.leverage,
            capital     = args.capital,
            pos_frac    = args.pos_frac,
            timeframe   = args.timeframe,
            device      = args.device,
        )

        n = res["n_trades"]
        roi    = res["roi"]
        sharpe = res["sharpe"]
        mdd    = res["mdd"]

        # 통계 불충분
        if n < args.min_trades:
            return -999.0

        # ROI 음수면 패널티
        if roi <= 0:
            return roi / 100.0   # -1 ~ 0 사이 작은 음수

        # 목적: Sharpe 최대화 (MDD 패널티 포함)
        score = sharpe - (mdd / 100.0) * args.mdd_penalty

        trial.set_user_attr("roi",      round(roi, 2))
        trial.set_user_attr("sharpe",   round(sharpe, 3))
        trial.set_user_attr("mdd",      round(mdd, 1))
        trial.set_user_attr("n_trades", n)
        trial.set_user_attr("wr",       round(res["wr"], 1))

        return score

    return objective


# ── 메인 ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Optuna Backtest Optimizer")
    parser.add_argument("--model-path",  default="checkpoints/quantum_v2/agent_best.pt")
    parser.add_argument("--symbol",      default="BTCUSDT")
    parser.add_argument("--timeframe",   default="15m")
    parser.add_argument("--start-date",  default="2025-09-01")
    parser.add_argument("--end-date",    default="2026-03-01")
    parser.add_argument("--capital",     type=float, default=100.0)
    parser.add_argument("--leverage",    type=float, default=10.0)
    parser.add_argument("--pos-frac",    type=float, default=0.5)
    parser.add_argument("--n-trials",    type=int,   default=60)
    parser.add_argument("--min-trades",  type=int,   default=10,
                        help="최소 거래 수 (미달 시 trial 무효)")
    parser.add_argument("--mdd-penalty", type=float, default=0.5,
                        help="MDD 패널티 계수 (score -= mdd% × coef)")
    parser.add_argument("--study-name",  default="bt_opt")
    parser.add_argument("--device",      default="cpu")
    args = parser.parse_args()

    os.makedirs("reports", exist_ok=True)
    db_path = f"reports/optuna_{args.study_name}.db"

    # ── 데이터 로드 (1회) ──────────────────────────────────────────
    df, feats = _load_data(args)

    # ── Study 생성 / 재개 ──────────────────────────────────────────
    storage  = f"sqlite:///{db_path}"
    sampler  = TPESampler(seed=42)
    study    = optuna.create_study(
        study_name     = args.study_name,
        storage        = storage,
        direction      = "maximize",
        sampler        = sampler,
        load_if_exists = True,
    )

    print(f"\n{'='*60}")
    print(f"  Optuna Backtest Optimizer")
    print(f"  Model  : {args.model_path}")
    print(f"  Period : {args.start_date} ~ {args.end_date}")
    print(f"  Trials : {args.n_trials}  (DB: {db_path})")
    print(f"{'='*60}\n")

    def _cb(study, trial):
        b = study.best_trial
        print(f"  Trial {trial.number:3d} | "
              f"conf={trial.params.get('confidence', 0):.2f} "
              f"tp={trial.params.get('tp_mult', 0):.1f} "
              f"sl={trial.params.get('sl_mult', 0):.2f} | "
              f"ROI={trial.user_attrs.get('roi', '?'):>8} "
              f"Sharpe={trial.user_attrs.get('sharpe', '?'):>6} "
              f"MDD={trial.user_attrs.get('mdd', '?'):>5}% "
              f"N={trial.user_attrs.get('n_trades', '?'):>4} | "
              f"Best→ #{b.number} score={b.value:.3f}")

    study.optimize(
        make_objective(args, df, feats),
        n_trials  = args.n_trials,
        callbacks = [_cb],
        show_progress_bar = False,
    )

    # ── 결과 출력 ─────────────────────────────────────────────────
    best = study.best_trial
    print(f"\n{'='*60}")
    print(f"  BEST TRIAL #{best.number}")
    print(f"{'='*60}")
    print(f"  confidence : {best.params['confidence']:.4f}")
    print(f"  tp_mult    : {best.params['tp_mult']:.4f}")
    print(f"  sl_mult    : {best.params['sl_mult']:.4f}")
    print(f"  ─────────────────────────────")
    print(f"  ROI        : {best.user_attrs.get('roi', '?')}%")
    print(f"  Sharpe     : {best.user_attrs.get('sharpe', '?')}")
    print(f"  MDD        : {best.user_attrs.get('mdd', '?')}%")
    print(f"  WR         : {best.user_attrs.get('wr', '?')}%")
    print(f"  Trades     : {best.user_attrs.get('n_trades', '?')}")
    print(f"{'='*60}\n")

    # 최적 파라미터로 재실행 명령어 출력
    print("  Run with best params:")
    print(f"  python scripts/backtest_model_v2.py \\")
    print(f"    --model-path {args.model_path} \\")
    print(f"    --start-date {args.start_date} --end-date {args.end_date} \\")
    print(f"    --confidence {best.params['confidence']:.4f} \\")
    print(f"    --tp-mult {best.params['tp_mult']:.4f} \\")
    print(f"    --sl-mult {best.params['sl_mult']:.4f} \\")
    print(f"    --leverage {args.leverage} --capital {args.capital}\n")

    # Importance plot
    try:
        import optuna.visualization as vis
        fig = vis.plot_param_importances(study)
        fig.write_image(f"reports/optuna_{args.study_name}_importance.png")
        print(f"  Importance plot → reports/optuna_{args.study_name}_importance.png")
    except Exception:
        pass


if __name__ == "__main__":
    main()
