"""
Optuna BC Hyperparameter Search — Quantum Trading V2
=====================================================
TPE sampler + MedianPruner + SQLite 저장 (재시작 가능)

Usage:
  python scripts/optuna_bc_search.py                    # 기본 (100 trials)
  python scripts/optuna_bc_search.py --trials 200       # 200 trials
  python scripts/optuna_bc_search.py --resume           # 이전 study 이어서
  python scripts/optuna_bc_search.py --trials 50 --study-name my_study

탐색 공간:
  seed            int   [0, 999]
  orth_weight     float [0.01, 0.15]  log scale
  parity_weight   float [0.02, 0.20]  log scale
  label_smoothing float [0.05, 0.20]
"""

import subprocess, sys, re, argparse, time, shutil
from pathlib import Path

# ── Optuna 설치 확인 ───────────────────────────────────────────
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners  import MedianPruner
except ImportError:
    print("[Setup] optuna 설치 중...")
    subprocess.run([sys.executable, "-m", "pip", "install", "optuna", "-q"], check=True)
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners  import MedianPruner

optuna.logging.set_verbosity(optuna.logging.WARNING)  # Optuna 내부 로그 최소화

# ── BC 실행 설정 ───────────────────────────────────────────────
BASE_CMD = [
    sys.executable, "scripts/pretrain_bc.py",
    "--symbol", "BTCUSDT", "--timeframe", "15m",
    "--start-date", "2019-01-31", "--end-date", "2025-09-01",
    "--device", "cuda",
]

RE_EPOCH = re.compile(
    r"^\s+(\d+)\s+[\d.]+\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%"
)  # group1=ep, group2=TrAcc, group3=ValAcc, group4=H%, group5=L%, group6=S%

SCREEN_CKPT = "checkpoints/optuna_screen"
FULL_CKPT   = "checkpoints/quantum_v2"


# ── 단일 BC trial 실행 ─────────────────────────────────────────
def run_trial_bc(trial: "optuna.Trial", epochs: int, ckpt_dir: str) -> float:
    """BC pretrain 실행 + Optuna 중간 보고 + Pruning 지원"""
    seed            = trial.suggest_int  ("seed",            0,    999)
    orth_weight     = trial.suggest_float("orth_weight",     0.01, 0.15, log=True)
    parity_weight   = trial.suggest_float("parity_weight",   0.02, 0.20, log=True)
    label_smoothing = trial.suggest_float("label_smoothing", 0.05, 0.20)

    cmd = BASE_CMD + [
        "--epochs",          str(epochs),
        "--seed",            str(seed),
        "--orth-weight",     f"{orth_weight:.4f}",
        "--parity-weight",   f"{parity_weight:.4f}",
        "--label-smoothing", f"{label_smoothing:.4f}",
        "--patience",        str(epochs),   # EarlyStop 비활성
        "--checkpoint-dir",  ckpt_dir,
    ]

    best_val = 0.0
    output_lines = []

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace"
    )

    for line in proc.stdout:
        line = line.rstrip()
        output_lines.append(line)
        m = RE_EPOCH.match(line)
        if m:
            ep      = int(m.group(1))
            val_acc = float(m.group(3))
            h_pct   = float(m.group(4))
            # HOLD collapse 필터: H%>85% 이면 사기 점수 무시
            valid = h_pct < 85.0
            if valid and val_acc > best_val:
                best_val = val_acc

            # Optuna 중간 보고 — collapse면 0 보고 (Pruner가 즉시 자름)
            reported = val_acc if valid else 0.0
            trial.report(reported, step=ep)
            tag = "⚠HOLD" if not valid else ""
            print(f"    Ep{ep:>2}  ValAcc={val_acc:5.2f}%  H%={h_pct:4.1f}%  best={best_val:.2f}%"
                  f"  [seed={seed} orth={orth_weight:.3f} par={parity_weight:.3f} ls={label_smoothing:.3f}] {tag}",
                  end="\r")

            # Pruning 판단
            if trial.should_prune():
                proc.terminate()
                proc.wait()
                print(f"\n  ✂ Pruned at Ep{ep} (ValAcc={val_acc:.2f}%)")
                raise optuna.TrialPruned()

    proc.wait()
    print()

    # 크래시 감지
    if best_val == 0.0:
        print("  ⚠ ValAcc=0 (크래시). 마지막 출력:")
        for l in output_lines[-8:]:
            print(f"    {l}")

    return best_val


# ── Objective ──────────────────────────────────────────────────
def make_objective(screen_epochs: int):
    def objective(trial: "optuna.Trial") -> float:
        return run_trial_bc(trial, screen_epochs, SCREEN_CKPT)
    return objective


# ── 최종 풀 학습 ───────────────────────────────────────────────
def run_full_training(params: dict, full_epochs: int) -> float:
    """최적 하이퍼파라미터로 풀 학습"""
    cmd = BASE_CMD + [
        "--epochs",          str(full_epochs),
        "--seed",            str(params["seed"]),
        "--orth-weight",     f"{params['orth_weight']:.4f}",
        "--parity-weight",   f"{params['parity_weight']:.4f}",
        "--label-smoothing", f"{params['label_smoothing']:.4f}",
        "--patience",        "7",    # 풀 학습은 EarlyStop 활성
        "--checkpoint-dir",  FULL_CKPT,
    ]

    best_val = 0.0
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace"
    )
    for line in proc.stdout:
        line = line.rstrip()
        print(line)
        m = RE_EPOCH.match(line)
        if m:
            val_acc = float(m.group(3))
            if val_acc > best_val:
                best_val = val_acc
    proc.wait()
    return best_val


# ── 메인 ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",        type=int,   default=100)
    parser.add_argument("--screen-epochs", type=int,   default=7,
                        help="스크리닝 에폭 (default=7)")
    parser.add_argument("--full-epochs",   type=int,   default=20,
                        help="최종 풀 학습 에폭 (default=20)")
    parser.add_argument("--threshold",     type=float, default=38.0,
                        help="풀 학습 진행 기준 ValAcc%% (default=38)")
    parser.add_argument("--top-k",         type=int,   default=3,
                        help="풀 학습 후보 수 (default=3)")
    parser.add_argument("--study-name",    default="bc_hpo_v1")
    parser.add_argument("--storage",       default="sqlite:///optuna_bc.db",
                        help="Optuna storage (default: SQLite 로컬)")
    parser.add_argument("--resume",        action="store_true",
                        help="기존 study 이어서 탐색")
    parser.add_argument("--n-startup",     type=int, default=15,
                        help="TPE 시작 전 랜덤 탐색 수 (default=15)")
    args = parser.parse_args()

    Path(SCREEN_CKPT).mkdir(parents=True, exist_ok=True)
    Path(FULL_CKPT).mkdir(parents=True, exist_ok=True)

    # ── Study 생성 / 로드 ──────────────────────────────────────
    sampler = TPESampler(
        n_startup_trials=args.n_startup,   # 초반엔 랜덤 탐색
        seed=42,                           # 탐색 자체의 재현성
        multivariate=True,                 # 파라미터 간 상관관계 고려
        constant_liar=True,                # 병렬 실행 시 중복 방지
    )
    pruner = MedianPruner(
        n_startup_trials=10,   # 10개 trial 후부터 pruning 시작
        n_warmup_steps=3,      # 각 trial의 첫 3 에폭은 pruning 면제
        interval_steps=1,
    )

    load_if_exists = args.resume
    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        load_if_exists=load_if_exists,
    )

    completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    pruned    = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    print("=" * 65)
    print(f"  Optuna BC Hyperparameter Search")
    print(f"  Study    : {args.study_name}  ({'재시작' if args.resume else '신규'})")
    print(f"  Storage  : {args.storage}")
    print(f"  Trials   : {args.trials}개  (완료={completed}, Pruned={pruned})")
    print(f"  Screen   : {args.screen_epochs} epochs per trial")
    print(f"  Threshold: ValAcc ≥ {args.threshold}% → 풀 학습 진행")
    print(f"  탐색 공간:")
    print(f"    seed            : int   [0, 999]")
    print(f"    orth_weight     : float [0.01, 0.15]  log")
    print(f"    parity_weight   : float [0.02, 0.20]  log")
    print(f"    label_smoothing : float [0.05, 0.20]")
    print("=" * 65)

    # ── Phase 1: Optuna 탐색 ───────────────────────────────────
    t0 = time.time()
    study.optimize(
        make_objective(args.screen_epochs),
        n_trials=args.trials,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    elapsed = time.time() - t0
    print(f"\n탐색 완료: {elapsed/60:.1f}분")

    # ── Phase 1 결과 요약 ───────────────────────────────────────
    done_trials = [t for t in study.trials
                   if t.state == optuna.trial.TrialState.COMPLETE]
    done_trials.sort(key=lambda t: t.value, reverse=True)

    print("\n" + "=" * 65)
    print(f"  📊 Top-10 Trial 결과")
    print(f"  {'Rank':>4}  {'Trial':>6}  {'ValAcc':>7}  {'seed':>5}  "
          f"{'orth':>6}  {'parity':>7}  {'ls':>5}")
    print("  " + "-" * 55)
    for rank, t in enumerate(done_trials[:10], 1):
        p = t.params
        print(f"  {rank:>4}  #{t.number:>5}  {t.value:>6.2f}%  "
              f"{p['seed']:>5}  {p['orth_weight']:>6.3f}  "
              f"{p['parity_weight']:>7.3f}  {p['label_smoothing']:>5.3f}")

    # Pruning 통계
    pruned_n  = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
    print(f"\n  Pruned: {pruned_n}/{len(study.trials)} trials (효율 {pruned_n/max(1,len(study.trials))*100:.0f}% 절약)")

    # ── Phase 2: Top-K 풀 학습 ──────────────────────────────────
    candidates = [t for t in done_trials if t.value >= args.threshold]
    if not candidates:
        print(f"\n  ⚠ threshold={args.threshold}% 통과 없음 → 상위 {args.top_k}개로 진행")
        candidates = done_trials[:args.top_k]
    else:
        candidates = candidates[:args.top_k]

    if not candidates:
        print("  ❌ 유효한 trial 없음. 탐색 실패.")
        return

    print(f"\n  🏋 풀 학습 대상: {len(candidates)}개 trials")
    print("=" * 65)

    full_results = {}
    for t in candidates:
        p = t.params
        print(f"\n[Full] Trial#{t.number}  seed={p['seed']}"
              f"  orth={p['orth_weight']:.3f}  par={p['parity_weight']:.3f}"
              f"  ls={p['label_smoothing']:.3f}")
        val = run_full_training(p, args.full_epochs)
        full_results[t.number] = (val, p)
        print(f"  → Final ValAcc={val:.2f}%")

    # ── 최종 우승 ───────────────────────────────────────────────
    best_trial_num = max(full_results, key=lambda k: full_results[k][0])
    best_val, best_params = full_results[best_trial_num]

    print("\n" + "=" * 65)
    print(f"  🏆 최적 결과")
    print(f"  ValAcc       : {best_val:.2f}%")
    print(f"  seed         : {best_params['seed']}")
    print(f"  orth_weight  : {best_params['orth_weight']:.4f}")
    print(f"  parity_weight: {best_params['parity_weight']:.4f}")
    print(f"  label_smooth : {best_params['label_smoothing']:.4f}")
    print(f"  체크포인트   : {FULL_CKPT}/agent_bc_pretrained.pt")
    print()
    print(f"  RL 학습 명령어:")
    print(f"  python scripts/train_quantum_v2.py \\")
    print(f"    --symbol BTCUSDT --timeframe 15m \\")
    print(f"    --start-date 2019-01-31 --end-date 2025-09-01 \\")
    print(f"    --n-folds 5 --epochs 30 --rolling-window 480 \\")
    print(f"    --device cuda --confidence 0.30 \\")
    print(f"    --pretrain-ckpt {FULL_CKPT}/agent_bc_pretrained.pt")
    print("=" * 65)


if __name__ == "__main__":
    main()
