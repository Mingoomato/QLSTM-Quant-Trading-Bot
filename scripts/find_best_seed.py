"""
BC Seed Finder — 최적 랜덤 시드 자동 탐색
=============================================
Usage:
  python scripts/find_best_seed.py
  python scripts/find_best_seed.py --seeds 30 --screen-epochs 5 --threshold 38
  python scripts/find_best_seed.py --seeds 50 --workers 5 --batch-size 1024  # A100 병렬
"""
import subprocess, sys, re, argparse, time
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Config (args 파싱 후 _build_base_cmd()로 완성) ─────────────
_BASE_CMD_FIXED = [
    sys.executable, "scripts/pretrain_bc.py",
    "--symbol", "BTCUSDT", "--timeframe", "15m",
    "--start-date", "2019-01-31", "--end-date", "2025-09-01",
]

def _build_base_cmd(device: str, batch_size: int) -> list:
    return _BASE_CMD_FIXED + ["--device", device, "--batch-size", str(batch_size)]

RE_EPOCH = re.compile(
    r"^\s+(\d+)\s+[\d.]+\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%\s+([\d.]+)%"
)

# ── 단일 BC 실행 + ValAcc 실시간 파싱 ──────────────────────────
def run_bc(seed: int, epochs: int, ckpt_dir: str, verbose: bool = False,
           device: str = "cuda", batch_size: int = 512,
           silent: bool = False) -> float:
    """silent=True: 병렬 실행 시 출력 억제 (결과만 반환)"""
    cmd = _build_base_cmd(device, batch_size) + [
        "--epochs", str(epochs),
        "--seed",   str(seed),
        "--checkpoint-dir", ckpt_dir,
        "--patience", str(epochs),          # EarlyStop 비활성 (전 에폭 실행)
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
        if verbose:
            print(line)
        m = RE_EPOCH.match(line)
        if m:
            ep      = int(m.group(1))
            val_acc = float(m.group(3))
            h_pct   = float(m.group(4))
            l_pct   = float(m.group(5))
            s_pct   = float(m.group(6))
            # HOLD collapse 필터: H%>85% 이면 사기 점수 (항상 HOLD 예측)
            valid = h_pct < 85.0
            if valid and val_acc > best_val:
                best_val = val_acc
            tag = "⚠HOLD" if not valid else ""
            if not verbose and not silent:
                print(f"    Ep{ep:>2}  ValAcc={val_acc:5.1f}%  H%={h_pct:4.1f}% L%={l_pct:4.1f}% S%={s_pct:4.1f}%"
                      f"  best={best_val:.1f}%  {tag}", end="\r")
    proc.wait()
    if not verbose and not silent:
        print()

    # 크래시 감지: ValAcc=0이면 마지막 10줄 출력
    if best_val == 0.0 and not silent:
        print(f"  ⚠ Seed={seed} ValAcc=0 (크래시 의심). 마지막 출력:")
        for l in output_lines[-10:]:
            print(f"    {l}")
    return best_val


# ── 메인 ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds",          type=int, default=20,
                        help="탐색할 시드 개수 (default=20)")
    parser.add_argument("--seed-start",     type=int, default=0,
                        help="시작 시드 번호 (default=0)")
    parser.add_argument("--screen-epochs",  type=int, default=5,
                        help="스크리닝 에폭 수 (default=5, 빠를수록 빠름)")
    parser.add_argument("--full-epochs",    type=int, default=20,
                        help="최종 풀 학습 에폭 (default=20)")
    parser.add_argument("--threshold",      type=float, default=37.0,
                        help="스크리닝 통과 기준 ValAcc %% (default=37)")
    parser.add_argument("--top-k",          type=int, default=3,
                        help="풀 학습 진행할 Top-K 시드 (default=3)")
    parser.add_argument("--ckpt-dir",       default="checkpoints/quantum_v2",
                        help="최종 체크포인트 저장 경로")
    parser.add_argument("--random",         action="store_true",
                        help="0-999 중 랜덤하게 시드 선택 (default: 순차)")
    parser.add_argument("--random-seed",    type=int, default=None,
                        help="랜덤 시드 선택 자체의 재현성 (미지정 시 매번 다름)")
    parser.add_argument("--device",         default="cuda",
                        help="학습 장치 (default: cuda)")
    parser.add_argument("--batch-size",     type=int, default=512,
                        help="배치 크기 (default: 512, A100은 1024 권장)")
    parser.add_argument("--workers",        type=int, default=1,
                        help="병렬 스크리닝 프로세스 수 (default=1, A100은 4~6 권장)")
    args = parser.parse_args()

    if args.random:
        rng = np.random.default_rng(args.random_seed)
        seeds = sorted(rng.choice(1000, size=args.seeds, replace=False).tolist())
    else:
        seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    screen_dir = "checkpoints/seed_screen"
    Path(screen_dir).mkdir(parents=True, exist_ok=True)
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  BC Seed Finder")
    print(f"  시드 범위   : {seeds[0]} ~ {seeds[-1]}  ({len(seeds)}개)")
    print(f"  스크리닝    : {args.screen_epochs} epochs per seed")
    print(f"  통과 기준   : ValAcc ≥ {args.threshold:.1f}%")
    print(f"  풀 학습     : Top-{args.top_k} → {args.full_epochs} epochs")
    print(f"  병렬 workers: {args.workers}  batch_size={args.batch_size}")
    print("=" * 60)

    # ── Phase 1: 스크리닝 ───────────────────────────────────────
    screen_results = {}
    t0 = time.time()
    done = [0]  # mutable counter for thread-safe progress

    if args.workers == 1:
        # ── 순차 실행 (기존 동작) ──────────────────────────────
        for i, seed in enumerate(seeds):
            elapsed = time.time() - t0
            eta_s   = elapsed / max(i, 1) * (len(seeds) - i)
            print(f"\n[{i+1}/{len(seeds)}] Seed={seed}  "
                  f"(elapsed={elapsed/60:.1f}m  ETA={eta_s/60:.1f}m)")
            val = run_bc(seed, args.screen_epochs, screen_dir,
                         device=args.device, batch_size=args.batch_size)
            screen_results[seed] = val
            mark = "✅" if val >= args.threshold else "  "
            print(f"  → Best ValAcc={val:.2f}%  {mark}")
    else:
        # ── 병렬 실행 (A100 다중 프로세스) ────────────────────
        print(f"  [병렬] {args.workers}개 동시 실행 중... (완료 순서로 출력)")
        print()
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    run_bc, seed, args.screen_epochs, screen_dir,
                    False, args.device, args.batch_size, True  # silent=True
                ): seed
                for seed in seeds
            }
            for future in as_completed(futures):
                seed = futures[future]
                val  = future.result()
                screen_results[seed] = val
                done[0] += 1
                elapsed = time.time() - t0
                mark = "✅" if val >= args.threshold else "  "
                print(f"  [{done[0]:>3}/{len(seeds)}] Seed={seed:>4}  "
                      f"ValAcc={val:5.2f}%  {mark}  "
                      f"(elapsed={elapsed/60:.1f}m)")

    # ── Phase 1 결과 요약 ───────────────────────────────────────
    total_elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"  스크리닝 완료: {total_elapsed/60:.1f}분")
    print("  📊 스크리닝 결과 (상위 10개)")
    print("  " + "-" * 40)
    ranked = sorted(screen_results.items(), key=lambda x: x[1], reverse=True)
    for rank, (seed, val) in enumerate(ranked[:10], 1):
        mark = "✅" if val >= args.threshold else "  "
        print(f"  {rank:>2}. Seed={seed:>4}  ValAcc={val:5.2f}%  {mark}")

    # ── Phase 2: Top-K 풀 학습 ──────────────────────────────────
    top_seeds = [s for s, _ in ranked[:args.top_k]]
    passing   = [s for s, v in ranked if v >= args.threshold]

    if not passing:
        print(f"\n  ⚠ threshold={args.threshold}% 통과 시드 없음.")
        print(f"  상위 {args.top_k}개로 풀 학습 진행.")
    else:
        top_seeds = (passing + [s for s, _ in ranked if s not in passing])[:args.top_k]

    print(f"\n  풀 학습 대상 시드: {top_seeds}")
    print("=" * 60)

    full_results = {}
    for seed in top_seeds:
        print(f"\n[Full] Seed={seed}  ({args.full_epochs} epochs)")
        val = run_bc(seed, args.full_epochs, args.ckpt_dir, verbose=False,
                     device=args.device, batch_size=args.batch_size)
        full_results[seed] = val
        print(f"  → Final ValAcc={val:.2f}%")

    # ── 최종 우승 시드 ──────────────────────────────────────────
    best_seed = max(full_results, key=full_results.get)
    best_val  = full_results[best_seed]

    print("\n" + "=" * 60)
    print(f"  최적 시드: {best_seed}  (ValAcc={best_val:.2f}%)")
    print(f"  체크포인트 : {args.ckpt_dir}/agent_bc_pretrained.pt")
    print()
    print(f"  RL 학습 명령어:")
    print(f"  python scripts/train_quantum_v2.py \\")
    print(f"    --symbol BTCUSDT --timeframe 15m \\")
    print(f"    --start-date 2019-01-31 --end-date 2025-09-01 \\")
    print(f"    --n-folds 5 --epochs 30 --rolling-window 480 \\")
    print(f"    --device cuda --confidence 0.38 \\")
    print(f"    --pretrain-ckpt {args.ckpt_dir}/agent_bc_pretrained.pt")
    print("=" * 60)


if __name__ == "__main__":
    main()
