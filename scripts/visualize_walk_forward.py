"""
visualize_walk_forward.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Walk-forward backtest 결과의 누적 손익(Equity Value) 곡선을 시각화합니다.

실행 방법:
  python scripts/visualize_walk_forward.py

입력 파일:
  reports/walk_forward_results.csv  (train_quantum_v2.py 실행 시 생성됨)

출력 파일:
  reports/viz/walk_forward_ev_curve.png
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import sys

# Windows CP949 콘솔에서 UTF-8 특수문자 출력 허용
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

def visualize_cumulative_ev_curve():
    """
    Walk-forward 결과 CSV 파일을 읽어 누적 손익 곡선을 시각화하고 이미지로 저장합니다.
    """
    input_csv = "reports/walk_forward_results.csv"
    output_png = "reports/viz/walk_forward_ev_curve.png"
    output_dir = os.path.dirname(output_png)

    # 1. 입력 파일 확인
    if not os.path.exists(input_csv):
        print(f"오류: 입력 파일이 존재하지 않습니다. '{input_csv}'")
        print("먼저 'python scripts/train_quantum_v2.py'를 실행하여 결과 파일을 생성해야 합니다.")
        return

    # 2. 데이터 로드 및 전처리
    try:
        df = pd.read_csv(input_csv)
        if 'ts' not in df.columns or 'cumulative_pnl_pct' not in df.columns:
            print(f"오류: '{input_csv}' 파일에 'ts' 또는 'cumulative_pnl_pct' 열이 없습니다.")
            return
        
        # 'ts' 열을 datetime 객체로 변환
        df['ts'] = pd.to_datetime(df['ts'])
        df = df.sort_values(by='ts')
        
    except Exception as e:
        print(f"데이터를 읽는 중 오류가 발생했습니다: {e}")
        return

    # 3. 시각화
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        # seaborn-v0_8-darkgrid 스타일이 없을 경우 기본 스타일 사용
        pass

    fig, ax = plt.subplots(figsize=(16, 8))

    ax.plot(df['ts'], df['cumulative_pnl_pct'], label='Walk-Forward Cumulative PnL', color='cyan')
    ax.fill_between(df['ts'], df['cumulative_pnl_pct'], alpha=0.15, color='cyan')

    # 4. 그래프 스타일링
    ax.set_title('Walk-Forward Cumulative Equity Curve', fontsize=18, pad=20)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative PnL (%)', fontsize=12)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    # y=0 수평선 추가
    ax.axhline(0, color='red', linestyle='--', linewidth=1, label='Break-Even Point')

    # X축 날짜 형식 설정
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    fig.autofmt_xdate()

    ax.legend()
    plt.tight_layout()

    # 5. 이미지 파일로 저장
    try:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(output_png, dpi=300)
        print(f"\n✅ 누적 손익 곡선 그래프 저장 완료: {os.path.abspath(output_png)}")
    except Exception as e:
        print(f"그래프를 저장하는 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    visualize_cumulative_ev_curve()
