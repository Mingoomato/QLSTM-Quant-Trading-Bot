# Grand Council Agenda — 온체인 펀더멘털 피처 확장 (2026-03-23)

> **의장**: CEO Demis
> **긴급 안건**: 온체인 펀더멘털 피처 도입 — 가격-파생 피처의 구조적 한계 돌파
> **회의 성격**: 훈련 결과 검토 + 전략 피벗 승인 + 즉시 실행 결정

---

## PART 0 — 현재 모델 상태 (훈련 로그 요약)

> 이 섹션은 Grand Council 전에 모든 멤버가 숙지해야 할 베이스라인입니다.
> 데이터: `logs/logs.md` (BC pre-training + RL 10-fold training 결과)

### 0-1. BC Pre-training 결과

| 항목 | 값 | 비고 |
|------|-----|------|
| Timeframe | **1h** | RL은 30m으로 다름 → 타임프레임 불일치 |
| 기간 | 2019-01-01 ~ 2022-12-31 | |
| 심볼 | BTC + SOL + ETH | |
| Best BalAcc | **39.21%** | random baseline = 33.3% |
| 개선폭 | +5.91%p | 통계적으로 유의한 신호 |
| Feature dim | 28 (V4) | Koopman + LDA + FR + OI + CVD |
| Koopman λ₁ | 0.9626 | Slow mode 1/5 (|λ|>0.95) |
| 결론 | ✅ RL 진행 가능 | 최소 임계값(36%+) 충족 |

### 0-2. RL Training 10-Fold 요약

**훈련 설정**: `train_quantum_v2.py` — 30m, 2023-01-01~2025-01-01, 3 심볼, expanding window

| Fold | Val 기간 | Best EV/trade | Best WR | Critic Loss | 비고 |
|------|----------|:---:|:---:|:---:|------|
| 1 | 2023-03~05 | +0.0555 | **49.7%** | 0.033 | Barren E5 (일시적) |
| 2 | 2023-05~07 | +0.0449 | 47.1% | 0.027 | EarlyStop E8 |
| 3 | 2023-07~09 | +0.0381 | 47.5% | 0.025 | **GN_HIGH** E5,6,9 |
| 4 | 2023-09~11 | **+0.0893** | **50.3%** | 0.024 | 🏆 Global Best |
| 5 | 2023-11~24-01 | +0.0668 | 50.0% | 0.026 | **GN=30.8** E8 (심각) |
| 6 | 2024-01~04 | +0.0865 | 46.9% | 0.026 | |
| 7 | 2024-04~06 | +0.0669 | 46.0% | 0.028 | EarlyStop E7 |
| 8 | 2024-06~08 | +0.0572 | **38.0%** | 0.033 | ⚠️ WR 급락 Barren E8 |
| 9 | 2024-08~10 | +0.0584 | 45.7% | 0.093 | Critic 3.8x 상승 |
| 10 | 2024-10~25-01 | +0.0844 | 44.9% | **0.222** | ⚠️ Critic **6.7x** 발산 |

**Global Best**: `checkpoints/quantum_v2/agent_best.pt` — EV=+0.0893 (Fold 4)

### 0-3. 훈련 로그에서 발견된 핵심 문제 5개

#### 🔴 문제 1: Critic Loss 발산 (Fold 1→10: 0.033→0.222, 6.7배)
```
Fold 1  Critic=0.033   (정상)
Fold 9  Critic=0.093   (3.8x)
Fold 10 Critic=0.222   (6.7x) ← 심각
```
- GAE advantage Â_t 오염 → policy gradient 오염
- Fold이 누적될수록 이전 Fold 발산 weight가 초기값으로 전달
- **해결**: Critic LR 분리 (`lr_critic = lr_classic × 0.1`), Fold 경계 Critic head 재초기화

#### 🔴 문제 2: WR 불안정 (Fold 8: 38.0%)
```
Fold 1~5 avg WR: 48.7%
Fold 6~10 avg WR: 44.5%  (4.2%p 하락)
Fold 8: 38.0%  ← 이상값
```
- 2024년 6~8월 구간: ETF 승인 후 기관 플로우 구조 변화
- 기존 가격-파생 피처로는 새 시장 구조 적응 실패
- **근거**: 이것이 온체인 피처가 필요한 직접적 증거

#### 🟡 문제 3: GN 폭발 (Fold 5, Epoch 8: GN=30.7)
```
Fold 3: GN=5.5, 6.5 (반복)
Fold 5, E8: GN=30.8 (심각 — gradient clipping 한계 초과)
```
- 발생 후 다음 epoch에서 회복 (graceful) — 치명적이지 않음
- **해결**: Global clip 1.0 → 0.5 + per-layer clip 추가 검토

#### 🟡 문제 4: BC-RL 타임프레임 불일치
```
BC pretrain: 1h timeframe (2019-2022)
RL training: 30m timeframe (2023-2025)
```
- BC 학습된 피처 분포 ≠ RL 추론 피처 분포
- 특히 Koopman 고유값이 타임프레임 의존적 (1h: λ₁=0.9626 vs 30m: λ₁=0.93~0.97)
- **해결**: BC도 30m으로 재실행하거나, RL warm-up 기간 연장

#### 🟢 문제 5: Backtest 미실행
- 현재 훈련 EV (in-sample/validation): +0.038 ~ +0.089
- **OOS backtest 결과 없음** — 실제 성능 미확인
- 이전 세션 경험: in-sample WR ~47-50% → OOS WR 26.4% (대폭 하락)
- **즉시 실행 필요**: `python scripts/backtest_model_v2.py`

### 0-4. 현재 상태 판단

```
✅ 훈련 완료: agent_best.pt (EV=+0.0893 in-sample)
❌ Backtest 미실행: OOS 성능 미확인
❌ Critic Loss 발산 진행 중: 장기 훈련 불안정
⚠️  타임프레임 불일치: BC(1h) vs RL(30m)
⚠️  Fold 8 WR 38%: 2024 H2 시장 구조 변화 징후
```

> **결론**: 훈련 결과는 in-sample에서 긍정적이나, 이전 경험 상 OOS에서 대폭 하락 가능. 동시에 Fold 8의 WR 급락은 가격-파생 피처의 구조적 한계를 실증적으로 보여줌. **온체인 피처 확장이 필요한 이유가 이 로그에 있다.**

---

## 핵심 배경 — 왜 지금 온체인인가?

**현재 시스템의 구조적 문제**:
- OOS 2026 Q1 WR = **26.4%** (랜덤 수준)
- QLSTM이 가격-파생 피처(FR, OI, CVD, 기술지표) 만으로는 일반화 실패 확인
- 가격-파생 피처는 모든 시장 참여자가 동시 관찰 → 즉각 반영 → 알파 소멸

**온체인 데이터의 정보 우위**:
- 블록체인은 공개되지만 해석 난이도가 높음 → **분석 우위 존재**
- 장기보유자(LTH) 행동, 채굴자 매도, 거래소 자금 흐름은 가격에 **수일~수주 선행**
- MVRV Z-Score / SOPR 기반 규칙 전략: 역사적 WR **71%** (Glassnode 보고서, 2021-2024)
- **FR_LONG + EMA200 기준선(WR 36.8%)** 위에 온체인 필터 추가 시 38~45% 기대 가능

---

## PART I — 온체인 피처 목록 (19-dim)

### 카테고리 A — 밸류에이션 (Valuation)

| 피처 | 정의 | 시그널 해석 | 선행성 |
|------|------|------------|--------|
| `mvrv_z` | (시총 - 실현가치) / 표준편차 | >3: 과열, <0: 저점 공포 → 매수 | 1~4주 |
| `sopr` | 이동 코인의 실현가격 / 취득가격 | <1: 손실 항복 매도 → 저점 신호 | 2~7일 |
| `nupl` | (시총 - 실현가치) / 시총 | <0: 강한 손실 구간 → 매수 | 1~3주 |
| `realized_price_dev` | 현재가 / 실현가격 비율 편차 | 지지/저항 레벨 확인 | 즉시 |

### 카테고리 B — 네트워크 활동 (Network Activity)

| 피처 | 정의 | 시그널 해석 | 선행성 |
|------|------|------------|--------|
| `active_addr_z` | 24h 활성 주소 수 Z-Score | 급증 → 참여 확대, 랠리 선행 | 1~3일 |
| `nvt_signal` | 시총 / 거래량 (30일 MA) | >150: 과대평가, <50: 저평가 | 1~2주 |
| `new_addr_growth` | 신규 주소 증가율 7일 MA | 급증 → 신규 유입, 상승 에너지 | 3~7일 |

### 카테고리 C — 공급 구조 (Supply Dynamics)

| 피처 | 정의 | 시그널 해석 | 선행성 |
|------|------|------------|--------|
| `lth_supply_ratio` | 장기보유자(155일+) 보유량 비율 | 증가: 강세, 감소: 분배 국면 | 1~4주 |
| `sth_sopr` | 단기보유자(155일 이하) SOPR | <1: 패닉 판매 → 저점 신호 | 1~3일 |
| `exchange_netflow_z` | 거래소 순유입/유출 Z-Score | 유입(+): 매도압, 유출(-): 축적 | 1~5일 |
| `stablecoin_ratio` | USDT/USDC 거래소 잔고 / BTC 시총 | 증가: 대기 매수 자금 확충 | 3~7일 |

### 카테고리 D — 채굴자 행동 (Mining / Hash Ribbons)

| 피처 | 정의 | 시그널 해석 | 선행성 |
|------|------|------------|--------|
| `hash_ribbon_buy` | MA30 > MA60 of Hash Rate | 채굴자 항복 후 회복 → 매수 | 2~6주 |
| `miner_outflow_z` | 채굴자 → 거래소 전송량 Z-Score | 급증: 채굴자 매도 압력 | 2~5일 |
| `difficulty_ribbon` | 현재 해시 / 200일 MA | 압축: 채굴자 스트레스 → 저점 | 1~3주 |

### 카테고리 E — 파생상품 온체인 Context

| 피처 | 정의 | 시그널 해석 | 선행성 |
|------|------|------------|--------|
| `oi_lth_ratio` | 미결제약정 / LTH 공급량 | 고OI + LTH 분배: 과열 경고 | 1~3일 |
| `perp_basis_z` | 무기한 - 현물 기저 Z-Score | ±3σ: 청산 캐스케이드 예측 | 즉시~1일 |
| `funding_oi_product_z` | FR × OI 적산 Z-Score | 레버리지 과열 복합 지표 | 즉시 |
| `open_interest_usd_z` | USD 기준 OI Z-Score | 절대 레버리지 수준 측정 | 즉시 |

**총 19개 신규 피처 → 기존 13-dim 구조적 피처와 통합 시 32-dim 피처 세트**

---

## PART II — 무료 API 소스

| 소스 | 제공 지표 | 접근 방법 | 비용 |
|------|-----------|----------|------|
| **BGeometrics** | MVRV Z-Score, NUPL, SOPR, LTH/STH Supply | REST API / JSON | **무료** |
| **Blockchain.com** | Active Addresses, NVT, Hash Rate, Miner Outflow | `api.blockchain.info/charts/` | **무료** (API key 불필요) |
| **CryptoQuant (공개)** | Exchange Netflow, Stablecoin Ratio | CSV 다운로드 | **무료 tier** |
| **Bybit 확장** | OI 히스토리, Funding 히스토리, Liquidation | `/v5/market/open-interest` 등 | **무료** (현재 사용 중) |
| **Glassnode** | 전체 온체인 지표 | REST API | 유료 $39~$799/월 |

**추천 무료 조합**: BGeometrics + Blockchain.com + Bybit 확장 → **핵심 14개 피처 무료 커버 가능**

```python
# BGeometrics MVRV Z-Score
import requests

def fetch_mvrv_z():
    r = requests.get("https://bgeometrics.com/api/mvrv-z", timeout=10)
    return r.json()

# Blockchain.com Active Addresses (무료, API key 불필요)
def fetch_active_addresses(days=90):
    url = f"https://api.blockchain.info/charts/n-unique-addresses?timespan={days}days&format=json&sampled=true"
    r = requests.get(url, timeout=10)
    return r.json()["values"]  # [{"x": timestamp, "y": count}, ...]

# Bybit OI 히스토리 (기존 사용 중)
def fetch_oi_history(symbol="BTCUSDT", interval="1h", limit=200):
    url = "https://api.bybit.com/v5/market/open-interest"
    params = {"category": "linear", "symbol": symbol, "intervalTime": interval, "limit": limit}
    r = requests.get(url, params=params, timeout=10)
    return r.json()["result"]["list"]
```

---

## PART III — 아키텍처 통합 설계

### 기존 파이프라인

```
Raw OHLCV + FR + OI + CVD
    │
features_structural.py (13-dim)
    │
QLSTM / 규칙 기반 전략
```

### 온체인 통합 후

```
Raw OHLCV + FR + OI + CVD          온체인 데이터 (비동기 캐시, 1h 갱신)
    │                                    │
features_structural.py (13-dim)    features_onchain.py (19-dim)
    │                                    │
    └──────────────┬─────────────────────┘
                   │
          feature_fusion.py (32-dim)
          ├── 타임스탬프 정렬 (일별 온체인 → 1h 보간)
          ├── 결측값: forward-fill 최대 48h
          └── Z-Score 정규화 (rolling 90일 window)
                   │
         QLSTM / 규칙 기반 전략
```

### 핵심 설계 원칙

1. **비동기 캐싱**: 온체인은 별도 캐시 파일에 저장, 메인 루프는 읽기만
2. **결측값 Forward-fill**: API 장애 시 최대 48h 마지막 유효값 사용 (온체인은 느리게 변함)
3. **Lookahead Bias 차단**: 일별 데이터는 해당 날 00:00 UTC 이후 봉에만 적용
4. **모듈 분리**: `features_structural.py` 수정 금지 → 새 `features_onchain.py` 추가

---

## PART IV — 구현 Phase 로드맵

### Phase 0 — 즉시 착수 (1주 목표)

```
[ ] scripts/fetch_onchain.py — BGeometrics + Blockchain.com API 연동
[ ] 90일 히스토리 수집: data/onchain/
    - mvrv_z_90d.csv
    - sopr_90d.csv
    - active_addr_90d.csv
    - exchange_netflow_90d.csv
[ ] Granger causality test: 온체인 → BTC 1h 수익률 선행성 검증
    - lag=[1,2,4,8,24]h → p<0.05 통과 피처만 Phase 1에 포함
[ ] Cross-correlation plot: 온체인 vs 가격 선행 lag 시각화
```

**담당**: Marvin (데이터 수집) + Radi (선행성 검증) + Viktor (Granger 검정 수학 검증)

```python
# Granger causality 즉시 실행 예시
from statsmodels.tsa.stattools import grangercausalitytests
import pandas as pd

mvrv = pd.read_csv("data/onchain/mvrv_z_90d.csv", parse_dates=["timestamp"])
btc_1h = pd.read_csv("data/btc_1h.csv", parse_dates=["timestamp"])

merged = mvrv.merge(btc_1h[["timestamp","close"]], on="timestamp")
merged["ret"] = merged["close"].pct_change()

results = grangercausalitytests(merged[["ret","mvrv_z"]].dropna(), maxlag=24)
# lag별 p-value 보고 → 유의한 선행 lag 확인
```

### Phase 1 — 구현 (2~3주)

```
[ ] src/data/onchain_fetcher.py: 비동기 API 클라이언트 (asyncio + aiohttp)
[ ] src/models/features_onchain.py: 19-dim 온체인 피처 빌더
[ ] src/models/feature_fusion.py: 타임스탬프 정렬 + 피처 병합
[ ] scripts/backtest_onchain.py: 온체인 필터 포함 규칙 기반 백테스트
    - A: FR_LONG + EMA200 (기준선, WR 36.8%)
    - B: A + mvrv_z < 0 필터
    - C: A + exchange_netflow_z < -1 (축적 신호)
    - D: A + sopr < 1.0 (항복 매도 확인)
    - E: B + C + D (Full 온체인 조합)
```

**담당**: Felix (피처 파이프라인) + Casandra (백테스트 통합) + Schwertz (신호 조합 설계)

### Phase 2 — ML 통합 및 검증 (4~6주)

```
[ ] QLSTM BC 사전학습: 32-dim 피처 세트 (구조적 13 + 온체인 19)
[ ] RL 파인튜닝: 온체인 피처 포함 에이전트 학습
[ ] OOS 검증: 2025 Q1~Q4 구간 A/B 비교
[ ] 피처 중요도 분석: Granger 통과 피처 vs RL에서 실제 사용 비교
```

**담당**: Darvin (훈련 파이프라인) + Viktor (OOS 검증)

### Phase 3 — 실시간 TUI 통합 (2개월+)

```
[ ] TUI 온체인 패널 추가: 실시간 MVRV-Z, SOPR, Exchange Netflow 표시
[ ] 매 1시간 백그라운드 fetch + 캐시 자동 갱신
[ ] 알림: MVRV-Z > 3.0 → "⚠ 과열 경고" / SOPR < 1.0 → "✅ 항복 저점"
```

---

## PART V — 기대 효과 및 리스크

### 시나리오별 기대 성능

| 시나리오 | WR 예상 | ROI 예상 | 근거 |
|----------|---------|---------|------|
| 온체인 필터만 (규칙 기반) | **38~45%** | +150~250% | MVRV/SOPR 역사적 효과 |
| QLSTM + 온체인 피처 | 32~40% | +80~150% | 피처 정보 증가 효과 |
| 최고 시나리오 | **48%+** | +400%+ | 사이클 정점/바닥 예측 성공 |
| 실패 시나리오 | <30% | 손실 | ETF 이후 온체인 신호 구조 변화 |

### 리스크 및 완화

| 리스크 | 심각도 | 완화 방법 |
|--------|--------|-----------|
| API 장애 (데이터 미수신) | 중 | Forward-fill 48h, 기준선으로 폴백 |
| Lookahead Bias | **높음** | Viktor 타임스탬프 정렬 검증 필수 |
| 과최적화 (피처 과다) | **높음** | Phase 0 Granger p<0.05 통과 피처만 선택 |
| ETF 이후 온체인 구조 변화 | 중 | Fold별 피처 중요도 모니터링 |
| Glassnode 유료 압박 | 낮음 | 무료 조합으로 충분 (검증 후 결정) |

---

## PART VI — Grand Council 결정 사항

### CEO (Demis) 결정

1. **Phase 0 즉시 착수 승인**: Marvin + Radi 1주 공수 투입 승인
2. **우선순위**: 온체인 Phase 0 착수 **AND** Gate 0 버그 수정 병행 (독립 팀)
3. **Glassnode 예산**: Phase 1 완료 후 무료 조합 성능 확인 → 그 때 결정
4. **판단 기준**: 2주 내 Granger 검정 결과 → p<0.05 피처 3개 이상이면 Phase 1 진입

### CTO (Viktor) 분석 과제

1. **Granger Causality 수행**: BGeometrics CSV로 BTC 1h 수익률 Granger 검정, lag 1~24h p-value 보고
2. **Lookahead Bias 감사**: 일별 온체인 데이터 → 1h 보간 시 미래 참조 없음을 수학적으로 검증
3. **기존 파이프라인 호환성**: 32-dim 피처가 현재 BC 사전학습 코드와 충돌 없는지 확인

### Alpha Lead (Radi) 분석 과제

1. **규칙 기반 빠른 검증**:
   ```bash
   python scripts/backtest_behavioral.py --signals fr --long-only --trend-ema 200 \
     --onchain-filter mvrv_z --mvrv-thr -0.5 --days 90
   ```
2. **SOPR 저점 매수 조합**: SOPR < 1.0 + FR_LONG 조합 WR 검증 (2023-2026 전구간)

### Quant Lead (Marvin) 구현 과제

1. `scripts/fetch_onchain.py` — BGeometrics + Blockchain.com API 연동 및 CSV 저장
2. 90일 히스토리 `data/onchain/` 폴더 생성 및 초기 데이터 수집

---

## PART VII — 참고 자료

**관련 논문**:
- Alessandretti et al. (2018) "Anticipating cryptocurrency prices using machine learning" — Sci. Reports
- Liu & Tsyvinski (2021) "Risks and Returns of Cryptocurrency" — Review of Financial Studies
- Cong et al. (2023) "Token-Based Platform Finance" — Journal of Financial Economics
- Delphi Digital (2024) "On-chain Metrics as Alpha Signals" — 연구 보고서

**WorldQuant BRAIN / IQC 2026**:
- 온체인 피처 기반 알파는 제출 경쟁에서 아직 희소 → **차별화 포인트**
- 논문 제목 후보: *"On-chain Enhanced Quantum LSTM for Cryptocurrency Alpha Generation"*

---

> **결론**: 온체인 피처 확장은 현재 가격-파생 피처의 구조적 한계를 돌파할 수 있는 **가장 현실적이고 즉각적인 방법**이다. 무료 API로 핵심 피처를 커버할 수 있으며, 규칙 기반 필터 단계에서도 즉각적인 WR 개선이 기대된다. Grand Council은 **Phase 0 즉시 착수를 승인**하고, 2주 내 Granger 검정 결과를 바탕으로 Phase 1 진입을 결정한다.

---

*Agenda 작성: 2026-03-23*
*제안: CEO (Demis) — Grand Council 최우선 안건*
*다음 회의: Phase 0 완료 후 Granger 검정 결과 공유 시*
