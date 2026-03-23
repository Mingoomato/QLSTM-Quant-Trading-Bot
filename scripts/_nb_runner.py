# == CELL ==
# -- 0. Imports & Configuration -----------------------------------------------
import sys, warnings, time as _time, json as _json, urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker

warnings.filterwarnings('ignore')

# -- paths
REPO_ROOT = Path().resolve()
while not (REPO_ROOT / 'src').exists() and REPO_ROOT != REPO_ROOT.parent:
    REPO_ROOT = REPO_ROOT.parent
DATA_DIR = REPO_ROOT / 'data'
OUT_DIR  = REPO_ROOT / 'reports'
OUT_DIR.mkdir(parents=True, exist_ok=True)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# -- study parameters
SYMBOL         = 'BTCUSDT'
THRESHOLD      = 2.0
THRESHOLD_HIGH = 3.0
WINDOW         = 60       # +/- 60 minutes
MIN_EVENT_GAP  = 3        # minimum 3h gap between events (1h index)
Z_WINDOW       = 50
BINANCE_BASE   = 'https://api.binance.com'

print(f'Repo root : {REPO_ROOT}')
print(f'Window    : +/-{WINDOW} minutes  ({WINDOW*2+1} 1m bars per event)')
print(f'Threshold : liq_long_z > {THRESHOLD} sigma')

# == CELL ==
# -- 1. Load 1h Structural Data (event detection) ----------------------------
t0 = _time.time()

def _load_ohlcv_1h(data_dir):
    files = sorted(data_dir.glob('training_BTCUSDT_1h_*.csv'))
    if not files:
        raise FileNotFoundError(f'No 1h OHLCV in {data_dir}')
    frames = [pd.read_csv(f, usecols=['ts','open','high','low','close','volume'],
                           encoding='utf-8') for f in files]
    df = pd.concat(frames, ignore_index=True)
    df['ts'] = pd.to_datetime(df['ts'], utc=True)
    return df.drop_duplicates('ts').sort_values('ts').set_index('ts')

def _load_aux(data_dir, glob_pat):
    files = sorted(data_dir.glob(glob_pat))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f, encoding='utf-8') for f in files],
                   ignore_index=True)
    # Handle both 'ts' (string datetime) and 'ts_ms' (epoch ms) column names
    if 'ts' in df.columns:
        df['ts'] = pd.to_datetime(df['ts'], utc=True)
    elif 'ts_ms' in df.columns:
        df['ts'] = pd.to_datetime(df['ts_ms'].astype('int64'), unit='ms', utc=True)
    else:
        return None
    return df.drop_duplicates('ts').sort_values('ts').set_index('ts')

master   = _load_ohlcv_1h(DATA_DIR)
taker_df = _load_aux(DATA_DIR, 'binance_taker_BTCUSDT_1h_*.csv')
oi_df    = _load_aux(DATA_DIR, 'oi_BTCUSDT_1h_*.csv')

for aux in [taker_df, oi_df]:
    if aux is not None:
        cols = [c for c in aux.columns if c not in master.columns]
        master = master.join(aux[cols], how='left')

print(f'1h master : {len(master):,} bars  '
      f'({master.index.min().date()} to {master.index.max().date()})')
print(f'Columns   : {master.columns.tolist()}')
print(f'Loaded in {_time.time()-t0:.1f}s')

# == CELL ==
# -- 2. Compute liq_long_z (structural liquidation proxy) --------------------
def _rz(s, w=Z_WINDOW):
    """Rolling z-score; NaN until window fills."""
    mu  = s.rolling(w, min_periods=w).mean()
    std = s.rolling(w, min_periods=w).std(ddof=1).replace(0, np.nan)
    return (s - mu) / std

# Lower wick * volume = proxy for long-stop-sweep liquidation volume
lower_wick = (master[['open','close']].min(axis=1) - master['low']).clip(lower=0)
liq_raw    = lower_wick * master['volume']
liq_long_z = _rz(liq_raw)
liq_long_z.name = 'liq_long_z'

print(f'liq_long_z non-NaN: {liq_long_z.notna().sum():,} bars')
print(f'Mean={liq_long_z.mean():.3f}  Std={liq_long_z.std():.3f}')
print(f'Bars > {THRESHOLD}s      : {(liq_long_z > THRESHOLD).sum():,}')
print(f'Bars > {THRESHOLD_HIGH}s : {(liq_long_z > THRESHOLD_HIGH).sum():,}')

# == CELL ==
# -- 3. Identify Events & Fetch 1m OHLCV from Binance Public API -------------
def identify_events(series, threshold, min_gap_h=MIN_EVENT_GAP):
    """De-clustered event timestamps where series > threshold."""
    cands = series[series > threshold].index
    if len(cands) == 0:
        return pd.DatetimeIndex([])
    evs = [cands[0]]
    for ts in cands[1:]:
        if (ts - evs[-1]).total_seconds() / 3600 >= min_gap_h:
            evs.append(ts)
    return pd.DatetimeIndex(evs)


def fetch_binance_1m(symbol, start_ms, end_ms, limit=1000):
    """Paginated 1m klines from Binance; no auth required."""
    all_bars, cur = [], start_ms
    while cur < end_ms:
        url = (f'{BINANCE_BASE}/api/v3/klines'
               f'?symbol={symbol}&interval=1m'
               f'&startTime={cur}&endTime={end_ms}&limit={limit}')
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = _json.loads(resp.read())
        if not data:
            break
        all_bars.extend(data)
        last_ts = data[-1][0]
        if last_ts >= end_ms or len(data) < limit:
            break
        cur = last_ts + 60_000
        _time.sleep(0.05)
    if not all_bars:
        return pd.DataFrame()
    df = pd.DataFrame(all_bars, columns=[
        'ts_open','open','high','low','close','volume',
        'ts_close','qvol','ntrades','tbase','tquote','ign'
    ])
    df['ts'] = pd.to_datetime(df['ts_open'].astype('int64'), unit='ms', utc=True)
    for c in ['open','high','low','close','volume']:
        df[c] = df[c].astype(float)
    return (df[['ts','open','high','low','close','volume']]
            .drop_duplicates('ts').sort_values('ts').set_index('ts'))


events_norm    = identify_events(liq_long_z, THRESHOLD)
events_extreme = identify_events(liq_long_z, THRESHOLD_HIGH)
print(f'Events > {THRESHOLD}s   : {len(events_norm):,}')
print(f'Events > {THRESHOLD_HIGH}s : {len(events_extreme):,}')

# Build batched API ranges (merge nearby windows to minimise calls)
FETCH_MARGIN = WINDOW + 5
_ms = lambda ts: int(ts.timestamp() * 1000)

ranges = [(ev - pd.Timedelta(minutes=FETCH_MARGIN),
           ev + pd.Timedelta(minutes=FETCH_MARGIN))
          for ev in events_norm]
ranges.sort(key=lambda x: x[0])
merged_r = []
for s, e in ranges:
    if merged_r and s <= merged_r[-1][1] + pd.Timedelta(minutes=5):
        merged_r[-1] = (merged_r[-1][0], max(merged_r[-1][1], e))
    else:
        merged_r.append([s, e])

print(f'Batched API calls : {len(merged_r)}')
frames_1m = []
for i, (s, e) in enumerate(merged_r):
    try:
        chunk = fetch_binance_1m(SYMBOL, _ms(s), _ms(e))
        if not chunk.empty:
            frames_1m.append(chunk)
    except Exception as exc:
        print(f'  [warn] range {i}: {exc}')
    if (i + 1) % 20 == 0:
        print(f'  {i+1}/{len(merged_r)} done')

price_1m = pd.DataFrame()
if frames_1m:
    price_1m = pd.concat(frames_1m)
    price_1m = price_1m[~price_1m.index.duplicated()].sort_index()
    print(f'1m bars fetched   : {len(price_1m):,}')
    print(f'Range             : {price_1m.index.min()} to {price_1m.index.max()}')
else:
    print('WARNING: No 1m data fetched — check network / Binance API.')

# == CELL ==
# -- 4. Extract Normalized Price Windows (+/-60 1m bars) ----------------------
OFFSETS = np.arange(-WINDOW, WINDOW + 1)   # -60 ... +60  (121 values)

def extract_windows(events, price_df, window=WINDOW):
    """
    For each 1h event timestamp, snap to nearest 1m bar, extract
    +/- window 1m closes, normalise to % return vs t=0.
    Returns ndarray (n_events, 2*window+1); NaN where bars unavailable.
    """
    if price_df.empty:
        return np.full((len(events), 2*window+1), np.nan)
    close  = price_df['close'].values
    ts_map = {ts: i for i, ts in enumerate(price_df.index)}
    rows   = []
    offsets = np.arange(-window, window+1)
    for ev in events:
        ev_fl = ev.floor('1min')
        if ev_fl not in ts_map:
            rows.append(np.full(2*window+1, np.nan))
            continue
        c   = ts_map[ev_fl]
        idx = c + offsets
        ok  = (idx >= 0) & (idx < len(close))
        row = np.full(2*window+1, np.nan)
        row[ok] = close[idx[ok]]
        p0 = row[window]
        if np.isnan(p0) or p0 == 0:
            rows.append(np.full(2*window+1, np.nan))
            continue
        rows.append((row / p0 - 1.0) * 100.0)
    return np.array(rows)


windows_norm    = extract_windows(events_norm,    price_1m)
windows_extreme = extract_windows(events_extreme, price_1m)

# Drop rows with >50% NaN (Binance history may not reach far back)
def _clean(mat):
    return mat[np.mean(np.isnan(mat), axis=1) < 0.5]

windows_norm    = _clean(windows_norm)
windows_extreme = _clean(windows_extreme)
print(f'Usable windows >{THRESHOLD}s   : {windows_norm.shape[0]}')
print(f'Usable windows >{THRESHOLD_HIGH}s : {windows_extreme.shape[0]}')

def _stats(mat):
    mean = np.nanmean(mat, axis=0)
    std  = np.nanstd(mat,  axis=0)
    n    = np.sum(~np.isnan(mat), axis=0)
    se   = std / np.sqrt(np.maximum(n, 1))
    return mean, std, se

mean_norm, std_norm, se_norm = _stats(windows_norm)
mean_ext,  std_ext,  se_ext  = _stats(windows_extreme)

# == CELL ==
# -- 5. Main Event Study Chart (6 panels) ------------------------------------
DARK_BG = '#0a0a0a'
ACCENT  = '#00d4aa'
RED     = '#ff4757'
ORANGE  = '#ffa502'
GRID    = '#1e1e1e'
TEXT    = '#e0e0e0'

fig = plt.figure(figsize=(18, 16), facecolor=DARK_BG)
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)
ax_main = fig.add_subplot(gs[0, :])
ax_heat = fig.add_subplot(gs[1, 0])
ax_dist = fig.add_subplot(gs[1, 1])
ax_wr   = fig.add_subplot(gs[2, 0])
ax_info = fig.add_subplot(gs[2, 1])

for ax in [ax_main, ax_heat, ax_dist, ax_wr, ax_info]:
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.6)

t = OFFSETS

# Panel 1: mean price trajectory +/- 2 SE band
ax = ax_main
if windows_norm.shape[0] > 0:
    ax.fill_between(t, mean_norm - 2*se_norm, mean_norm + 2*se_norm,
                    alpha=0.15, color=ACCENT)
    ax.plot(t, mean_norm, color=ACCENT, lw=2.0,
            label=f'Mean +/-2SE  N={windows_norm.shape[0]}  (z>{THRESHOLD})')
if windows_extreme.shape[0] > 0:
    ax.fill_between(t, mean_ext - 2*se_ext, mean_ext + 2*se_ext,
                    alpha=0.18, color=RED)
    ax.plot(t, mean_ext, color=RED, lw=2.5, ls='--',
            label=f'Extreme +/-2SE  N={windows_extreme.shape[0]}  (z>{THRESHOLD_HIGH})')
ax.axvline(0, color='white', lw=1.0, ls=':')
ax.axhline(0, color=GRID, lw=0.8)
ax.axvspan(-WINDOW, 0, alpha=0.04, color='red')
ax.axvspan(0, WINDOW, alpha=0.04, color='green')
ax.set_xlabel('Offset from event (minutes)', color=TEXT, fontsize=10)
ax.set_ylabel('Avg price return vs t=0 (%)', color=TEXT, fontsize=10)
ax.set_title(
    f'BTCUSDT: Average Price Trajectory +/-{WINDOW} Minutes Around liq_long_z Spike\n'
    f'(1m bars | threshold={THRESHOLD}s | {len(events_norm)} raw events | '
    f'{windows_norm.shape[0]} usable)',
    color=TEXT, fontsize=12, fontweight='bold'
)
ax.legend(framealpha=0.2, labelcolor=TEXT, fontsize=9)
ax.set_xlim(-WINDOW, WINDOW)

# Panel 2: heatmap of all paths
ax = ax_heat
if windows_norm.shape[0] > 0:
    clip = np.nanpercentile(np.abs(windows_norm), 99)
    im = ax.imshow(np.clip(windows_norm, -clip, clip),
                   aspect='auto', cmap='RdYlGn',
                   extent=[-WINDOW, WINDOW, 0, windows_norm.shape[0]],
                   vmin=-clip, vmax=clip)
    plt.colorbar(im, ax=ax, label='% return').ax.tick_params(colors=TEXT)
    ax.axvline(0, color='white', lw=0.8, ls=':')
ax.set_title('All Event Windows (heatmap)', color=TEXT, fontsize=10)
ax.set_xlabel('Offset (minutes)', color=TEXT)
ax.set_ylabel('Event index', color=TEXT)

# Panel 3: post-event return distributions
ax = ax_dist
for offset, color, label in [(10,'#ffd700','+10min'),(30,ORANGE,'+30min'),(60,RED,'+60min')]:
    if windows_norm.shape[0] > 0:
        vals = windows_norm[:, WINDOW + offset]
        clean = vals[~np.isnan(vals)]
        if len(clean) > 0:
            ax.hist(clean, bins=30, alpha=0.5, color=color, label=label, density=True)
ax.axvline(0, color='white', lw=1.0, ls=':')
ax.set_title('Post-Event Return Distribution', color=TEXT, fontsize=10)
ax.set_xlabel('% return vs t=0', color=TEXT)
ax.set_ylabel('Density', color=TEXT)
ax.legend(framealpha=0.2, labelcolor=TEXT, fontsize=8)

# Panel 4: % events where price falls (SHORT signal quality curve)
ax = ax_wr
if windows_norm.shape[0] > 0:
    post = windows_norm[:, WINDOW:]
    wr   = np.nanmean(post < 0, axis=0) * 100
    xax  = np.arange(0, WINDOW + 1)
    ax.plot(xax, wr, color=ACCENT, lw=1.8)
    ax.axhline(50, color='white', lw=0.8, ls='--', alpha=0.4)
    ax.fill_between(xax, wr, 50, where=(wr > 50), alpha=0.20, color=RED,
                    label='Falls more often')
    ax.fill_between(xax, wr, 50, where=(wr <= 50), alpha=0.20, color=ACCENT,
                    label='Rises more often')
ax.set_title('% Events Where Price Falls Post-Event\n(SHORT signal quality)', color=TEXT, fontsize=10)
ax.set_xlabel('Minutes after event', color=TEXT)
ax.set_ylabel('% events price falls', color=TEXT)
ax.set_ylim(0, 100)
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:.0f}%'))
ax.legend(framealpha=0.2, labelcolor=TEXT, fontsize=8)

# Panel 5: stats table
ax = ax_info
ax.axis('off')
def _fmt(offset):
    if windows_norm.shape[0] == 0:
        return 'N/A', 'N/A', 'N/A'
    v = windows_norm[:, WINDOW + offset]
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return 'N/A', 'N/A', 'N/A'
    return (f'{np.mean(v):+.3f}%', f'{np.median(v):+.3f}%', f'{np.mean(v<0)*100:.1f}%')

table_data = [_fmt(off) for off in [5, 10, 20, 30, 45, 60]]
table_rows = [[lbl, *row] for lbl, row in
              zip(['+5m','+10m','+20m','+30m','+45m','+60m'], table_data)]
tbl = ax.table(cellText=table_rows,
               colLabels=['Horizon','Mean ret','Median ret','% fell'],
               cellLoc='center', loc='center', bbox=[0, 0.1, 1, 0.8])
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
for (r, c), cell in tbl.get_celld().items():
    cell.set_facecolor('#1a1a2e' if r == 0 else DARK_BG)
    cell.set_text_props(color=TEXT)
    cell.set_edgecolor(GRID)
ax.set_title('Post-Event Statistics (1m resolution)', color=TEXT, fontsize=10)

fig.suptitle(
    'Jose / Risk Manager  --  Liquidation Cascade Event Study  (+/-60 min, 1m bars)',
    y=0.997, color=TEXT, fontsize=13, fontweight='bold'
)
out_path = OUT_DIR / 'liq_impact_analysis_1m.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
plt.close(fig)
print(f'Chart saved -> {out_path}')

# == CELL ==
# -- 6. Statistical Summary Table ---------------------------------------------
print('=' * 72)
print('LIQUIDATION CASCADE IMPACT -- MINUTE-LEVEL STATISTICAL SUMMARY')
print('=' * 72)
print(f'Dataset  : BTCUSDT 1h  '
      f'{master.index.min().date()} to {master.index.max().date()}')
print(f'Price    : 1m bars from Binance public REST API')
print(f'Window   : +/-{WINDOW} minutes  ({WINDOW*2+1} 1m bars per event)')
print(f'Threshold: liq_long_z > {THRESHOLD} sigma')
print(f'Events   : {len(events_norm):,} raw  ->  '
      f'{windows_norm.shape[0]:,} usable (>=50% data coverage)')
print()
hdr = f"{'Horizon':>8}  {'Mean%':>9}  {'Median%':>9}  {'Std%':>8}  {'P(fall)':>7}  {'t-stat':>7}"
print(hdr)
print('-' * len(hdr))
for offset in [1, 5, 10, 15, 20, 30, 45, 60]:
    if windows_norm.shape[0] == 0:
        break
    vals  = windows_norm[:, WINDOW + offset]
    clean = vals[~np.isnan(vals)]
    if len(clean) < 3:
        continue
    m  = np.mean(clean)
    md = np.median(clean)
    s  = np.std(clean, ddof=1)
    p  = np.mean(clean < 0) * 100
    t  = m / (s / np.sqrt(len(clean)))
    print(f'+{offset:>6}m  {m:>+9.4f}  {md:>+9.4f}  {s:>8.4f}  '
          f'{p:>6.1f}%  {t:>+7.3f}')

# == CELL ==
# -- 7. Alpha Half-Life Estimation (minute-level) ----------------------------
from scipy.optimize import curve_fit

def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)

if windows_norm.shape[0] >= 5:
    post     = windows_norm[:, WINDOW:]
    abs_mean = np.abs(np.nanmean(post, axis=0))
    t_arr    = np.arange(0, WINDOW + 1, dtype=float)
    mask     = (t_arr >= 1) & (abs_mean > 0)
    try:
        popt, _ = curve_fit(exp_decay, t_arr[mask], abs_mean[mask],
                            p0=[abs_mean[mask][0], 15.0], maxfev=5000)
        A_fit, tau_fit = popt
        half_life = tau_fit * np.log(2)
        print(f'Exponential decay fit   A={A_fit:.4f}%  tau={tau_fit:.1f}min')
        print(f'Alpha half-life         = {half_life:.1f} minutes')
        if half_life < 5:
            msg = '< 5min -- must execute within the same 1m candle'
        elif half_life < 20:
            msg = f'~{half_life:.0f}min -- viable for 5m entries'
        else:
            msg = f'> 20min -- viable for 15m-1h entries'
        print(f'Risk implication        : {msg}')
    except RuntimeError as e:
        print(f'Curve fit failed: {e}')
else:
    print('Insufficient events for half-life estimation.')

