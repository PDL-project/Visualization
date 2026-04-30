import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

def find_korean_font():
    candidates = ['Malgun Gothic', 'NanumGothic', 'AppleGothic', 'Gulim', 'Dotum']
    available = {f.name for f in fm.fontManager.ttflist}
    for c in candidates:
        if c in available:
            return c
    return None

korean_font = find_korean_font()
if korean_font:
    plt.rcParams['font.family'] = korean_font
plt.rcParams['axes.unicode_minus'] = False

CONFIG_DENOM = {1: 9, 2: 6, 3: 5, 4: 6}

TASK_ALIAS = {
    '2_open_all_cabinets': '2_close_all_cabinets',
}

def is_valid_task(task):
    """All canonical tasks start with 1_/2_/3_/4_. Rejects header strings like 'Task', 'mode'."""
    return len(task) > 2 and task[0].isdigit() and task[1] == '_'

def normalize(task):
    task = task.strip()
    return TASK_ALIAS.get(task, task)

# Read file
csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '메인 실험.CSV')
with open(csv_path, 'r', encoding='cp949') as f:
    raw_lines = f.readlines()

def parse_line(line):
    return next(csv.reader([line]))

# Section detection
# Known baseline names and their data format:
#   fmt_A: LaMMA-P, H-AIM, Ours  → task=col[1], sr=col[4] (TRUE/FALSE)
#   fmt_B: SmartLLM, PDL_central → task=col[2], finished=col[9] (0/1)
#   fmt_C: COELA                 → talk_iter=col[1], task=col[3], finished=col[10] (0/1)

FMT_A = {'LaMMA-P', 'H-AIM', 'Ours'}
FMT_B = {'SmartLLM', 'PDL_central'}
FMT_C = {'COELA'}

def detect_baseline(col0):
    """Return (canonical_baseline_name, raw_col0_belonged_to_it)."""
    s = col0.strip()
    # header rows: 'method', 'SmartLLM', 'PDL_central', 'COELA', or 'method,...'
    for name in list(FMT_A) + list(FMT_B) + list(FMT_C):
        if s == name or s == f'베이스라인명: {name}':
            return name
    return None

# Build task→success dicts
task_dicts = {
    'LaMMA-P':    {},
    'H-AIM':      {},
    'Ours':       {},
    'SmartLLM':   {},
    'PDL_central':{},
    'COELA_1':    {},
    'COELA_2':    {},
}

def get_config(task):
    """Return config number from task prefix (1_→1, 2_→2, ...), or None."""
    c = task[0]
    return int(c) if c in '1234' else None

def record(d, task, val):
    """Record val for task; keep max across duplicates. All valid tasks accepted."""
    t = normalize(task)
    d[t] = max(d.get(t, 0), val)

current_section = None   # name like 'LaMMA-P'

for lineno, raw in enumerate(raw_lines, start=1):
    cols = parse_line(raw)

    bl = detect_baseline(cols[0]) if cols else None
    if bl is not None:
        current_section = bl

    if current_section is None:
        continue

    # Format A: LaMMA-P / H-AIM / Ours
    if current_section in FMT_A:
        if len(cols) < 5:
            continue
        task = cols[1].strip()
        if not is_valid_task(task):   # skip blanks and header strings like 'Task'
            continue
        val = 1 if cols[4].strip().upper() == 'TRUE' else 0
        record(task_dicts[current_section], task, val)

    # Format B: SmartLLM / PDL_central
    elif current_section in FMT_B:
        if len(cols) < 10:
            continue
        task = cols[2].strip()
        if not is_valid_task(task):   # skip blanks and header strings like 'task_folder', 'mode'
            continue
        try:
            val = int(float(cols[9].strip()))
        except ValueError:
            continue
        record(task_dicts[current_section], task, val)

    # Format C: COELA  (DMRS-1D = talk_iter 1, DMRS-2D = talk_iter 2)
    elif current_section in FMT_C:
        if len(cols) < 11:
            continue
        task = cols[3].strip()
        if not is_valid_task(task):   # skip blanks and header strings like 'task_folder'
            continue
        try:
            val = int(float(cols[10].strip()))
            talk_iter = int(float(cols[1].strip()))
        except ValueError:
            continue
        key = f'COELA_{talk_iter}'
        if key in task_dicts:
            record(task_dicts[key], task, val)

# Compute SR: successes / fixed denominator
def compute_sr(task_dict):
    sr = {}
    for cfg, denom in CONFIG_DENOM.items():
        successes = sum(v for t, v in task_dict.items()
                        if get_config(t) == cfg and v == 1)
        sr[cfg] = successes / denom
    return sr

sr_all = {bl: compute_sr(task_dicts[bl]) for bl in task_dicts}

# Validation report
print("=" * 72)
print(f"Denominators: Config1={CONFIG_DENOM[1]}, Config2={CONFIG_DENOM[2]}, "
      f"Config3={CONFIG_DENOM[3]}, Config4={CONFIG_DENOM[4]}  (total 26)")
print("=" * 72)
for bl, sr in sr_all.items():
    print(f"{bl}:")
    for cfg in CONFIG_DENOM:
        tasks_run = {t: v for t, v in task_dicts[bl].items() if get_config(t) == cfg}
        n_success = sum(v for v in tasks_run.values() if v == 1)
        pct = sr[cfg] * 100
        print(f"  Config{cfg}: {pct:5.1f}%  "
              f"({n_success} success / {len(tasks_run)} run / {CONFIG_DENOM[cfg]} denom)  "
              f"tasks={sorted(tasks_run.keys())}")
print("=" * 72)

# Plot
baseline_order = ['SmartLLM', 'COELA_2', 'COELA_1', 'LaMMA-P', 'H-AIM', 'PDL_central', 'Ours']
display_labels = {
    'LaMMA-P':     'LaMMA-P',
    'H-AIM':       'H-AIM',
    'Ours':        'HiF-P(Ours)',
    'SmartLLM':    'SMART-LLM',
    'PDL_central': 'HiF-P(Central)',
    'COELA_1':     'DMRS-1D',
    'COELA_2':     'DMRS-2D',
}

colors  = ['#708090', '#87A96B', '#2F5D62', '#E67E7E', '#967BB6', '#7B9ACC', '#003BFF']
markers = ['o', 's', '^', 'D', 'v', 'P', '*']
linestyles = {
    'SmartLLM':    '--',
    'COELA_2':     '-',
    'COELA_1':     '--',
    'LaMMA-P':     '--',
    'H-AIM':       '-',
    'PDL_central': '-',
    'Ours':        '-',
}
x_labels = ['Config1', 'Config2', 'Config3', 'Config4']

fig, ax = plt.subplots(figsize=(9, 5))

for idx, bl in enumerate(baseline_order):
    sr = sr_all[bl]
    y_vals = [sr[cfg] for cfg in [1, 2, 3, 4]]
    ax.plot(
        range(4), y_vals,
        label=display_labels[bl],
        color=colors[idx],
        marker=markers[idx],
        linewidth=1.8,
        markersize=7,
        linestyle=linestyles[bl],
    )

ax.set_xticks(range(4))
ax.set_xticklabels(x_labels, fontsize=11)
ax.set_xlabel('Config', fontsize=12)
ax.set_ylabel('Success Rate (SR)', fontsize=12)
ax.set_title('Success Rate by Config and Baseline', fontsize=13)
ax.set_ylim(-0.05, 1.05)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
ax.legend(loc='upper right', fontsize=9, framealpha=0.85)
ax.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'main_experiment_SR.png')
plt.savefig(out_path, dpi=150)
print(f"Saved → {out_path}")
