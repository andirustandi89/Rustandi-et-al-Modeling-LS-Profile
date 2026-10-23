# Membuat Figure 2 dan Supplementary Table S1 dari output eksperimen.
# Komentar dibuat ringkas agar alur eksperimen mudah diikuti saat dijalankan lokal.

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CROSS = ROOT / 'OUTPUT' / '05_CROSS_DATASET'
FIG = ROOT / 'OUTPUT' / '10_FIGURES'
SUP = ROOT / 'SUPPLEMENTARY'
FIG.mkdir(parents=True, exist_ok=True)
SUP.mkdir(parents=True, exist_ok=True)

rank_file = CROSS / '10_average_rank.csv'
if rank_file.is_file():
    raw = pd.read_csv(rank_file)
    raw = raw[raw['source'].astype(str).eq('resample_mean')].copy()
    metric_order = ['initial_cost', 'final_cost', 'dbi', 'si', 'iterations']
    method_order = ['CARE-KM', 'Cao', 'Bai', 'K-Modes++', 'Random', 'Huang', 'HR-2023']
    plot_df = raw[raw['metric'].isin(metric_order)].pivot(
        index='method', columns='metric', values='average_rank'
    )
    plot_df = plot_df.reindex([m for m in method_order if m in plot_df.index])
    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = list(range(len(metric_order)))
        for method, row in plot_df.iterrows():
            ax.plot(x, [row[m] for m in metric_order], marker='o', label=method)
        ax.set_xticks(x, ['Initial cost', 'Final cost', 'DBI', 'SI', 'Iterations'])
        ax.set_ylabel('Average rank (lower is better)')
        ax.set_title('Average ranks across ten categorical benchmark datasets')
        ax.grid(axis='y', alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG / 'figure_2_average_ranks.png', dpi=220)
        plt.close(fig)

src = ROOT / 'OUTPUT' / '06_FSLSM_PROFILES' / '13_table_05b_fslsm_strength_profiles.csv'
if not src.is_file():
    src = ROOT / 'OUTPUT' / '06_FSLSM_PROFILES' / '08_cluster_to_fslsm_mapping.csv'
if src.is_file():
    pd.read_csv(src).to_csv(SUP / 'Table_S1_Full_FSLSM_Profiles.csv', index=False)

print('Artefak figure dan supplementary selesai dibuat.')
