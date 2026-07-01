#!/usr/bin/env python3
"""Generate manuscript and supplementary figures with publication-style layouts."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.lines import Line2D

base = Path(__file__).resolve().parents[1]
res = base / 'results'
figdir = base / 'figures'
figdir.mkdir(exist_ok=True)

mpl.rcParams.update({
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8.2,
    'ytick.labelsize': 8.6,
    'legend.fontsize': 8,
    'axes.linewidth': 0.8,
})

COL = {
    'navy':'#1f4e79','blue':'#4c78a8','teal':'#3b8b8c','green':'#59a14f','orange':'#f28e2b',
    'purple':'#7b6fd0','red':'#b2182b','gray':'#7f7f7f','dark':'#2b2b2b','lightgray':'#f5f6f8',
    'midgray':'#d7dce2','paleblue':'#eaf2fb','paleteal':'#e6f4f1','paleorange':'#fff3e0',
    'palegreen':'#edf7ed','palepurple':'#f0ecfb','palered':'#fdebed'
}

def savefig(fig, name):
    fig.savefig(figdir/name, bbox_inches='tight', pad_inches=0.06)
    plt.close(fig)

def box(ax, x, y, w, h, text, fc='white', ec='#333333', lw=1.0, fs=8.5, wt='normal', r=0.02):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.012,rounding_size={r}',
                                facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2))
    ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fs, weight=wt,
            color=COL['dark'], linespacing=1.15, zorder=3)

def arrow(ax, start, end, c='#333333', lw=1.2, ms=10, rad=0.0, style='-', z=5):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=ms, linewidth=lw,
                                color=c, linestyle=style, connectionstyle=f'arc3,rad={rad}',
                                zorder=z, shrinkA=1, shrinkB=1))

# Architecture diagram
fig, ax = plt.subplots(figsize=(7.8, 4.85))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
for y, h, fc in [(0.72, 0.20, '#fbfbfc'), (0.32, 0.35, '#f7f9fb'), (0.10, 0.17, '#fbfbfc')]:
    ax.add_patch(Rectangle((0.02, y), 0.96, h, facecolor=fc, edgecolor='none'))
ax.text(0.035, 0.91, 'A. Dynamic change detection', fontsize=8.4, weight='bold', color=COL['navy'])
ax.text(0.035, 0.665, 'B. Candidate generation and selection-filtered admission', fontsize=8.4, weight='bold', color=COL['navy'])
ax.text(0.035, 0.27, 'C. Feedback and diagnostics', fontsize=8.4, weight='bold', color=COL['navy'])
box(ax, 0.07, 0.775, 0.16, 0.09, 'Dynamic\nenvironment', COL['paleblue'], COL['blue'], 1.2, 8.6, 'bold')
box(ax, 0.29, 0.775, 0.16, 0.09, 'Change\nmonitoring', 'white', COL['dark'], 1.0, 8.4)
arrow(ax, (0.23, 0.82), (0.29, 0.82)); ax.text(0.26, 0.847, 'state change', ha='center', fontsize=6.9, color=COL['gray'])
ax.add_patch(FancyBboxPatch((0.06,0.36),0.45,0.245,boxstyle='round,pad=0.012,rounding_size=0.025',
                            facecolor='white', edgecolor=COL['midgray'], linewidth=0.9, zorder=1))
ax.text(0.285, 0.588, 'proposal sources (over-generated pool)', ha='center', va='center', fontsize=8.3, weight='bold')
sources = [
    (0.085,0.505,0.18,0.06,'Elite survivors\n$ q_e = 0.35 $',COL['palegreen'],COL['green']),
    (0.305,0.505,0.18,0.06,'Memory archive\n$ q_m = 0.30 $',COL['paleteal'],COL['teal']),
    (0.085,0.405,0.18,0.06,'Temporal prediction\n$ q_p = 0.25 $',COL['paleorange'],COL['orange']),
    (0.305,0.405,0.18,0.06,'Diversity immigrants\n$ q_d = 0.10 $',COL['palepurple'],COL['purple']),
]
for x, y, w, h, t, fc, ec in sources:
    box(ax, x, y, w, h, t, fc, ec, 1.0, 7.3, 'normal', 0.016)
arrow(ax, (0.37,0.775), (0.18,0.605), lw=1.0, ms=9, rad=0.05)
arrow(ax, (0.37,0.775), (0.40,0.605), lw=1.0, ms=9, rad=-0.05)
box(ax, 0.585, 0.45, 0.15, 0.115, 'Evaluated\nproposal pool\n$U_t$', COL['lightgray'], COL['dark'], 1.0, 8.0, 'bold')
ax.text(0.66, 0.425, 'all candidates are scored\nunder current objectives', ha='center', va='top', fontsize=6.8, color=COL['gray'])
for _, y, _, _, _, _, ec in sources:
    arrow(ax, (0.485, y+0.03), (0.585, 0.505), c=ec, lw=1.0, ms=8, rad=0.02)
box(ax, 0.80, 0.535, 0.15, 0.095, 'NSGA-II\nenvironmental\nselection', COL['palered'], COL['red'], 1.2, 7.8, 'bold')
box(ax, 0.80, 0.37, 0.15, 0.09, 'Redeployed\npopulation\n$P_t$', 'white', COL['dark'], 1.0, 8.0, 'bold')
arrow(ax, (0.735,0.507), (0.80,0.585), c=COL['red'], lw=1.2, ms=10)
arrow(ax, (0.875,0.535), (0.875,0.46), c=COL['red'], lw=1.2, ms=10)
ax.text(0.875, 0.647, 'filter: no reserved slots', ha='center', va='bottom', fontsize=6.8, color=COL['red'])
box(ax, 0.58, 0.155, 0.18, 0.075, 'Pareto set +\ndecision metrics', 'white', COL['dark'], 0.9, 7.6)
box(ax, 0.29, 0.155, 0.17, 0.075, 'Archive update\n+ diagnostics', 'white', COL['teal'], 0.9, 7.6)
arrow(ax, (0.81,0.39), (0.745,0.23), lw=1.0, ms=9, rad=0.12)
arrow(ax, (0.58,0.19), (0.46,0.19), c=COL['teal'], lw=1.0, ms=9)
arrow(ax, (0.375,0.23), (0.395,0.505), c=COL['teal'], lw=0.9, ms=8, style='--', rad=-0.05)
arrow(ax, (0.34,0.23), (0.365,0.775), c=COL['gray'], lw=0.8, ms=8, style='--', rad=0.10)
ax.text(0.50, 0.045, 'Prediction, memory, elite and diversity generate proposals; only current-state selection determines final admission.',
        ha='center', fontsize=8.0, color=COL['dark'])
savefig(fig, 'figure1_architecture.pdf')

# Graphical abstract
fig, ax = plt.subplots(figsize=(11.0, 3.55))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ax.text(0.03, 0.88, 'Selection-filtered predictive redeployment', fontsize=14, weight='bold', ha='left', color=COL['dark'])
ax.text(0.03, 0.79, 'Prediction proposes candidates; current-state selection decides admission.', fontsize=9.5, ha='left', color=COL['gray'])
box(ax, 0.035, 0.43, 0.12, 0.18, 'dynamic\nchange', COL['paleblue'], COL['blue'], 1.2, 9, 'bold')
arrow(ax, (0.155,0.52), (0.205,0.52), c=COL['blue'], lw=1.5, ms=13)
cards = [('memory','recent + historical',COL['paleteal'],COL['teal']), ('prediction','centroid movement',COL['paleorange'],COL['orange']), ('diversity','immigrant seeds',COL['palepurple'],COL['purple']), ('elites','best survivors',COL['palegreen'],COL['green'])]
for (label, sublabel, fc, ec), y in zip(cards, [0.65,0.49,0.33,0.17]):
    box(ax, 0.21, y, 0.17, 0.105, f'{label}\n{sublabel}', fc, ec, 1.0, 8.2)
    arrow(ax, (0.38, y+0.052), (0.48,0.52), c=ec, lw=1.2, ms=11, rad=(0.54-y)*0.22)
box(ax, 0.48, 0.37, 0.15, 0.26, 'evaluated\nproposal pool', COL['lightgray'], COL['dark'], 1.0, 9, 'bold')
np.random.seed(7)
for i in range(46):
    ax.add_patch(Circle((0.505+np.random.rand()*0.10, 0.400+np.random.rand()*0.19), 0.006,
                        color=[COL['teal'],COL['orange'],COL['purple'],COL['green'],'#a8a8a8'][i%5], alpha=0.85, zorder=4))
ax.text(0.555, 0.335, 'current objectives', fontsize=7.7, ha='center', color=COL['gray'])
arrow(ax, (0.63,0.50), (0.70,0.50), c=COL['red'], lw=1.5, ms=14)
box(ax, 0.70, 0.40, 0.11, 0.20, 'NSGA-II\nselection\nfilter', COL['palered'], COL['red'], 1.2, 8.7, 'bold')
ax.text(0.755, 0.37, 'no reserved\nprediction slots', ha='center', va='top', fontsize=7.5, color=COL['red'])
arrow(ax, (0.81,0.50), (0.865,0.50), c=COL['red'], lw=1.5, ms=14)
ax.add_patch(FancyBboxPatch((0.865,0.22),0.11,0.51,boxstyle='round,pad=0.012,rounding_size=0.02',
                            facecolor='white', edgecolor=COL['dark'], linewidth=1.0, zorder=2))
ax.text(0.920,0.625,'redeployed\npopulation',ha='center',va='center',fontsize=8.8,weight='bold',color=COL['dark'],zorder=4)
x0,y0,w,h = 0.888,0.285,0.070,0.205
ax.plot([x0,x0,x0+w], [y0+h,y0,y0], color=COL['midgray'], lw=0.8, zorder=3)
fx=np.linspace(x0+0.006, x0+w-0.006, 8); fy=y0+h-0.028-0.13*np.sqrt(np.linspace(0,1,8))
ax.plot(fx,fy,color=COL['navy'],lw=1.3,zorder=4); ax.scatter(fx,fy,s=11,color=COL['navy'],zorder=5)
for _ in range(12):
    ax.scatter(x0+0.01+np.random.rand()*(w-0.02), y0+0.018+np.random.rand()*(h-0.035), s=6, color='#bdbdbd', alpha=0.45, zorder=3)
ax.text(0.50,0.060,'Useful predictions can accelerate recovery; harmful predictions must compete and may be filtered out.',ha='center',fontsize=9.1,color=COL['dark'])
savefig(fig, 'graphical_abstract.pdf')

# Overall interval plots
alg_order = ['NSGA-II','DNSGA-II','MNSGA-II','PPS-NSGA-II','DMOEA/D','ARVLP-NSGA-II','KTR-NSGA-II','ABR-NSGA-II','RDMOEA-NSGA-II','PRR-NSGA-II']
ov = pd.read_csv(res/'overall_summary.csv').set_index('algorithm').loc[alg_order].reset_index()

def interval_plot(df, metric, xlabel, name, higher=True):
    d = df.copy().sort_values(f'{metric}_mean', ascending=not higher).iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.1, 4.55))
    for sp in ['top','right','left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#444444')
    ax.grid(axis='x', color='#dddddd', lw=0.6, zorder=0)
    for i, row in d.iterrows():
        alg, mean, sd = row.algorithm, row[f'{metric}_mean'], row[f'{metric}_std']
        if alg == 'PRR-NSGA-II': color, marker, ms, lw = COL['red'], 'D', 6.8, 1.7
        elif alg == 'ABR-NSGA-II': color, marker, ms, lw = COL['blue'], 's', 6.8, 1.7
        else: color, marker, ms, lw = COL['gray'], 'o', 5.0, 1.1
        ax.errorbar(mean, i, xerr=sd, fmt=marker, markersize=ms, color=color, ecolor=color,
                    elinewidth=lw, capsize=3.2, capthick=lw, markerfacecolor='white', markeredgewidth=1.5, zorder=4)
        ax.text(mean, i+0.24, f'{mean:.3f}', ha='center', va='bottom', fontsize=7.1, color=color)
    ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d.algorithm); ax.tick_params(axis='y', length=0)
    ax.set_xlabel(xlabel)
    ax.text(0.0, 1.02, 'mean ± standard deviation across 20 scenarios and 30 runs', transform=ax.transAxes,
            ha='left', va='bottom', fontsize=9.0, weight='bold', color=COL['dark'])
    ax.text(0.0, -0.15, 'higher is better' if higher else 'lower is better', transform=ax.transAxes, fontsize=7.8, color=COL['gray'])
    leg=[Line2D([0],[0],marker='D',color=COL['red'],label='PRR-NSGA-II',markerfacecolor='white',lw=0,markersize=7,markeredgewidth=1.5),
         Line2D([0],[0],marker='s',color=COL['blue'],label='ABR-NSGA-II',markerfacecolor='white',lw=0,markersize=7,markeredgewidth=1.5),
         Line2D([0],[0],marker='o',color=COL['gray'],label='Other baselines',markerfacecolor='white',lw=0,markersize=6,markeredgewidth=1.3)]
    ax.legend(handles=leg, loc='lower right', frameon=True, framealpha=0.95, edgecolor='#dddddd')
    ax.set_xlim(max(0, (d[f'{metric}_mean']-d[f'{metric}_std']).min()-0.02), (d[f'{metric}_mean']+d[f'{metric}_std']).max()*1.06)
    ax.margins(y=0.055); fig.tight_layout(); savefig(fig, name)

for metric, xlabel, higher, name in [
    ('HV','Normalized hypervolume',True,'overall_HV.pdf'),
    ('IGD','Inverted generational distance',False,'overall_IGD.pdf'),
    ('Stability','Stability',True,'overall_Stability.pdf'),
    ('DecisionDrift','Decision drift',False,'overall_DecisionDrift.pdf'),
    ('QDR','Quality-drift ratio',True,'overall_QDR.pdf')]:
    interval_plot(ov, metric, xlabel, name, higher)

def interval_panel(ax, df, metric, xlabel, panel_title, higher=True, show_legend=False):
    d = df.copy().sort_values(f'{metric}_mean', ascending=not higher).iloc[::-1].reset_index(drop=True)
    for sp in ['top', 'right', 'left']:
        ax.spines[sp].set_visible(False)
    ax.spines['bottom'].set_color('#444444')
    ax.grid(axis='x', color='#dddddd', lw=0.55, zorder=0)
    for i, row in d.iterrows():
        alg, mean, sd = row.algorithm, row[f'{metric}_mean'], row[f'{metric}_std']
        if alg == 'PRR-NSGA-II':
            color, marker, ms, lw = COL['red'], 'D', 5.7, 1.45
        elif alg == 'ABR-NSGA-II':
            color, marker, ms, lw = COL['blue'], 's', 5.7, 1.45
        else:
            color, marker, ms, lw = COL['gray'], 'o', 4.2, 0.95
        ax.errorbar(mean, i, xerr=sd, fmt=marker, markersize=ms, color=color, ecolor=color,
                    elinewidth=lw, capsize=2.8, capthick=lw, markerfacecolor='white',
                    markeredgewidth=1.25, zorder=4)
        ax.text(mean, i + 0.22, f'{mean:.3f}', ha='center', va='bottom', fontsize=6.7, color=color)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d.algorithm)
    ax.tick_params(axis='y', length=0)
    ax.set_xlabel(xlabel)
    ax.set_title(panel_title, loc='left', fontsize=14, weight='bold', pad=26)
    ax.text(0.0, 1.025, 'mean +/- standard deviation across 20 scenarios and 30 runs',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=8.2, weight='bold', color=COL['dark'])
    ax.text(0.0, -0.20, 'higher is better' if higher else 'lower is better',
            transform=ax.transAxes, fontsize=7.2, color=COL['gray'])
    ax.margins(y=0.055)
    if show_legend:
        leg = [
            Line2D([0], [0], marker='D', color=COL['red'], label='PRR-NSGA-II',
                   markerfacecolor='white', lw=0, markersize=6, markeredgewidth=1.4),
            Line2D([0], [0], marker='s', color=COL['blue'], label='ABR-NSGA-II',
                   markerfacecolor='white', lw=0, markersize=6, markeredgewidth=1.4),
            Line2D([0], [0], marker='o', color=COL['gray'], label='Other baselines',
                   markerfacecolor='white', lw=0, markersize=5, markeredgewidth=1.2),
        ]
        ax.legend(handles=leg, loc='lower right', frameon=True, framealpha=0.95, edgecolor='#dddddd', fontsize=7.2)

fig = plt.figure(figsize=(13.1, 6.25))
gs = fig.add_gridspec(2, 6, height_ratios=[1, 1], hspace=0.66, wspace=0.62)
panel_specs = [
    (fig.add_subplot(gs[0, 0:2]), 'HV', 'Normalized hypervolume', '(a) Hypervolume', True),
    (fig.add_subplot(gs[0, 2:4]), 'IGD', 'Inverted generational distance', '(b) IGD', False),
    (fig.add_subplot(gs[0, 4:6]), 'QDR', 'Quality-drift ratio', '(c) QDR', True),
    (fig.add_subplot(gs[1, 1:3]), 'Stability', 'Stability', '(d) Stability', True),
    (fig.add_subplot(gs[1, 3:5]), 'DecisionDrift', 'Decision drift', '(e) Decision drift', False),
]
for ax, metric, xlabel, title, higher in panel_specs:
    interval_panel(ax, ov, metric, xlabel, title, higher, show_legend=True)
fig.savefig(figdir/'overall_metrics_panel.pdf', bbox_inches='tight', pad_inches=0.08)
fig.savefig(figdir/'overall_metrics_panel.png', bbox_inches='tight', pad_inches=0.08, dpi=300)
plt.close(fig)

# Ablation plot
abl = pd.read_csv(res/'ablation_all20_overall_summary.csv')
abl_order = ['PRR-complete','PRR-no-prediction','PRR-no-memory','PRR-no-diversity','PRR-direct-insertion','PRR-no-elite']
labels = {'PRR-complete':'Complete PRR','PRR-no-prediction':'No prediction','PRR-no-memory':'No memory','PRR-no-diversity':'No diversity','PRR-direct-insertion':'Direct insertion','PRR-no-elite':'No elite'}
abl = abl.set_index('algorithm').loc[abl_order].reset_index(); abl['label'] = abl.algorithm.map(labels)
d = abl.iloc[::-1].reset_index(drop=True); ref = float(abl.loc[abl.algorithm=='PRR-complete','HV_mean'].iloc[0])
fig, ax = plt.subplots(figsize=(7.1, 3.85))
for sp in ['top','right','left']:
    ax.spines[sp].set_visible(False)
ax.spines['bottom'].set_color('#444444'); ax.grid(axis='x', color='#dddddd', lw=0.6)
ax.axvline(ref, color=COL['red'], ls='--', lw=1.1, zorder=1)
for i, row in d.iterrows():
    alg, mean, sd = row.algorithm, row.HV_mean, row.HV_std
    if alg == 'PRR-complete': color, marker, ms, lw = COL['red'], 'D', 6.8, 1.7
    elif alg == 'PRR-no-prediction': color, marker, ms, lw = '#666666', 'x', 7.0, 1.4
    elif mean > ref: color, marker, ms, lw = COL['blue'], 'o', 6.2, 1.4
    else: color, marker, ms, lw = COL['gray'], 'o', 5.8, 1.2
    ax.errorbar(mean, i, xerr=sd, fmt=marker, markersize=ms, color=color, ecolor=color,
                elinewidth=lw, capsize=3.2, capthick=lw, markerfacecolor='white', markeredgewidth=1.5, zorder=4)
    ax.text(mean, i+0.20, f'{mean:.3f}', ha='center', va='bottom', fontsize=7.1, color=color)
ax.set_yticks(np.arange(len(d))); ax.set_yticklabels(d.label); ax.tick_params(axis='y', length=0)
ax.set_xlabel('Hypervolume')
ax.text(0, 1.03, 'component-risk ablation, mean ± standard deviation', transform=ax.transAxes, ha='left', va='bottom', fontsize=9.0, weight='bold')
ax.text(0, -0.17, 'higher is better; dashed line marks complete PRR', transform=ax.transAxes, fontsize=7.8, color=COL['gray'])
ax.set_xlim(max(0, (d.HV_mean-d.HV_std).min()-0.02), (d.HV_mean+d.HV_std).max()*1.06)
fig.tight_layout(); savefig(fig, 'ablation_hv.pdf')

# Scenario-level plots, emphasizing PRR and ABR
summ = pd.read_csv(res/'summary_by_scenario_algorithm.csv')
for metric, label in [('HV','Scenario-wise hypervolume'), ('IGD','Scenario-wise IGD')]:
    scenarios = list(summ['scenario'].unique())
    fig, ax = plt.subplots(figsize=(11.0, 4.9))
    for alg in alg_order:
        sub = summ[summ.algorithm == alg].set_index('scenario').loc[scenarios]
        if alg == 'PRR-NSGA-II':
            ax.plot(range(len(scenarios)), sub[f'{metric}_mean'], marker='D', linewidth=2.0, markersize=4.2, color=COL['red'], label=alg, zorder=5)
        elif alg == 'ABR-NSGA-II':
            ax.plot(range(len(scenarios)), sub[f'{metric}_mean'], marker='s', linewidth=2.0, markersize=4.2, color=COL['blue'], label=alg, zorder=5)
        else:
            ax.plot(range(len(scenarios)), sub[f'{metric}_mean'], marker='o', linewidth=0.9, markersize=2.8, color='#9a9a9a', alpha=0.55, label=alg, zorder=2)
    ax.set_xticks(range(len(scenarios))); ax.set_xticklabels(scenarios, rotation=45, ha='right')
    ax.set_ylabel(label); ax.grid(axis='y', color='#dddddd', lw=0.6)
    for sp in ['top','right']: ax.spines[sp].set_visible(False)
    ax.legend(fontsize=7, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.24), frameon=False)
    ax.text(0.0, 1.02, f'{label} by scenario', transform=ax.transAxes, ha='left', va='bottom', fontsize=9.0, weight='bold')
    fig.tight_layout(); savefig(fig, f'scenario_{metric}.pdf')

# Generation-budget sensitivity
sen = pd.read_csv(res/'sensitivity_summary.csv')
sub = sen[sen.parameter == 'generations_per_state'].sort_values('value')
fig, ax = plt.subplots(figsize=(6.8, 4.05))
ax.errorbar(sub['value'], sub['HV_mean'], yerr=sub['HV_std'], marker='o', markersize=6.0, linewidth=1.8,
            color=COL['blue'], ecolor=COL['blue'], capsize=4, markerfacecolor='white', markeredgewidth=1.4)
ax.set_xlabel('Generations per environmental state'); ax.set_ylabel('Hypervolume')
ax.grid(axis='y', color='#dddddd', lw=0.6)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.text(0.0, 1.02, 'Generation-budget sensitivity, mean ± standard deviation', transform=ax.transAxes,
        ha='left', va='bottom', fontsize=9.0, weight='bold')
fig.tight_layout(); savefig(fig, 'sensitivity_generations_hv.pdf')

print(f'Figures written to {figdir}')
