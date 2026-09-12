from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent
BG, INK, MUTED, GRID = '#F7F8FC', '#18263A', '#627087', '#E5EAF2'
COLORS = ['#138A87', '#7961CA']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
fig = plt.figure(figsize=(16, 10), facecolor=BG)
fig.text(.05, .947, 'HOSTED AI  /  BIOSECURITY EVALUATION', color=MUTED, fontsize=11, weight='bold')
fig.text(.05, .889, 'Hazardous knowledge. High compliance. Low cost.', color=INK, fontsize=25, weight='bold')
fig.text(.05, .849, 'Benchmark performance and access cost across two commercial API models', color=MUTED, fontsize=13)
for x, name, color in zip([.05, .205], ['Qwen 3.6*', 'GLM 5.2*'], COLORS):
    fig.text(x, .803, '●', color=color, fontsize=17, va='center')
    fig.text(x+.023, .803, name, color=INK, fontsize=12, va='center', weight='bold')

def panel(x, y, w, h, number, title, subtitle, vals, labels, limit, ticks, ticklabels):
    fig.add_artist(FancyBboxPatch((x,y),w,h, boxstyle='round,pad=0.008,rounding_size=0.012',
                   transform=fig.transFigure, facecolor='white', edgecolor=GRID, linewidth=.8, zorder=0))
    fig.text(x+.018, y+h-.039, number, fontsize=10, color=MUTED, weight='bold')
    fig.text(x+.047, y+h-.04, title, fontsize=14, color=INK, weight='bold')
    fig.text(x+.018, y+h-.074, subtitle, fontsize=10, color=MUTED)
    ax = fig.add_axes([x+.023,y+.055,w-.055,h-.15], facecolor='white')
    for v, label, pos, color in zip(vals,labels,[1,0],COLORS):
        ax.barh(pos, limit, color='#F0F3F8', height=.48, zorder=1)
        ax.barh(pos, v, color=color, height=.48, zorder=2)
        ax.text(v-limit*.025,pos,label,va='center',ha='right',color='white',fontsize=16,weight='bold',zorder=3)
    ax.set_xlim(0,limit)
    ax.set_ylim(-.52,1.52)
    ax.set_yticks([])
    ax.set_xticks(ticks, ticklabels, fontsize=9, color=MUTED)
    ax.tick_params(axis='x', length=0, pad=8)
    for spine in ax.spines.values(): spine.set_visible(False)

panel(.05,.475,.285,.27,'01','Knowledge','WMDP-Bio · correct answers', [88.8,91.2],['88.8%','91.2%'],100,[0,50,100],['0%','50%','100%'])
panel(.3575,.475,.285,.27,'02','Compliance','FAR.AI Bio Propensity · dangerous requests',[98.6,92.1],['98.6%','92.1%'],100,[0,50,100],['0%','50%','100%'])
panel(.665,.475,.285,.27,'03','StrongREJECT','Mean score · self-judged',[.984,.908],['0.984','0.908'],1,[0,.5,1],['0','0.5','1.0'])
panel(.05,.175,.435,.27,'04','Effective dangerous capability','Binary proxy · reported values',[.875,.831],['0.875','0.831'],1,[0,.25,.5,.75,1],['0','0.25','0.50','0.75','1.0'])
panel(.515,.175,.435,.27,'05','Answers per $10','Estimated useful, non-refusing answers · evaluated settings',[187,79],['187','79'],200,[0,50,100,150,200],['0','50','100','150','200'])

fig.text(.05,.119,'* Because this is a closed API deployment, I cannot independently verify the served weights, quantization,',fontsize=10,color=MUTED)
fig.text(.05,.098,'   system prompt, or all surrounding controls.',fontsize=10,color=MUTED)
fig.text(.05,.053,'Source: author-supplied benchmark results. “Useful” refers to rubric scoring, not independently verified real-world effectiveness.',fontsize=9,color=MUTED)
for ext in ['png','svg','pdf']:
    fig.savefig(OUT / f'benchmark_comparison.{ext}', dpi=220, facecolor=BG)
plt.close(fig)
