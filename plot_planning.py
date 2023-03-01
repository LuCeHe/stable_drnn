import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from alif_sg.tools.plot_tools import lsc_colors, lsc_clean_comments

expsid = 'rnns'  # rnns transfeff

extra_name = ''
title = ''
if expsid == 'transfeff':
    all_comments = [
        '',
        f'findLSC',
        f'findLSC_radius',
        f'findLSC_supsubnpsd',
    ]
    activations = ['relu', 'tanh', 'swish']
    archs = ['EfficientNet\nCIFAR100' + r' $\uparrow$', 'Transformer\nEnglish-German' + ' $\downarrow$']
    bbox_to_anchor = (-.2, -.3)
    # shift = 1.5
elif expsid == 'rnns':
    title = 'layers'
    all_comments = [
        '',
        # f'findLSC',
        f'findLSC_radius',
        'findLSC_radius_targetnorm:.5',
        # f'findLSC_supsubnpsd',
    ]
    if not title == 'layers':
        archs = ['sl-MNIST' + r' $\uparrow$', 'SHD' + r' $\uparrow$', 'PTB' + r' $\downarrow$']
    else:
        archs = [1, 3, 5, 7]
    activations = ['LSTM', 'ALIF', 'ALIFb']
    bbox_to_anchor = (-.7, -.3)
    extra_name = '_stack'
else:
    raise ValueError
shift = (len(all_comments) + 1) / 2 - 1

fig, axs = plt.subplots(1, len(archs), figsize=(6, 3), sharey=True,
                        gridspec_kw={'wspace': .05, 'hspace': .1})
X = np.arange(len(activations))
w = 1 / (len(all_comments) + 1)

for j, arch in enumerate(archs):
    for i, c in enumerate(all_comments):
        data = .3 + .65 * np.random.rand(len(activations))
        error = .02 + .07 * np.random.rand(len(activations))
        axs[j].bar(X + i * w, data, yerr=error, width=w, color=lsc_colors[c], label=lsc_clean_comments(c))

    axs[j].set_xticks([r + shift * w for r in range(len(activations))], activations, rotation=10)
    axs[j].set_title(arch, weight='bold')

    if not j == 0:
        axs[j].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

for ax in axs.reshape(-1):
    for pos in ['right', 'left', 'bottom', 'top']:
        ax.spines[pos].set_visible(False)

fig.text(0.5, 0.7, 'blueprint plot', va='center', ha='center', rotation='horizontal', weight='bold')

legend_elements = [Line2D([0], [0], color=lsc_colors[n], lw=4, label=lsc_clean_comments(n))
                   for n in all_comments]
plt.legend(ncol=len(all_comments), handles=legend_elements, loc='lower center', bbox_to_anchor=bbox_to_anchor)
fig.suptitle(title, weight='bold')

plot_filename = f'experiments/{expsid}_{title}.pdf'
fig.savefig(plot_filename, bbox_inches='tight')
plt.show()
