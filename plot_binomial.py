import os
import numpy as np
import matplotlib.pyplot as plt


from scipy import special as sp

bound = lambda l, t: sp.binom(t + l + 2, t) / t
bound = lambda l, t: sp.binom(t + l, t)

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')




def plot_binomial():


    fontsize = 18
    linewidth = 2
    fig, axs = plt.subplot_mosaic([['i)', 'ii)', 'iii)']], layout='constrained', figsize=(6, 3))

    for label, ax in axs.items():
        ax.set_title(label, fontfamily='serif', loc='left', fontsize=fontsize)
        ax.set_yscale('log')

        if label == 'i)':
            T, dL = 10000, 5

            ts = np.linspace(1, T, 1000)
            y = bound(dL, ts)

            ax.plot(ts, y, color='#018E32', linewidth=linewidth)
            ax.set_xlabel(r'$T$', fontsize=fontsize)
            ax.set_yticks([1e8, 1e16])

            # axs[0].set_ylabel(r'$\frac{1}{T}\binom{T + \Delta l +2}{T}$', fontsize=fontsize + 6)
            # ax.set_ylabel('Descent paths number\nin rectangular grid', fontsize=fontsize * 1.1, labelpad=20)
            ax.set_ylabel('# descent paths\nin rectangular grid', fontsize=fontsize * 1)

        elif label == 'ii)':

            T, dL = 5, 100

            ls = np.linspace(1, dL, 1000)
            y = bound(ls, T)

            ax.plot(ls, y, color='#018E32', linewidth=linewidth)
            ax.set_xlabel(r'$\Delta l$', fontsize=fontsize)
            ax.set_yticks([1e4, 1e7])

        elif label == 'iii)':

            T = 10000
            ts = np.linspace(1, T, 1000)
            y = bound(ts / 100, ts)

            ax.plot(ts, y, color='#018E32', linewidth=linewidth)
            ax.set_xlabel(r'$100\Delta l=T$', fontsize=fontsize)
            ax.set_yticks([1e110, 1e220])

        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)

        xlabels = [f'{int(x / 1000)}K' if x > 1000 else int(x) for x in ax.get_xticks()]
        ax.set_xticklabels(xlabels)
        ax.minorticks_off()
        ax.tick_params(axis='both', which='major', labelsize=fontsize * .9)
        ax.yaxis.tick_right()

        pathplot = os.path.join(CDIR, 'experiments', 'subexp.pdf')
        fig.savefig(pathplot)

        plt.show()


if __name__ == '__main__':
    plot_binomial()
