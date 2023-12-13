import numpy as np
import matplotlib as mpl

from pyaromatics.stay_organized.mpl_tools import load_plot_settings

# mpl = load_plot_settings(mpl=mpl)

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Fixing random state for reproducibility
np.random.seed(17)  # 12
fig, axs = plt.subplots(nrows=1, figsize=(7, 5))

Nx = 6
Ny = 4

colors = 'k'
area = 50  # (30 * np.random.rand(N))**2  # 0 to 15 point radii
fontsize = 20
onernnffn = True

x, y = np.meshgrid(np.linspace(0, 1, Nx), np.linspace(0, 1, Ny))
x = x.flatten()
y = y.flatten()
xs = np.unique(x)
ys = np.unique(y)
axs.scatter(x, y, s=area, c=colors, alpha=1., facecolors='none')
n_moves = 3

colors = plt.get_cmap('tab20')
for nm in range(n_moves):
    # zero means move down, 1 means move left
    moves = [0] * (Ny - 1) + [1] * (Nx - Ny)
    np.random.shuffle(moves)

    c = colors(.4 + nm / (n_moves - 1) * .4)
    pos = np.array([0.27, 0.25]) if nm == 0 else np.array([.1, .1]) if nm == 1 else np.array([.08, -.08])
    plt.text(*pos, f"$c_{nm}$", fontsize=fontsize, rotation=0, color=c)

    xi, yi = -1, -1
    for move in moves:
        nxi, nyi = (xi - 1, yi) if not move == 0 else (xi - 1, yi - 1)
        print(move, ':', xi, yi, nxi, nyi)
        if move == 1:
            x_tail, y_tail, x_head, y_head = (
                xs[xi] - 1 / Nx / 8, ys[yi], xs[nxi] + 1 / Nx / 8, ys[nyi]
            )
        else:
            x_tail, y_tail, x_head, y_head = (
                xs[xi] - 1 / Nx / 8, ys[yi] - 1 / Nx / 8, xs[nxi] + 1 / Nx / 8, ys[nyi] + 1 / Nx / 8
            )

        edgecolor = None
        arrow = mpatches.FancyArrowPatch(
            (x_tail, y_tail), (x_head, y_head), mutation_scale=10, color=c, edgecolor=edgecolor
        )
        axs.add_patch(arrow)
        xi, yi = nxi, nyi

if onernnffn:
    c = 'k'
    # zero means move down, 1 means move left
    moves = [0] * (Ny - 1)

    plt.text(1 / Nx / 5, .5, "$c_{FFN}$", fontsize=fontsize, rotation=0, color=c)

    xi, yi = -Nx, -1
    for move in moves:
        nxi, nyi = (xi, yi) if not move == 0 else (xi, yi - 1)
        x_tail, y_tail, x_head, y_head = (
            xs[xi], ys[yi] - 1 / Nx / 8, xs[nxi], ys[nyi] + 1 / Nx / 8
        )

        edgecolor = None
        arrow = mpatches.FancyArrowPatch(
            (x_tail, y_tail), (x_head, y_head), mutation_scale=10, color=c, edgecolor=edgecolor
        )
        axs.add_patch(arrow)
        xi, yi = nxi, nyi

    c = 'r'
    # zero means move down, 1 means move left
    moves = [1] * (Nx - 1)
    plt.text(.5, -1 / Ny / 2, r"$c_{1\text{-}RNN}$", fontsize=fontsize, rotation=0, color=c)

    xi, yi = -1, -Ny
    for move in moves:
        nxi, nyi = (xi - 1, yi)
        yshift = .01 if xi < -3 else 0
        x_tail, y_tail, x_head, y_head = (
            xs[xi] - 1 / Nx / 8, ys[yi] - yshift, xs[nxi] + 1 / Nx / 8, ys[nyi] - yshift
        )

        edgecolor = None
        arrow = mpatches.FancyArrowPatch(
            (x_tail, y_tail), (x_head, y_head), mutation_scale=10, color=c, edgecolor=edgecolor
        )
        axs.add_patch(arrow)
        xi, yi = nxi, nyi

if not onernnffn:
    plt.text(1 + .5 / Nx / 2, 1 - 1 / Ny / 8, 'L', fontsize=fontsize, rotation=0)
    plt.text(1 + .5 / Nx / 2, 0 / Ny - .5 / Ny / 6, 'l', fontsize=fontsize, rotation=0)
    plt.text(1 + .5 / Nx / 2, 1 / Ny + 1.5 / Ny / 6, 'l + 1', fontsize=fontsize, rotation=0)

    plt.text(1 - 1 / Nx / 10, 1 + .5 / Ny / 2, 't', fontsize=fontsize, rotation=0)
    plt.text(0 / Nx - .3 / Nx / 5, 1 + .5 / Ny / 2, "t'", fontsize=fontsize, rotation=0)
    plt.text(1 / Nx - 1 / Nx / 5, 1 + .5 / Ny / 2, "t' + 1", fontsize=fontsize, rotation=0)

    # plt.text(.3 / Nx - 2 / Nx / 8, 1 / Ny / 4 + 1.5 / Ny, r"$\frac{\partial h_{t'+2, l+2}}{\partial h_{t'+1, l+1}}$",
    #          fontsize=fontsize+4, rotation=0)
    plt.text(.95 / Nx, 1 / Ny / 4 + 1.5 / Ny, r"$M_k$", fontsize=fontsize + 4, rotation=0)
    # make a nice dark blue
    color = (0, 0, 1, .5)
    plt.text(.8, 1.1, r"$J^{T,L}_{t',l}$", fontsize=fontsize, rotation=0, color=color)
    plot_filename = r'./experiments/grad_grid.pdf'

else:
    plt.text(1 + .5 / Nx / 2, 1 - 1 / Ny / 8, 'L', fontsize=fontsize, rotation=0)
    plt.text(1 + .5 / Nx / 2, 0 / Ny - .5 / Ny / 6, 'l=0', fontsize=fontsize, rotation=0)

    plt.text(1 - 1 / Nx / 10, 1 + .5 / Ny / 2, 'T', fontsize=fontsize, rotation=0)
    plt.text(0 / Nx - 1.3 / Nx / 5, 1 + .5 / Ny / 2, "t=0", fontsize=fontsize, rotation=0)
    # plt.text(1 / Nx - 1 / Nx / 5, 1 + .5 / Ny / 2, "t' + 1", fontsize=fontsize, rotation=0)
    plot_filename = r'./experiments/grad_gridvsffn1rnn.pdf'

plt.axis('off')

fig.savefig(plot_filename, bbox_inches='tight')
plt.show()
