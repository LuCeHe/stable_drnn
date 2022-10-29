import numpy as np
import matplotlib as mpl

from GenericTools.stay_organized.mpl_tools import load_plot_settings

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

mpl = load_plot_settings(mpl=mpl)

# Fixing random state for reproducibility
np.random.seed(7)
fig, axs = plt.subplots(nrows=1, figsize=(7, 4))

Nx = 8
Ny = 4
# base = np.linspace(0, 1, Nx)
ys = np.linspace(0, 1, (Nx - Ny + 1))
xs = np.linspace(0, 1, Nx)
y = ys.tolist() * Nx
x = np.repeat(xs, (Nx - Ny + 1))
colors = 'k'
area = 100  # (30 * np.random.rand(N))**2  # 0 to 15 point radii

axs.scatter(x, y, s=area, c=colors, alpha=1., facecolors='none')

# zero means move down, 1 means move left
moves_1 = [0] * (Ny - 1) + [1] * (Nx - Ny)
moves_2 = [0] * (Ny - 1) + [1] * (Nx - Ny)

np.random.shuffle(moves_1)
np.random.shuffle(moves_2)

for c, moves in zip(['#E08108', '#049A25'], [moves_1, moves_2]):
    xi, yi = -1, -1
    for move in moves:
        nxi, nyi = (xi - 1, yi) if move == 0 else (xi - 1, yi - 1)

        if move == 0:
            x_tail, y_tail, x_head, y_head = (xs[xi] - 1 / Nx / 8, ys[yi], xs[nxi] + 1 / Nx / 8, ys[nyi])
        else:
            x_tail, y_tail, x_head, y_head = (
                xs[xi] - 1 / Nx / 8, ys[yi] - 1 / Nx / 8, xs[nxi] + 1 / Nx / 8, ys[nyi] + 1 / Nx / 8)

        arrow = mpatches.FancyArrowPatch((x_tail, y_tail), (x_head, y_head), mutation_scale=10, color=c)
        axs.add_patch(arrow)
        xi, yi = nxi, nyi

plt.text(1 + 1 / Nx / 2, 1 - 1 / Ny / 8, '$L$', fontsize=14, rotation=0)
plt.text(1 + 1 / Nx / 2, 0 / Ny - 1 / Ny / 6, '$l$', fontsize=14, rotation=0)
plt.text(1 + 1 / Nx / 2, 1 / Ny - 1 / Ny / 8, '$l+1$', fontsize=14, rotation=0)

plt.text(1 - 1 / Nx / 10, 1 + 1 / Ny / 2, '$t$', fontsize=14, rotation=0)
plt.text(0 / Nx - 0 / Nx / 5, 1 + 1 / Ny / 2, "$t'$", fontsize=14, rotation=0)
plt.text(1 / Nx - 1 / Nx / 5, 1 + 1 / Ny / 2, "$t'+1$", fontsize=14, rotation=0)

plt.text(1 / Nx + 1 / Nx/8, 1 / Ny / 4 + 1 / Ny, r"$\frac{\partial h_{t'+2, l+1}}{\partial h_{t'+1, l+1}}$",
         fontsize=14,         rotation=0)

plt.axis('off')

plot_filename = r'experiments/grad_grid.pdf'
fig.savefig(plot_filename, bbox_inches='tight')

plt.show()


