import time

from GenericTools.stay_organized.submit_jobs import dict2iter

print(1)
# measure time
start = time.time()

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GenericTools.stay_organized.pandardize import experiments_to_pandas, complete_missing_exps
from GenericTools.stay_organized.standardize_strings import shorten_losses

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
# EXPERIMENTS = os.path.join(CDIR, 'experiments')
EXPERIMENTS = r'D:\work\alif_sg\experiments'
GEXPERIMENTS = [
    # os.path.join(CDIR, 'good_experiments', '2022-12-16--ffn'),
    # os.path.join(CDIR, 'good_experiments'),
    r'D:\work\alif_sg\good_experiments\2022-12-16--ffn'
]

plot_norms_evol = False
plot_norms_evol_1 = False
lrs_plot = False
plot_losses = False
missing_exps = False
remove_incomplete = False
truely_remove = False

print(2, time.time() - start)
start = time.time()

metric = 'val_acc M'  # 'val_acc max'   'val_loss min'
expsid = 'ffnandcnns'  # effnet als ffnandcnns
h5path = os.path.join(EXPERIMENTS, f'summary_{expsid}.h5')

df = experiments_to_pandas(
    h5path=h5path, zips_folder=GEXPERIMENTS, unzips_folder=EXPERIMENTS, experiments_identifier=expsid,
    exclude_files=['cout.txt'], exclude_columns=['_mean list', '_var list'], check_for_new=True
)

print(list(df.columns))
print(3, time.time() - start)

start = time.time()

# select only rows with width 10
df['time_elapsed'] = pd.to_timedelta(df['time_elapsed'], unit='s')

df = df[df['width'] == 128]
df = df[df['layers'] == 30]
# df = df[df['dataset'] == 'cifar100']
# df = df[~df['comments'].str.contains('findLSC_radius_supn')]
# df = df[df['comments'].str.contains('heinit') | df['comments'].str.contains('findLSC_supnpsd2')]
# df = df[~df['activation'].str.contains('sin')]


if plot_norms_evol:
    for _, row in df.iterrows():

        fig, axs = plt.subplots(2, 3)
        fig.suptitle(row['comments'])

        for k in row.keys():
            try:
                if '_mean' in k:
                    curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                    axs[0, 0].plot(curve)
                    axs[0, 0].title.set_text('mean')

                if '_var' in k:
                    curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                    axs[0, 1].plot(curve)
                    axs[0, 1].title.set_text('variance')

                if 'LSC_losses' in k:
                    curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                    axs[1, 0].plot(curve)
                    axs[1, 0].title.set_text('lsc loss')

                if 'LSC_norms' in k:
                    curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                    axs[1, 1].plot(curve)
                    axs[1, 1].title.set_text('lsc norm')

                if 'sparse_categorical_accuracy list' == k:
                    axs[1, 2].plot(row[k])
                    axs[1, 2].title.set_text('acc')

                if 'val_sparse_categorical_accuracy list' == k:
                    axs[1, 2].plot(row[k], '--')
                    axs[1, 2].title.set_text('acc')

                if 'loss list' == k:
                    axs[0, 2].plot(row[k])
                    axs[0, 2].title.set_text('loss')

                if 'val_loss list' == k:
                    axs[0, 2].plot(row[k], '--')
                    axs[0, 2].title.set_text('loss')

            except Exception as e:
                print(e)

        plt.show()

if plot_norms_evol_1:
    df = df[df['width'] == 128]
    df = df[df['layers'] == 30]

    print(df.shape)
    fig, axs = plt.subplots(df.shape[0], 6)

    for j, row in df.iterrows():
        i = j - 1

        axs[i, 0].set_ylabel(row['comments'])
        print('-' * 20)
        kernel_names = sorted(np.unique([k.replace('_mean', '') for k in row.keys() if 'kernel:0_mean' in k]))
        print(kernel_names)
        print(len(kernel_names))
        for k in row.keys():
            if not 'bias' in k:
                try:
                    if '_mean' in k:
                        print(k)
                        # print(row['comments'],row['path'], k)
                        if isinstance(row[k], str):
                            print(row[k][:10])
                        else:
                            print(row[k])
                        curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                        axs[i, 0].plot(curve)
                        axs[i, 0].title.set_text('mean')

                    if '_var' in k:
                        # print(row[k])
                        curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                        axs[i, 1].plot(curve)
                        axs[i, 1].title.set_text('variance')

                    if 'LSC_losses' in k:
                        curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                        axs[i, 2].plot(curve)
                        axs[i, 2].title.set_text('lsc loss')

                    if 'LSC_norms' in k:
                        curve = np.array([float(x) for x in row[k][1:-1].split(',')])
                        axs[i, 3].plot(curve)
                        axs[i, 3].title.set_text('lsc norm')

                    if 'sparse_categorical_accuracy list' == k:
                        axs[i, 4].plot(row[k])
                        axs[i, 4].title.set_text('acc')

                    if 'val_sparse_categorical_accuracy list' == k:
                        axs[i, 4].plot(row[k], '--')
                        axs[i, 4].title.set_text('acc')

                    if 'loss list' == k:
                        axs[i, 5].plot(row[k])
                        axs[i, 5].title.set_text('loss')

                    if 'val_loss list' == k:
                        axs[i, 5].plot(row[k], '--')
                        axs[i, 5].title.set_text('loss')

                except Exception as e:
                    print(e)

    plt.show()

columns_containing = ['_var']

new_column_names = {c_name: shorten_losses(c_name) for c_name in df.columns}
df.rename(columns=new_column_names, inplace=True)
df = df.rename(columns={'test_loss': 'test_loss m', 'test_acc': 'test_acc M'})

plot_only = [
    'act', 'pre_eps', 'eps', 'dataset',
    # 'spe', 'width', 'depth',
    'seed', 'lr', 'comments',
    'val_acc M', 'val_loss m', 'test_acc M', 'test_loss m',
    # 'acc max','loss min',
    'LSC_norms i', 'LSC_norms f',
     'ep M', 'time_elapsed', 'hostname', 'path',
]

odf = df
df = df[plot_only]

print(df.columns)
df = df.sort_values(by=metric)
print(df.to_string())

# import sys
# sys.exit()

metrics_oi = [
    # 'loss min',
    'val_loss m', 'test_loss m',
    # 'acc max',
    'val_acc M', 'test_acc M',
    'LSC_norms i', 'LSC_norms f',
]

group_cols = ['lr', 'comments', 'act', 'dataset']
counts = df.groupby(group_cols).size().reset_index(name='counts')

metrics_oi = [shorten_losses(m) for m in metrics_oi]
mdf = df.groupby(
    group_cols, as_index=False
).agg({m: ['mean', 'std'] for m in metrics_oi})

for m in metrics_oi:
    mdf['mean_{}'.format(m)] = mdf[m]['mean']
    mdf['std_{}'.format(m)] = mdf[m]['std']
    mdf = mdf.drop([m], axis=1)
mdf = mdf.droplevel(level=1, axis=1)

mdf['counts'] = counts['counts']
mdf = mdf.sort_values(by='mean_' + metric)

print(mdf.to_string())

# sort mdf by lr
mdf = mdf.sort_values(by='lr')

colors = {'findLSC_radius': [0.43365406, 0.83304796, 0.58958684], '': [0.24995383, 0.49626022, 0.35960801],
          'findLSC': [0.74880857, 0.9167003, 0.50021289], 'findLSC_supnpsd2': [0.69663182, 0.25710645, 0.19346206],
          'findLSC_supsubnpsd': [0.2225346, 0.06820208, 0.9836983], 'heinit': [0.96937357, 0.28256986, 0.26486611]}

if lrs_plot:
    from matplotlib.lines import Line2D

    comments = mdf['comments'].unique()
    activations = sorted(mdf['act'].unique())
    datasets = sorted(mdf['dataset'].unique())
    comments = sorted(mdf['comments'].unique())
    comments = ['', 'heinit', 'findLSC', 'findLSC_radius', 'findLSC_supnpsd2', 'findLSC_supsubnpsd']

    print(comments)

    # figsize=(4, 2)
    fig, axs = plt.subplots(
        len(datasets), len(activations), figsize=(5, 3), sharey='row',
        gridspec_kw={'wspace': .1, 'hspace': .1},
    )

    if len(datasets) == 1:
        axs = np.array([axs])
    if len(activations) == 1:
        axs = np.array([axs]).T


    def clean_comments(c):
        if c == 'findLSC':
            return 'sub ($L_2$)'
        if 'radius' in c:
            return r'sub ($\rho$)'
        if c == '':
            return 'Glorot'
        if c == 'heinit':
            return 'He'
        c = c.replace('findLSC_', '')
        c = c.replace('npsd', '')
        c = c.replace('2', '')
        return c


    for i, dataset in enumerate(datasets):
        ddf = mdf[mdf['dataset'] == dataset]
        for j, a in enumerate(activations):
            adf = ddf[ddf['act'] == a]
            axs[0, j].set_title(a, weight='bold')
            for c in comments:
                idf = adf[adf['comments'] == c]
                ys = idf['mean_' + metric].values
                yerrs = idf['std_' + metric].values
                xs = idf['lr'].values
                axs[i, j].plot(xs, ys, color=colors[c], label=clean_comments(c))
                # axs[i, j].fill_between(xs, ys - yerrs / 2, ys + yerrs / 2, alpha=0.5, color=colors[c])

            if not i == len(datasets) - 1:
                axs[i, j].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            if not j == 0:
                axs[i, j].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

                # x axis log scale
    axs[-1, -1].set_xlabel('Learning rate')
    axs[0, 0].set_ylabel('Accuracy')

    legend_elements = [Line2D([0], [0], color=colors[n], lw=4, label=clean_comments(n))
                       for n in comments]
    plt.legend(ncol=3, handles=legend_elements, loc='lower center', bbox_to_anchor=(-.1, -1.))

    # add a vertical text to the plot, to indicate the dataset, one for each row
    for i, dataset in enumerate(datasets):
        fig.text(-0.03, 0.7 - i * .45, dataset, va='center', rotation='vertical', weight='bold')

    # plt.legend()

    for ax in axs.reshape(-1):
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)
        ax.locator_params(axis='y', nbins=5)
        ax.set_xscale('log')

    plt.show()
    plot_filename = f'experiments/ffn_relu.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')

if plot_losses:
    df = odf
    # df = df[df['dataset'] == 'cifar100']
    # df = df[df['comments'].str.contains('supsubnpsd')]

    activations = sorted(df['activation'].unique())

    fig, axs = plt.subplots(1, len(activations), figsize=(6, 3))
    metric = 'LSC_norms list'  # 'val_acc list' 'loss list' LSC_norms
    # print([c for c in df.columns if 'list' in c])

    for i, a in enumerate(activations):
        adf = df[df['activation'] == a]
        axs[i].set_title(a)

        for _, row in adf.iterrows():
            c = row['comments']
            axs[i].plot(row[metric], color=colors[c])

    plt.legend()
    plt.show()

print('Maximal time elapsed is: {}'.format(df['time_elapsed'].max()))

if remove_incomplete:
    import shutil

    # df = odf
    ids = [
        # 'findLSC_radius_supn'
    ]
    rdfs = []
    for c in ids:
        rdf = df[df['comments'].str.contains(c)]
        rdfs.append(rdf)

    # from LSC_norms final column, select those that are epsilon away from 1
    epsilon = 0.09
    rdf = df[abs(df['LSC_norms f'] - 1) > epsilon]
    print(rdf.to_string())
    print(rdf.shape, odf.shape)

    # rdfs.append(rdf)

    # remove one seed from those that have more than 4 seeds
    brdf = mdf[mdf['counts'] > 4]
    # print(rdf.to_string())

    for _, row in brdf.iterrows():
        # print('-' * 80)
        srdf = df[
            (df['lr'] == row['lr'])
            & (df['comments'] == row['comments'])
            & (df['activation'] == row['activation'])]

        # no duplicates
        gsrdf = srdf.drop_duplicates(subset=['seed'])

        # remainder
        rdf = srdf[~srdf.apply(tuple, 1).isin(gsrdf.apply(tuple, 1))]
        # rdfs.append(rdf)

    rdf= df[df['val_loss min'].isna()]
    print(rdf.to_string())
    rdfs.append(rdf)

    if truely_remove:
        for rdf in rdfs:
            print(rdf['comments'])
            paths = rdf['path'].values
            for p in paths:
                print('Removing {}'.format(p))
                exps_path = p
                gexp_path = os.path.join(GEXPERIMENTS[0], os.path.split(p)[1] + '.zip')
                print(exps_path)
                print(gexp_path)
                print(os.path.exists(exps_path), os.path.exists(gexp_path))
                if os.path.exists(exps_path):
                    shutil.rmtree(exps_path)
                if os.path.exists(gexp_path):
                    os.remove(gexp_path)

if missing_exps:
    # columns of interest
    coi = ['seed', 'activation', 'lr', 'comments', 'dataset', 'epochs', 'steps_per_epoch']
    import pandas as pd

    sdf = df

    sdf.drop([c for c in sdf.columns if c not in coi], axis=1, inplace=True)

    experiments = []
    experiment = {
        'comments': [
            '', 'findLSC', 'findLSC_supsubnpsd', 'findLSC_supnpsd2', 'findLSC_radius', 'heinit',
        ],
        'activation': ['sin', 'relu', 'cos'], 'dataset': ['cifar10', 'cifar100'],
        'layers': [30], 'width': [128], 'lr': [1e-3, 3.16e-4, 1e-4, 3.16e-5, 1e-5],
        'epochs': [50], 'steps_per_epoch': [-1], 'pretrain_epochs': [30], 'seed': list(range(4)),
    }
    experiments.append(experiment)

    ds = dict2iter(experiments)
    print(ds[0])
    complete_missing_exps(sdf, ds, coi)
