import time

from GenericTools.stay_organized.submit_jobs import dict2iter
from alif_sg.tools.plot_tools import lsc_colors, lsc_clean_comments

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from GenericTools.stay_organized.pandardize import experiments_to_pandas, complete_missing_exps
from GenericTools.stay_organized.standardize_strings import shorten_losses


def clean_title(c):
    t = ''
    if 'ppl' in c:
        t = 'Perplexity'
    if 'acc' in c:
        t = 'Accuracy'
    if 'loss' in c:
        t = 'Loss'

    if 'val' in c:
        t = 'Validation ' + t
    return t


FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
EXPERIMENTS = r'D:\work\alif_sg\experiments'
GEXPERIMENTS = [
    # os.path.join(CDIR, 'good_experiments', '2022-12-16--ffn'),
    # os.path.join(CDIR, 'good_experiments'),
    # r'D:\work\alif_sg\good_experiments\2022-12-16--ffn'
    r'D:\work\alif_sg\good_experiments\2023-01-01--effnet',
    r'D:\work\alif_sg\good_experiments\2023-01-15--transf',
]

plot_norms_evol = False
plot_norms_evol_1 = False
lrs_plot = False
bar_plot = False
plot_losses = True
missing_exps = False
remove_incomplete = False
truely_remove = False

metric = 'val_acc M'  # 'val_acc M'   'val_loss m'
expsid = 'effnet'  # effnet als ffnandcnns transf
h5path = os.path.join(EXPERIMENTS, f'summary_{expsid}.h5')
force_keep_column = ['LSC_norms list', 'val_sparse_categorical_accuracy list', 'val_loss list']

df = experiments_to_pandas(
    h5path=h5path, zips_folder=GEXPERIMENTS, unzips_folder=EXPERIMENTS, experiments_identifier=expsid,
    exclude_files=['cout.txt'], exclude_columns=['_mean ', '_var ', ' list'], check_for_new=True,
    force_keep_column=force_keep_column
)

if expsid == 'effnet':
    from alif_sg.tools.config import default_eff_lr

    df = df[df['comments'].str.contains('newarch')]
    df['comments'] = df['comments'].str.replace('newarch_', '')
    df['comments'] = df['comments'].str.replace('pretrained_', '')
    df['lr'] = df.apply(
        lambda x: default_eff_lr(x['activation'], x['lr'], x['batch_normalization']) if x['lr']==-1 else x['lr'],
        axis=1)

# select only rows with width 10
df['time_elapsed'] = pd.to_timedelta(df['time_elapsed'], unit='s')

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
odf = df.copy()
print(list(odf.columns))
df = df.rename(columns={'test_loss': 'test_loss m', 'test_acc': 'test_acc M'})

metrics_oi = ['val_acc M', 'test_acc M', 'val_loss m', 'test_loss m', 'LSC_norms i', 'LSC_norms f']
stats_oi = ['mean', 'std']  # ['mean', 'std']
group_cols = ['lr', 'comments', 'act', 'dataset']
if 'ffnandcnns' in expsid:
    plot_only = [
        'act', 'pre_eps', 'eps', 'dataset',
        # 'spe', 'width', 'depth',
        'seed', 'lr', 'comments',
        'val_acc M', 'val_loss m', 'test_acc M', 'test_loss m',
        # 'acc max','loss min',
        'LSC_norms i', 'LSC_norms f',
        'ep M', 'time_elapsed', 'hostname', 'path',
    ]
elif 'effnet' in expsid:
    plot_only = [
        'act', 'eps', 'dataset', 'batch_normalization',
        'seed', 'lr', 'comments',
        'val_acc M', 'val_loss m', 'test_acc M', 'test_loss m',
        'LSC_norms i', 'LSC_norms f',
        'ep M', 'time_elapsed', 'hostname', 'path',
    ]
    group_cols = ['lr', 'comments', 'act', 'dataset', 'batch_normalization']


elif 'transf' in expsid:
    plot_only = [
        'act', 'eps', 'dataset',
        'seed', 'lr', 'comments',
        'val_acc M', 'val_loss m',
        # 'test_acc M', 'test_loss m',
        'val_ppl M', 'val_ppl m',
        # 'test_ppl M', 'test_ppl m',
        # 'LSC_norms i', 'LSC_norms f',
        'ep M', 'time_elapsed', 'hostname', 'path',
    ]
    metrics_oi = [
        'val_acc M', 'val_loss m', 'val_ppl m',
        # 'LSC_norms i', 'LSC_norms f'
    ]
    # group_cols = ['lr', 'comments', 'act']
    df['dataset'] = 'ende'
    metric = 'val_ppl m'  # 'val_acc M'   'val_loss min'

df = df[plot_only]
df = df.sort_values(by=metric)
print(df.to_string())
df['comments'] = df['comments'].str.replace('_preprocessinput', '')

counts = df.groupby(group_cols).size().reset_index(name='counts')

metrics_oi = [shorten_losses(m) for m in metrics_oi]
mdf = df.groupby(
    group_cols, as_index=False
).agg({m: stats_oi for m in metrics_oi})

for m in metrics_oi:
    for s in stats_oi:
        mdf[f'{s}_{m}'] = mdf[m][s]
    mdf = mdf.drop([m], axis=1)
mdf = mdf.droplevel(level=1, axis=1)

mdf['counts'] = counts['counts']
mdf = mdf.sort_values(by='mean_' + metric)

print(mdf.to_string())

if 'effnet' in expsid:
    # remove string from column comments in the df
    bn = 0
    mdf = mdf[~mdf['comments'].eq('deslice_findLSC_truersplit_meanaxis')]
    no_LSC_string = 'deslice_' if bn == 1 else 'meanaxis_deslice_'
    mdf = mdf[
        mdf['comments'].eq(no_LSC_string)
        | (
                mdf['comments'].str.contains('findLSC')
                & mdf['comments'].str.contains('meanaxis')
        )
        ]
    mdf['comments'] = mdf['comments'].str.replace('meanaxis_', '')
    mdf = mdf[mdf['batch_normalization'].eq(bn)]
    mdf['comments'] = mdf['comments'].str.replace('deslice_', '')
    mdf['comments'] = mdf['comments'].str.replace('pretrained_', '')
    mdf['comments'] = mdf['comments'].str.replace('truersplit_', '')
    mdf['comments'] = mdf['comments'].str.replace('sameemb_', '')

    mdf['comments'] = mdf['comments'].replace(r'^\s*$', 'heinit', regex=True)



if lrs_plot:
    from matplotlib.lines import Line2D

    mdf = mdf.sort_values(by='lr')

    comments = mdf['comments'].unique()
    activations = sorted(mdf['act'].unique())
    if 'ffnandcnns' in expsid:
        activations = ['relu', 'sin', 'cos']
        ncol = 3
        bbox_to_anchor = (-.7, -1.)
    elif 'effnet' in expsid or 'transf' in expsid:
        ncol = 4
        bbox_to_anchor = (-.3, -.4)

    datasets = sorted(mdf['dataset'].unique())
    comments = sorted(mdf['comments'].unique())

    # comments = ['', 'heinit', 'findLSC', 'findLSC_radius', 'findLSC_supnpsd2', 'findLSC_supsubnpsd']

    print(comments)

    fig, axs = plt.subplots(
        len(datasets), len(activations), figsize=(5, 3), sharey='row',
        gridspec_kw={'wspace': .3, 'hspace': .1},
    )
    fontsize = 12
    linewidth = 1

    if len(datasets) == 1:
        axs = np.array([axs])
    if len(activations) == 1:
        axs = np.array([axs]).T

    for i, dataset in enumerate(datasets):
        ddf = mdf[mdf['dataset'] == dataset]
        for j, a in enumerate(activations):
            adf = ddf[ddf['act'] == a]
            title = a if not 'relu' in a else 'ReLU'
            axs[0, j].set_title(title, weight='bold', fontsize=fontsize)
            for c in comments:
                idf = adf[adf['comments'] == c]
                ys = idf['mean_' + metric].values
                yerrs = idf['std_' + metric].values
                xs = idf['lr'].values
                axs[i, j].plot(xs, ys, color=lsc_colors[c], label=lsc_clean_comments(c), linewidth=linewidth)
                axs[i, j].fill_between(xs, ys - yerrs / 2, ys + yerrs / 2, alpha=0.5, color=lsc_colors[c])

            if not i == len(datasets) - 1:
                axs[i, j].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            if not j == 0:
                axs[i, j].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

                # x axis log scale
    axs[-1, -1].set_xlabel('Learning rate', fontsize=fontsize)
    axs[0, 0].set_ylabel(clean_title(metric), fontsize=fontsize)

    legend_elements = [Line2D([0], [0], color=lsc_colors[n], lw=4, label=lsc_clean_comments(n))
                       for n in comments]
    plt.legend(ncol=ncol, handles=legend_elements, loc='lower center', bbox_to_anchor=bbox_to_anchor)

    if len(datasets) > 1:
        # add a vertical text to the plot, to indicate the dataset, one for each row
        for i, dataset in enumerate(datasets):
            fig.text(-0.03, 0.7 - i * .45, dataset, va='center', rotation='vertical', weight='bold')

    else:
        fig.text(-0.03, 0.5, datasets[0], va='center', rotation='vertical', weight='bold')

    # plt.legend()

    for ax in axs.reshape(-1):
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)
        ax.locator_params(axis='y', nbins=5)
        ax.set_xscale('log')

    plt.show()
    plot_filename = f'experiments/{expsid}_relu.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')

if bar_plot:
    from matplotlib.lines import Line2D

    mdf = mdf.sort_values(by='lr')

    comments = mdf['comments'].unique()
    activations = sorted(mdf['act'].unique())

    datasets = sorted(mdf['dataset'].unique())
    comments = sorted(mdf['comments'].unique())

    # comments = ['', 'heinit', 'findLSC', 'findLSC_radius', 'findLSC_supnpsd2', 'findLSC_supsubnpsd']

    print(comments)

    # figsize=(4, 2)
    fig, axs = plt.subplots(
        1, 1, figsize=(5, 3), sharey='row',
        gridspec_kw={'wspace': .3, 'hspace': .1},
    )
    if len(datasets) == 1:
        axs = np.array([axs])
    if len(activations) == 1:
        axs = np.array([axs]).T

    X = np.arange(len(activations))
    w = 1 / (len(comments) + 1)

    for i, c in enumerate(comments):
        data = []
        error = []

        for a in activations:
            adf = mdf[mdf['act'] == a]
            print(a)

            # select best lr
            #fixme:
            if bn ==1:
                lrdf = adf[adf['comments'] == 'heinit']
                lrdf = lrdf[lrdf['mean_' + metric] == lrdf['mean_' + metric].max()]
                lr = lrdf['lr'].values[0]
                idf = adf[adf['lr'].astype(float).eq(lr)]
                iidf = idf[idf['comments'] == c]
            else:
                iidf = adf[adf['comments'] == c]
            print(iidf.to_string())

            data.append(iidf['mean_' + metric].values[0])
            error.append(iidf['std_' + metric].values[0])
        axs[0].bar(X + i * w, data, yerr=error, width=w, color=lsc_colors[c], label=lsc_clean_comments(c))

    legend_elements = [Line2D([0], [0], color=lsc_colors[n], lw=4, label=lsc_clean_comments(n))
                       for n in comments]
    plt.legend(ncol=4, handles=legend_elements, loc='lower center')

    for ax in axs.reshape(-1):
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)
        ax.locator_params(axis='y', nbins=5)

    plot_filename = f'experiments/{expsid}_bars.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')

    plt.show()

if plot_losses:
    df = odf

    df['comments'] = df['comments'].str.replace('deslice_', '')
    df['comments'] = df['comments'].str.replace('_preprocessinput', '')

    activations = sorted(df['act'].unique())

    fig, axs = plt.subplots(1, len(activations), figsize=(6, 3))
    metric = 'LSC_norms list'  # 'val_acc list' 'loss list' LSC_norms

    for i, a in enumerate(activations):
        adf = df[df['act'] == a]
        axs[i].set_title(a)

        for _, row in adf.iterrows():
            c = row['comments'].replace('meanaxis_', '')
            c = c.replace('_meanaxis', '')
            c = c.replace('_truersplit', '')
            # print(c, lsc_colors[c], row[metric])
            # if 'truersplit' in c:
            #     print(row[metric])
            axs[i].plot(row[metric], color=lsc_colors[c])

    plt.legend()
    plt.show()

print('Maximal time elapsed is: {}'.format(df['time_elapsed'].max()))
print('Minimal time elapsed is: {}'.format(df['time_elapsed'].min()))

if remove_incomplete:
    import shutil

    # df = odf
    ids = [
        'findLSC_supn',
        'findLSC_logradius',
        'findLSC'
    ]
    rdfs = []
    for c in ids:
        rdf = df[df['comments'].str.contains(c)]
        # print(rdf.shape)
        # rdfs.append(rdf)


    rdf = df[
        df['batch_normalization'].eq(0)
        & df['act'].eq('tanh')
        & df['comments'].str.contains('findLSC')
    ]
    rdfs.append(rdf)

    print(rdf.to_string())
    print(rdf.shape, odf.shape)


    # from LSC_norms final column, select those that are epsilon away from 1
    epsilon = 0.09
    epsilon = 2.
    rdf = df[abs(df['LSC_norms f'] - 1) > epsilon]
    # print(rdf.to_string())
    # print(rdf.shape, odf.shape)

    # rdfs.append(rdf)

    # remove one seed from those that have more than 4 seeds
    brdf = mdf[mdf['counts'] > 4]
    # print(rdf.to_string())

    for _, row in brdf.iterrows():
        # print('-' * 80)
        srdf = df[
            (df['lr'] == row['lr'])
            & (df['comments'] == row['comments'])
            & (df['act'] == row['act'])
            & (df['dataset'] == row['dataset'])
            ]

        # no duplicates
        gsrdf = srdf.drop_duplicates(subset=['seed'])

        # remainder
        rdf = srdf[~srdf.apply(tuple, 1).isin(gsrdf.apply(tuple, 1))]
        print(rdf.shape, odf.shape)
        # print(rdf.to_string())
        # rdfs.append(rdf)


    rdf = df[df['eps'] < 50]
    # rdfs.append(rdf)

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

    # print(sdf.to_string())
    experiments = []
    if 'ffnandcnns' in expsid:
        coi = ['seed', 'activation', 'lr', 'comments', 'dataset', 'epochs', 'steps_per_epoch']

        experiment = {
            'comments': [
                '', 'findLSC', 'findLSC_supsubnpsd', 'findLSC_supnpsd2', 'findLSC_radius', 'heinit',
            ],
            'activation': ['sin', 'relu', 'cos'], 'dataset': ['cifar10', 'cifar100'],
            'layers': [30], 'width': [128], 'lr': [1e-3, 3.16e-4, 1e-4, 3.16e-5, 1e-5],
            'epochs': [50], 'steps_per_epoch': [-1], 'pretrain_epochs': [30], 'seed': list(range(4)),
        }
        experiments.append(experiment)

    elif 'effnet' in expsid:
        coi = ['seed', 'activation', 'lr', 'comments', 'epochs', 'steps_per_epoch']

        incomplete_comments = 'deslice_'
        all_comments = [
            incomplete_comments,
            incomplete_comments + f'findLSC',
            incomplete_comments + f'findLSC_supsubnpsd',
            incomplete_comments + f'findLSC_radius',
        ]

        all_comments = [c + '_preprocessinput' for c in all_comments]

        experiment = {
            'seed': list(range(4)), 'comments': all_comments, 'epochs': [100], 'steps_per_epoch': [-1],
            'lr': [1e-3, 3.16e-4, 1e-4, 3.16e-5, 1e-5], 'activation': ['swish', 'relu'],
        }
        experiments.append(experiment)

    ds = dict2iter(experiments)
    sdf = odf
    sdf.drop([c for c in sdf.columns if c not in coi], axis=1, inplace=True)

    complete_missing_exps(sdf, ds, coi)
