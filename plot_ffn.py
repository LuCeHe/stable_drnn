from pyaromatics.stay_organized.submit_jobs import dict2iter
from alif_sg.tools.plot_tools import lsc_colors, lsc_clean_comments, clean_title

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pyaromatics.stay_organized.pandardize import experiments_to_pandas, complete_missing_exps
from pyaromatics.stay_organized.standardize_strings import shorten_losses

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
EXPERIMENTS = r'D:\work\alif_sg\experiments'


plot_norms_evol = False
plot_norms_evol_1 = False
lrs_plot = False
lrs_plot_2 = False
bar_plot = False
plot_losses = False

missing_exps = True
remove_incomplete = False
truely_remove = False

expsid = 'effnet'  # effnet als ffnandcnns transf

if expsid == 'ffnandcnns':
    GEXPERIMENTS = [r'D:\work\alif_sg\good_experiments\2022-12-16--ffn']
elif expsid == 'effnet':
    GEXPERIMENTS = [r'D:\work\alif_sg\good_experiments\2023-01-01--effnet']

metric = 'val_acc M'  # 'val_acc M'   'val_loss m' test_acc
h5path = os.path.join(EXPERIMENTS, f'summary_{expsid}.h5')
force_keep_column = ['LSC_norms list', 'val_sparse_categorical_accuracy list', 'val_loss list',
                     'encoder_norm list', 'decoder_norm list']
columns_to_remove = [
    '_var', '_mean', 'sparse_categorical_crossentropy', 'bpc', 'artifacts',
    'experiment_dependencies', 'experiment_sources', 'experiment_repositories', 'host_os',

    # 'sparse_categorical_accuracy', 'loss',
    'LSC_losses', 'rec_norms', 'fail_trace', 'list']
# force_keep_column = [
#     'LSC_norms list', 'batch ',
#     'val_sparse_mode_accuracy list', 'val_perplexity list',
#     'v_sparse_mode_accuracy list', 'v_perplexity list',
#     't_sparse_mode_accuracy list', 't_perplexity list',
# ]
force_keep_column = ['LSC_norms list', 'val_sparse_categorical_accuracy list', 'val_loss list',
                     'encoder_norm list', 'decoder_norm list']
# force_keep_column = []

df = experiments_to_pandas(
    h5path=h5path, zips_folder=GEXPERIMENTS, unzips_folder=EXPERIMENTS, experiments_identifier=expsid,
    exclude_files=['cout.txt'], exclude_columns=columns_to_remove, check_for_new=True,
    force_keep_column=force_keep_column
)

if expsid == 'effnet':
    from alif_sg.tools.config import default_eff_lr

    df = df[df['comments'].str.contains('newarch')]
    df['comments'] = df['comments'].str.replace('newarch_', '')
    df['comments'] = df['comments'].str.replace('pretrained_', '')
    df['lr'] = df.apply(
        lambda x: default_eff_lr(x['activation'], x['lr'], x['batch_normalization']) if x['lr'] == -1 else x['lr'],
        axis=1)

# select only rows with width 10
df['time_elapsed'] = pd.to_timedelta(df['time_elapsed'], unit='s')
df['val_loss argm'] = df['val_loss list'].apply(np.argmin)
df['conveps'] = df['val_loss len'] - df['val_loss argm']
# df['val_acc argM'] = df['val_sparse_categorical_accuracy list'].apply(np.argmax)
# df['conveps'] = df['val_sparse_categorical_accuracy len'] - df['val_acc argM']

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

metrics_oi = ['val_acc M', 'test_acc M', 'val_loss m', 'test_loss m', 'LSC_norms i', 'LSC_norms f', 'conveps']
stats_oi = ['mean', 'std']  # ['mean', 'std']
group_cols = ['lr', 'comments', 'act', 'dataset']
bn = 1
if 'ffnandcnns' in expsid:
    plot_only = [
        'act', 'pre_eps', 'eps', 'dataset',
        # 'spe', 'width', 'depth',
        'seed', 'lr', 'comments',
        'val_acc M', 'val_loss m', 'test_acc M', 'test_loss m',
        # 'acc max','loss min',
        'LSC i', 'LSC f', 'LSC a',
        'conveps',
        'ep M', 'time_elapsed', 'hostname', 'path',
    ]

elif 'effnet' in expsid:
    metrics_oi = ['val_acc M', 'test_acc M', 'LSC i', 'LSC f']
    plot_only = [
        'act', 'eps', 'dataset', 'batch_normalization',
        'seed', 'lr', 'comments',
        'val_acc M', 'val_loss m', 'test_acc M', 'test_loss m',
        'LSC i', 'LSC f', 'LSC a',
        'ep M', 'time_elapsed', 'hostname', 'path',
    ]
    group_cols = ['lr', 'comments', 'act', 'dataset', 'batch_normalization']
    stats_oi = ['mean']

elif 'transf' in expsid:
    plot_only = [
        'act', 'eps', 'dataset',
        'seed', 'lr', 'comments', 'batch_size',
        'val_ppl m',
        'LSC i', 'LSC f', 'LSC a',
        'encoder_norm i', 'encoder_norm f',
        'decoder_norm i', 'decoder_norm f',
        'ep M', 'time_elapsed', 'hostname', 'path',
        'dataset',
    ]
    metrics_oi = [
        'val_ppl m',
        'LSC f',
        'encoder_norm f', 'decoder_norm f',
        'encoder_norm i', 'decoder_norm i',
    ]
    group_cols = ['lr', 'comments', 'act']
    df['dataset'] = 'ende'
    metric = 'val_ppl m'  # 'val_acc M'   'val_loss min'
    stats_oi = ['mean']

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

    if 'ffnandcnns' in expsid:
        activations = ['relu', 'sin', 'cos']
        ncol = 3
        bbox_to_anchor = (-.7, -1.)

        mdf = mdf[~mdf['comments'].eq('findLSC')]
    elif 'effnet' in expsid or 'transf' in expsid:
        ncol = 4
        bbox_to_anchor = (-.3, -.4)

    activations = sorted(mdf['act'].unique())
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
                axs[i, j].plot(xs, ys, color=lsc_colors(c), label=lsc_clean_comments(c), linewidth=linewidth)
                axs[i, j].fill_between(xs, ys - yerrs / 2, ys + yerrs / 2, alpha=0.5, color=lsc_colors(c))

            if not i == len(datasets) - 1:
                axs[i, j].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

            if not j == 0:
                axs[i, j].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

                # x axis log scale
    axs[-1, -1].set_xlabel('Learning rate', fontsize=fontsize)
    axs[0, 0].set_ylabel(clean_title(metric), fontsize=fontsize)

    comments = ['', 'heinit', 'findLSC_radius', 'findLSC_supnpsd2', 'findLSC_supsubnpsd', ]

    legend_elements = [Line2D([0], [0], color=lsc_colors(n), lw=4, label=lsc_clean_comments(n))
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
    plot_filename = os.path.join(EXPERIMENTS, f'{expsid}_relu.pdf')
    fig.savefig(plot_filename, bbox_inches='tight')

if lrs_plot_2:
    from matplotlib.lines import Line2D

    mdf = mdf[mdf['comments'].str.contains('adabelief')]
    mdf['comments'] = mdf['comments'].str.replace('_adabelief', '')
    mdf = mdf.sort_values(by='lr')
    print(mdf.to_string())

    if 'ffnandcnns' in expsid:
        activations = ['relu', 'sin', 'cos']
        ncol = 5
        bbox_to_anchor = (-3.1, -.8)

        mdf = mdf[~mdf['comments'].eq('findLSC')]
    elif 'effnet' in expsid or 'transf' in expsid:
        ncol = 4
        bbox_to_anchor = (-.3, -.4)

    activations = sorted(mdf['act'].unique())
    datasets = sorted(mdf['dataset'].unique())
    comments = sorted(mdf['comments'].unique())

    # comments = ['', 'heinit', 'findLSC', 'findLSC_radius', 'findLSC_supnpsd2', 'findLSC_supsubnpsd']

    print(comments)

    fig, axs = plt.subplots(
        1, len(datasets) * len(activations), figsize=(8, 1.5), sharey='row',
        gridspec_kw={'wspace': .4, 'hspace': .1},
    )

    fontsize = 12
    linewidth = 1

    if len(datasets) == 1:
        axs = np.array([axs])
    if len(activations) == 1:
        axs = np.array([axs]).T

    import itertools

    # itertool all combinations of datasets and activations
    datacts = itertools.product(datasets, activations)
    print(datacts)

    for i, (dataset, a) in enumerate(datacts):
        ddf = mdf[mdf['dataset'] == dataset]
        # for j, a in enumerate(activations):
        adf = ddf[ddf['act'] == a]
        title = a if not 'relu' in a else 'ReLU'
        axs[i].set_title(title, fontsize=fontsize)
        for c in comments:
            idf = adf[adf['comments'] == c]
            ys = idf['mean_' + metric].values
            yerrs = idf['std_' + metric].values
            xs = idf['lr'].values
            axs[i].plot(xs, ys, color=lsc_colors(c), label=lsc_clean_comments(c), linewidth=linewidth)
            axs[i].fill_between(xs, ys - yerrs / 2, ys + yerrs / 2, alpha=0.5, color=lsc_colors(c))

        # if not i == len(datasets) - 1:
        #     axs[i, j].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        #
        if not i == 0:
            axs[i].tick_params(axis='y', which='both', left=False, right=False, labelleft=False)

            # x axis log scale
    axs[-1].set_xlabel('Learning rate', fontsize=fontsize)
    axs[0].set_ylabel(clean_title(metric), fontsize=fontsize)

    comments = ['', 'heinit', 'findLSC_radius', 'findLSC_supnpsd2', 'findLSC_supsubnpsd', ]

    legend_elements = [Line2D([0], [0], color=lsc_colors(n), lw=4, label=lsc_clean_comments(n))
                       for n in comments]
    plt.legend(ncol=ncol, handles=legend_elements, loc='lower center', bbox_to_anchor=bbox_to_anchor)

    if len(datasets) > 1:
        # add a vertical text to the plot, to indicate the dataset, one for each row
        for i, dataset in enumerate(datasets):
            fig.text(0.265 + i * .456, 1.15, dataset, va='center', weight='bold', fontsize=1.1 * fontsize)

    else:
        fig.text(-0.03, 0.5, datasets[0], va='center', rotation='vertical', weight='bold')

    # plt.legend()

    for i, ax in enumerate(axs.reshape(-1)):
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)
        ax.locator_params(axis='y', nbins=5)
        ax.set_xscale('log')

        pos = ax.get_position()
        dx = 0 if i < 3 else .06
        ax.set_position([pos.x0 + dx, pos.y0, pos.width, pos.height])

    plot_filename = os.path.join(EXPERIMENTS, f'{expsid}_relu.pdf')
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.show()

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
            print(a, c)

            # select best lr
            # fixme:
            if bn == 1:
                lrdf = adf[adf['comments'] == 'heinit']
                lrdf = lrdf[lrdf['mean_' + metric] == lrdf['mean_' + metric].max()]
                lr = lrdf['lr'].values[0]
                idf = adf[adf['lr'].astype(float).eq(lr)]
                iidf = idf[idf['comments'] == c]
            else:
                iidf = adf[adf['comments'] == c]
                iidf = iidf[iidf['mean_val_acc M'] == iidf['mean_val_acc M'].max()]

            print(iidf.to_string())

            data.append(iidf['mean_' + metric].values[0])
            error.append(iidf['std_' + metric].values[0])
        axs[0].bar(X + i * w, data, yerr=error, width=w, color=lsc_colors(c), label=lsc_clean_comments(c))

    legend_elements = [Line2D([0], [0], color=lsc_colors(n), lw=4, label=lsc_clean_comments(n))
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
    df['comments'] = df['comments'].str.replace('pretrained_', '')

    activations = sorted(df['act'].unique())

    fig, axs = plt.subplots(1, len(activations), figsize=(6, 3))
    metric = 'decoder_norm list'  # 'val_acc list' 'loss list' LSC_norms encoder_norm decoder_norm

    for i, a in enumerate(activations):
        adf = df[df['act'] == a]
        axs[i].set_title(a)

        for _, row in adf.iterrows():
            c = row['comments'].replace('meanaxis_', '')
            c = c.replace('_meanaxis', '')
            c = c.replace('_truersplit', '')
            c = c.replace('sameemb_', '')
            c = c.replace('chunked_', '')
            c = c.replace('_deslice', '')

            axs[i].plot(row[metric], color=lsc_color(c), label=c)

    plt.legend()
    plt.show()

print('Maximal time elapsed is: {}'.format(df['time_elapsed'].max()))
print('Minimal time elapsed is: {}'.format(df['time_elapsed'].min()))

if remove_incomplete:
    rdfs = []
    import shutil


    if expsid == 'ffnandcnns':
        df = df[df['comments'].str.contains('adabelief')]
    elif expsid == 'effnet':
        df = df[df['comments'].str.contains('onlypretrain')]
    else:
        raise NotImplementedError

    # from LSC_norms final column, select those that are epsilon away from 1
    epsilon = 0.02
    # epsilon = 2.
    print(df.to_string())
    print('\n\nRemove if too far from target radius')
    # rdf = df[abs(df['LSC f'] - 1) > epsilon]
    df['vs_epsilon'] = ((abs(df['LSC a'] - 1) > epsilon)
                        & df['comments'].str.contains('onlyloadpretrained')) \
                       | ((abs(df['LSC f'] - 1) > epsilon)
                          & df['comments'].str.contains('onlypretrain'))

    # rdf = df[
    #     df['comments'].str.contains('findLSC')
    #     & df['vs_epsilon']
    #     ]
    rdf = df[df['vs_epsilon']
        ]

    print(rdf.to_string())
    print(rdf.shape, odf.shape, df.shape, df[df['comments'].str.contains('pretrain')].shape)

    rdfs.append(rdf)

    print('Remove onlypretrain of the onlyloadpretrained that did not satisfy the lsc')
    nrdf = rdf[rdf['comments'].str.contains('onlyloadpretrained')].copy()
    # rdf['comments'] = rdf['comments'].str.replace('onlyloadpretrained', 'onlypretrain')

    listem = []
    for _, row in nrdf.iterrows():
        act = row['act']
        dataset = row['dataset']
        seed = row['seed']
        comments = row['comments'].replace('onlyloadpretrained', 'onlypretrain')
        # print(net, task, seed, stack, comments)
        rdf = df[
            df['act'].eq(act)
            & df['dataset'].eq(dataset)
            & df['seed'].eq(seed)
            & df['comments'].eq(comments)
            ]
        listem.append(rdf)
        rdfs.append(rdf)

    if len(listem) > 0:
        listem = pd.concat(listem)

        print(listem.to_string())
        print(listem.shape, df.shape)

    print('Remove if it didnt converge')
    # rdf = df[df['conveps'] < 8]
    #
    # print(rdf.to_string())
    # print(rdf.shape, df.shape)
    # rdfs.append(rdf)


    print('Remove repeated experiments')
    brdf = mdf[mdf['counts'] > 4]
    for _, row in brdf.iterrows():
        srdf = df[
            (df['lr'] == row['lr'])
            & (df['comments'] == row['comments'])
            & (df['act'] == row['act'])
            & (df['dataset'] == row['dataset'])
            ]

        # order wrt path column
        srdf = srdf.sort_values(by=['path'], ascending=False)

        # no duplicates
        gsrdf = srdf.drop_duplicates(subset=['seed'])

        # remainder
        rdf = srdf[~srdf.apply(tuple, 1).isin(gsrdf.apply(tuple, 1))]
        print(srdf.to_string())
        print(rdf.to_string())
        print(rdf.shape, odf.shape)

    allrdfs = pd.concat(rdfs)
    allrdfs = allrdfs.drop_duplicates()
    print(f'Remove {allrdfs.shape} of {df.shape}')

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
        os.remove(h5path)

if missing_exps:
    # columns of interest

    # print(sdf.to_string())
    experiments = []
    if 'ffnandcnns' in expsid:
        # coi = ['seed', 'act', 'lr', 'comments', 'dataset', 'eps', 'spe']
        coi = ['seed', 'act', 'comments', 'dataset', 'eps', 'spe']
        flags = ['_onlypretrain', '_onlyloadpretrained']
        flags = ['_onlypretrain']
        # all_comments = ['', 'findLSC_supsubnpsd', 'findLSC_supnpsd2', 'findLSC_radius', 'heinit', ]
        comments = ['findLSC_supsubnpsd', 'findLSC_supnpsd2', 'findLSC_radius', ]
        all_comments = lambda x: [c + f'_adabelief{x}' for c in comments]

        experiment = lambda x: {
            'comments': all_comments(x),
            'act': ['sin', 'relu', 'cos'], 'dataset': ['cifar10', 'cifar100'],
            'layers': [30], 'width': [128], 'lr': [1e-3, 3.16e-4, 1e-4, 3.16e-5, 1e-5],
            'eps': [50], 'spe': [-1], 'pretrain_epochs': [30], 'seed': list(range(4)),
        }
        exps = lambda x: [experiment(x)]

    elif 'effnet' in expsid:
        coi = ['seed', 'act', 'lr', 'comments', 'eps', 'spe', 'batch_normalization', 'dataset']
        flags = ['_onlypretrain']
        incomplete_comments = 'newarch_findLSC_lscvar'
        all_comments = [incomplete_comments + f'_onlypretrain']
        seeds = list(range(4))
        experiment = lambda x: {
            'seed': seeds, 'comments': all_comments, 'eps': [150], 'spe': [-1],
            'lr': [-1], 'act': ['swish', 'relu', 'tanh'], 'batch_normalization': [1],
            'dataset': ['cifar10', 'cifar100', 'mnist'],
        }
        exps = lambda x: [experiment(x)]

    sdf = odf
    sdf['comments'] = sdf['comments'].apply(
        lambda x: x + '_onlypretrain' if 'onlypretrain' not in x and 'onlyloadpretrained' not in x
        else x
    )
    print(sdf.head().to_string())
    sdf.drop([c for c in sdf.columns if c not in coi], axis=1, inplace=True)

    for add_flag in flags:
        # add string pretrain to comments if it's not there
        ds = dict2iter(exps(add_flag))

        df, experiments = complete_missing_exps(sdf, ds, coi)
        new_exps = []
        for e in experiments:
            ne = e.copy()
            ne.update({'pretrain_epochs': [100]})
            ne.update({'epochs': [int(e['eps'][0])]})
            ne.update({'steps_per_epoch': [int(e['spe'][0])]})
            ne.update({'activation': e['act']})
            # ne.update({'comments': [e['comments'][0] + '_onlypretrain']})
            del ne['act'], ne['eps'], ne['spe']
            # print(ne)
            new_exps.append(ne)

        print(f'experiments{add_flag} = ', new_exps)
        print('# ', len(new_exps))
