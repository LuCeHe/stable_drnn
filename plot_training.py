import os, json

from alif_sg.tools.admin_model_removal import remove_pretrained_extra

from pyaromatics.stay_organized.submit_jobs import dict2iter
from tqdm import tqdm
import pandas as pd
import matplotlib as mpl

import pickle
from matplotlib.lines import Line2D
from pyaromatics.stay_organized.mpl_tools import load_plot_settings
from pyaromatics.stay_organized.pandardize import experiments_to_pandas, complete_missing_exps
from pyaromatics.stay_organized.standardize_strings import shorten_losses
from pyaromatics.stay_organized.utils import str2val
# from alif_sg.neural_models.recLSC import load_LSC_model
from alif_sg.tools.plot_tools import *

mpl, pd = load_plot_settings(mpl=mpl, pd=pd)

import matplotlib.pyplot as plt
from datetime import timedelta, datetime

from pyaromatics.stay_organized.plot_tricks import large_num_to_reasonable_string

FMT = '%Y-%m-%dT%H:%M:%S'
from pyaromatics.stay_organized.unzip import unzip_good_exps

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
EXPERIMENTS = r'D:\work\alif_sg\experiments'
GEXPERIMENTS = [
    # os.path.join(CDIR, 'good_experiments'),
    # os.path.join(CDIR, 'good_experiments', '2022-11-07--complete_set_of_exps'),
    # r'D:\work\alif_sg\experiments',
    # r'D:\work\alif_sg\good_experiments\2022-12-21--rnn',
    # r'D:\work\alif_sg\good_experiments\2023-01-20--rnn-v2',
    # r'D:\work\alif_sg\good_experiments\2023-09-01--rnn-lru-first',
    # r'D:\work\alif_sg\good_experiments\2023-10-10--s5lru',
    r'D:\work\alif_sg\good_experiments\2023-11-01--ptblif',
    # r'D:\work\alif_sg\good_experiments\2023-11-10--decolletc',
]

expsid = 'fluctuations'  # effnet als ffnandcnns s5lru mnl fluctuations _decolle
h5path = os.path.join(EXPERIMENTS, f'summary_{expsid}.h5')

lsc_epsilon = 0.02  # 0.02

check_for_new = True
plot_losses = False
one_exp_curves = False
pandas_means = True
show_per_tasknet = True
make_latex = False
make_good_latex = False
nice_bar_plot = False

plot_lsc_vs_naive = False
plot_dampenings_and_betas = False
plot_norms_pretraining = False
plot_weights = False
plot_pretrained_weights = False
plot_init_lrs = False
plot_lrs = False
plot_bars = False
plot_new_bars = False
chain_norms = False
lruptb2latex = False

missing_exps = True
remove_incomplete = False
truely_remove = False
truely_remove_pretrained = False
check_all_norms = False

# sparse_mode_accuracy sparse_categorical_crossentropy bpc sparse_mode_accuracy_test_10
# val_sparse_mode_accuracy test_perplexity
metric = 'v_mode_acc'  # 'v_ppl min'
metric = 'val_ppl m'  # 'v_ppl min'
metric = 'test_ppl'  # 'v_ppl min'
metric = shorten_losses(metric)
metrics_oi = [
    # 't_ppl min', 't_mode_acc max', 'v_ppl min', 'v_mode_acc max',
    # 't_ppl', 't_mode_acc', 'v_ppl', 'v_mode_acc',
    'val_ppl m', 'val_mode_acc M', 'test_ppl', 'test_mode_acc',
    # 'LSC_norms i', 'LSC_norms f', 'LSC_norms mean',
    'ma_norm', 'ni',
    # 'n_params',
    'conveps',
    # 'final_norms_mean', 'final_norms_std', 'best_std_ma_norm', 'std_ma_norm',
]

metrics_oi = [shorten_losses(m) for m in metrics_oi]

plot_only = [
    'seed', 'net', 'task', 'stack', 'comments', 'path', 'lr',  # 'lr f',
    'n_neurons', 'batch_size',
    'optimizer_name', 'lr_schedule', 'host_hostname',
    'v_ppl argm', 'v_ppl len',
]

columns_to_remove = [
    '_var', '_mean', 'sparse_categorical_crossentropy', 'bpc', 'artifacts',
    'experiment_dependencies', 'experiment_sources', 'experiment_repositories', 'host_os',
    'sparse_categorical_accuracy', 'LSC_losses', 'rec_norms', 'fail_trace', 'list', 'weights_shapes',
]
force_keep_column = [
    'LSC_norms list', 'batch ',
    'val_sparse_mode_accuracy list', 'val_perplexity list',
    'v_sparse_mode_accuracy list', 'v_perplexity list',
    't_sparse_mode_accuracy list', 't_perplexity list',
    'final_norms_mean', 'final_norms_std'
]

group_cols = [
    'net', 'task', 'comments', 'stack', 'lr',  # 'lr f',
    'n_neurons', 'optimizer_name', 'batch_size', 'lr_schedule'
]
task_flag = 'task'  # task dataset
net_flag = 'net'  # net lru
depth_flag = 'stack'  # 'stack'
stats_oi = ['mean', 'std']

plot_metric = 'rec_norms list'
plot_metric = 'val_ppl list'

task_name_pairs = [
    ('SHD', 'heidelberg'),
    ('sl-MNIST', 'sl_mnist'),
    ('PTB', 'wordptb'),
]

if expsid == 's5lru':
    GEXPERIMENTS = [
        r'D:\work\alif_sg\good_experiments\2023-10-10--s5lru',
    ]
    plot_metric = 'l0_lnorms list'
    task_flag = 'dataset'  # task dataset
    net_flag = 'lru'  # net lru
    depth_flag = 'n_depth'  # 'stack'

    metric = 'val_acc M'  # 'v_ppl min'
    metrics_oi = [
        # 'val_loss m', 'test_loss m',
        # 'train_loss i', 'train_loss f',
        'val_acc M', 'test_acc M',
        'conveps_val_acc', 'val_acc len', 'n_params', 'time_elapsed',
        *[f'l{i}_tnorms f' for i in range(8)],
        *[f'l{i}_lnorms f' for i in range(8)]
    ]

    plot_only = [
        'jax_seed', 'lru', 'dataset', 'n_depth', 'comments',
        'val_acc argM', 'path', 'eps', 'spe', 'bsz'
    ]
    group_cols = ['lru', 'dataset', 'comments', 'n_depth']

    columns_to_remove = ['experiment_sources']
    force_keep_column = []
    # stats_oi = ['mean']
    stats_oi = ['mean']
    task_name_pairs = [
        ('sCIFAR-3', 'cifar-classification'),
        ('sCIFAR-1', 'lra-cifar-classification'),
        ('Text', 'imdb-classification'),
        ('ListOps', 'listops-classification'),
        ('Retrieval', 'aan-classification'),
        ('Pathfinder', 'pathfinder-classification'),
        ('PathX', 'pathx-classification'),
        ('MNIST', 'mnist-classification'),
    ]

elif expsid == 'mnl':

    GEXPERIMENTS = [
        r'D:\work\alif_sg\good_experiments\2023-11-01--ptblif',
    ]

    task_flag = 'task_name'  # task dataset
    net_flag = 'net_name'  # net lru

    plot_only = [
        'seed', 'net_name', 'task_name', 'stack', 'comments', 'path', 'lr',  # 'lr f',
        'n_neurons', 'batch_size', 'n_params',
        'optimizer_name', 'lr_schedule', 'host_hostname',
        'v_ppl argm', 'v_ppl len',
    ]
    group_cols = [
        'net_name', 'task_name', 'n_params', 'comments', 'stack', 'lr',
        'n_neurons', 'optimizer_name', 'batch_size', 'lr_schedule',
    ]

    metrics_oi = [
        'val_ppl m', 'test_ppl',
        'conveps',
    ]
    stats_oi = ['mean']

elif expsid == 'fluctuations':

    GEXPERIMENTS = [
        r'D:\work\alif_sg\good_experiments\2023-11-10--decolletc',
    ]
    task_flag = 'dataset'
    net_flag = 'dataset'
    depth_flag = 'dataset'

    plot_only = [
        'seed', 'dataset', 'comments', 'path', 'hostname',  # 'lr f',
        'stop_time', 'log_dir', 'n_params',
        # 'valid_acc argM',
    ]
    group_cols = [
        'dataset', 'comments', 'n_params'
    ]

    metrics_oi = [
        # 'train_acc M', 'valid_acc M', 'valid_loss m',
        'test_acc',
        'conveps_valid_acc', 'conveps_valid_loss',
        'valid_acc len', 'valid_loss len', 'time_elapsed',
    ]
    stats_oi = ['mean']
    metric = 'test_acc'  # 'v_ppl min'

elif expsid == '_decolle':

    GEXPERIMENTS = [
        r'D:\work\alif_sg\good_experiments\2023-11-10--decolletc',
    ]
    task_flag = 'datasetname'
    net_flag = 'datasetname'
    depth_flag = 'datasetname'

    plot_only = [
        'seed', 'datasetname', 'comments', 'path', 'hostname',  # 'lr f',
        'stop_time', 'log_dir', 'n_params',
        'test_losses argm', 'test_accs argM', 'test_accs len',
        'val_loss argm', 'val_loss len', 'val_acc argM', 'val_loss len',
    ]

    group_cols = [
        'datasetname', 'comments',
    ]

    metrics_oi = [
        # 'test_losses m', 'train_losses m',
        'test_acc M',
        'test_losses len', 'conveps_test_losses', 'conveps_test_accs', 'time_elapsed',
        'conveps_val_loss', 'conveps_val_acc',
    ]
    stats_oi = ['mean']
    metric = 'test_acc M'  # 'v_ppl min'
    plot_metric = 'test_losses list'

plot_only += metrics_oi
print('plot_only', plot_only)

df = experiments_to_pandas(
    h5path=h5path, zips_folder=GEXPERIMENTS, unzips_folder=EXPERIMENTS, experiments_identifier=expsid,
    exclude_files=['cout.txt', 'text_indices.txt', 'text_sentences.txt'],
    check_for_new=check_for_new,
    exclude_columns=columns_to_remove, force_keep_column=force_keep_column
)
print(list(df.columns))

for flag in [task_flag, net_flag]:
    if flag in df.columns:
        df[flag] = df[flag].astype(str)

if 'als' == expsid:
    df['stack'] = df['stack'].fillna(-1).astype(int)
    df['stack'] = df['stack'].replace(-1, 'None')
    df['stack'] = df['stack'].astype(str)
    df['comments'] = df['comments'].str.replace('simplereadout', 'embproj')
    df['batch_size'] = df['batch_size'].astype(str)
    df['ni'] = df['ni'].astype(float)
    # df['comments'] = df['comments'].str.replace('_pretrained', '')
    df['comments'] = df['comments'].astype(str)
    df.replace(['nan'], np.nan, inplace=True)

if 'mnl' in expsid:
    df['comments'] = df.apply(
        lambda row: '_'.join([
            c for c in row['comments'].split('_')
            if not 'taskmean' in c
               and not 'taskvar' in c]
        ), axis=1
    )

    df['comments'] = df.apply(
        lambda row: ''.join([c for c in row['comments'].split('**') if not 'folder' in c]), axis=1
    )

new_column_names = {c_name: shorten_losses(c_name) for c_name in df.columns}
df.rename(columns=new_column_names, inplace=True)

for m in ['v_ppl', 'val_ppl', 'val_acc', 'val_loss', 'valid_acc', 'valid_loss', 'test_losses', 'test_accs']:
    argm = 'argm' if ('ppl' in m or 'loss' in m) else 'argM'
    if f'{m} {argm}' in df.columns and f'{m} len' in df.columns:
        df['conveps_' + m] = df[f'{m} len'].astype(float) - df[f'{m} {argm}'].astype(float)

print(df.columns)
for c in ['t_ppl', 't_^acc', 'v_ppl', 'v_^acc']:
    # if column doesn't exist, create a NaN column
    if c not in df.columns:
        df[c] = np.nan

for cname in ['net', 'task']:
    if cname in df.columns and isinstance(df[cname], pd.DataFrame):
        c = df[cname].iloc[:, 0].fillna(df[cname].iloc[:, 1])
        df = df.drop([cname], axis=1)
        df[cname] = c

print(list(df.columns))

if plot_pretrained_weights:
    n_bins = 50
    cols = 5
    wspace, hspace = .1, 1.1
    pretrained_path = r'D:\work\alif_sg\good_experiments\pmodels'
    models_files = [f for f in os.listdir(pretrained_path) if f.endswith('.h5')]
    plots_path = r'D:\work\alif_sg\good_experiments\plots'
    os.makedirs(plots_path, exist_ok=True)
    for mf in tqdm(models_files):
        plot_filename = os.path.join(plots_path, mf.replace('.h5', '.png'))
        if not os.path.exists(plot_filename):

            path_pretrained = os.path.join(pretrained_path, mf)
            model = load_LSC_model(path_pretrained)

            weights = model.get_weights()
            weight_names = [weight.name for layer in model.layers for weight in layer.weights]
            # print(weight_names)
            fig, axs = plt.subplots(
                int(len(weights) / cols + 1), cols, gridspec_kw={'wspace': wspace, 'hspace': hspace},
                figsize=(10, 3)
            )
            for i, ax in zip(range(len(weights)), axs.reshape(-1)):
                w = weights[i]
                wn = weight_names[i]
                counts, bins = np.histogram(w.flatten(), bins=n_bins)

                highest = max(counts) / sum(counts) / (bins[1] - bins[0])

                ax.hist(bins[:-1], bins, weights=counts)

                if highest > 100:
                    ax.set_ylim([0, 20])
                ax.set_xlabel(clean_weight_name(wn), fontsize=12)
                ax.locator_params(axis='x', nbins=2)

            for i, ax in enumerate(axs.reshape(-1)):
                for pos in ['right', 'left', 'bottom', 'top']:
                    ax.spines[pos].set_visible(False)

                if i >= len(weights):
                    ax.set_visible(False)  # to remove last plot

            axs[0, 0].set_ylabel('Density', fontsize=12)

            plt.suptitle(f"{mf}", y=1.01, fontsize=16)
            fig.savefig(plot_filename, bbox_inches='tight')
            # plt.show()

if plot_losses:
    df['comments'] = df['comments'].str.replace('allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_', '')
    # df = df[df[task_flag].str.contains('wordptb')]

    # plot_metric = 'val_^acc list'
    # plot_metric = 'LSC list'
    # plot_metric = 'norms dec layer 1 list'  # enc depth rec dec
    # tasks = df['task'].unique()
    # nets = df['net'].unique()

    flags = [task_flag, net_flag]
    # flags = [task_flag, depth_flag]
    assert len(flags) == 2
    fu = {f: sorted(df[f].unique()) for f in flags}
    # tasks = df[task_flag].unique()
    # nets = df[net_flag].unique()
    # depths = df[depth_flag].unique()
    for k, v in fu.items():
        print(k)
        print(v)
    # print(tasks)
    # print(nets)
    print(len(fu[flags[0]]))
    fig, axs = plt.subplots(
        len(fu[flags[0]]), len(fu[flags[1]]), figsize=(6, 3),
        gridspec_kw={'wspace': .2, 'hspace': 0.8}
    )
    # add axis to axs if one dimensional
    if len(fu[flags[0]]) == 1:
        axs = np.expand_dims(axs, axis=0)
    if len(fu[flags[1]]) == 1:
        axs = np.expand_dims(axs, axis=1)

    for i in range(len(fu[flags[0]])):
        for j in range(len(fu[flags[1]])):
            print('-===-' * 30)
            # print(task, net)
            # idf = idf[idf[task_flag].str.contains(task) & idf[net_flag].str.contains(net)]
            idf = df.copy()
            for k, f in enumerate(flags):
                c = i if k == 0 else j
                idf = idf[idf[f].str.contains(fu[f][c])]
            print(idf.to_string())
            for _, row in idf.iterrows():
                n = row['comments']
                print(n)
                if isinstance(row[plot_metric], list):
                    linestyle = '-' if not 'clipping' in row['comments'] else '--'
                    axs[i, j].plot(row[plot_metric], color=lsc_colors(n), label=lsc_clean_comments(n),
                                   linestyle=linestyle)

            axs[i, j].set_title(f'{fu[flags[1]][j]} layers')
    metric_ = plot_metric.replace('val_ppl', 'Validation Perplexity').replace(' list', '')
    axs[0, 0].set_ylabel(metric_, fontsize=14)
    for ax in axs.reshape(-1):
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)
            # ax.set_ylim([100, 300])

    comments = df['comments'].unique()
    comments = [c for c in comments if not 'clipping' in c]
    legend_elements = [Line2D([0], [0], color=lsc_colors(n), lw=4, label=lsc_clean_comments(n)) for n in comments]
    plt.legend(ncol=2, handles=legend_elements, loc='lower center', bbox_to_anchor=(-.2, -.5))

    plt.show()

    plotpath = os.path.join(EXPERIMENTS, 'losses_lru.pdf')
    fig.savefig(plotpath, bbox_inches='tight')

if 'net' in df.columns:
    # df.loc[df['comments'].str.contains('noalif'), 'net'] = 'LIF'
    df.loc[df['net'].eq('maLSNNb'), 'net'] = 'ALIFb'
    df.loc[df['net'].eq('maLSNN'), 'net'] = 'ALIF'

if task_flag in df.columns:
    for nice, brute in task_name_pairs:
        df.loc[df[task_flag].eq(brute), task_flag] = nice

# eps column stays eps if not equal to None else it becomes the content of v_mode_acc len
# df.loc[df['eps'].isnull(), 'eps'] = df.loc[df['eps'].isnull(), f'{metric} len']

if 'v_^acc len' in df.columns:
    # # FIXME: 14 experiments got nans in the heidelberg task validation, plot them anyway?
    print(list(df.columns))
    print('v_mode_acc nans:', df['v_^acc len'].isna().sum())
    print('t_ppl nans:', df['t_ppl list'].isna().sum())

    df['v_ppl'] = df['v_ppl m']

    df['t_ppl'] = df.apply(
        lambda row: row['t_ppl list'][int(row['v_ppl argm'])] if isinstance(row['t_ppl list'], list) else np.nan,
        axis=1
    )
    df['v_^acc'] = df['v_^acc M']
    df['t_^acc'] = df.apply(
        lambda row: row['t_^acc list'][int(row['v_^acc argM'])] if isinstance(row['t_^acc list'], list) else np.nan,
        axis=1
    )

for c_name in columns_to_remove:
    df = df[df.columns.drop(list(df.filter(regex=c_name)))]

if metric in df.keys():
    df = df.sort_values(by=metric)

print(list(df.columns))
if not plot_only is None:
    df = df.reindex(plot_only, axis=1)
    plotdf = df[plot_only]
    print(plotdf.to_string())

if pandas_means:
    # group_cols = ['net', 'task', 'comments', 'stack', 'lr']

    counts = df.groupby(group_cols).size().reset_index(name='counts')
    # stats = ['mean', 'std']
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

    # mdf = mdf.sort_values(by='mean_' + metric)
    print(mdf.shape)
    if 'comments' in mdf.columns and not mdf.shape[0] == 0:
        mdf['comments'] = mdf['comments'].str.replace('__', '_', regex=True)
    for time in ['time_elapsed', 'mean_time_elapsed', 'std_time_elapsed']:
        if time in mdf.columns:
            mdf[time] = mdf[time].apply(lambda x: timedelta(seconds=x) if not x != x else x)

    for k in ['mean_n_params', 'n_params']:
        if k in mdf.columns:
            mdf[k] = mdf[k].apply(lambda x: large_num_to_reasonable_string(x, 1))

    if not show_per_tasknet:
        # sort by mean metric
        mdf = mdf.sort_values(by='mean_' + metric, ascending=False)
        mdf = mdf.sort_values(by=task_flag, ascending=False)

        print(mdf.to_string())

    elif not mdf.empty:
        tasks = sorted(np.unique(mdf[task_flag]))
        nets = sorted(np.unique(mdf[net_flag]))
        stacks = sorted(np.unique(mdf[depth_flag]))

        xdf = mdf.copy()
        xdf['comments'] = xdf['comments'].str.replace('allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_', '')
        # for stack in stacks:
        for net in nets:
            for task in tasks:
                # for net in nets:
                print('-===-' * 30)
                # print(task, net, stack)
                # print(task, stack)
                print(task, net)

                idf = xdf[
                    xdf[task_flag].eq(task)
                    & xdf[net_flag].eq(net)
                    # & xdf[depth_flag].eq(stack)
                    ]

                cols = [c for c in idf.columns if not net_flag in c and not task_flag in c]
                if not 'PTB' in task:
                    # idf = idf.sort_values(by='mean_' + metric, ascending=False)
                    cols = [c for c in cols if not 'ppl' in c]
                else:
                    # idf = idf.sort_values(by='mean_v_ppl', ascending=True)
                    cols = [c for c in cols if not 'acc' in c]

                idf.rename(columns=new_column_names, inplace=True)
                idf = idf[cols]

                idf = idf.sort_values(by='mean_' + metric, ascending=True)
                for c in idf.columns:
                    if 'ppl' in c:
                        idf[c] = idf[c].apply(
                            lambda x: '%.3f' % x if isinstance(x, np.floating) and int(x) < 1000 else str(x)
                        )

                print(idf.to_string())
    sdf = mdf.copy()
    if expsid == 'mnl':
        print('-===-' * 30)
        sdf = sdf[
            (sdf['comments'].str.contains('allns_36_dropout:.2_embproj_pretrained_maxlen:300_mlminputs_mlmeps:3')
             | sdf['comments'].str.contains('allns_36_dropout:.1_embproj_pretrained_maxlen:300_mlminputs_mlmeps:3'))
            & sdf['lr'].eq(0.03)
            ]

    # print(sdf.to_string())

if lruptb2latex:
    ffnlsc = False
    xdf = mdf.copy()

    if not ffnlsc:
        xdf = xdf[~xdf['comments'].str.contains('ffnlsc')]
    else:
        xdf = xdf[
            ~xdf['comments'].str.contains('findLSC')
            | xdf['comments'].str.contains('ffnlsc')
            ]
        xdf['comments'] = xdf['comments'].str.replace('_ffnlsc', '')

    xdf = xdf[xdf['task'].str.contains('PTB')]
    xdf = xdf[
        (idf['comments'].str.contains('allns_36_dropout:.2_embproj_pretrained_maxlen:300_mlminputs_mlmeps:3')
         | idf['comments'].str.contains('allns_36_dropout:.1_embproj_pretrained_maxlen:300_mlminputs_mlmeps:3'))
        & idf['lr'].eq(0.03)
        ]
    xdf['clipping'] = xdf['comments'].str.contains('clipping')

    new_column_names = {
        'mean_val_ppl m': 'mean_val_ppl',
        'std_val_ppl m': 'std_val_ppl',
    }
    xdf.rename(columns=new_column_names, inplace=True)

    coi = ['comments', 'clipping', 'stack', 'mean_val_ppl', 'std_val_ppl', 'mean_test_ppl', 'std_test_ppl']
    xdf = xdf[coi]
    coif = ['comments', 'vppl', 'tppl']

    xdf['comments'] = xdf['comments'].str.replace('_clipping', '')
    xdf['comments'] = xdf['comments'].str.replace('findLSC_radius', 'LSC')
    xdf['comments'] = xdf['comments'].str.replace('allns_36_dropout:.0_', '')
    xdf['comments'] = xdf['comments'].str.replace(
        'allns_36_dropout:.1_embproj_pretrained_maxlen:300_mlminputs_mlmeps:3', '')
    xdf['comments'] = xdf['comments'].str.replace(
        'allns_36_dropout:.2_embproj_pretrained_maxlen:300_mlminputs_mlmeps:3', '')
    xdf['comments'] = xdf['comments'].map(lambda x: x.lstrip('_'))

    # write $\rho=1$ as comments if comments is LSC
    xdf['comments'] = xdf.apply(lambda row: r'$\rho=1$' if 'LSC' == row['comments'] else row['comments'], axis=1)
    xdf['comments'] = xdf.apply(
        lambda row: r'$\rho=0.5$' if 'LSC_targetnorm:.5' == row['comments'] else row['comments'], axis=1)
    xdf['comments'] = xdf.apply(
        lambda row: r'$\overline{\rho}_t=0.5$' if 'LSC_targetnorm:.5_unbalanced' == row['comments'] else row[
            'comments'], axis=1)
    xdf['comments'] = xdf.apply(lambda row: r'default' if '' == row['comments'] else row['comments'], axis=1)
    xdf['comments'] = xdf.apply(lambda row: row['comments'] + ' + clip' if row['clipping'] else row['comments'], axis=1)

    depths = sorted(np.unique(xdf['stack']))

    full_latex = ''
    for d in depths:
        idf = xdf[xdf['stack'].eq(d)]
        # sort by clipping column
        idf = idf.sort_values(by='clipping', ascending=True)

        metrics_cols = [c for c in idf.columns if 'ppl' in c]
        for m in metrics_cols:
            mode = 'max' if 'acc' in m and not 'std' in m else 'min'
            idf[f'best_{m}'] = idf.groupby(['clipping'])[m].transform(mode)
            idf[m] = idf.apply(bolden_best(m), axis=1)

        idf['vppl'] = idf.apply(compactify_metrics('ppl', data_split='val_'), axis=1)
        idf['tppl'] = idf.apply(compactify_metrics('ppl', data_split='test_'), axis=1)

        idf = idf[coif]

        latex_df = idf.to_latex(index=False, escape=False).replace('{lll}', '{lcc}')
        latex_df = latex_df.replace('comments', r'\textbf{' + f'Depth {d} LRU' + r'}')
        latex_df = latex_df.replace('vppl', r'\textbf{\shortstack{validation \\ perplexity} }')
        latex_df = latex_df.replace('tppl', r'\textbf{\shortstack{test \\ perplexity} }')

        #
        import re

        latex_df = re.sub(' +', ' ', latex_df)
        # print('\n\n')
        full_latex += latex_df

    # full_latex = full_latex.replace(r'\end{tabular}\n\begin{tabular}{lcc}', '')
    full_latex = full_latex.replace('\n\\end{tabular}\n\\begin{tabular}{lcc}', '')
    full_latex = '\\begin{table}\n' + full_latex + '\end{table}'
    print('\n\n')
    print(full_latex)

if nice_bar_plot:
    df = df.copy()
    df = df[~df['net'].str.contains('rsimplernn')]
    df = df[df['comments'].str.contains('onlyloadpretrained')]

    pickle_path = os.path.join(EXPERIMENTS, 'results_bar_plot.pkl')
    if not os.path.exists(pickle_path):
        results = {}
        for stack in ['None', '5']:
            idf = df[df['stack'].eq(stack)]
            five_df = idf[
                idf['comments'].str.contains('targetnorm:.5')
                & idf['comments'].str.contains('findLSC')
                ]

            five_better_than_one = 0
            five_better_than_none = 0
            tot_5 = 0
            tot_none = 0
            for _, row in five_df.iterrows():
                # print('-' * 80)
                # print(row['comments'])
                same_one_df = idf[
                    ~idf['comments'].str.contains('targetnorm:.5')
                    & idf['comments'].str.contains('findLSC')
                    & idf['net'].str.contains(row['net'])
                    & idf['seed'].eq(row['seed'])
                    & idf['task'].str.contains(row['task'])
                    ]
                # print(same_one_df.to_string())
                if not same_one_df.empty:
                    tot_5 += 1
                    metric = 't_^acc' if not row['task'] == 'PTB' else 't_ppl'
                    if same_one_df[metric].values[0] < row[metric] and not row['task'] == 'PTB':
                        five_better_than_one += 1
                    elif same_one_df[metric].values[0] > row[metric] and row['task'] == 'PTB':
                        five_better_than_one += 1

                same_none_df = idf[
                    ~idf['comments'].str.contains('findLSC')
                    & idf['net'].str.contains(row['net'])
                    & idf['seed'].eq(row['seed'])
                    & idf['task'].str.contains(row['task'])
                    ]
                # print(same_none_df.to_string())

                if not same_none_df.empty:
                    tot_none += 1
                    metric = 't_^acc' if not row['task'] == 'PTB' else 't_ppl'
                    if same_none_df[metric].values[0] < row[metric] and 'acc' in metric:
                        five_better_than_none += 1
                    elif same_none_df[metric].values[0] > row[metric] and 'ppl' in metric:
                        five_better_than_none += 1

            print(f'five_better_than_one (stack {stack}): ', five_better_than_one / tot_5)
            print(f'five_better_than_none (stack {stack}):', five_better_than_none / tot_none)

            results[stack] = {
                'five_better_than_one': five_better_than_one / tot_5,
                'five_better_than_none': five_better_than_none / tot_none,
            }

        for stack in ['None', '5']:
            idf = df[
                df['stack'].eq(stack)
                & df['net'].str.contains('ALIF')
                ]
            one_df = idf[
                idf['comments'].str.contains('findLSC')
            ]

            one_better_than_none = 0
            tot = 0
            for _, row in one_df.iterrows():
                # print('-' * 80)
                # print(row['comments'])
                same_none_df = idf[
                    ~idf['comments'].str.contains('findLSC')
                    & idf['net'].eq(row['net'])
                    & idf['seed'].eq(row['seed'])
                    & idf['task'].str.contains(row['task'])
                    ]
                if not same_none_df.empty:
                    tot += 1
                    metric = 't_^acc' if not row['task'] == 'PTB' else 't_ppl'
                    if same_none_df[metric].values[0] < row[metric] and 'acc' in metric:
                        one_better_than_none += 1
                    elif same_none_df[metric].values[0] > row[metric] and 'ppl' in metric:
                        one_better_than_none += 1

            print(f'one_better_than_none (stack {stack}): ', one_better_than_none / tot)

            results[stack].update({
                'one_better_than_none': one_better_than_none / tot,
            })

        for stack in ['None', '5']:
            idf = df[
                df['stack'].eq(stack)
                & ~df['net'].str.contains('ALIF')
                ]
            one_df = idf[
                idf['comments'].str.contains('findLSC')
            ]

            one_better_than_none = 0
            tot = 0
            for _, row in one_df.iterrows():
                # print('-' * 80)
                # print(row['comments'])
                same_none_df = idf[
                    ~idf['comments'].str.contains('findLSC')
                    & idf['net'].eq(row['net'])
                    & idf['seed'].eq(row['seed'])
                    & idf['task'].str.contains(row['task'])
                    ]
                if not same_none_df.empty:
                    tot += 1
                    metric = 't_^acc' if not row['task'] == 'PTB' else 't_ppl'
                    if same_none_df[metric].values[0] < row[metric] and 'acc' in metric:
                        one_better_than_none += 1
                    elif same_none_df[metric].values[0] > row[metric] and 'ppl' in metric:
                        one_better_than_none += 1

            print(f'one_better_than_none no ALIF (stack {stack}): ', one_better_than_none / tot)

            results[stack].update({
                'one_better_than_none_noalif': one_better_than_none / tot,
            })

        with open(pickle_path, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(pickle_path, 'rb') as handle:
            results = pickle.load(handle)

    print(results)

    import numpy as np
    import matplotlib.pyplot as plt

    # set width of bar
    barWidth = 0.33
    fig, axs = plt.subplots(1, 1, figsize=(6, 4))

    # set height of bar
    print(list(results['None'].keys()))
    stack_2 = 100 * np.array(list(results['None'].values()))
    stack_5 = 100 * np.array(list(results['5'].values()))

    # Set position of bar on X axis
    br1 = np.arange(len(stack_2))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    c_2 = '#C6C580'
    c_5 = '#609208'

    # Make the plot
    axs.bar(br1, stack_2, color=c_2, width=barWidth, label='2 layers')
    axs.bar(br2, stack_5, color=c_5, width=barWidth, label='5 layers')
    axs.axhline(y=50, color='k', linestyle='--')
    axs.set_yticks([0, 25, 50, 75])

    # for ax in axs.reshape(-1):
    for pos in ['right', 'left', 'bottom', 'top']:
        axs.spines[pos].set_visible(False)

    # Adding Xticks
    # plt.xlabel('Branch', fontweight='bold', fontsize=15)
    plt.ylabel('Percentage\nof experiments', fontweight='bold', fontsize=17)
    plt.xticks([r + barWidth / 2 for r in range(len(stack_2))],
               # ['half > one', 'half > none', 'one > none\n(ALIFs)', ],
               [r'$\rho_t=0.5$' + '\nbetter than\n' + r'$\rho_t=1$',
                r'$\rho_t=0.5$' + '\nbetter than\nnone',
                r'$\rho_t=1$' + '\nbetter than\nnone\n(ALIFs)',
                r'$\rho_t=1$' + '\nbetter than\nnone\n(no ALIFs)',
                ],
               rotation=10, fontsize=15)

    legend_elements = [
        Line2D([0], [0], color=c_2, lw=4, label='2 layers'),
        Line2D([0], [0], color=c_5, lw=4, label='5 layers'),
    ]

    plt.legend(ncol=2, handles=legend_elements, loc='center', bbox_to_anchor=(.5, .995), fontsize=15)

    plot_filename = f'experiments/nicebars.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.show()

if make_good_latex:

    mdf['target'] = mdf['comments'].apply(
        lambda x: 0.5 if 'targetnorm:.5' in x else 1 if 'findLSC' in x else np.nan)
    mdf['diff_target'] = abs(mdf['mean_LSC f'] - mdf['target'])
    mdf['vs_epsilon'] = mdf['diff_target'] > lsc_epsilon

    tab_types = {
        'task': ['sl-MNIST', 'SHD', 'PTB'],
        'stack': [1, 3, 5, 7]
    }

    # indrnn
    net_types = {
        'all': ['ssimplernn', 'GRU', 'LSTM', 'ALIF', 'ALIFb'],  # 'rsimplernn',
        'nolsnns': ['ssimplernn', 'GRU', 'LSTM'],  # 'rsimplernn',
        'lsnns': ['ALIF', 'ALIFb'],
    }

    idf = mdf.copy()
    idf = idf[idf['comments'].str.contains('onlyloadpretrained')]
    idf['net'] = pd.Categorical(idf['net'], categories=net_types['all'], ordered=True)
    idf['task'] = pd.Categorical(idf['task'], categories=tab_types['task'], ordered=True)
    idf = idf[~(idf['net'].str.contains('ALIF') & idf['comments'].str.contains(r'targetnorm:.5'))]

    ntype = 'all'
    tttype = 'task5'  # stack task task5
    ttype = ''.join([i for i in tttype if not i.isdigit()])
    data_split = 't_'  # t_ v_

    if ttype == 'task':
        if not '5' in tttype:
            idf = idf[idf['stack'].eq('None')]
        else:
            idf = idf[idf['stack'].eq('5')]
    else:
        idf = idf[~idf['stack'].eq('None')]
        idf = idf[idf['task'].eq('SHD')]

    # select only rows that have any of the models above in the net column
    idf = idf[idf['net'].isin(net_types[ntype])]

    metrics_cols = [c for c in idf.columns if 'ppl' in c or 'acc' in c]
    for m in metrics_cols:
        mode = 'max' if 'acc' in m and not 'std' in m else 'min'
        idf[f'best_{m}'] = idf.groupby([ttype, 'net'])[m].transform(mode)
        idf[m] = idf.apply(bolden_best(m), axis=1)

    idf['ppl'] = idf.apply(compactify_metrics('ppl', data_split=data_split), axis=1)
    idf['acc'] = idf.apply(compactify_metrics('^acc', data_split=data_split), axis=1)

    idf = idf.round({'mean_LSC f': 3, 'std_LSC f': 3})
    idf['lsc'] = idf.apply(compactify_metrics('LSC f', data_split=''), axis=1)
    idf['metric'] = idf.apply(choose_metric, axis=1)

    coi = ['net', ttype, 'comments', 'metric', 'lsc']
    idf = idf[coi]
    print(idf.to_string())

    # clean comments
    idf['comments'] = idf['comments'].str.replace('allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_findLSC_', '')
    idf['comments'] = idf['comments'].str.replace('_onlyloadpretrained', '')
    idf['comments'] = idf['comments'].str.replace('allns_36_simplereadout_nogradreset_dropout:.3_timerepeat:2', 'none')
    idf['comments'] = idf['comments'].str.replace('allns_36_embproj_nogradreset_dropout:.3_timerepeat:2', 'none')
    idf['comments'] = idf['comments'].str.replace('radius_targetnorm:.5', '$\rho_t=0.5$')
    idf['comments'] = idf['comments'].str.replace('radius', '$\rho_t=1$')

    # Make a table that has a column net, followed by a column comments and a column for each task.
    # For that, you need to pivot since idf now has a task column.
    df = idf.copy()

    # reshape dataframe
    df = pd.melt(df, id_vars=['net', ttype, 'comments'], value_vars=['metric', 'lsc'], var_name='type')
    df = df.pivot_table(values='value', index=['net', 'comments', 'type'], columns=ttype, aggfunc=np.sum)
    df = df.rename_axis(columns=None).reset_index().rename_axis(index=None)

    # reorder 'type' column
    df['type'] = pd.Categorical(df['type'], categories=['metric', 'lsc'], ordered=True)
    df = df.sort_values(['net', 'comments', 'type'])

    # group by 'net' and 'comments'
    df.rename(columns={'comments': 'LSC'}, inplace=True)
    df = df.set_index(['net', 'LSC', 'type'])
    df.columns.names = [ttype]

    print(df.to_string())

    print('\n\n\n')
    lls = ''.join('l' * len(tab_types[ttype]))
    ccs = ''.join('c' * len(tab_types[ttype]))
    latex_df = df.to_latex(index=True, escape=False).replace('{lll' + lls, '{llr' + ccs)

    import re

    latex_df = re.sub(' +', ' ', latex_df)

    latex_df = latex_df.replace('\midrule \n\midrule', '\midrule')
    latex_df = latex_df.replace('\midrule\n\midrule', '\midrule')

    for net in net_types['all']:
        if net == 'ALIF':
            latex_df = latex_df.replace(net + ' ', r'\midrule\midrule' + '\n ' + net)

        elif net == 'ssimplernn':
            latex_df = latex_df.replace('\\midrule\n' + net, r'\\ \toprule\midrule' + ' \n ' + net)

        else:
            latex_df = latex_df.replace(net, r'\midrule' + '\n ' + net)

    latex_df = latex_df.replace(r'\bottomrule', r'\midrule\bottomrule')
    latex_df = latex_df.replace('lsc', r'$\rho$')

    latex_df = latex_df.replace('ssimplernn', 'RNN \, $\sigma$')
    latex_df = latex_df.replace('rsimplernn', 'RNN \, $ReLU$')
    latex_df = latex_df.replace('sl-MNIST', r'sl-MNIST $\uparrow$')
    latex_df = latex_df.replace('SHD', r'SHD $\uparrow$')
    latex_df = latex_df.replace('PTB', r'PTB $\downarrow$')
    latex_df = latex_df.replace('type', '')

    for task in ['sl-MNIST', 'SHD', 'PTB', 'net', 'LSC', 'type', ttype]:
        latex_df = latex_df.replace(task, r'\textbf{' + task + '}')

    latex_df = latex_df.replace('stack', 'depth')

    # loop over the lines of the latex table
    ref_line = 1000000
    new_latex_df = ''
    net_name = None
    for i, line in enumerate(latex_df.split('\n')):

        if not 'metric' in line and not 'task' in line and not 'rule' in line \
                and not 'depth' in line and not 'net' in line:
            line = line.replace('&', r'& \scriptsize')
        if r'\rho_t=0.5' in line:
            line = line.replace(r'$\textbf{', r'$\cellcolor{NiceGreen!25}\textbf{')
        if r'\rho_t=1' in line:
            line = line.replace(r'$\textbf{', r'$\cellcolor{NiceOrange!25}\textbf{')
        if r'none' in line:
            line = line.replace(r'$\textbf{', r'$\cellcolor{NiceBlue!25}\textbf{')

        line = re.sub(' +', '', line)

        if 'ALIF' in line:
            ref_line = i
            if 'ALIFb' in line:
                net_name = 'ALIFb'
            else:
                net_name = 'ALIF'
            pass

        elif i == ref_line + 2:
            new_latex_df += net_name + line + '\n'

        elif not i == ref_line + 1:

            new_latex_df += line + '\n'

    latex_df = new_latex_df
    latex_df = latex_df.replace('ALIFb', 'ALIF$_{\pm}$')
    latex_df = latex_df.replace('ALIF ', 'ALIF$_{+}$ ')

    latex_df = latex_df + '\caption{' + f'{tttype}' + '}\n\end{table}'
    latex_df = '\\begin{table}[h]\n\centering\n' + latex_df
    print(latex_df)

if plot_init_lrs:
    idf = mdf

    # idf = idf.dropna(subset=['lr'])
    print(idf.to_string())

    tasks = ['sl-MNIST', 'SHD', 'PTB']
    lrs = np.unique(idf['lr'])
    nets = ['LSTM', 'ALIF', 'ALIFb']

    fig, axs = plt.subplots(1, len(tasks), gridspec_kw={'wspace': .2, 'hspace': 0.8}, figsize=(14, 3))
    colors = lambda net: '#FF5733' if net == 'ALIF' else '#1E55A9'
    for i, task in enumerate(tasks):
        for net in nets:
            iidf = idf[idf['task'].eq(task) & idf['net'].eq(net)]
            iidf = iidf.sort_values(by='lr')
            print(iidf.columns)
            vppls = iidf[f'mean_{metric}'].values
            lrs = iidf['lr'].values
            stds = iidf[f'std_{metric}'].values
            axs[i].plot(lrs, vppls, color=colors(net))
            axs[i].fill_between(lrs, vppls - stds / 2, vppls + stds / 2, alpha=0.5, color=colors(net))
            axs[i].set_xscale('log')

            iidf = iidf.sort_values(by=f'mean_{metric}')
            print(f"{net} on {task} got best vPPL {iidf[f'mean_{metric}'].values[0]} for {iidf['lr'].values[0]}")

        axs[i].set_title(task)
        axs[i].set_xticks([1e-2, 1e-3, 1e-4, 1e-5])
        axs[i].locator_params(axis='y', nbins=5)
    axs[0].set_ylabel('Perplexity')
    axs[0].set_xlabel('Learning Rate')

    for ax in axs.reshape(-1):
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)

    legend_elements = [
        Line2D([0], [0], color=colors(net), lw=4, label=net)
        for net in nets
    ]

    plt.legend(handles=legend_elements, loc='center', bbox_to_anchor=(-2.15, .5))

    plot_filename = f'experiments/lrs.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')

    plt.show()

if plot_lrs:
    stack = '1'  # 1 None 7

    idf = mdf.copy()
    idf = idf[idf['stack'].eq(stack)]

    idf['comments'] = idf['comments'].str.replace('allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_', '')

    tasks = idf['task'].unique()
    nets = idf['net'].unique()
    comments = idf['comments'].unique()

    # tasks = ['sl-MNIST', 'SHD', 'PTB']
    lrs = np.unique(idf['lr'])
    idf = idf.sort_values(by='lr')

    # nets = ['LSTM', 'ALIF', 'ALIFb']

    fig, axs = plt.subplots(len(nets), len(tasks), gridspec_kw={'wspace': .2, 'hspace': 0.8}, figsize=(14, 3))
    if len(nets) == 1:
        axs = np.array([axs])
    if len(tasks) == 1:
        axs = np.array([axs]).T

    colors = lambda net: '#FF5733' if net == 'ALIF' else '#1E55A9'
    for j, task in enumerate(tasks):
        for i, net in enumerate(nets):
            iidf = idf[idf['task'].eq(task) & idf['net'].eq(net)]
            iidf = iidf.sort_values(by='lr')

            for c in comments:
                cdf = iidf[iidf['comments'].eq(c)]
                print('-==*-' * 20)
                print(cdf.to_string())
                means, stds, lrs = cdf[f'mean_{metric}'].values, cdf[f'std_{metric}'].values, cdf['lr'].values
                axs[i, j].plot(lrs, means, color=lsc_colors[c], label=c)
                axs[i, j].fill_between(lrs, means - stds / 2, means + stds / 2, alpha=0.5, color=lsc_colors[c])
            axs[i, j].set_xscale('log')

            iidf = iidf.sort_values(by=f'mean_{metric}')

            if not iidf.shape[0] == 0:
                print(f"{net} on {task} got best vPPL {iidf[f'mean_{metric}'].values[0]} for {iidf['lr'].values[0]}")

        axs[0, j].set_title(task)
        # axs[i,0].set_xticks([1e-2, 1e-3, 1e-4, 1e-5])
        # axs[i,0].locator_params(axis='y', nbins=5)
    axs[0, 0].set_ylabel('Validation Perplexity')
    axs[-1, -1].set_xlabel('Learning Rate')

    for ax in axs.reshape(-1):
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)

    legend_elements = [Line2D([0], [0], color=lsc_colors[n], lw=4, label=n) for n in comments]
    plt.legend(ncol=3, handles=legend_elements, loc='lower center')  # , bbox_to_anchor=(-.1, -1.))

    plot_filename = f'experiments/lrs.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')

    plt.show()

if plot_new_bars:
    type_plot = 'layers'  # layers tasks
    idf = mdf.copy()

    nets = ['LSTM', 'GRU']
    tasks = ['sl-MNIST', 'SHD', 'PTB']

    if type_plot == 'tasks':
        idf = idf[idf['stack'].eq('None')]
        subplots = tasks
        col_id = 'task'
        bbox_to_anchor = (-.9, -.5)

    elif type_plot == 'layers':
        idf = idf[~idf['stack'].eq('None')]
        subplots = sorted(idf['stack'].unique())
        col_id = 'stack'
        print(subplots)
        bbox_to_anchor = (-1.1, -.5)
    else:
        raise NotImplementedError
    idf = idf[idf['comments'].str.contains('_onlyloadpretrained')]
    idf['comments'] = idf['comments'].str.replace('allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_', '')
    idf['comments'] = idf['comments'].str.replace('_onlyloadpretrained', '')
    idf['comments'] = idf['comments'].str.replace('onlyloadpretrained', '')
    comments = idf['comments'].unique()
    shift = (len(comments) + 1) / 2 - 1

    data_split = 't_'  # 'v_'
    # metric = data_split + '^acc'  # 'v_^acc'

    fig, axs = plt.subplots(1, len(subplots), figsize=(6, 3), gridspec_kw={'wspace': .4, 'hspace': .1})
    X = np.arange(len(nets))
    w = 1 / (len(comments) + 1)

    for j, t in enumerate(subplots):
        for i, c in enumerate(comments):
            metric = data_split + '^acc' if not t == 'PTB' else data_split + 'ppl'
            iidf = idf[idf[col_id].eq(t) & idf['comments'].eq(c)]
            # sort net column anti alphabetically
            iidf = iidf.sort_values(by='net', ascending=False)
            print(iidf.to_string())
            data = iidf[f'mean_{metric}'].values
            error = iidf[f'std_{metric}'].values / 2
            axs[j].bar(X + i * w, data, yerr=error, width=w, color=lsc_colors(c), label=lsc_clean_comments(c))

        axs[j].set_xticks([r + shift * w for r in range(len(nets))], nets, rotation=10)
        axs[j].set_title(t + (r' $\downarrow$' if t == 'PTB' else (r' $\uparrow$' if t in ['sl-MNIST', 'SHD'] else '')),
                         weight='bold')
        axs[j].locator_params(axis='y', nbins=3)
        # axs[j].xaxis.set_major_locator(plt.MaxNLocator(3))

        if t == 'PTB':
            axs[j].set_ylim(75, 125)
        elif t == 'sl-MNIST':
            axs[j].set_ylim(0.93, 0.975)
        else:
            axs[j].set_ylim(0.82, 0.97)

    for ax in axs.reshape(-1):
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)

    legend_elements = [Line2D([0], [0], color=lsc_colors(n), lw=4, label=lsc_clean_comments(n)) for n in comments]
    plt.legend(ncol=len(comments), handles=legend_elements, loc='lower center', bbox_to_anchor=bbox_to_anchor)

    plot_filename = f'experiments/{expsid}_{type_plot}.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')
    plt.show()
    plt.show()

if plot_bars:
    stack = 'None'  # 1 None 7

    idf = mdf.copy()
    idf = idf[idf['stack'].eq(stack)]

    idf['comments'] = idf['comments'].str.replace('allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_', '')

    tasks = idf['task'].unique()
    nets = idf['net'].unique()
    comments = idf['comments'].unique()

    # tasks = ['sl-MNIST', 'SHD', 'PTB']
    lrs = np.unique(idf['lr'])
    idf = idf.sort_values(by='lr')

    nets = ['LSTM', 'GRU']

    fig, axs = plt.subplots(len(nets), len(tasks), gridspec_kw={'wspace': .2, 'hspace': 0.8}, figsize=(14, 3))
    if len(nets) == 1:
        axs = np.array([axs])

    if len(tasks) == 1:
        axs = np.array([axs]).T

    colors = lambda net: '#FF5733' if net == 'ALIF' else '#1E55A9'
    for j, task in enumerate(tasks):
        for i, net in enumerate(nets):
            iidf = idf[idf['task'].eq(task) & idf['net'].eq(net)]
            iidf = iidf.sort_values(by='lr')

            for c in comments:
                cdf = iidf[iidf['comments'].eq(c)]
                print('-==*-' * 20)
                print(cdf.to_string())
                means, stds, lrs = cdf[f'mean_{metric}'].values, cdf[f'std_{metric}'].values, cdf['lr'].values
                axs[i, j].plot(lrs, means, color=lsc_colors[c], label=c)
                axs[i, j].fill_between(lrs, means - stds / 2, means + stds / 2, alpha=0.5, color=lsc_colors[c])
            axs[i, j].set_xscale('log')

            iidf = iidf.sort_values(by=f'mean_{metric}')

            if not iidf.shape[0] == 0:
                print(f"{net} on {task} got best vPPL {iidf[f'mean_{metric}'].values[0]} for {iidf['lr'].values[0]}")

        axs[0, j].set_title(task)
        # axs[i,0].set_xticks([1e-2, 1e-3, 1e-4, 1e-5])
        # axs[i,0].locator_params(axis='y', nbins=5)
    axs[0, 0].set_ylabel('Validation Perplexity')
    axs[-1, -1].set_xlabel('Learning Rate')

    for ax in axs.reshape(-1):
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)

    legend_elements = [Line2D([0], [0], color=lsc_colors(n), lw=4, label=n) for n in comments]
    plt.legend(ncol=3, handles=legend_elements, loc='lower center')  # , bbox_to_anchor=(-.1, -1.))

    plot_filename = f'experiments/bars.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')

    plt.show()

if plot_norms_pretraining:
    moi = 'norms'  # losses norms
    ref = 0 if metric == 'losses' else 1
    fig, axs = plt.subplots(len(nets), len(tasks), figsize=(6, 2), gridspec_kw={'wspace': .05})

    if len(nets) == 1:
        axs = axs[None]

    cmap = plt.cm.get_cmap('Paired')
    norms = [0.1, 1, 2, 3, -1]
    colors = cmap(np.arange(len(norms)) / len(norms))
    for i, n in enumerate(nets):
        for j, t in enumerate(tasks):
            idf = df[(df['net'].eq(n)) & (df['task'].eq(t))]

            for index, row in idf.iterrows():
                if 'LSC_' + moi in row.keys():
                    if isinstance(row['LSC_' + moi], str):
                        normpow = str2val(row['comments'], 'normpow', float, default=1)

                        metric = [float(s) for s in row['LSC_' + moi][1:-1].split(', ')]
                        axs[i, j].plot(metric, color=colors[norms.index(normpow)], label=normpow)

            axs[i, j].set_title(f'{n}: {t}')

    axs[i, j].legend()
    plt.show()

if remove_incomplete:
    import shutil

    # plotdf = plotdf[plotdf['comments'].str.contains('onlypretrain')]
    rdfs = []

    print('\n\n')
    print('-=***=-' * 10)
    print('\n\n')

    print('Eliminate non converged')
    rdf = plotdf[
        (plotdf['conveps_valid_acc'] < 13)
        | (plotdf['conveps_valid_loss'] < 13)
    ]
    ardf = rdf.copy()
    print(rdf.to_string())
    print(rdf.shape, df.shape)
    rdfs.append(rdf)

    print('Eliminate if f_norms_std too large')
    # rdf = plotdf[
    #     (plotdf['f_norms_std'] > .2)
    #     & plotdf['comments'].str.contains('findLSC')
    #     ]
    # ardf = rdf.copy()
    # print(rdf.to_string())
    # print(rdf.shape, df.shape)
    # rdfs.append(rdf)

    print('Eliminate if std_ma_norm too large')
    # rdf = plotdf[
    #     (plotdf['std_ma_norm'] > .2)
    #     & plotdf['comments'].str.contains('findLSC')
    #     # & plotdf['comments'].str.contains('onlypretrain')
    #     ]
    # brdf = rdf.copy()
    # print(rdf.to_string())
    # print(rdf.shape, df.shape)
    # rdfs.append(rdf)

    print('Eliminate if best_std_ma_norm too large')
    # rdf = plotdf[
    #     (plotdf['best_std_ma_norm'] > .2)
    #     & plotdf['comments'].str.contains('findLSC')
    #     & plotdf['comments'].str.contains('onlypretrain')
    #     ]
    # print(rdf.to_string())
    # print(rdf.shape, df.shape)
    # rdfs.append(rdf)

    print('Eliminate if not close enough to target norm')

    # from LSC_norms final column, select those that are epsilon away from 1
    # make a column target equal to .5 if targetnorm:.5 is in comments else 1 if findLSC is in comments
    # else nan
    # plotdf['target'] = plotdf['comments'].apply(
    #     lambda x: 0.5 if 'targetnorm:.5' in x else 1 if 'findLSC' in x else np.nan
    # )

    # plotdf['diff_target'] = abs(plotdf['LSC f'] - plotdf['target'])
    # plotdf['vs_epsilon'] = plotdf['diff_target'] > lsc_epsilon
    # plotdf['vs_epsilon'] = ((abs(plotdf['LSC a'] - plotdf['target']) > lsc_epsilon)
    #                         & plotdf['comments'].str.contains('onlyloadpretrained')) \
    #                        | ((abs(plotdf['LSC f'] - plotdf['target']) > lsc_epsilon)
    #                           & plotdf['comments'].str.contains('onlypretrain'))
    #
    # rdf = plotdf[
    #     plotdf['comments'].str.contains('findLSC')
    #     & plotdf['vs_epsilon']
    #     ]
    # print(rdf.to_string())
    # print(rdf.shape, df.shape)
    # rdfs.append(rdf)
    # 105

    print('Check if LSC didnt change much')
    # irdf = rdf[
    #     abs(rdf['LSC f'] - rdf['LSC i']) < lsc_epsilon
    #     ]
    # print(irdf.to_string())
    # print(irdf.shape, df.shape)

    print('Remove onlypretrain of the onlyloadpretrained that did not satisfy the lsc')
    # nrdf = rdf[rdf['comments'].str.contains('onlyloadpretrained')].copy()
    # ardf = ardf[ardf['comments'].str.contains('onlyloadpretrained')]
    # brdf = brdf[brdf['comments'].str.contains('onlyloadpretrained')]
    #
    # # concatenate these 3 pandas
    # nrdf = pd.concat([nrdf, ardf, brdf])
    #
    # listem = []
    # for _, row in nrdf.iterrows():
    #     net = row['net']
    #     task = row['task']
    #     seed = row['seed']
    #     stack = row['stack']
    #     comments = row['comments'].replace('onlyloadpretrained', 'onlypretrain')
    #     # print(net, task, seed, stack, comments)
    #     rdf = plotdf[
    #         plotdf['net'].eq(net)
    #         & plotdf['task'].eq(task)
    #         & plotdf['seed'].eq(seed)
    #         & plotdf['stack'].eq(stack)
    #         & plotdf['comments'].eq(comments)
    #         ]
    #     listem.append(rdf)
    #     rdfs.append(rdf)
    #
    # if len(listem) > 0:
    #     listem = pd.concat(listem)
    #     print(listem.to_string())
    #     print(listem.shape, df.shape)

    # print('Keep pretraining')
    # rdf = plotdf[plotdf['comments'].str.contains('findLSC')]
    # rdfs.append(rdf)
    # print(rdf.head().to_string())
    # print(rdf.shape, df.shape)

    print('Remove if it didnt converge')
    # plotdf['conveps'] = plotdf['v_ppl len'] - plotdf['v_ppl argm']
    # rdf = plotdf[
    #     plotdf['conveps'] < 8
    #     ]
    #
    # print(rdf.to_string())
    # print(rdf.shape, df.shape)
    # rdfs.append(rdf)

    # 86

    print('Remove lsc na')
    # rdf = plotdf[
    #     plotdf['LSC f'].isna()
    # ]
    # remove_models = rdf.copy()
    # print(rdf.to_string())
    # print(rdf.shape, df.shape)
    # rdfs.append(rdf)

    print('Remove ppl and acc na and inf')
    # rdf = plotdf[
    #     plotdf['comments'].str.contains('onlyloadpretrained')
    #     & (
    #             plotdf['t_ppl'].isna()
    #             | plotdf['v_ppl'].isna()
    #             | plotdf['t_^acc'].isna()
    #             | plotdf['v_^acc'].isna()
    #             # or is infinity
    #             | plotdf['t_ppl'].eq(np.inf)
    #             | plotdf['v_ppl'].eq(np.inf)
    #             | plotdf['t_^acc'].eq(np.inf)
    #             | plotdf['v_^acc'].eq(np.inf)
    #     )
    #     ]
    # infrdf = rdf.copy()
    # print(rdf.to_string())
    # print(rdf.shape, df.shape)
    # rdfs.append(rdf)

    print('Remove pretrain that gave ppl and acc na and inf')
    # for _, row in infrdf.iterrows():
    #     net = row['net']
    #     task = row['task']
    #     seed = row['seed']
    #     stack = row['stack']
    #     comments = row['comments'].replace('onlyloadpretrained', 'onlypretrain')
    #     rdf = plotdf[
    #         plotdf['net'].eq(net)
    #         & plotdf['task'].eq(task)
    #         & plotdf['seed'].eq(seed)
    #         & plotdf['stack'].eq(stack)
    #         & plotdf['comments'].eq(comments)
    #         ]
    #     print(rdf.to_string())
    #     print(rdf.shape, df.shape)
    #     rdfs.append(rdf)

    print('Remove repeated experiments')
    brdf = mdf[mdf['counts'] > 4]
    print(brdf.to_string())

    for _, row in brdf.iterrows():
        print('-' * 80)
        srdf = plotdf[
            # (df['lr'] == row['lr'])
            (plotdf['comments'].eq(row['comments']))
            & (plotdf[depth_flag] == row[depth_flag])
            & (plotdf[task_flag] == row[task_flag])
            & (plotdf[net_flag] == row[net_flag])
            ].copy()

        # order wrt path column
        srdf = srdf.sort_values(by=['path'], ascending=False)

        # no duplicates
        gsrdf = srdf.drop_duplicates(subset=['jax_seed'])

        # remainder
        rdf = srdf[~srdf.apply(tuple, 1).isin(gsrdf.apply(tuple, 1))]
        print(srdf.to_string())
        print(rdf.to_string())
        print(rdf.shape)
        rdfs.append(rdf)

    allrdfs = pd.concat(rdfs)
    allrdfs = allrdfs.drop_duplicates()
    print(f'Remove {allrdfs.shape} of {plotdf.shape}')
    # trueallrdfs = allrdfs.drop_duplicates(subset=['seed', 'task', 'net', 'comments', 'stack'])
    # print(f'Remove actually {trueallrdfs.shape} of {plotdf.shape}')
    # allrdfs = allrdfs[allrdfs['comments'].str.contains('onlypretrain')]
    # print(f'Remove instead {allrdfs.shape} of {plotdf.shape}')

    if truely_remove_pretrained:

        # sdf = pd.read_hdf(h5path, 'df')
        sdf = remove_models.copy()
        print(sdf.head().to_string())
        sdf.loc[sdf['task'].str.contains('SHD'), 'task'] = 'heidelberg'
        sdf.loc[sdf['task'].str.contains('sl-MNIST'), 'task'] = 'sl_mnist'
        sdf.loc[sdf['task'].str.contains('PTB'), 'task'] = 'wordptb'

        sdf.loc[sdf['net'].eq('ALIFb'), 'net'] = 'maLSNNb'
        sdf.loc[sdf['net'].eq('ALIF'), 'net'] = 'maLSNN'

        coi = ['seed', 'task', 'net', 'comments', 'stack']
        experiments = []

        for _, row in sdf.iterrows():
            experiments.append({c: [row[c]] for c in coi})
        print(experiments)
        print(f'Experiments to remove: {len(experiments)}')
        folder = r'D:\work\alif_sg\good_experiments\pmodels'
        print(experiments)
        remove_pretrained_extra(experiments, remove_opposite=False, folder=folder, truely_remove=False)

    if truely_remove:
        for rdf in [allrdfs]:
            print(rdf['comments'])
            paths = rdf['path'].values
            for i, p in enumerate(paths):
                print(f'{i + 1}/{len(paths)} Removing {p}')
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

if missing_exps and expsid == 'als' and 'rnn-lru-' in GEXPERIMENTS[0]:
    print('Missing experiments')
    print(df.columns)
    # columns of interest
    coi = [
        'task', 'net', 'seed', 'stack', 'lr', 'n_neurons', 'comments',
        'batch_size', 'lr_schedule', 'optimizer_name'
    ]

    sdf = df.copy()
    print(sdf.head().to_string())

    for nice, brute in task_name_pairs:
        sdf.loc[df[task_flag].eq(nice), task_flag] = brute

    sdf.rename(columns={
        'eps': 'epochs', 'spe': 'steps_per_epoch'
    }, inplace=True)

    sdf.drop([c for c in sdf.columns if c not in coi], axis=1, inplace=True)
    sdf = sdf[sdf['task'].eq('wordptb')]

    sdf = sdf[sdf['n_neurons'].eq(256)]
    sdf = sdf[sdf['batch_size'].eq('64')]

    sdf = sdf[sdf['comments'].str.contains('maxlen:300_mlminputs_mlmeps:3')]
    sdf = sdf[sdf['lr'].eq(0.03)]
    sdf = sdf.astype(
        {
            'net': 'string', 'task': 'string', 'stack': 'string', 'comments': 'string',
            'lr_schedule': 'string', 'optimizer_name': 'string',
            'batch_size': 'float64', 'n_neurons': 'int64', 'seed': 'int64',
        }
    )
    sdf = sdf.astype({'batch_size': 'int64'})

    print(f'Experiments already done: {sdf.shape[0]}, len cols = {len(sdf.columns)}')

    # print column type
    print(sdf.dtypes)

    print(sdf.head().to_string())

    seed = 0
    n_seeds = 4
    seeds = [l + seed for l in range(n_seeds)]

    types = ['b']
    tasks = ['wordptb']
    stacks = ['3', '6']
    schedules = ['cosine_no_restarts']
    base_comment = 'allns_36_embproj_pretrained_'
    optimizer_name = 'SWAAdaBeliefLA'

    experiments = []
    for typ in types:
        for stack in stacks:
            comment = base_comment

            if str(stack) == '3':
                comment = comment.replace('allns_36', 'allns_36_dropout:.2')
            elif str(stack) == '6':
                comment = comment.replace('allns_36', 'allns_36_dropout:.1')
            else:
                raise NotImplementedError

            batch_sizes = [64]  # ['None']
            if typ == 'a':
                comments = [
                    comment + 'maxlen:150_mlminputs_mlmeps:3',
                    comment + 'maxlen:300_mlminputs_mlmeps:3',
                ]
                lrs = [3e-2]

            elif typ == 'b':
                comments = [
                    comment + 'maxlen:300_mlminputs_mlmeps:3',
                    comment + 'maxlen:300_mlminputs_mlmeps:3_findLSC_radius',
                    comment + 'maxlen:300_mlminputs_mlmeps:3_findLSC_radius_targetnorm:.5',
                    comment + 'maxlen:300_mlminputs_mlmeps:3_findLSC_radius_targetnorm:.5_unbalanced',
                    comment + 'maxlen:300_mlminputs_mlmeps:3_ffnlsc_findLSC_radius',
                    comment + 'maxlen:300_mlminputs_mlmeps:3_ffnlsc_findLSC_radius_targetnorm:.5',
                    comment + 'maxlen:300_mlminputs_mlmeps:3_ffnlsc_findLSC_radius_targetnorm:.5_unbalanced',
                    comment + 'maxlen:300_mlminputs_mlmeps:3_clipping',
                    comment + 'maxlen:300_mlminputs_mlmeps:3_findLSC_radius_clipping',
                    comment + 'maxlen:300_mlminputs_mlmeps:3_findLSC_radius_targetnorm:.5_clipping',
                    comment + 'maxlen:300_mlminputs_mlmeps:3_findLSC_radius_targetnorm:.5_unbalanced_clipping',
                    comment + 'maxlen:300_mlminputs_mlmeps:3_ffnlsc_findLSC_radius_clipping',
                    comment + 'maxlen:300_mlminputs_mlmeps:3_ffnlsc_findLSC_radius_targetnorm:.5_clipping',
                    comment + 'maxlen:300_mlminputs_mlmeps:3_ffnlsc_findLSC_radius_targetnorm:.5_unbalanced_clipping',
                ]
                lrs = [3e-2]

            else:
                raise NotImplementedError

            experiment = {
                'task': tasks,
                'net': nets, 'seed': seeds, 'stack': [stack],
                'lr': lrs,
                'n_neurons': [256],
                'comments': comments,
                'batch_size': batch_sizes,
                'lr_schedule': schedules, 'optimizer_name': [optimizer_name],
            }
            experiments.append(experiment)

    ds = dict2iter(experiments)
    _, experiments_left = complete_missing_exps(sdf, ds, coi)
    np.random.shuffle(experiments_left)
    experiments = experiments_left

    print(f'experiments =', experiments)
    print(f'# {len(experiments)}/{len(ds)}')

if missing_exps and expsid == 'fluctuations':
    print('Missing experiments')
    # columns of interest
    coi = [
        'seed', 'comments', 'epochs', 'dataset'
    ]

    sdf = df.copy()
    sdf['epochs'] = -1
    print(sdf.head().to_string())

    sdf.drop([c for c in sdf.columns if c not in coi], axis=1, inplace=True)
    print(f'Experiments already done: {sdf.shape[0]}, len cols = {len(sdf.columns)}')

    seed = 0
    n_seeds = 4
    seeds = [l + seed for l in range(n_seeds)]

    base_comments = ['deep', '']
    conds = ['', 'condI', 'condIV', 'condI_IV', 'condI_continuous', 'condIV_continuous', 'condI_IV_continuous']
    conds = [
        '', 'condIV', 'normcurv', 'condIV_continuous_normcurv', 'condIV_normcurv',
        'condIV_continuous_normcurv_oningrad', 'condIV_normcurv_oningrad',
        'condI_IV', 'condI_IV_continuous', 'condI_IV_continuous_oningrad',
        'condIV_forwback', 'condIV_normcurv_forwback',
        'condI_forwback', 'condI_normcurv_forwback',
        'condI_IV_forwback', 'condI_IV_normcurv_forwback',
    ]

    base_comments = ['smorms3_deep', 'smorms3', 'adabelief_deep', 'adabelief', ]
    base_comments = ['smorms3_deep', 'smorms3']


    experiments = []

    for dataset in ['dvs', 'shd', 'cifar10']:

        if dataset == 'dvs' or dataset == 'cifar10':
            more_lrs = [5e-5]
        else:
            more_lrs = [5e-1]

        more_lrs = []
        conds = ['', 'condIV', 'condIV_continuous']
        comments = [b if c == '' else c if b == '' else f'{b}_{c}' for b in base_comments for c in conds]
        comments_1 = [c + f'_lr:{lr}' for c in comments for lr in [5e-2, 5e-3, 5e-4] + more_lrs]

        conds = ['condIV_sgoutn', 'condIV_continuous_sgoutn']
        comments = [b if c == '' else c if b == '' else f'{b}_{c}' for b in base_comments for c in conds]
        comments_2 = [c + f'_lr:{lr}' for c in comments for lr in [5e-2, 5e-3, 5e-4] + more_lrs]

        conds = ['condIV_sgoutn_fanout', 'condIV_continuous_sgoutn_fanout']
        comments = [b if c == '' else c if b == '' else f'{b}_{c}' for b in base_comments for c in conds]
        comments_3 = [c + f'_lr:{lr}' for c in comments for lr in [5e-3] + more_lrs]

        if dataset == 'shd':
            conds = ['condIV_sgoutn_rfanout', 'condIV_continuous_sgoutn_rfanout']
            comments = [b if c == '' else c if b == '' else f'{b}_{c}' for b in base_comments for c in conds]
            comments_4 = [c + f'_lr:{lr}' for c in comments for lr in [5e-3] + more_lrs]
            comments_3 = comments_3 + comments_4

        comments_ = comments_1 + comments_2 + comments_3

        experiment = {
            'seed': seeds, 'epochs': [-1],
            'comments': comments_,
            'dataset': [dataset]
        }
        experiments.append(experiment)




    experiments = []
    conds = [
        'muone',
        'condIV_muone',

        'muone_noreg',
        'condIV_muone_noreg',
        'noreg',
        'condIV_noreg',

        'muone_regp5',
        'condIV_muone_regp5',
        'regp5',
        'condIV_regp5',

        'muone_regp5:.25',
        'regp5:.25',
        'muone_regp5:.75',
        'regp5:.75',
    ]
    comments = [b if c == '' else c if b == '' else f'{b}_{c}' for b in base_comments for c in conds]
    for dataset in ['dvs', 'shd', 'cifar10']:

        if dataset == 'dvs' or dataset == 'cifar10':
            lrs= [5e-4]
        else:
            lrs = [5e-2]
        comments = [c + f'_lr:{lr}' for c in comments for lr in lrs]

        experiment = {
            'seed': seeds, 'epochs': [-1],
            'comments': comments,
            'dataset': [dataset]
        }
        experiments.append(experiment)

    ds = dict2iter(experiments)
    _, experiments_left = complete_missing_exps(sdf, ds, coi)
    np.random.shuffle(experiments_left)
    experiments = experiments_left

    print(f'experiments =', experiments)
    print(f'# {len(experiments)}/{len(ds)}')

    exps_shd, exps_dvs, exps_cif = [], [], []
    for ex in experiments:
        if ex['dataset'][0] == 'shd':
            exps_shd.append(ex)
        elif ex['dataset'][0] == 'dvs':
            exps_dvs.append(ex)
        else:
            exps_cif.append(ex)

    print(f'experiments_shd =', exps_shd)
    print(f'# {len(exps_shd)}/{len(ds)}')
    print(f'experiments_dvs =', exps_dvs)
    print(f'# {len(exps_dvs)}/{len(ds)}')
    print(f'experiments_cif =', exps_cif)
    print(f'# {len(exps_cif)}/{len(ds)}')

if missing_exps and expsid == '_decolle':
    print('Missing experiments')
    # columns of interest
    coi = [
        'seed', 'comments', 'datasetname'
    ]

    sdf = df.copy()
    # sdf['epochs'] = -1
    # print(sdf.head().to_string())

    sdf.drop([c for c in sdf.columns if c not in coi], axis=1, inplace=True)
    print(f'Experiments already done: {sdf.shape[0]}, len cols = {len(sdf.columns)}')

    seed = 0
    n_seeds = 4
    seeds = [l + seed for l in range(n_seeds)]
    experiments = []
    base_comments = ['', 'condI', 'condIV', 'condI_IV']
    base_comments = ['']
    sgcurves = ['sgcurve:dfastsigmoid', 'sgcurve:triangular', 'sgcurve:rectangular', ]
    conds = [
        '', 'normcurv', 'condIV_continuous_normcurv', 'condIV_normcurv', 'condIV_continuous_normcurv_oningrad',
        'condIV_normcurv_oningrad', 'condI_IV', 'condI_IV_continuous', 'condI_IV_continuous_oningrad',
        'condIV_forwback', 'condIV_normcurv_forwback',
        'condI_forwback', 'condI_normcurv_forwback',
        'condI_IV_forwback', 'condI_IV_normcurv_forwback',
    ]

    comments = [b if c == '' else c if b == '' else f'{b}_{c}' for b in base_comments for c in conds]
    experiment = {
        'seed': seeds, 'datasetname': ['dvs', 'nmnist'],
        'comments': comments,
    }
    experiments.append(experiment)

    experiments = []

    base_comments = [
        '', 'condIV_continuous', 'condIV',
        'condIV_continuous_sgoutn', 'condIV_sgoutn',
        'condI_continuous_sgoutn', 'condI_sgoutn',
        'condI_IV_continuous_sgoutn', 'condI_IV_sgoutn',
    ]
    sgcurves = ['sgcurve:dfastsigmoid_adabelief', 'sgcurve:dfastsigmoid_adamax']
    comments = [b if c == '' else c if b == '' else f'{b}_{c}' for b in base_comments for c in sgcurves]
    experiment = {
        'seed': seeds, 'datasetname': ['dvs'],
        'comments': comments,
    }
    experiments.append(experiment)

    ds = dict2iter(experiments)
    print(ds)
    _, experiments_left = complete_missing_exps(sdf, ds, coi)
    np.random.shuffle(experiments_left)
    experiments = experiments_left

    print(f'experiments =', experiments)
    print(f'# {len(experiments)}/{len(ds)}')

    exps_nmnist, exps_dvs = [], []
    for ex in experiments:
        if ex['datasetname'][0] == 'dvs':
            exps_dvs.append(ex)
        else:
            exps_nmnist.append(ex)

    print(f'experiments_dvs =', exps_dvs)
    print(f'# {len(exps_dvs)}/{len(ds)}')
    print(f'experiments_nmnist =', exps_nmnist)
    print(f'# {len(exps_nmnist)}/{len(ds)}')

if missing_exps and not expsid == 's5lru' and False:
    # columns of interest
    coi = ['seed', 'task', 'net', 'comments', 'stack']

    import pandas as pd

    # sdf = pd.read_hdf(h5path, 'df')
    sdf = df.copy()

    sdf.loc[df['task'].eq('SHD'), 'task'] = 'heidelberg'
    sdf.loc[df['task'].eq('sl-MNIST'), 'task'] = 'sl_mnist'
    sdf.loc[df['task'].eq('PTB'), 'task'] = 'wordptb'
    sdf.loc[df['net'].eq('ALIFb'), 'net'] = 'maLSNNb'
    sdf.loc[df['net'].eq('ALIF'), 'net'] = 'maLSNN'
    sdf['comments'] = sdf['comments'].str.replace('_timerepeat:2', '_timerepeat:2_pretrained')
    fsdf = sdf.copy()

    sdf.drop([c for c in sdf.columns if c not in coi], axis=1, inplace=True)
    # print('Existing exps')
    # print(sdf.to_string())

    # add_flag = '_onlyloadpretrained'  # _onlyloadpretrained _onlypretrain
    # only_if_good_lsc = True
    seed = 0
    n_seeds = 4
    seeds = [l + seed for l in range(n_seeds)]

    net_types = {
        'nolsnns': ['LSTM', 'GRU', 'rsimplernn', 'ssimplernn'],  # 'indrnn',
        'lsnns': ['maLSNN', 'maLSNNb'],
    }
    tasks = ['heidelberg', 'sl_mnist', 'wordptb']

    incomplete_comments = 'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_pretrained'

    for add_flag in ['_onlyloadpretrained', '_onlypretrain']:  # ['_onlyloadpretrained', '_onlypretrain']:
        if add_flag == '_onlyloadpretrained':
            good_lsc_options = [True, False]
        else:
            good_lsc_options = [True]

        for only_if_good_lsc in good_lsc_options:
            if only_if_good_lsc:
                all_comments = [
                    incomplete_comments + f'_findLSC_radius' + add_flag,
                    incomplete_comments + f'_findLSC_radius_targetnorm:.5' + add_flag,
                ]
            else:
                all_comments = []

                if '_onlyloadpretrained' in add_flag:
                    all_comments.append(incomplete_comments + add_flag)

            experiments = []
            for nt, nets in net_types.items():
                comments = all_comments

                if nt == 'lsnns':
                    comments = [c for c in all_comments if not 'targetnorm:.5' in c]
                else:
                    comments = all_comments

                experiment = {
                    'task': tasks,
                    'net': nets, 'seed': seeds, 'stack': ['None'],
                    'comments': comments,
                }
                experiments.append(experiment)

                experiment = {
                    'task': ['heidelberg'],
                    'net': nets, 'seed': seeds, 'stack': ['1', '3', '5', '7'],
                    'comments': comments,
                }
                experiments.append(experiment)

                experiment = {
                    'task': ['sl_mnist', 'wordptb'],
                    'net': nets, 'seed': seeds, 'stack': ['5'],
                    'comments': comments,
                }
                experiments.append(experiment)

            ds = dict2iter(experiments)
            ldf, experiments_left = complete_missing_exps(sdf, ds, coi)
            np.random.shuffle(experiments_left)
            experiments = experiments_left

            # print('experiments_left', experiments_left)
            # print(len(experiments_left))

            if '_onlyloadpretrained' in add_flag and only_if_good_lsc and False:
                fsdf = fsdf[fsdf['comments'].str.contains('onlypretrain')]
                fsdf['comments'] = fsdf['comments'].str.replace('onlypretrain', 'onlyloadpretrained')

                fsdf['target'] = fsdf['comments'].apply(
                    lambda x: 0.5 if 'targetnorm:.5' in x else 1 if 'findLSC' in x else np.nan)
                fsdf['diff_target'] = abs(fsdf['LSC f'] - fsdf['target'])
                fsdf['vs_epsilon'] = fsdf['diff_target'] < lsc_epsilon

                fsdf = fsdf[fsdf['vs_epsilon']]
                fsdf.drop([c for c in fsdf.columns if c not in coi], axis=1, inplace=True)

                # intersection = fsdf[fsdf == ldf].dropna()
                intersection = pd.concat([ldf, fsdf], ignore_index=True)
                intersection = intersection[intersection.duplicated()]
                # intersection = pd.merge(fsdf, ldf, on=coi, how="inner")
                # print(fsdf.shape, ldf.shape, intersection.shape)

                # concatenate intersection and fsdf
                # print('ndf.shape, intersection.shape, fsdf.shape')
                ndf = pd.concat([intersection, fsdf], ignore_index=True)
                # remove duplicates
                # print(ndf.shape, intersection.shape, fsdf.shape)
                ndf.drop_duplicates(inplace=True)
                # print(ndf.shape, intersection.shape, fsdf.shape)

                # do the same for ldf
                # print('ndf.shape, intersection.shape, ldf.shape')
                ndf = pd.concat([intersection, ldf], ignore_index=True)
                # print(ndf.shape, intersection.shape, ldf.shape)
                ndf.drop_duplicates(inplace=True)
                # print(ndf.shape, intersection.shape, ldf.shape)

                ldf, experiments_left = complete_missing_exps(sdf, intersection, coi)
                np.random.shuffle(experiments_left)
                experiments = experiments_left

            np.random.shuffle(experiments)
            # print(add_flag, only_if_good_lsc)
            print(f'experiments{add_flag}_{str(only_if_good_lsc)} =', experiments)
            print('#', len(experiments))

            # for e in experiments:
            # if e['net'] == ['maLSNN']:
            # print(e)

if missing_exps and expsid == 's5lru':
    # columns of interest
    coi = ['jax_seed', 'dataset', 'lru', 'comments', 'epochs', 'steps_per_epoch', 'bsz']

    # import pandas as pd

    sdf = df.copy()

    for nice, brute in task_name_pairs:
        sdf.loc[df[task_flag].eq(nice), task_flag] = brute

    sdf.rename(columns={
        'eps': 'epochs', 'spe': 'steps_per_epoch'
    }, inplace=True)

    print(f'Experiments already done: {sdf.shape[0]}, len cols = {len(sdf.columns)}')
    sdf.drop([c for c in sdf.columns if c not in coi], axis=1, inplace=True)
    sdf = sdf.astype({'lru': 'string', 'dataset': 'string', 'comments': 'string'})

    seed = 0
    n_seeds = 2
    seeds = [l + seed for l in range(n_seeds)]

    experiments = []
    datasets = [
        'cifar-classification',
        'imdb-classification',
        'listops-classification',
        'aan-classification',
        # 'pathfinder-classification',
        # 'pathx-classification'
    ]

    ci = lambda islru: 'default' if not islru else 'defaultlru_lruv3'
    for lru in [True, False]:
        for dataset in datasets:
            if 'pathx' in dataset:
                bsz = 8
            elif 'pathfinder' in dataset or 'aan' in dataset:
                bsz = 16
            else:
                bsz = 32

            experiment = {
                'jax_seed': seeds,
                'epochs': [300], 'steps_per_epoch': [-1], 'dataset': [dataset], 'bsz': [bsz],
                'lru': [str(lru)],
                'comments': [
                    ci(lru),
                    ci(lru) + '_pretrain_targetnorm:1',
                    ci(lru) + '_pretrain_targetnorm:0.5',
                    # ci(lru) + '_pretrain_unbalanced',
                    ci(lru) + '_clipping',
                    # ================================================
                    ci(lru) + '_emaopt',
                    ci(lru) + '_pretrain_targetnorm:1_emaopt',
                    ci(lru) + '_pretrain_targetnorm:0.5_emaopt',
                    ci(lru) + '_clipping_emaopt',

                ],
            }
            experiments.append(experiment)

    ds = dict2iter(experiments)
    _, experiments_left = complete_missing_exps(sdf, ds, coi)
    np.random.shuffle(experiments_left)
    experiments = experiments_left

    np.random.shuffle(experiments)
    print(f'experiments =', experiments)
    print(f'# {len(experiments)}/{len(ds)}')

if check_all_norms:
    dirs = [d for d in os.listdir(EXPERIMENTS) if 'als' in d][:1000]
    print(dirs)


    def task_marker(task):
        if task == 'heidelberg':
            return 'o'
        elif task == 'sl_mnist':
            return 's'
        elif task == 'wordptb':
            return 'd'
        else:
            return 'x'


    def net_marker(net):
        if net == 'maLSNN':
            return 'o'
        if net == 'maLSNNb':
            return 'x'
        elif net == 'LSTM':
            return '_'
        elif net == 'GRU':
            return '3'
        elif net == 'rsimplernn':
            return 'd'
        elif net == 'ssimplernn':
            return 's'
        else:
            return '.'


    # make subplots
    fig, axs = plt.subplots(2, 2, figsize=(10, 5))
    exceptions = []
    tasks = ['heidelberg', 'sl_mnist', 'wordptb']
    nets = ['maLSNN', 'maLSNNb', 'LSTM', 'GRU', 'rsimplernn', 'ssimplernn']

    # dirs = dirs[:10]
    for i, d in tqdm(enumerate(dirs), total=len(dirs)):
        # if i > 230 and i < 235:
        try:
            # print('-' * 30)
            config_path = os.path.join(EXPERIMENTS, d, '1', 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            # print(i)
            # print(config['comments'])
            # print(config['net'])
            # print(config['task'])
            # print()

            c = 'g' if 'targetnorm:.5' in config['comments'] else 'r'
            c = c if 'findLSC' in config['comments'] else 'k'
            results_path = os.path.join(EXPERIMENTS, d, 'other_outputs', 'results.json')
            with open(results_path, 'r') as f:
                results = json.load(f)
            norms = results['save_norms']
            keys = list(norms.keys())
            last_batch = np.unique([k[:8] for k in keys])[-1]
            keys = [k for k in keys if last_batch in k]
            keys.sort()

            for k in keys:
                # print(k)

                if not norms[k] == [-1] and not norms[k] == []:
                    if not 'dec' in k:
                        axs[0, 0].scatter(i, norms[k][-1], c=c, marker=net_marker(config['net']))
                        axs[1, 0].scatter(i, norms[k][-1], c=c, marker=task_marker(config['task']))
                    else:
                        axs[0, 1].scatter(i, norms[k][-1], c=c, marker=net_marker(config['net']),
                                          label=config['net'])
                        axs[1, 1].scatter(i, norms[k][-1], c=c, marker=task_marker(config['task']),
                                          label=config['task'])

        except Exception as e:
            exceptions.append(e)

    print('Exceptions:')
    for i, e in enumerate(exceptions):
        print(f'{i}/{len(exceptions)}', e)

    for i, ax in enumerate(axs.reshape(-1)):
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)

    net_elements = [Line2D([0], [0], color='k', lw=2, label=n, linestyle='None', marker=net_marker(n), markersize=5)
                    for n in nets]
    task_elements = [Line2D([0], [0], color='k', lw=2, label=t, linestyle='None', marker=task_marker(t), markersize=5)
                     for t in tasks]
    axs[0, 1].legend(ncol=3, handles=net_elements, fontsize=10)
    axs[1, 1].legend(ncol=3, handles=task_elements, fontsize=10)

    plot_filename = f'experiments/manynormsperexps.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')

    plt.show()
