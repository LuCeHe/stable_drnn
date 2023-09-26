import os, json, copy, time, shutil, math

from alif_sg.tools.admin_model_removal import remove_pretrained_extra
from pyaromatics.keras_tools.esoteric_layers.rate_voltage_reg import RateVoltageRegularization

from pyaromatics.stay_organized.submit_jobs import dict2iter
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib as mpl

import pickle
from matplotlib.lines import Line2D
from pyaromatics.stay_organized.mpl_tools import load_plot_settings
from pyaromatics.stay_organized.pandardize import experiments_to_pandas, complete_missing_exps
from pyaromatics.stay_organized.standardize_strings import shorten_losses
from pyaromatics.stay_organized.utils import str2val
from alif_sg.neural_models.recLSC import load_LSC_model
from alif_sg.tools.plot_tools import *
from sg_design_lif.neural_models import maLSNN

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
    r'D:\work\alif_sg\good_experiments\2023-09-01--rnn-lru-first',
]

expsid = 'als'  # effnet als ffnandcnns
h5path = os.path.join(EXPERIMENTS, f'summary_{expsid}.h5')

lsc_epsilon = 0.02  # 0.02

check_for_new = True
plot_losses = True
one_exp_curves = False
pandas_means = True
show_per_tasknet = False
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

missing_exps = False
remove_incomplete = False
truely_remove = False
truely_remove_pretrained = False
check_all_norms = False

task = 'ps_mnist'  # heidelberg wordptb sl_mnist all ps_mnist
incomplete_comments = '36_embproj_nogradreset_dropout:.3_timerepeat:2_lscdepth:1_pretrained_'

# sparse_mode_accuracy sparse_categorical_crossentropy bpc sparse_mode_accuracy_test_10
# val_sparse_mode_accuracy test_perplexity
metric = 'v_mode_acc'  # 'v_ppl min'
metric = shorten_losses(metric)
optimizer_name = 'SWAAdaBelief'  # SGD SWAAdaBelief
metrics_oi = [
    # 't_ppl min', 't_mode_acc max', 'v_ppl min', 'v_mode_acc max',
    't_ppl', 't_mode_acc', 'v_ppl', 'v_mode_acc', 'val_ppl m', 'val_mode_acc M',
    'LSC_norms i', 'LSC_norms f', 'LSC_norms mean',
    'final_norms_mean', 'final_norms_std', 'best_std_ma_norm', 'std_ma_norm',
]
metrics_oi = [shorten_losses(m) for m in metrics_oi]

plot_only = ['seed', 'net', 'task', 'n_params', 'stack', 'comments', 'path', 'lr', 'host_hostname',
             'v_ppl argm', 'v_ppl len', ] + metrics_oi

columns_to_remove = [
    '_var', '_mean', 'sparse_categorical_crossentropy', 'bpc', 'loss', 'artifacts',
    'experiment_dependencies', 'experiment_sources', 'experiment_repositories', 'host_os',
    'sparse_categorical_accuracy', 'LSC_losses', 'rec_norms', 'fail_trace', 'list', 'weights_shapes'
]
force_keep_column = [
    'LSC_norms list', 'batch ',
    'val_sparse_mode_accuracy list', 'val_perplexity list',
    'v_sparse_mode_accuracy list', 'v_perplexity list',
    't_sparse_mode_accuracy list', 't_perplexity list',
    'final_norms_mean', 'final_norms_std'
]

df = experiments_to_pandas(
    h5path=h5path, zips_folder=GEXPERIMENTS, unzips_folder=EXPERIMENTS, experiments_identifier=expsid,
    exclude_files=['cout.txt'], check_for_new=check_for_new,
    exclude_columns=columns_to_remove, force_keep_column=force_keep_column
)
# df = df[~df['comments'].str.contains('randlsc')]

# print(df.to_string())
# df = df[~df['stack'].str.contains('4:3', na=False)]
df['stack'] = df['stack'].fillna(-1).astype(int)
df = df.replace(-1, 'None')
df['stack'] = df['stack'].astype(str)
df['comments'] = df['comments'].str.replace('simplereadout', 'embproj')
df['batch_size'] = df['batch_size'].astype(str)
df['comments'] = df['comments'].str.replace('_pretrained', '')
df['comments'] = df['comments'].astype(str)
df.replace(['nan'], np.nan, inplace=True)

new_column_names = {c_name: shorten_losses(c_name) for c_name in df.columns}
df.rename(columns=new_column_names, inplace=True)

for c in ['t_ppl', 't_^acc', 'v_ppl', 'v_^acc']:
    # if column doesn't exist, create a NaN column
    if c not in df.columns:
        df[c] = np.nan

for cname in ['net', 'task']:
    if isinstance(df[cname], pd.DataFrame):
        c = df[cname].iloc[:, 0].fillna(df[cname].iloc[:, 1])
        df = df.drop([cname], axis=1)
        df[cname] = c

if 'net' in df.columns: df['net'] = df['net'].astype(str)
if 'task' in df.columns: df['task'] = df['task'].astype(str)

if 'n_params' in df.columns:
    df['n_params'] = df['n_params'].apply(lambda x: large_num_to_reasonable_string(x, 1))

print(list(df.columns))


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

    plot_metric = 'rec_norms list'
    plot_metric = 'val_ppl list'
    # plot_metric = 'val_^acc list'
    # plot_metric = 'LSC list'
    # plot_metric = 'norms dec layer 1 list'  # enc depth rec dec
    tasks = df['task'].unique()
    nets = df['net'].unique()
    comments = df['comments'].unique()
    comments = [c.replace('_onlypretrain', '') for c in comments]
    df = df[~df['comments'].str.contains('randlsc')]
    print(comments)
    print(tasks)
    print(nets)
    fig, axs = plt.subplots(len(tasks), len(nets), figsize=(len(nets) * 3, len(tasks) * 3),
                            gridspec_kw={'wspace': .2, 'hspace': 0.8})
    # add axis to axs if one dimensional
    if len(tasks) == 1:
        axs = np.expand_dims(axs, axis=0)
    if len(nets) == 1:
        axs = np.expand_dims(axs, axis=1)

    for i, task in enumerate(tasks):
        for j, net in enumerate(nets):
            if not net == 'LIF':
                print('-===-' * 30)
                print(task, net)
                idf = df[df['task'].str.contains(task) & df['net'].str.contains(net)]

                for _, row in idf.iterrows():
                    n = row['comments']
                    # c = 'r' if 'LSC' in row['comments'] else 'b'
                    # c = 'r' if 'supsub' in row['comments'] else 'b'
                    if isinstance(row[plot_metric], list):
                        # print(row['loss list'])
                        print('epochs', len(row[plot_metric]), n)
                        axs[i, j].plot(row[plot_metric], color=lsc_colors(n), label=lsc_clean_comments(n))
                    else:
                        print(row[plot_metric], row['path'], row['comments'])

                axs[i, j].set_title(f'{task} {net}')

    legend_elements = [Line2D([0], [0], color=lsc_colors(n), lw=4, label=lsc_clean_comments(n)) for n in comments]
    plt.legend(ncol=3, handles=legend_elements, loc='lower center')  # , bbox_to_anchor=(-.1, -1.))

    plt.show()

if 'net' in df.columns:
    # df.loc[df['comments'].str.contains('noalif'), 'net'] = 'LIF'
    df.loc[df['net'].eq('maLSNNb'), 'net'] = 'ALIFb'
    df.loc[df['net'].eq('maLSNN'), 'net'] = 'ALIF'

if 'task' in df.columns:
    df.loc[df['task'].eq('heidelberg'), 'task'] = 'SHD'
    df.loc[df['task'].eq('sl_mnist'), 'task'] = 'sl-MNIST'
    df.loc[df['task'].eq('wordptb'), 'task'] = 'PTB'

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
    # plotdf = plotdf[plotdf['task'].str.contains('PTB')]
    print(plotdf.to_string())

    print('\n\n\n')
    adf = plotdf[
        plotdf['task'].str.contains('PTB')
        & plotdf['net'].str.contains('ALIF')
        & plotdf['comments'].str.contains('target')
        ]
    print(adf.to_string())


if pandas_means:
    # group_cols = ['net', 'task', 'comments', 'stack', 'lr']
    group_cols = ['net', 'task', 'comments', 'stack']

    counts = df.groupby(group_cols).size().reset_index(name='counts')
    # stats = ['mean', 'std']
    metrics_oi = [shorten_losses(m) for m in metrics_oi]
    stats_oi = ['mean', 'std']
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

    tasks = sorted(np.unique(mdf['task']))
    nets = sorted(np.unique(mdf['net']))
    stacks = sorted(np.unique(mdf['stack']))

    mdf['comments'] = mdf['comments'].str.replace('__', '_', regex=True)
    print(mdf.to_string())

    if show_per_tasknet:
        xdf = mdf.copy()
        xdf['comments'] = xdf['comments'].str.replace('allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_', '')
        for stack in stacks:
            for task in tasks:
                for net in nets:
                    print('-===-' * 30)
                    print(task, net, stack)

                    idf = xdf[
                        xdf['task'].eq(task)
                        & xdf['net'].eq(net)
                        & xdf['stack'].eq(stack)
                        ]

                    cols = idf.columns
                    if not 'PTB' in task:
                        idf = idf.sort_values(by='mean_' + metric, ascending=False)
                        cols = [c for c in cols if not 'ppl' in c]
                    else:
                        idf = idf.sort_values(by='mean_v_ppl', ascending=True)
                        cols = [c for c in cols if not 'acc' in c]

                    idf.rename(columns=new_column_names, inplace=True)
                    idf = idf[cols]

                    print(idf.to_string())


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
    plt.xticks([r + barWidth/2 for r in range(len(stack_2))],
               # ['half > one', 'half > none', 'one > none\n(ALIFs)', ],
               [r'$\rho_t=0.5$' + '\nbetter than\n' + r'$\rho_t=1$',
                r'$\rho_t=0.5$' + '\nbetter than\nnone',
                r'$\rho_t=1$'+'\nbetter than\nnone\n(ALIFs)',
                r'$\rho_t=1$'+'\nbetter than\nnone\n(no ALIFs)',
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

    print('Eliminate if f_norms_std too large')
    rdf = plotdf[
        (plotdf['f_norms_std'] > .2)
        & plotdf['comments'].str.contains('findLSC')
        ]
    ardf = rdf.copy()
    print(rdf.to_string())
    print(rdf.shape, df.shape)
    rdfs.append(rdf)

    print('Eliminate if std_ma_norm too large')
    rdf = plotdf[
        (plotdf['std_ma_norm'] > .2)
        & plotdf['comments'].str.contains('findLSC')
        # & plotdf['comments'].str.contains('onlypretrain')
        ]
    brdf = rdf.copy()
    print(rdf.to_string())
    print(rdf.shape, df.shape)
    rdfs.append(rdf)

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
    plotdf['target'] = plotdf['comments'].apply(
        lambda x: 0.5 if 'targetnorm:.5' in x else 1 if 'findLSC' in x else np.nan
    )

    # plotdf['diff_target'] = abs(plotdf['LSC f'] - plotdf['target'])
    # plotdf['vs_epsilon'] = plotdf['diff_target'] > lsc_epsilon
    plotdf['vs_epsilon'] = ((abs(plotdf['LSC a'] - plotdf['target']) > lsc_epsilon)
                            & plotdf['comments'].str.contains('onlyloadpretrained')) \
                           | ((abs(plotdf['LSC f'] - plotdf['target']) > lsc_epsilon)
                              & plotdf['comments'].str.contains('onlypretrain'))

    rdf = plotdf[
        plotdf['comments'].str.contains('findLSC')
        & plotdf['vs_epsilon']
        ]
    print(rdf.to_string())
    print(rdf.shape, df.shape)
    rdfs.append(rdf)
    # 105

    print('Check if LSC didnt change much')
    irdf = rdf[
        abs(rdf['LSC f'] - rdf['LSC i']) < lsc_epsilon
        ]
    print(irdf.to_string())
    print(irdf.shape, df.shape)

    print('Remove onlypretrain of the onlyloadpretrained that did not satisfy the lsc')
    nrdf = rdf[rdf['comments'].str.contains('onlyloadpretrained')].copy()
    ardf = ardf[ardf['comments'].str.contains('onlyloadpretrained')]
    brdf = brdf[brdf['comments'].str.contains('onlyloadpretrained')]

    # concatenate these 3 pandas
    nrdf = pd.concat([nrdf, ardf, brdf])

    listem = []
    for _, row in nrdf.iterrows():
        net = row['net']
        task = row['task']
        seed = row['seed']
        stack = row['stack']
        comments = row['comments'].replace('onlyloadpretrained', 'onlypretrain')
        # print(net, task, seed, stack, comments)
        rdf = plotdf[
            plotdf['net'].eq(net)
            & plotdf['task'].eq(task)
            & plotdf['seed'].eq(seed)
            & plotdf['stack'].eq(stack)
            & plotdf['comments'].eq(comments)
            ]
        listem.append(rdf)
        rdfs.append(rdf)

    if len(listem) > 0:
        listem = pd.concat(listem)
        print(listem.to_string())
        print(listem.shape, df.shape)

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
    rdf = plotdf[
        plotdf['LSC f'].isna()
    ]
    remove_models = rdf.copy()
    print(rdf.to_string())
    print(rdf.shape, df.shape)
    rdfs.append(rdf)

    print('Remove ppl and acc na and inf')
    rdf = plotdf[
        plotdf['comments'].str.contains('onlyloadpretrained')
        & (
                plotdf['t_ppl'].isna()
                | plotdf['v_ppl'].isna()
                | plotdf['t_^acc'].isna()
                | plotdf['v_^acc'].isna()
                # or is infinity
                | plotdf['t_ppl'].eq(np.inf)
                | plotdf['v_ppl'].eq(np.inf)
                | plotdf['t_^acc'].eq(np.inf)
                | plotdf['v_^acc'].eq(np.inf)
        )
        ]
    infrdf = rdf.copy()
    print(rdf.to_string())
    print(rdf.shape, df.shape)
    rdfs.append(rdf)

    print('Remove pretrain that gave ppl and acc na and inf')
    for _, row in infrdf.iterrows():
        net = row['net']
        task = row['task']
        seed = row['seed']
        stack = row['stack']
        comments = row['comments'].replace('onlyloadpretrained', 'onlypretrain')
        rdf = plotdf[
            plotdf['net'].eq(net)
            & plotdf['task'].eq(task)
            & plotdf['seed'].eq(seed)
            & plotdf['stack'].eq(stack)
            & plotdf['comments'].eq(comments)
            ]
        print(rdf.to_string())
        print(rdf.shape, df.shape)
        rdfs.append(rdf)

    print('Remove repeated experiments')
    brdf = mdf[mdf['counts'] > 4]
    print(brdf.to_string())

    for _, row in brdf.iterrows():
        print('-' * 80)
        srdf = plotdf[
            # (df['lr'] == row['lr'])
            (plotdf['comments'].eq(row['comments']))
            & (plotdf['stack'] == row['stack'])
            & (plotdf['task'] == row['task'])
            & (plotdf['net'] == row['net'])
            ].copy()

        # order wrt path column
        srdf = srdf.sort_values(by=['path'], ascending=False)

        # no duplicates
        gsrdf = srdf.drop_duplicates(subset=['seed'])

        # remainder
        rdf = srdf[~srdf.apply(tuple, 1).isin(gsrdf.apply(tuple, 1))]
        print(srdf.to_string())
        print(rdf.to_string())
        print(rdf.shape)
        rdfs.append(rdf)

    allrdfs = pd.concat(rdfs)
    allrdfs = allrdfs.drop_duplicates()
    print(f'Remove {allrdfs.shape} of {plotdf.shape}')
    trueallrdfs = allrdfs.drop_duplicates(subset=['seed', 'task', 'net', 'comments', 'stack'])
    print(f'Remove actually {trueallrdfs.shape} of {plotdf.shape}')
    allrdfs = allrdfs[allrdfs['comments'].str.contains('onlypretrain')]
    print(f'Remove instead {allrdfs.shape} of {plotdf.shape}')

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

if missing_exps:
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
