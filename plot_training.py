import os, json, copy, time, shutil

from GenericTools.keras_tools.esoteric_layers import AddLossLayer, AddMetricsLayer
from GenericTools.keras_tools.esoteric_layers.rate_voltage_reg import RateVoltageRegularization

from GenericTools.stay_organized.submit_jobs import dict2iter
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib as mpl

import pickle
from matplotlib.lines import Line2D
from GenericTools.stay_organized.mpl_tools import load_plot_settings
from GenericTools.stay_organized.pandardize import experiments_to_pandas, complete_missing_exps
from GenericTools.stay_organized.standardize_strings import shorten_losses
from GenericTools.stay_organized.utils import str2val
from alif_sg.neural_models.recLSC import remove_pretrained_extra
from alif_sg.tools.plot_tools import *
from sg_design_lif.neural_models import maLSNN

mpl, pd = load_plot_settings(mpl=mpl, pd=pd)

import matplotlib.pyplot as plt
from datetime import timedelta, datetime

from GenericTools.stay_organized.plot_tricks import large_num_to_reasonable_string

FMT = '%Y-%m-%dT%H:%M:%S'
from GenericTools.stay_organized.unzip import unzip_good_exps

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
EXPERIMENTS = r'D:\work\alif_sg\experiments'
GEXPERIMENTS = [
    # os.path.join(CDIR, 'good_experiments'),
    # os.path.join(CDIR, 'good_experiments', '2022-11-07--complete_set_of_exps'),
    # r'D:\work\alif_sg\experiments',
    # r'D:\work\alif_sg\good_experiments\2022-12-21--rnn',
    r'D:\work\alif_sg\good_experiments\2023-01-20--rnn-v2',
]

expsid = 'als'  # effnet als ffnandcnns
h5path = os.path.join(EXPERIMENTS, f'summary_{expsid}.h5')

check_for_new = True
plot_losses = False
one_exp_curves = False
pandas_means = True
show_per_tasknet = True
make_latex = False
missing_exps = True
plot_lsc_vs_naive = False
plot_dampenings_and_betas = False
plot_norms_pretraining = False
plot_weights = False
plot_init_lrs = False
plot_lrs = False
plot_bars = False

remove_incomplete = False
truely_remove = False
truely_remove_pretrained = False
remove_saved_model = False

task = 'ps_mnist'  # heidelberg wordptb sl_mnist all ps_mnist
incomplete_comments = '36_embproj_nogradreset_dropout:.3_timerepeat:2_lscdepth:1_pretrained_'

# sparse_mode_accuracy sparse_categorical_crossentropy bpc sparse_mode_accuracy_test_10
# val_sparse_mode_accuracy test_perplexity
metric = 'v_mode_acc'  # 'v_ppl min'
metric = shorten_losses(metric)
optimizer_name = 'SWAAdaBelief'  # SGD SWAAdaBelief
metrics_oi = [
    # 't_ppl min', 't_mode_acc max', 'v_ppl min', 'v_mode_acc max',
    't_ppl', 't_mode_acc', 'v_ppl', 'v_mode_acc',
    'LSC_norms i', 'LSC_norms f'
]
metrics_oi = [shorten_losses(m) for m in metrics_oi]

plot_only = ['eps', 'net', 'task', 'n_params', 'stack', 'comments', 'path', 'lr', 'seed', 'host_hostname'] + metrics_oi
columns_to_remove = [
    'heaviside', '_test', 'weight', 'sLSTM_factor', 'save_model', 'clipnorm', 'GPU', 'batch_size',
    'continue_training', 'embedding', 'lr_schedule', 'loss_name', 'seed', 'stack', 'stop_time',
    'convergence', 'n_neurons', 'optimizer_name', 'LSC', ' list', 'artifacts', 'command', 'heartbeat', 'meta',
    'resources', 'host', 'start_time', 'status', 'experiment', 'result',
]
columns_to_remove = []
columns_to_remove = [
    '_var', '_mean', 'sparse_categorical_crossentropy', 'bpc', 'loss', 'artifacts',
    'experiment_dependencies', 'experiment_sources', 'experiment_repositories', 'host_os',
    'sparse_categorical_accuracy', 'LSC_losses', 'rec_norms', 'fail_trace', 'list']
force_keep_column = [
    'LSC_norms list', 'batch ',
    'val_sparse_mode_accuracy list', 'val_perplexity list',
    'v_sparse_mode_accuracy list', 'v_perplexity list',
    't_sparse_mode_accuracy list', 't_perplexity list',
]

df = experiments_to_pandas(
    h5path=h5path, zips_folder=GEXPERIMENTS, unzips_folder=EXPERIMENTS, experiments_identifier=expsid,
    exclude_files=['cout.txt'], check_for_new=check_for_new,
    exclude_columns=columns_to_remove, force_keep_column=force_keep_column
)
# df = df[~df['comments'].str.contains('randlsc')]

df = df[~df['stack'].str.contains('4:3', na=False)]
df['stack'] = df['stack'].fillna(-1).astype(int)
df = df.replace(-1, 'None')
df['stack'] = df['stack'].astype(str)
df['batch_size'] = df['batch_size'].astype(str)
df['comments'] = df['comments'].str.replace('_pretrained', '')
df['comments'] = df['comments'].astype(str)

new_column_names = {c_name: shorten_losses(c_name) for c_name in df.columns}
df.rename(columns=new_column_names, inplace=True)

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
norms_cols = [c for c in df.columns if 'save_norms' in c and 'batch 0' in c and 'list' in c]
for c in norms_cols:
    new_c = c.replace('_batch 0 ', '')
    tag = new_c.replace('save_norms', '')
    title = new_c.replace('save_norms', 'norms ')
    tag_cols = [c for c in df.columns if tag in c]

    print(title)
    df[title] = df.apply(
        lambda row:
        np.concatenate([row[k] if isinstance(row[k], list) else [] for k in tag_cols]).tolist() \
            if len(tag_cols) > 0 else [],
        axis=1
    )

print(list(df.columns))

if plot_losses:
    df['comments'] = df['comments'].str.replace('allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_', '')

    plot_metric = 'rec_norms list'
    # plot_metric = 'val_perplexity list'
    # plot_metric = 'val_^acc list'
    # plot_metric = 'LSC list'
    plot_metric = 'norms dec layer 1 list'  # enc depth rec dec
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
    df.loc[df['comments'].str.contains('noalif'), 'net'] = 'LIF'
    df.loc[df['net'].str.contains('maLSNNb'), 'net'] = 'ALIFb'
    df.loc[df['net'].str.contains('maLSNN'), 'net'] = 'ALIF'

if 'task' in df.columns:
    df.loc[df['task'].str.contains('heidelberg'), 'task'] = 'SHD'
    df.loc[df['task'].str.contains('sl_mnist'), 'task'] = 'sl-MNIST'
    df.loc[df['task'].str.contains('wordptb'), 'task'] = 'PTB'

# eps column stays eps if not equal to None else it becomes the content of v_mode_acc len
df.loc[df['eps'].isnull(), 'eps'] = df.loc[df['eps'].isnull(), f'{metric} len']

if 'v_^acc len' in df.columns:
    # # FIXME: 14 experiments got nans in the heidelberg task validation, plot them anyway?
    print(list(df.columns))
    print('v_mode_acc nans:', df['v_^acc len'].isna().sum())
    print('t_ppl nans:', df['t_ppl list'].isna().sum())
    # df['v_ppl argm'] = df['v_ppl argm'].astype(int)
    # df['v_^acc argM'] = df['v_^acc argM'].astype(int)

    df['v_ppl'] = df['v_ppl m']
    # df['t_ppl'] = df.apply(lambda row: row['t_ppl list'][row['v_ppl argm']], axis=1)
    df['v_^acc'] = df['v_^acc M']
    # df['t_^acc'] = df.apply(lambda row: row['t_^acc list'][row['v_^acc argM']], axis=1)

    # FIXME: following is incorrect, correct it as soon as you get rid of the NaNs
    df['t_ppl'] = df['t_ppl m']
    df['t_^acc'] = df['t_^acc M']

for c_name in columns_to_remove:
    df = df[df.columns.drop(list(df.filter(regex=c_name)))]

if metric in df.keys():
    df = df.sort_values(by=metric)

print(list(df.columns))
if not plot_only is None:
    plotdf = df[plot_only]
    # plotdf = plotdf[plotdf['task'].str.contains('PTB')]
    print(plotdf.to_string())

if one_exp_curves:
    for _ in range(6):
        plt.close()
        try:
            print(df['path'].sample(1).values[0])
            id = '2022-12-06--20-00-13--0826--als_'
            id = '2022-12-06--19-55-03--9822--als_'
            id = os.path.split(df['path'].sample(1).values[0])[1]
            print(id)
            res_path = os.path.join(EXPERIMENTS, id, 'other_outputs', 'results.json')
            config_path = os.path.join(EXPERIMENTS, id, '1', 'config.json')

            with open(res_path) as f:
                res = json.load(f)
            with open(config_path) as f:
                con = json.load(f)

            fig, axs = plt.subplots(2, 2)
            fig.suptitle(con['comments'])

            for k in res.keys():
                if 'mean' in k:
                    curve = np.array([float(x) for x in res[k][1:-1].split(',')])
                    axs[0, 0].plot(curve)
                    axs[0, 0].title.set_text('mean')

                if 'var' in k:
                    print(res[k])
                    curve = np.array([float(x) for x in res[k][1:-1].split(',')])
                    axs[0, 1].plot(curve)
                    axs[0, 1].title.set_text('variance')

                if 'LSC_losses' in k:
                    curve = np.array([float(x) for x in res[k][1:-1].split(',')])
                    axs[1, 0].plot(curve)
                    axs[1, 0].title.set_text('lsc loss')

                if 'LSC_norms' in k:
                    curve = np.array([float(x) for x in res[k][1:-1].split(',')])
                    axs[1, 1].plot(curve)
                    axs[1, 1].title.set_text('lsc norm')

        except Exception as e:
            print(e)

        plt.show()

if pandas_means:
    # group_cols = ['net', 'task', 'comments', 'stack', 'lr']
    group_cols = ['net', 'task', 'comments', 'stack']
    df['lr'] = df['lr i']
    # group_cols = ['net', 'task', 'comments', 'stack', 'lr i']
    counts = df.groupby(group_cols).size().reset_index(name='counts')
    stats = ['mean']  # ['mean', 'std']
    metrics_oi = [shorten_losses(m) for m in metrics_oi]
    stats_oi = ['mean']  # ['mean', 'std']
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

                    if not 'wordptb' in task:
                        idf = idf.sort_values(by='mean_' + metric, ascending=False)
                    else:
                        idf = idf.sort_values(by='mean_v_ppl', ascending=True)

                    # new_column_names = {c_name: shorten_losses(c_name) for c_name in idf.columns}
                    # print(new_column_names)
                    idf.rename(columns=new_column_names, inplace=True)

                    print(idf.to_string())

    print('-===-' * 30)
    print(mdf[mdf['counts'] > 4].to_string())

    if make_latex:

        net = 'LSTM'  # ALIF LSTM

        idf = mdf[mdf['net'].str.contains(net)]
        # print(mdf[mdf['net'].str.contains('ALIF')].shape, mdf[mdf['net'].str.contains('LSTM')].shape)

        idf = idf[~idf['comments'].str.contains('reoldspike')]
        idf = idf[~idf['comments'].str.contains('savelscweights')]

        # print(idf.to_string())

        metrics_cols = [c for c in idf.columns if 'ppl' in c or 'acc' in c]
        for m in metrics_cols:
            mode = 'max' if 'acc' in m and not 'std' in m else 'min'
            idf[f'best_{m}'] = idf.groupby(['task'])[m].transform(mode)
            idf[m] = idf.apply(bolden_best(m), axis=1)

        idf['ppl'] = idf.apply(compactify_metrics('ppl min'), axis=1)
        idf['acc'] = idf.apply(compactify_metrics('mode_acc max'), axis=1)
        idf['metric'] = idf.apply(choose_metric, axis=1)
        idf = idf[idf.columns.drop(list(idf.filter(regex='acc')) + list(idf.filter(regex='ppl')))]
        idf = idf[idf.columns.drop(['counts', 'initializer', 'net'])]

        idf['comments'] = idf['comments'].str.replace('34_embproj_nogradreset_dropout:.3_timerepeat:2_', '', regex=True)
        idf['comments'] = idf['comments'].str.replace('find', '', regex=True)
        idf['comments'] = idf['comments'].str.replace('normpow:-1', r'\\infty', regex=True)
        idf['comments'] = idf['comments'].str.replace('normpow:', '', regex=True)
        idf['comments'] = idf['comments'].str.replace('_gausslsc', ' + g', regex=True)
        idf['comments'] = idf['comments'].str.replace('_berlsc', ' + b', regex=True)
        idf['comments'] = idf['comments'].str.replace('_randwlsc', ' + c', regex=True)
        idf['comments'] = idf['comments'].str.replace('_shufflelsc', ' + s', regex=True)
        idf['comments'] = idf['comments'].str.replace('gaussbeta', r'\\beta', regex=True)
        idf['comments'] = idf['comments'].str.replace(r'LSC_2_\\beta', r'\\beta LSC_2', regex=True)
        idf['comments'] = idf['comments'].str.replace('_lscdepth:1_lscout:0', '^{(d)}', regex=True)
        idf['comments'] = idf['comments'].str.replace('_lscdepth:1_lscout:1', '^{(dr)}', regex=True)
        # idf['comments'] = idf['comments'].str.replace('_', '', regex=True)
        idf = idf[~idf['comments'].str.contains('timerepeat:1')]
        idf['comments'] = idf['comments'].replace(r'^\s*$', 'no LSC', regex=True)
        idf['comments'] = idf['comments'].replace(r'\\beta$', r'\\beta no LSC', regex=True)
        idf['comments'] = idf['comments'].replace(r'\\beta', r'$\\beta$', regex=True)

        conditions = np.unique(idf['comments'])
        tasks = ['sl-MNIST', 'SHD', 'PTB']
        order_conditions = [
            'no LSC', 'LSC_1', 'LSC_2', r'LSC_\infty',
            'LSC_2 + g', 'LSC_2 + b', 'LSC_2 + s', 'LSC_2 + c',
            r'$\beta$ no LSC', r'$\beta$ LSC_2', r'$\beta$ LSC_2 + g', r'$\beta$ LSC_2 + b',
            r'$\beta$ LSC_2 + s', r'$\beta$ LSC_2 + c',
            'LSC_2^{(d)}', 'LSC_2^{(dr)}'
        ]

        idf['comments'] = pd.Categorical(idf['comments'], order_conditions)

        pdf = pd.pivot_table(idf, values='metric', index=['comments'], columns=['task'], aggfunc=np.sum)
        pdf = pdf.replace([0], '-')

        pdf = pdf[tasks]

        for task in tasks:
            pdf[task + ' val'] = pdf[task].str.split("/", n=1, expand=True)[0]
            pdf[task + ' test'] = pdf[task].str.split("/", n=1, expand=True)[1]
            pdf.drop(columns=[task], inplace=True)

        # print(pdf.to_string())

        print(pdf.to_latex(index=True, escape=False))

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

    plot_filename = f'experiments/bars.pdf'
    fig.savefig(plot_filename, bbox_inches='tight')

    plt.show()

if plot_weights:
    create_pickles = False
    plot_1 = False
    plot_2 = True

    net = 'LSTM'  # ALIF LSTM
    task = 'sl-MNIST'  # sl_mnist heidelberg
    gauss_beta = False
    kwargs = dict(histtype='step', alpha=1., density=True, lw=1)

    axs = None
    cols = 3 if net == 'LSTM' else 6
    n_bins = 50

    if create_pickles:
        for normpow in [1, -1, 2]:
            path = get_path(df, normpow, task, net, gauss_beta)
            _, exp_identifiers = os.path.split(path)

            model_path = os.path.join(path, 'trained_models', 'lsc', 'model_weights_lsc_before.h5')
            if not os.path.exists(model_path):
                shutil.rmtree(path)

                # os.remove(path)
                _ = unzip_good_exps(
                    GEXPERIMENTS, EXPERIMENTS,
                    exp_identifiers=[exp_identifiers], except_folders=[],
                    unzip_what=['model_', '.txt', '.json', '.csv'], except_files=['cout.txt']
                )

        import tensorflow as tf

        for norm in [1, 2, -1]:
            hists_path = os.path.join(EXPERIMENTS, f'hists_{net}_{task}_gb{gauss_beta}_normpow{norm}.pickle')
            path = get_path(df, norm, task, net, gauss_beta)

            hist_dict = {}

            for befaft in ['before', 'after']:
                json_path = os.path.join(path, 'trained_models', 'lsc', f'model_config_lsc_{befaft}.json')
                h5_path = os.path.join(path, 'trained_models', 'lsc', f'model_weights_lsc_{befaft}.h5')

                with open(json_path) as json_file:
                    json_config = json_file.read()

                model = tf.keras.models.model_from_json(
                    json_config, custom_objects={
                        'maLSNN': maLSNN, 'RateVoltageRegularization': RateVoltageRegularization,
                        'AddLossLayer': AddLossLayer,
                        'SparseCategoricalCrossentropy': tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=True),
                        'AddMetricsLayer': AddMetricsLayer
                    }
                )
                model.load_weights(h5_path)
                weights = model.get_weights()
                weight_names = [weight.name for layer in model.layers for weight in layer.weights]
                hist_dict[befaft] = {}
                print(weight_names)

                for i in range(len(weights)):
                    if not os.path.exists(hists_path):
                        w = weights[i]
                        wn = weight_names[i]
                        counts, bins = np.histogram(w.flatten(), bins=n_bins)
                        print(counts)
                        print(wn)
                        if not 'switch' in wn:
                            hist_dict[befaft][wn] = (bins, counts)

            if not os.path.exists(hists_path):
                pickle.dump(hist_dict, open(hists_path, 'wb'))

    if plot_1:
        from matplotlib.ticker import FormatStrFormatter

        norms = [None, 1, 2, -1]
        for norm_id in norms:
            norm = norm_id if norm_id is not None else 1
            hists_path = os.path.join(EXPERIMENTS, f'hists_{net}_{task}_gb{gauss_beta}_normpow{norm}.pickle')
            hist_dict = pickle.load(open(hists_path, 'rb'))  # Unpickling the object

            befafts = ['before'] if norm_id is None else ['after']
            for befaft in befafts:
                n_weights = len(hist_dict[befaft]) if os.path.exists(hists_path) else len(weights)
                if axs is None:
                    wspace = 0.1 if net == 'LSTM' else 0.4
                    hspace = 1.6 if net == 'LSTM' else 1.6
                    fig, axs = plt.subplots(
                        int(n_weights / cols + 1), cols, gridspec_kw={'wspace': wspace, 'hspace': hspace},
                        figsize=(10, 3)
                    )

                for ax, i in zip(axs.flat, range(n_weights)):
                    wn = list(hist_dict[befaft].keys())[i]
                    histogram = hist_dict[befaft][wn]

                    bins = histogram[0]
                    counts = histogram[1]
                    highest = max(counts) / sum(counts) / (bins[1] - bins[0])

                    ax.hist(bins[:-1], bins, weights=counts, color=color_nid(norm_id), ec=color_nid(norm_id), **kwargs)

                    if highest > 100:
                        ax.set_ylim([0, 20])
                    ax.set_xlabel(clean_weight_name(wn), fontsize=12)
                    ax.locator_params(axis='x', nbins=2)

                    if max(bins) > 1:
                        ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    else:
                        ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))

                    if highest > 1:
                        ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
                    else:
                        scientific_notation = "{:e}".format(highest)
                        ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{scientific_notation[-1]}f'))

                    ax.tick_params(axis='both', which='major', labelsize=8)

        for i, ax in enumerate(axs.reshape(-1)):
            for pos in ['right', 'left', 'bottom', 'top']:
                ax.spines[pos].set_visible(False)

            if i >= len(hist_dict[befaft]):
                ax.set_visible(False)  # to remove last plot

        axs[0, 0].set_ylabel('Density', fontsize=12)

        legend_elements = [
            Line2D([0], [0], color=color_nid(norm_id), lw=4, label=clean_nid(norm_id))
            for norm_id in [None, 1, 2, -1]
        ]

        fig.legend(ncol=5, handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -.2), fontsize=12)

        plt.suptitle(f"{net} weights with LSC pretraining on {task}", y=1.01, fontsize=16)
        plot_filename = f'experiments/weights_{net}_{task}_gb{gauss_beta}.pdf'
        fig.savefig(plot_filename, bbox_inches='tight')
        plt.show()

    if plot_2:

        lstm_wns = ['encoder_0_0/lstm_cell/recurrent_kernel:0', 'encoder_1_0/lstm_cell_1/kernel:0']
        alif_wns = ['encoder_0_0/ma_lsnn/thr:0', 'encoder_0_0/ma_lsnn/beta:0']

        assert len(lstm_wns) == len(alif_wns)
        fig, axs = plt.subplots(
            2, len(lstm_wns), gridspec_kw={'wspace': .3, 'hspace': 0.8}, figsize=(7, 3)
        )

        norms = [None, 1, 2, -1]
        for norm_id in norms:
            norm = norm_id if norm_id is not None else 1
            # hists_path = os.path.join(EXPERIMENTS, f'hists_{net}_{task}_gb{gauss_beta}_normpow{norm}.pickle')
            # hist_dict = pickle.load(open(hists_path, 'rb'))  # Unpickling the object

            lstm_path = os.path.join(EXPERIMENTS, f'hists_LSTM_{task}_gb{gauss_beta}_normpow{norm}.pickle')
            lstm_dict = pickle.load(open(lstm_path, 'rb'))
            alif_path = os.path.join(EXPERIMENTS, f'hists_ALIF_{task}_gb{gauss_beta}_normpow{norm}.pickle')
            alif_dict = pickle.load(open(alif_path, 'rb'))

            color = color_nid(norm_id)

            befafts = ['before'] if norm_id is None else ['after']
            for befaft in befafts:
                # color = '#097B2A' if befaft == 'before' else '#40DE6E'

                for i, wn in enumerate(alif_wns):
                    histogram = alif_dict[befaft][wn]

                    bins = histogram[0]
                    counts = histogram[1]
                    axs[1, i].hist(bins[:-1], bins, weights=counts, color=color, ec=color, **kwargs)
                    axs[1, i].set_xlabel(clean_weight_name(wn))

                    highest = max(counts) / sum(counts) / (bins[1] - bins[0])
                    if highest > 100:
                        axs[1, i].set_ylim([0, 40])

                for i, wn in enumerate(lstm_wns):
                    histogram = lstm_dict[befaft][wn]

                    bins = histogram[0]
                    counts = histogram[1]
                    axs[0, i].hist(bins[:-1], bins, weights=counts, color=color, ec=color, **kwargs)
                    axs[0, i].set_xlabel(clean_weight_name(wn))
        axs[0, 0].set_ylabel('Density', fontsize=12)

        fig.text(0.05, 0.25, 'ALIF', ha='center', va='center', fontsize=16, rotation=90)
        fig.text(0.05, 0.74, 'LSTM', ha='center', va='center', fontsize=16, rotation=90)

        for ax in axs.reshape(-1):
            ax.locator_params(axis='y', nbins=3)

            for pos in ['right', 'left', 'bottom', 'top']:
                ax.spines[pos].set_visible(False)

        legend_elements = [
            Line2D([0], [0], color=color_nid(norm_id), lw=4, label=clean_nid(norm_id))
            for norm_id in [None, 1, 2, -1]
        ]

        fig.legend(ncol=5, handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -.25), fontsize=12)

        plot_filename = f'experiments/weights_aliflstm_{task}.pdf'
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

if plot_lsc_vs_naive:

    from matplotlib.lines import Line2D

    if task == 'all':
        tasks = np.unique(df['task'])
    else:
        tasks = [task]

    # idf = df[df['optimizer_name'].str.contains(optimizer_name)]
    idf = copy.deepcopy(df)
    idf['comments'] = idf['comments'].str.replace('_timerepeat:2', '')
    idf['comments'] = idf['comments'].str.replace('timerepeat:2', '')

    for task in tasks:
        iidf = idf[idf['task'].str.contains(task)]
        # idf = idf[idf['d'].str.contains('2022-07-06--')]
        # idf = idf[idf['epochs'].eq(1000)]

        print(iidf.to_string())
        n_plots = 10
        colors_for_type = {
            'LSC1': 'Blues',
            # 'dampening:1.': 'Oranges',
            'randominit': 'Reds',
            'lsc1': 'Greens',
            # 'LSC_dampening:1.': 'Purples',
            # 'original': 'Purples',
            '': 'Purples',
            'LSC2': 'Oranges',
            # 'LSC2_ingain:1.414': 'Greens',
        }

        # types = ['LSC', 'dampening:1.', 'randominit', 'lscc', 'LSC_dampening:1.', 'original', '']
        types = colors_for_type.keys()
        m = metric.replace('val_', '')
        # types = ['LSC', 'randominit', 'original']
        fig, axs = plt.subplots(1, 2, figsize=(6, 2), sharey=True, gridspec_kw={'wspace': .05})
        for i in range(2):
            for comment in types:
                iiidf = iidf[iidf['comments'].eq(comment.replace('_timerepeat:2', ''))]
                print(iiidf.to_string())
                # print(colors[comment])
                cmap = plt.cm.get_cmap(colors_for_type[comment])
                colors = cmap(np.arange(iiidf.shape[0]) / iiidf.shape[0])
                for j, (_, row) in enumerate(iiidf.iterrows()):
                    d = row['d']
                    h = histories[d][m if i == 0 else 'val_' + m]
                    axs[i].plot(h, color=colors[j], label=comment)

        axs[0].set_title('train ' + task)
        axs[1].set_title('validation')
        axs[0].set_ylabel('accuracy')
        axs[1].set_xlabel('training epoch')

        custom_lines = [Line2D([0], [0], color=plt.cm.get_cmap(colors_for_type[type])(0.5), lw=4) for type in types]

        axs[1].legend(custom_lines, [t.replace('init', '').replace('original', 'reference') for t in types],
                      loc='lower right', framealpha=0.9)

        for ax in axs.reshape(-1):
            for pos in ['right', 'left', 'bottom', 'top']:
                ax.spines[pos].set_visible(False)

        axs[1].tick_params(labelleft=False, left=False)
        pathplot = os.path.join(CDIR, 'experiments', f'lscvsrandom_{task}.png')
        fig.savefig(pathplot, bbox_inches='tight')

        plt.show()

if plot_dampenings_and_betas:
    init = 'LSC2'  # lsc LSC LSC2
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    idf = df[df['task'].str.contains(task)]
    idf = idf[idf['optimizer_name'].str.contains(optimizer_name)]
    iidf = idf[idf['comments'].str.contains(init + '_beta')]  # cdr gra blg
    iidf['betas'] = iidf['comments'].str.replace(init + '_beta:', '').values.astype(float)
    iidf = iidf.sort_values(by='betas')

    mdf = iidf.groupby(
        ['net', 'task', 'initializer', 'betas'], as_index=False
    ).agg({m: ['mean', 'std'] for m in metrics_oi})

    for m in metrics_oi:
        mdf['mean_{}'.format(m)] = mdf[m]['mean']
        mdf['std_{}'.format(m)] = mdf[m]['std']
        mdf = mdf.drop([m], axis=1)
    mdf = mdf.sort_values(by='betas')

    print(mdf.to_string())

    color = plt.cm.Oranges(3 / 6)

    metric = metric.replace('val_', '')
    means = mdf['mean_val_' + metric]
    stds = mdf['std_val_' + metric]
    betas = mdf['betas']

    axs[0].plot(betas, means, color=color)
    axs[0].fill_between(betas, means - stds, means + stds, alpha=0.5, color=color)

    idf = df[df['task'].str.contains(task)]
    idf = idf[idf['optimizer_name'].str.contains(optimizer_name)]
    iidf = idf[idf['comments'].str.contains(init + '_dampening')]  # cdr gra blg
    iidf['dampening'] = iidf['comments'].str.replace(init + '_dampening:', '').values.astype(float)
    iidf = iidf.sort_values(by='dampening')

    mdf = iidf.groupby(
        ['net', 'task', 'initializer', 'dampening'], as_index=False
    ).agg({m: ['mean', 'std'] for m in metrics_oi})

    for m in metrics_oi:
        mdf['mean_{}'.format(m)] = mdf[m]['mean']
        mdf['std_{}'.format(m)] = mdf[m]['std']
        mdf = mdf.drop([m], axis=1)
    mdf = mdf.sort_values(by='dampening')

    color = plt.cm.Oranges(3 / 6)

    means = mdf['mean_val_' + metric]
    stds = mdf['std_val_' + metric]
    dampening = mdf['dampening']
    axs[1].plot(dampening, means, color=color)
    axs[1].fill_between(dampening, means - stds, means + stds, alpha=0.5, color=color)

    axs[0].set_ylabel('accuracy')
    axs[0].set_xlabel(r'$\beta$')
    axs[1].set_xlabel('$\gamma$')

    # LSC values
    # axs.axvline(x=value, color='k', linestyle='--')

    # axs[1].set_ylabel(metric)
    # axs[1].set_title(task)

    for ax in axs.reshape(-1):
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)

    pathplot = os.path.join(CDIR, 'experiments', 'beta_dampening.png')
    fig.savefig(pathplot, bbox_inches='tight')

    plt.show()

if remove_incomplete:
    import shutil

    rdfs = []

    print('-=***=-' * 10)
    print('Eliminate if not close enough to target norm')
    # from LSC_norms final column, select those that are epsilon away from 1
    epsilon = 0.09
    # epsilon = 0.2
    # make a column target equal to .5 if targetnorm:.5 is in comments else 1 if findLSC is in comments
    # else nan
    plotdf['target'] = plotdf['comments'].apply(
        lambda x: 0.5 if 'targetnorm:.5' in x else 1 if 'findLSC' in x else np.nan)
    rdf = plotdf[
        abs(plotdf['LSC f'] - plotdf['target']) > epsilon
        ]
    print(rdf.to_string())
    print(rdf.shape, df.shape)
    rdfs.append(rdf)

    # remove LSC that didn't record LSC norms

    print('remove old settings')
    # rdf = plotdf[plotdf['comments'].eq('allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_findLSC')]
    rdf = plotdf[
        plotdf['comments'].str.contains('onlypretrain')
    ]
    # rdfs.append(rdf)
    # print(rdf.shape)

    # remove repeated
    # remove one seed from those that have more than 4 seeds
    brdf = mdf[mdf['counts'] > 4]

    print('-=***=-' * 10)
    print('Count > 4')
    print(brdf.to_string())

    for _, row in brdf.iterrows():
        srdf = plotdf[
            # (df['lr'] == row['lr'])
            (plotdf['comments'].eq(row['comments']))
            & (plotdf['stack'] == row['stack'])
            & (plotdf['task'] == row['task'])
            & (plotdf['net'] == row['net'])
            ].copy()

        # no duplicates
        gsrdf = srdf.drop_duplicates(subset=['seed'])

        # remainder
        rdf = srdf[~srdf.apply(tuple, 1).isin(gsrdf.apply(tuple, 1))]
        print(rdf.shape)
        rdfs.append(rdf)

    allrdfs = pd.concat(rdfs)
    allrdfs = allrdfs.drop_duplicates()
    print(f'Remove {allrdfs.shape} of {plotdf.shape}')

    if truely_remove_pretrained:

        # sdf = pd.read_hdf(h5path, 'df')
        sdf = allrdfs.copy()

        sdf.loc[df['task'].str.contains('SHD'), 'task'] = 'heidelberg'
        sdf.loc[df['task'].str.contains('sl-MNIST'), 'task'] = 'sl_mnist'
        sdf.loc[df['task'].str.contains('PTB'), 'task'] = 'wordptb'

        sdf.loc[df['net'].str.contains('ALIFb'), 'net'] = 'maLSNNb'
        sdf.loc[df['net'].str.contains('ALIF'), 'net'] = 'maLSNN'

        coi = ['seed', 'task', 'net', 'comments', 'stack']
        experiments = []

        for _, row in sdf.iterrows():
            experiments.append({c: [row[c]] for c in coi})
        print(experiments)
        print(f'Experiments to remove: {len(experiments)}')
        folder = r'D:\work\alif_sg\good_experiments\pmodels'
        remove_pretrained_extra(experiments, remove_opposite=False, folder=folder)

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

if missing_exps:
    # columns of interest
    coi = ['seed', 'task', 'net', 'comments', 'stack']

    import pandas as pd

    # sdf = pd.read_hdf(h5path, 'df')
    sdf = df.copy()

    sdf.loc[df['task'].str.contains('SHD'), 'task'] = 'heidelberg'
    sdf.loc[df['task'].str.contains('sl-MNIST'), 'task'] = 'sl_mnist'
    sdf.loc[df['task'].str.contains('PTB'), 'task'] = 'wordptb'
    sdf.loc[df['net'].str.contains('ALIFb'), 'net'] = 'maLSNNb'
    sdf.loc[df['net'].str.contains('ALIF'), 'net'] = 'maLSNN'

    # sdf['lr'] = sdf['lr i']

    sdf.drop([c for c in sdf.columns if c not in coi], axis=1, inplace=True)
    # substitute the string _timerepeat:2 by _timerepeat:2_pretrained_ in the comments column
    sdf['comments'] = sdf['comments'].str.replace('_timerepeat:2', '_timerepeat:2_pretrained')
    # sdf['comments'] = sdf['comments'].str.replace('_onlypretrain', '')

    add_flag = '_onlyloadpretrained'  # _onlyloadpretrained _onlypretrain
    seed = 0
    n_seeds = 4
    seeds = [l + seed for l in range(n_seeds)]

    incomplete_comments = 'allns_36_embproj_nogradreset_dropout:.3_timerepeat:2_pretrained_'

    experiments = []

    all_comments = [
        # incomplete_comments,
        # incomplete_comments + f'findLSC',
        # incomplete_comments + f'findLSC_supsubnpsd',
        incomplete_comments + f'findLSC_radius' + add_flag,
        incomplete_comments + f'findLSC_radius_targetnorm:.5' + add_flag,
        # incomplete_comments + f'findLSC_radius_targetnorm:.5_randlsc',
        # incomplete_comments + f'findLSC_supsubnpsd_deslice',
    ]

    nets = ['LSTM', 'maLSNN', 'maLSNNb']
    tasks = ['heidelberg', 'sl_mnist', 'wordptb']
    experiment = {
        'task': tasks,
        'net': nets, 'seed': seeds, 'stack': ['None'],
        'comments': all_comments,
    }
    experiments.append(experiment)

    experiment = {
        'task': ['heidelberg'],
        'net': nets, 'seed': seeds, 'stack': ['1', '3', '5', '7'],
        'comments': all_comments,
    }
    experiments.append(experiment)

    ds = dict2iter(experiments)
    print(ds[0])
    experiments_left = complete_missing_exps(sdf, ds, coi)
    np.random.shuffle(experiments_left)
    experiments = experiments_left
    # experiments = []
    # for e in experiments_left:
    #     e['comments'] = [e['comments'][0] + '_onlypretrain']
    #     print(e)
    #     experiments.append(e)

    print(experiments)
    print(len(experiments))

    # experiments_1 = [e for e in experiments_left if e['stack'][0] in ['7', '5']]
    # experiments_2 = [e for e in experiments_left if not e['stack'][0] in ['7', '5']]
    # print(len(experiments_1), len(experiments_2), len(experiments_left))
