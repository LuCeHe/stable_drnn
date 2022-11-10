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
from GenericTools.stay_organized.pandardize import experiments_to_pandas
from GenericTools.stay_organized.standardize_strings import shorten_losses
from GenericTools.stay_organized.utils import str2val
from alif_sg.tools.plot_tools import clean_weight_name, get_path, clean_nid, color_nid, bolden_best, compactify_metrics, \
    choose_metric
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
GEXPERIMENTS = os.path.join(CDIR, 'good_experiments', '2022-10-10--good_for_initial_tables')
GEXPERIMENTS = os.path.join(CDIR, 'good_experiments')

expsid = 'als'  # effnet als
h5path = os.path.join(EXPERIMENTS, f'summary_{expsid}.h5')
# CSVPATH = r'D:\work\alif_sg\good_experiments\2022-08-20--learned-LSC\summary.h5'
# HSITORIESPATH = os.path.join(EXPERIMENTS, 'histories.json')

pandas_means = True
show_per_tasknet = False
make_latex = False
missing_exps = True
plot_lsc_vs_naive = False
plot_dampenings_and_betas = False
plot_norms_pretraining = False
plot_losses = False
plot_weights = False
plot_lrs = False

task_name = 'ps_mnist'  # heidelberg wordptb sl_mnist all ps_mnist

# sparse_mode_accuracy sparse_categorical_crossentropy bpc sparse_mode_accuracy_test_10
# val_sparse_mode_accuracy test_perplexity
metric = 'v_ppl min'
optimizer_name = 'SWAAdaBelief'  # SGD SWAAdaBelief
metrics_oi = [
    # 'val_sparse_mode_accuracy', 'val_perplexity', 'val_sparse_categorical_crossentropy',
    #           'test_sparse_mode_accuracy', 'test_perplexity',
    't_ppl min', 't_mode_acc max', 'v_ppl min', 'v_mode_acc max',
    # 'final_epochs'
]

plot_only = ['net_name', 'n_params', 'comments', 'epochs', 'initializer', 'optimizer_name', 'steps_per_epoch',
             'task_name', 'path', 'lr'] + metrics_oi
columns_to_remove = [
    'heaviside', '_test', 'weight', 'sLSTM_factor', 'save_model', 'clipnorm', 'GPU', 'batch_size',
    'continue_training', 'embedding', 'lr_schedule', 'loss_name', 'seed', 'stack', 'stop_time',
    'convergence', 'n_neurons', 'optimizer_name', 'LSC', ' list', 'artifacts', 'command', 'heartbeat', 'meta',
    'resources', 'host', 'start_time', 'status', 'experiment', 'result',
]
columns_to_remove = []

df = experiments_to_pandas(
    h5path=h5path, zips_folder=GEXPERIMENTS, unzips_folder=EXPERIMENTS, experiments_identifier=expsid,
    exclude_files=['cout.txt']
)

print(list(df.columns))

if 'n_params' in df.columns:
    df['n_params'] = df['n_params'].apply(lambda x: large_num_to_reasonable_string(x, 1))

df = df[~df['comments'].str.contains('test')]

if 'net_name' in df.columns:
    df.loc[df['comments'].str.contains('noalif'), 'net_name'] = 'LIF'
    df.loc[df['net_name'].str.contains('maLSNN'), 'net_name'] = 'ALIF'

df.loc[df['task_name'].str.contains('heidelberg'), 'task_name'] = 'SHD'
df.loc[df['task_name'].str.contains('sl_mnist'), 'task_name'] = 'sl-MNIST'
df.loc[df['task_name'].str.contains('wordptb'), 'task_name'] = 'PTB'

for c_name in columns_to_remove:
    df = df[df.columns.drop(list(df.filter(regex=c_name)))]

new_column_names = {c_name: shorten_losses(c_name) for c_name in df.columns}

df.rename(columns=new_column_names, inplace=True)

df['v_ppl argmin'] = df['v_ppl argmin'].astype(int)
df['v_mode_acc argmax'] = df['v_mode_acc argmax'].astype(int)

df['v_ppl'] = df.apply(lambda row: np.nanmin(row['v_ppl list']), axis=1)
df['t_ppl'] = df.apply(lambda row: row['t_ppl list'][row['v_ppl argmin']], axis=1)
df['v_mode_acc'] = df.apply(lambda row: np.nanmax(row['v_mode_acc list']), axis=1)

# # FIXME: 14 experiments got nans in the heidelberg task validation, plot them anyway?
# print('v_mode_acc nans:', df['v_mode_acc argmax'].isna().sum())
# df = df[~df['v_mode_acc argmax'].isna()]

df['t_mode_acc'] = df.apply(lambda row: row['t_mode_acc list'][row['v_mode_acc argmax']], axis=1)

df = df[df['lr'].isna()]

if metric in df.keys():
    df = df.sort_values(by=metric)

if not plot_only is None:
    df = df[plot_only]

print(list(df.columns))
# print(df['experiment'])
# print(df['host'])
print(df.to_string())

if pandas_means:
    # group_cols = ['net_name', 'task_name', 'initializer', 'comments', 'lr']
    group_cols = ['net_name', 'task_name', 'initializer', 'comments']
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

    mdf = mdf.sort_values(by='mean_' + metric)
    mdf['counts'] = counts['counts']

    tasks = np.unique(mdf['task_name'])
    nets = np.unique(mdf['net_name'])

    mdf['comments'] = mdf['comments'].str.replace('__', '_', regex=True)
    print(mdf.to_string())

    if show_per_tasknet:
        for task in tasks:
            for net in nets:
                if not net == 'LIF':
                    print('-===-' * 30)
                    print(task, net)
                    idf = mdf[mdf['task_name'].str.contains(task) & mdf['net_name'].str.contains(net)]
                    print(idf.to_string())

    if make_latex:

        net = 'ALIF'  # ALIF LSTM

        idf = mdf[mdf['net_name'].str.contains(net)]
        # print(mdf[mdf['net_name'].str.contains('ALIF')].shape, mdf[mdf['net_name'].str.contains('LSTM')].shape)

        idf = idf[~idf['comments'].str.contains('reoldspike')]
        idf = idf[~idf['comments'].str.contains('savelscweights')]

        # print(idf.to_string())

        metrics_cols = [c for c in idf.columns if 'ppl' in c or 'acc' in c]
        for m in metrics_cols:
            mode = 'max' if 'acc' in m and not 'std' in m else 'min'
            idf[f'best_{m}'] = idf.groupby(['task_name'])[m].transform(mode)
            idf[m] = idf.apply(bolden_best(m), axis=1)

        idf['ppl'] = idf.apply(compactify_metrics('ppl min'), axis=1)
        idf['acc'] = idf.apply(compactify_metrics('mode_acc max'), axis=1)
        idf['metric'] = idf.apply(choose_metric, axis=1)
        idf = idf[idf.columns.drop(list(idf.filter(regex='acc')) + list(idf.filter(regex='ppl')))]
        idf = idf[idf.columns.drop(['counts', 'initializer', 'net_name'])]

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
        print('here!')
        print(idf.to_string())

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
        print(idf.to_string())

        pdf = pd.pivot_table(idf, values='metric', index=['comments'], columns=['task_name'], aggfunc=np.sum)
        pdf = pdf.replace([0], '-')

        pdf = pdf[tasks]

        print(conditions)

        print(idf.to_string())
        print(pdf.to_string())


        print(pdf.to_latex(index=True, escape=False))

if plot_lrs:
    idf = mdf
    idf = idf.dropna(subset=['lr'])
    print(idf.to_string())

    tasks = ['sl-MNIST', 'SHD', 'PTB']
    lrs = np.unique(idf['lr'])
    nets = ['LSTM', 'ALIF']

    fig, axs = plt.subplots(1, len(tasks), gridspec_kw={'wspace': .2, 'hspace': 0.8}, figsize=(14, 3))
    colors = lambda net_name: '#FF5733' if net_name == 'ALIF' else '#1E55A9'
    for i, task in enumerate(tasks):
        for net in nets:
            iidf = idf[idf['task_name'].eq(task) & idf['net_name'].eq(net)]
            iidf = iidf.sort_values(by='lr')

            vppls = iidf['mean_v_ppl min'].values
            lrs = iidf['lr'].values
            stds = iidf['std_v_ppl min'].values
            axs[i].plot(lrs, vppls, color=colors(net))
            axs[i].fill_between(lrs, vppls - stds / 2, vppls + stds / 2, alpha=0.5, color=colors(net))
            axs[i].set_xscale('log')

            iidf = iidf.sort_values(by='mean_v_ppl min')
            print(f"{net} on {task} got best vPPL {iidf['mean_v_ppl min'].values[0]} for {iidf['lr'].values[0]}")

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

if missing_exps:
    # columns of interest
    coi = ['seed', 'task_name', 'net_name', 'comments']
    import pandas as pd

    sdf = pd.read_hdf(h5path, 'df')

    sdf.drop([c for c in sdf.columns if c not in coi], axis=1, inplace=True)

    seed = 0
    n_seeds = 4
    seeds = [l + seed for l in range(n_seeds)]
    incomplete_comments = '34_embproj_nogradreset_dropout:.3_timerepeat:2_'

    experiments = []
    experiment = {
        # 'task_name': ['heidelberg', 'ps_mnist', 's_mnist', 'ss_mnist', 'sps_mnist', 'sl_mnist'],
        'task_name': ['heidelberg', 'sl_mnist'],
        'net_name': ['maLSNN', 'LSTM'], 'seed': seeds,
        'comments': [
            incomplete_comments,
            incomplete_comments + f'findLSC_normpow:1',
            incomplete_comments + f'findLSC_normpow:-1',
            incomplete_comments + f'findLSC_normpow:2',
            incomplete_comments + f'findLSC_normpow:2_shufflelsc',
            incomplete_comments + f'findLSC_normpow:2_berlsc',
            incomplete_comments + f'findLSC_normpow:2_gausslsc',
        ],
    }
    experiments.append(experiment)

    experiment = {
        # 'task_name': ['heidelberg', 'ps_mnist', 's_mnist', 'ss_mnist', 'sps_mnist', 'sl_mnist'],
        'task_name': ['wordptb'],
        'net_name': ['maLSNN', 'LSTM'], 'seed': seeds,
        'comments': [
            incomplete_comments,
            incomplete_comments + f'findLSC_normpow:1',
            incomplete_comments + f'findLSC_normpow:-1',
            incomplete_comments + f'findLSC_normpow:2',
            incomplete_comments + f'findLSC_normpow:2_shufflelsc',
            incomplete_comments + f'findLSC_normpow:2_randwlsc',
        ],
    }
    experiments.append(experiment)

    experiment = {
        # 'task_name': ['heidelberg', 'ps_mnist', 's_mnist', 'ss_mnist', 'sps_mnist', 'sl_mnist'],
        'task_name': ['heidelberg', 'sl_mnist', 'wordptb'],
        'net_name': ['maLSNN', 'LSTM'], 'seed': seeds,
        'comments': [
            incomplete_comments + f'findLSC_normpow:2_lscdepth:1_lscout:0',
            incomplete_comments + f'findLSC_normpow:2_lscdepth:1_lscout:1',
        ],
    }
    experiments.append(experiment)

    experiment = {
        'task_name': ['heidelberg', 'wordptb', 'sl_mnist'],
        'net_name': ['maLSNN'], 'seed': seeds,
        'comments': [
            incomplete_comments + f'_gaussbeta',
            incomplete_comments + f'findLSC_normpow:2_gaussbeta',
            incomplete_comments + f'findLSC_normpow:2_gaussbeta_berlsc',
            incomplete_comments + f'findLSC_normpow:2_gaussbeta_gausslsc',
            incomplete_comments + f'findLSC_normpow:2_gaussbeta_shufflelsc',
            incomplete_comments + f'findLSC_normpow:2_gaussbeta_randwlsc',
        ],
    }
    experiments.append(experiment)

    ds = dict2iter(experiments)
    print(ds[0])

    data = {k: [] for k in coi}
    for d in ds:
        for k in data.keys():
            insertion = d[k]
            data[k].append(insertion)

    all_exps = pd.DataFrame.from_dict(data)
    # print(all_exps.to_string())

    # remove the experiments that were run successfully
    df = pd.concat([sdf, all_exps])
    df = df.drop_duplicates(keep=False)

    keys = list(all_exps.columns.values)
    i1 = all_exps.set_index(keys).index
    i2 = df.set_index(keys).index
    df = df[i2.isin(i1)]

    sdf = sdf.drop_duplicates()

    # df = df[~df['task_name'].str.contains('wordptb1')]
    # df = df[~df['task_name'].str.contains('wordptb')]
    # df = df[df['task_name'].str.contains('wordptb')]

    print('left, done, all: ', df.shape, sdf.shape, all_exps.shape)
    print('left')
    print(df.to_string())

    experiments = []
    for index, row in df.iterrows():
        experiment = {k: [row[k]] for k in df.columns}
        experiments.append(experiment)

    print(experiments)

if plot_weights:
    create_pickles = False
    plot_1 = False
    plot_2 = True

    net_name = 'LSTM'  # ALIF LSTM
    task_name = 'sl-MNIST'  # sl_mnist heidelberg
    gauss_beta = False
    kwargs = dict(histtype='step', alpha=1., density=True, lw=1)

    for normpow in [1, -1, 2]:
        path = get_path(df, normpow, task_name, net_name, gauss_beta)
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

    axs = None

    cols = 3 if net_name == 'LSTM' else 6
    n_bins = 50

    if create_pickles:
        import tensorflow as tf

        for norm in [1, 2, -1]:
            hists_path = os.path.join(EXPERIMENTS, f'hists_{net_name}_{task_name}_gb{gauss_beta}_normpow{norm}.pickle')
            path = get_path(df, norm, task_name, net_name, gauss_beta)

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
            hists_path = os.path.join(EXPERIMENTS, f'hists_{net_name}_{task_name}_gb{gauss_beta}_normpow{norm}.pickle')
            hist_dict = pickle.load(open(hists_path, 'rb'))  # Unpickling the object

            befafts = ['before'] if norm_id is None else ['after']
            for befaft in befafts:
                n_weights = len(hist_dict[befaft]) if os.path.exists(hists_path) else len(weights)
                if axs is None:
                    wspace = 0.1 if net_name == 'LSTM' else 0.4
                    hspace = 1.6 if net_name == 'LSTM' else 1.6
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

        plt.suptitle(f"{net_name} weights with LSC pretraining on {task_name}", y=1.01, fontsize=16)

        plot_filename = f'experiments/weights_{net_name}_{task_name}_gb{gauss_beta}.pdf'

        fig.savefig(plot_filename, bbox_inches='tight')

        plt.show()

    if plot_2:

        lstm_wns = ['encoder_0_0/lstm_cell/recurrent_kernel:0', 'encoder_1_0/lstm_cell_1/kernel:0']
        alif_wns = ['encoder_0_0/ma_lsnn/thr:0', 'encoder_0_0/ma_lsnn/beta:0']

        fig, axs = plt.subplots(
            1, len(lstm_wns + alif_wns), gridspec_kw={'wspace': .3, 'hspace': 0.8}, figsize=(8, 3)
        )

        norms = [None, 1, 2, -1]
        for norm_id in norms:
            norm = norm_id if norm_id is not None else 1
            # hists_path = os.path.join(EXPERIMENTS, f'hists_{net_name}_{task_name}_gb{gauss_beta}_normpow{norm}.pickle')
            # hist_dict = pickle.load(open(hists_path, 'rb'))  # Unpickling the object

            lstm_path = os.path.join(EXPERIMENTS, f'hists_LSTM_{task_name}_gb{gauss_beta}_normpow{norm}.pickle')
            lstm_dict = pickle.load(open(lstm_path, 'rb'))
            alif_path = os.path.join(EXPERIMENTS, f'hists_ALIF_{task_name}_gb{gauss_beta}_normpow{norm}.pickle')
            alif_dict = pickle.load(open(alif_path, 'rb'))

            color = color_nid(norm_id)

            befafts = ['before'] if norm_id is None else ['after']
            for befaft in befafts:
                # color = '#097B2A' if befaft == 'before' else '#40DE6E'

                for i, wn in enumerate(alif_wns):
                    histogram = alif_dict[befaft][wn]

                    bins = histogram[0]
                    counts = histogram[1]
                    axs[i + len(lstm_wns)].hist(bins[:-1], bins, weights=counts, color=color, ec=color, **kwargs)
                    axs[i + len(lstm_wns)].set_xlabel(clean_weight_name(wn))

                    highest = max(counts) / sum(counts) / (bins[1] - bins[0])
                    if highest > 100:
                        axs[i + len(lstm_wns)].set_ylim([0, 40])

                for i, wn in enumerate(lstm_wns):
                    histogram = lstm_dict[befaft][wn]

                    bins = histogram[0]
                    counts = histogram[1]
                    axs[i].hist(bins[:-1], bins, weights=counts, color=color, ec=color, **kwargs)
                    axs[i].set_xlabel(clean_weight_name(wn))

        fig.text(0.70, .98, 'ALIF', ha='center', va='center', fontsize=16)
        fig.text(0.29, .98, 'LSTM', ha='center', va='center', fontsize=16)

        for ax in axs.reshape(-1):
            ax.locator_params(axis='y', nbins=3)

            for pos in ['right', 'left', 'bottom', 'top']:
                ax.spines[pos].set_visible(False)

        legend_elements = [
            Line2D([0], [0], color=color_nid(norm_id), lw=4, label=clean_nid(norm_id))
            for norm_id in [None, 1, 2, -1]
        ]

        fig.legend(ncol=5, handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -.25), fontsize=12)

        plot_filename = f'experiments/weights_aliflstm_{task_name}.pdf'
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
            idf = df[(df['net_name'].eq(n)) & (df['task_name'].eq(t))]

            for index, row in idf.iterrows():
                if 'LSC_' + moi in row.keys():
                    if isinstance(row['LSC_' + moi], str):
                        normpow = str2val(row['comments'], 'normpow', float, default=1)

                        metric = [float(s) for s in row['LSC_' + moi][1:-1].split(', ')]
                        axs[i, j].plot(metric, color=colors[norms.index(normpow)], label=normpow)

            axs[i, j].set_title(f'{n}: {t}')

    axs[i, j].legend()
    plt.show()

if plot_losses:
    moi = 'sparse_mode_accuracy'  # losses norms sparse_mode_accuracy perplexity
    ref = 0 if metric == 'losses' else 1
    fig, axs = plt.subplots(len(nets), len(tasks), figsize=(6, 2), gridspec_kw={'wspace': .05})

    if len(nets) == 1:
        axs = axs[None]

    cmap = plt.cm.get_cmap('Paired')
    norms = [0.1, 1, 2, 3, -1]
    colors = cmap(np.arange(len(norms)) / len(norms))
    for i, n in enumerate(nets):
        for j, t in enumerate(tasks):
            idf = df[(df['net_name'].eq(n)) & (df['task_name'].eq(t))]

            for index, row in idf.iterrows():
                if 'findLSC_' in row['comments']:
                    normpow = str2val(row['comments'], 'normpow', float, default=1)
                    color = colors[norms.index(normpow)]
                else:
                    color = 'red'
                    normpow = '-'
                metric = histories[row['d']]['val_' + moi]
                axs[i, j].plot(metric, color=color, label=normpow)

            axs[i, j].set_title(f'{n}: {t}')

    axs[i, j].legend()
    plt.show()

if plot_lsc_vs_naive:

    from matplotlib.lines import Line2D

    if task_name == 'all':
        tasks = np.unique(df['task_name'])
    else:
        tasks = [task_name]

    # idf = df[df['optimizer_name'].str.contains(optimizer_name)]
    idf = copy.deepcopy(df)
    idf['comments'] = idf['comments'].str.replace('_timerepeat:2', '')
    idf['comments'] = idf['comments'].str.replace('timerepeat:2', '')

    for task in tasks:
        iidf = idf[idf['task_name'].str.contains(task)]
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

    idf = df[df['task_name'].str.contains(task_name)]
    idf = idf[idf['optimizer_name'].str.contains(optimizer_name)]
    iidf = idf[idf['comments'].str.contains(init + '_beta')]  # cdr gra blg
    iidf['betas'] = iidf['comments'].str.replace(init + '_beta:', '').values.astype(float)
    iidf = iidf.sort_values(by='betas')

    mdf = iidf.groupby(
        ['net_name', 'task_name', 'initializer', 'betas'], as_index=False
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

    idf = df[df['task_name'].str.contains(task_name)]
    idf = idf[idf['optimizer_name'].str.contains(optimizer_name)]
    iidf = idf[idf['comments'].str.contains(init + '_dampening')]  # cdr gra blg
    iidf['dampening'] = iidf['comments'].str.replace(init + '_dampening:', '').values.astype(float)
    iidf = iidf.sort_values(by='dampening')

    mdf = iidf.groupby(
        ['net_name', 'task_name', 'initializer', 'dampening'], as_index=False
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
    # axs[1].set_title(task_name)

    for ax in axs.reshape(-1):
        for pos in ['right', 'left', 'bottom', 'top']:
            ax.spines[pos].set_visible(False)

    pathplot = os.path.join(CDIR, 'experiments', 'beta_dampening.png')
    fig.savefig(pathplot, bbox_inches='tight')

    plt.show()
