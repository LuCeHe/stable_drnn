import os, json, copy

from GenericTools.stay_organized.submit_jobs import dict2iter
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib as mpl

from GenericTools.stay_organized.mpl_tools import load_plot_settings
from GenericTools.stay_organized.pandardize import experiments_to_pandas
from GenericTools.stay_organized.standardize_strings import shorten_losses
from GenericTools.stay_organized.utils import str2val

mpl, pd = load_plot_settings(mpl=mpl, pd=pd)

import matplotlib.pyplot as plt
from datetime import timedelta, datetime

from GenericTools.stay_organized.plot_tricks import large_num_to_reasonable_string

FMT = '%Y-%m-%dT%H:%M:%S'
from GenericTools.stay_organized.unzip import unzip_good_exps

FILENAME = os.path.realpath(__file__)
CDIR = os.path.dirname(FILENAME)
EXPERIMENTS = os.path.join(CDIR, 'experiments')
GEXPERIMENTS = os.path.join(CDIR, 'good_experiments')

expsid = 'als'  # effnet als
h5path = os.path.join(EXPERIMENTS, f'summary_{expsid}.h5')
# CSVPATH = r'D:\work\alif_sg\good_experiments\2022-08-20--learned-LSC\summary.h5'
# HSITORIESPATH = os.path.join(EXPERIMENTS, 'histories.json')

pandas_means = False
make_latex = False
missing_exps = True
plot_lsc_vs_naive = False
plot_dampenings_and_betas = False
plot_norms_pretraining = False
plot_losses = False

task_name = 'ps_mnist'  # heidelberg wordptb sl_mnist all ps_mnist

# sparse_mode_accuracy sparse_categorical_crossentropy bpc sparse_mode_accuracy_test_10
# val_sparse_mode_accuracy test_perplexity
metric = 'v_ppl'
optimizer_name = 'SWAAdaBelief'  # SGD SWAAdaBelief
metrics_oi = [
    # 'val_sparse_mode_accuracy', 'val_perplexity', 'val_sparse_categorical_crossentropy',
    #           'test_sparse_mode_accuracy', 'test_perplexity',
    't_ppl', 't_mode_acc', 'v_ppl', 'v_mode_acc',
    # 'final_epochs'
]

plot_only = ['net_name', 'n_params', 'comments', 'epochs', 'initializer', 'optimizer_name', 'steps_per_epoch',
             'task_name', 'path'] + metrics_oi
columns_to_remove = [
    'heaviside', '_test', 'weight', 'sLSTM_factor', 'save_model', 'clipnorm', 'GPU', 'batch_size',
    'continue_training', 'embedding', 'lr_schedule', 'loss_name', 'lr', 'seed', 'stack', 'stop_time',
    'convergence', 'n_neurons', 'optimizer_name', 'LSC', ' list', 'artifacts', 'command', 'heartbeat', 'meta',
    'resources', 'host', 'start_time', 'status', 'experiment', 'result',
]
columns_to_remove = []

df = experiments_to_pandas(
    h5path=h5path, zips_folder=GEXPERIMENTS, unzips_folder=EXPERIMENTS, experiments_identifier=expsid,
    exclude_files=['cout.txt']
)

print(df.columns)
if 'host' in df.columns:
    df['where'] = df['host'].apply(lambda x: x['hostname'])

if 'n_params' in df.columns:
    df['n_params'] = df['n_params'].apply(lambda x: large_num_to_reasonable_string(x, 1))

if 'stop_time' in df.columns and 'start_time' in df.columns:
    df['duration_experiment'] = df['stop_time'].apply(lambda x: datetime.strptime(x.split('.')[0], FMT)) - \
                                df['start_time'].apply(lambda x: datetime.strptime(x.split('.')[0], FMT))

df = df[~df['comments'].str.contains('test')]

if 'net_name' in df.columns:
    df.loc[df['comments'].str.contains('noalif'), 'net_name'] = 'LIF'
    df.loc[df['net_name'].str.contains('maLSNN'), 'net_name'] = 'ALIF'

for c_name in columns_to_remove:
    df = df[df.columns.drop(list(df.filter(regex=c_name)))]

new_column_names = {c_name: shorten_losses(c_name) for c_name in df.columns}

df.rename(columns=new_column_names, inplace=True)
# df = df[[c for c in df if c not in ['d', 'duration_experiment']] + ['d', 'duration_experiment']]


if metric in df.keys():
    df = df.sort_values(by=metric)

if not plot_only is None:
    df = df[plot_only]

print(list(df.columns))
# print(df['experiment'])
# print(df['host'])
print(df.to_string())

if pandas_means:
    show_per_tasknet = False
    group_cols = ['net_name', 'task_name', 'initializer', 'comments']
    counts = df.groupby(group_cols).size().reset_index(name='counts')

    metrics_oi = [shorten_losses(m) for m in metrics_oi]
    mdf = df.groupby(
        group_cols, as_index=False
    ).agg({m: ['mean', 'std'] for m in metrics_oi})

    print(mdf.columns)
    for m in metrics_oi:
        mdf['mean_{}'.format(m)] = mdf[m]['mean']
        mdf['std_{}'.format(m)] = mdf[m]['std']
        mdf = mdf.drop([m], axis=1)

    mdf = mdf.sort_values(by='mean_' + metric)
    mdf['counts'] = counts['counts']

    tasks = np.unique(mdf['task_name'])
    nets = np.unique(mdf['net_name'])

    if show_per_tasknet:
        for task in tasks:
            for net in nets:
                if not net == 'LIF':
                    print('-===-' * 30)
                    print(task, net)
                    idf = mdf[mdf['task_name'].str.contains(task) & mdf['net_name'].str.contains(net)]
                    print(idf.to_string())

    print(mdf.to_string())


    # print('Max experiment length: ', max(df['duration_experiment']))

    def cm_(metric='ppl'):
        c = 1
        if 'acc' in metric:
            c = 100

        def compactify_metrics(row):
            mtppl = round(c * row[f'mean_t_{metric}'].values[0], 2)
            stppl = round(c * row[f'std_t_{metric}'].values[0], 2)
            mvppl = round(c * row[f'mean_v_{metric}'].values[0], 2)
            svppl = round(c * row[f'std_v_{metric}'].values[0], 2)

            return f"{str(mvppl)}\pm{str(svppl)}/{str(mtppl)}\pm{str(stppl)}"

        return compactify_metrics


    def choose_metric(row):
        if row['task_name'].values[0] == 'wordptb':
            metric = row['ppl']
        else:
            metric = row['acc']
        return metric


    if make_latex:
        net = 'LSTM'
        idf = mdf[mdf['net_name'].str.contains(net)]
        idf['ppl'] = idf.apply(cm_('ppl'), axis=1)
        idf['acc'] = idf.apply(cm_('mode_acc'), axis=1)
        idf['metric'] = idf.apply(choose_metric, axis=1)
        # idf = idf[idf.columns.drop(list(idf.filter(regex='std')) + list(idf.filter(regex='mean')))]
        idf = idf[idf.columns.drop(list(idf.filter(regex='acc')) + list(idf.filter(regex='ppl')))]
        idf = idf[idf.columns.drop(['counts', 'initializer', 'net_name'])]
        # idf = idf[idf.columns.drop(['initializer', 'net_name'])]

        idf['comments'] = idf['comments'].str.replace('32_embproj_nogradreset_dropout:.3_timerepeat:2_', '', regex=True)
        idf['comments'] = idf['comments'].str.replace('find', '', regex=True)
        idf['comments'] = idf['comments'].str.replace('normpow:-1', r'\\infty', regex=True)
        idf['comments'] = idf['comments'].str.replace('normpow:', '', regex=True)
        idf['comments'] = idf['comments'].str.replace('_gausslsc', ' + g', regex=True)
        idf['comments'] = idf['comments'].str.replace('_berlsc', ' + b', regex=True)
        idf['comments'] = idf['comments'].str.replace('_gaussbeta', r'\\beta', regex=True)
        idf['comments'] = idf['comments'].str.replace('_lscdepth:1_lscout:0', '^{(d)}', regex=True)
        idf['comments'] = idf['comments'].str.replace('_lscdepth:1_lscout:1', '^{(dr)}', regex=True)
        idf = idf[~idf['comments'].str.contains('timerepeat:1')]

        conditions = np.unique(idf['comments'])
        tasks = ['sl_mnist', 'heidelberg', 'wordptb']
        order_conditions = ['', 'LSC_1', 'LSC_2', r'LSC_\infty', 'LSC_2 + g', 'LSC_2 + b',
                            r'LSC_2\beta', r'LSC_2\beta + g', r'LSC_2\beta + b',
                            'LSC_2^{(d)}', 'LSC_2^{(dr)}']

        idf['comments'] = pd.Categorical(idf['comments'], order_conditions)
        pdf = pd.pivot_table(idf, values='metric', index=['comments'], columns=['task_name'], aggfunc=np.sum)
        pdf = pdf.replace([0], '-')

        for t in tasks:
            pdf[t] = pdf[''][t]
            # mdf['std_{}'.format(m)] = mdf[m]['std']
        pdf = pdf.drop([''], axis=1)

        print(pdf.columns)
        # pdf = pdf[tasks]

        # pdf.sort_values('comments')
        # dfs = []
        # for task in tasks:
        #     tdf = idf[~idf['task_name'].str.contains(task)]
        #     tdf = tdf[tdf.columns.drop(['task_name'])]
        #
        #     dfs.append(tdf)

        print(conditions)

        print(idf.to_string())
        print(pdf.to_string())

        print(pdf.to_latex(index=True, escape=False))

if missing_exps:
    # columns of interest
    coi = ['seed', 'task_name', 'net_name', 'comments']
    import pandas as pd

    sdf = pd.read_hdf(h5path, 'df')

    sdf.drop([c for c in sdf.columns if c not in coi], axis=1, inplace=True)

    seed = 0
    n_seeds = 4
    seeds = [l + seed for l in range(n_seeds)]
    incomplete_comments = '32_embproj_nogradreset_dropout:.3_timerepeat:2_'

    experiments = []
    experiment = {
        # 'task_name': ['heidelberg', 'ps_mnist', 's_mnist', 'ss_mnist', 'sps_mnist', 'sl_mnist'],
        'task_name': ['heidelberg', 'wordptb', 'sl_mnist'],
        'net_name': ['maLSNN', 'LSTM'], 'seed': seeds,
        'comments': [
            incomplete_comments,
            incomplete_comments + f'findLSC_normpow:1',
            incomplete_comments + f'findLSC_normpow:-1',
            incomplete_comments + f'findLSC_normpow:2',
            incomplete_comments + f'findLSC_normpow:2_lscdepth:1_lscout:0',
            incomplete_comments + f'findLSC_normpow:2_lscdepth:1_lscout:1',
            incomplete_comments + f'findLSC_normpow:2_berlsc',
            incomplete_comments + f'findLSC_normpow:2_gausslsc',
        ],
    }
    experiments.append(experiment)

    experiment = {
        # 'task_name': ['heidelberg', 'ps_mnist', 's_mnist', 'ss_mnist', 'sps_mnist', 'sl_mnist'],
        'task_name': ['heidelberg', 'wordptb', 'sl_mnist'],
        'net_name': ['maLSNN'], 'seed': seeds,
        'comments': [
            incomplete_comments + f'findLSC_normpow:2_gaussbeta',
            incomplete_comments + f'findLSC_normpow:2_gaussbeta_berlsc',
            incomplete_comments + f'findLSC_normpow:2_gaussbeta_gausslsc',
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
    df = df[df['task_name'].str.contains('wordptb')]

    print('left, done, all: ', df.shape, sdf.shape, all_exps.shape)
    print('left')
    print(df.to_string())

    experiments = []
    for index, row in df.iterrows():
        experiment = {k: [row[k]] for k in df.columns}
        experiments.append(experiment)

    print(experiments)

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
