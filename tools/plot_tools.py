import numpy as np
def clean_weight_name(wn):
    if 'encoder' in wn or 'reg' in wn:
        layer = wn.split('_')[1]
    else:
        layer = 'R'

    wn = wn.split('/')[-1].split(':')[0]
    if 'input_weights' in wn:
        wn = 'W_{' + f'in, {layer}' + '}'
    elif 'recurrent_weights' in wn:
        wn = 'W_{' + f'rec, {layer}' + '}'
    elif 'tau_adaptation' in wn:
        wn = r'\tau^{\vartheta}_{' + f'{layer}' + '}'
    elif 'tau' in wn:
        wn = r'\tau^{y}_{' + f'{layer}' + '}'
    elif 'thr' in wn:
        wn = r'b^{\vartheta}_{' + f'{layer}' + '}'
    elif 'beta' in wn:
        wn = r'\beta_{' + f'{layer}' + '}'
    elif 'recurrent_kernel' in wn:
        wn = r'U_{j,' + f'{layer}' + '}'
    elif 'kernel' in wn:
        wn = r'W_{j,' + f'{layer}' + '}'
    elif 'bias' in wn:
        wn = r'b_{j,' + f'{layer}' + '}'
    elif 'switch' in wn:
        wn = r'switch_{' + f'{layer}' + '}'
    else:
        wn = f'{wn}_{layer}'

    if layer == 'R':
        wn = wn.replace('j,', '')

    return f'${wn}$'


def get_path(df, normpow, task_name, net_name, gauss_beta):
    comment = f'_normpow:{normpow}_lscdepth:1_lscout:1'

    idf = df[df['comments'].str.contains('savelscweights')]
    idf = idf[idf['comments'].str.contains(comment)]
    idf = idf[idf['task_name'].str.contains(task_name)]
    idf = idf[idf['net_name'].str.contains(net_name)]
    if gauss_beta:
        idf = idf[idf['comments'].str.contains('gaussbeta')]
    else:
        idf = idf[~idf['comments'].str.contains('gaussbeta')]

    print(idf.to_string())
    print(idf.head(3)['path'].values)
    path = idf.head(1)['path'].values[0]
    return path


def color_nid(norm_id):
    # list of tab20 colors
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
              '#17becf']
    # colors =[]
    if norm_id is None:
        return '#1f77b4'  # '#097B2A'
    elif norm_id == 1:
        return '#ff7f0e'  # '#40DE6E'
    elif norm_id == 2:
        return '#2ca02c'  # '#B94D0C'
    elif norm_id == -1:
        return '#9467bd'  # '#0C58B9'
    else:
        raise ValueError('norm_id not in [None, 1, 2, -1]')


def clean_nid(norm_id):
    if norm_id is None:
        return 'no LSC'
    elif norm_id == 1:
        return '$p=1$'
    elif norm_id == 2:
        return '$p=2$'
    elif norm_id == -1:
        return '$p=\infty$'
    else:
        raise ValueError('norm_id not in [None, 1, 2, -1]')


def compactify_metrics(metric='ppl', data_split='test', round_to=2):
    # assert data_split in [None, 'test', 'val', 't_', 'v_']
    def cm(row):
        print(row.keys())
        if data_split is None:
            mt = row[f'mean_t_{metric}']
            st = row[f'std_t_{metric}']
            mv = row[f'mean_v_{metric}']
            sv = row[f'std_v_{metric}']

            output = f"${str(mv)}\pm{str(sv)}$/${str(mt)}\pm{str(st)}$"
        else:
            m = row[f'mean_{data_split}{metric}']
            s = row[f'std_{data_split}{metric}']
            output = f"${str(m)}\pm{str(s)}$"

        return output

    return cm


def choose_metric(row):
    if row['task'] == 'PTB':
        metric = row['ppl']
    else:
        metric = row['acc']
    return metric


def bolden_best(metric='mean_t_ppl'):
    c = 1
    if 'acc' in metric:
        c = 100

    def bb(row):
        value = row[f'{metric}']  # .values[0]
        if value == row[f'best_{metric}']:
            bolden = True
        else:
            bolden = False

        value = round(c * value, 2)
        if bolden:
            value = r'\textbf{' + str(value) + '}'
        else:
            value = f'{value}'
        return value

    return bb


def find_length(x):
    if isinstance(x, dict):
        l = len(x['batch 0 layer 0'])
    elif isinstance(x, list):
        l = len(x)
    else:
        l = None
    return l


def summary_lsc(x):
    if isinstance(x, dict):
        if 'batch 1 layer 0' in x.keys():
            x = x['batch 1 layer 0']
        else:
            x = x['batch 0 layer 0']

        l = f"{round(x[0][0], 2)}/{round(x[-1][-1], 2)}"
    elif isinstance(x, list):
        l = f"{round(x[0], 2)}/{round(x[-1], 2)}"
    else:
        l = None
    return l


def recnorms_list(x):
    if isinstance(x, dict):
        if 'batch 1 layer 0' in x.keys():
            x = x['batch 1 layer 0']
        else:
            x = x['batch 0 layer 0']

        l = [item for sublist in x for item in sublist]
    elif isinstance(x, list):
        l = x
    else:
        l = None
    return l


def reorganize(x):
    if isinstance(x['LSC_norms'], list):
        x = x['LSC_norms']
    elif isinstance(x['LSC_norms'], str):
        x = [float(i) for i in x['LSC_norms'][1:-1].split(', ')]
    else:
        x = x['rec_norms']

    return x


lsc_colors_dict = {
    'findLSC_radius': [0.43365406, 0.83304796, 0.58958684],
    'findLSC_radius_targetnorm:.5': [0.43365406, 0.43304796, 0.58958684],
    'findLSC_radius_targetnorm:.5_randlsc': [0.73365406, 0.43304796, 0.58958684],
    'findLSC_supsubnpsd_deslice': [0.73365406, 0.23304796, 0.28958684],
    '': [0.24995383, 0.49626022, 0.35960801],
    'findLSC': [0.74880857, 0.9167003, 0.50021289], 'findLSC_supnpsd2': [0.69663182, 0.25710645, 0.19346206],
    'findLSC_supsubnpsd': [0.2225346, 0.06820208, 0.9836983], 'heinit': [0.96937357, 0.28256986, 0.26486611],
    'findLSC_radius_truersplit': [0.2, 0.83304796, 0.58958684],
    'findLSC_supsubnpsd_truersplit': [0.2, 0.3, 0.58958684],
    'findLSC_truersplit': [0.2, 0.3, 0.8],

}


def lsc_colors(name):
    name  = name.replace('_onlypretrain', '')
    name  = name.replace('_onlyloadpretrained', '')
    if name in lsc_colors_dict.keys():
        return lsc_colors_dict[name]
    else:
        # assign random color
        return np.random.rand(3)


def lsc_clean_comments(c):
    if c == 'findLSC':
        c = 'sub ($L_2$)'
    c = c.replace('findLSC_', '')

    if 'targetnorm:.5' in c:
        c = r'sub ($\rho_{1/2}$)'

    if 'radius' in c:
        c = r'sub ($\rho$)'

    if c == '':
        c = 'Glorot'
        c = 'default'

    if c == 'heinit':
        c = 'He'

    c = c.replace('npsd', '')
    # c = c.replace('2', '')
    return c
