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
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    # colors =[]
    if norm_id is None:
        return '#1f77b4' #'#097B2A'
    elif norm_id == 1:
        return '#ff7f0e' #'#40DE6E'
    elif norm_id == 2:
        return '#2ca02c' #'#B94D0C'
    elif norm_id == -1:
        return '#9467bd' #'#0C58B9'
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


def compactify_metrics(metric='ppl'):

    def cm(row):
        mt = row[f'mean_t_{metric}']
        st = row[f'std_t_{metric}']
        mv = row[f'mean_v_{metric}']
        sv = row[f'std_v_{metric}']

        return f"${str(mv)}\pm{str(sv)}$/${str(mt)}\pm{str(st)}$"

    return cm


def choose_metric(row):
    if row['task_name'] == 'PTB':
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
