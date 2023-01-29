
def default_eff_lr(activation, lr):

    if lr < 0:
        if activation in ['swish', 'relu']:
            lr = 0.001
        elif activation in ['tanh']:
            lr = 0.01
        else:
            lr = 0.001

    return lr