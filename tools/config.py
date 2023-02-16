
def default_eff_lr(activation, lr, batch_normalization=1):
    if batch_normalization == 1:
        if lr < 0:
            if activation in ['swish', 'relu']:
                lr = 0.001
            elif activation in ['tanh']:
                lr = 0.01
            else:
                lr = 0.001
    else:
        if lr < 0:
            if activation in ['swish', 'relu']:
                lr = 0.001
            elif activation in ['tanh']:
                lr = 0.000316
            else:
                lr = 0.001

    print('hei', lr)
    return lr