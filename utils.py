import numpy as np


def make_batch(data, batch_size):
    nbatch = len(data) // batch_size
    trim_num = nbatch * batch_size
    data = np.asarray(data[:trim_num]).reshape([batch_size, nbatch])
    return data


def get_batch(opt, data, i):
    # data: [batch_size, original_seq_len]
    # We need to get smaller data seq according to bptt_len
    seq_len = min(opt.bptt_len, data.shape[1] - i - 1)
    context = data[:, i:i+seq_len]
    target = data[:, i+1:i+1+seq_len].reshape(-1)
    return context, target
