#-*-coding:utf-8-*-
def to_seed(seed=0):
    print('seed: ' + str(seed))
    import numpy as np
    np.random.seed(seed)
    import random as rn
    rn.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 