import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--just_val', type=bool, default=False)
parser.add_argument('--running_time', type=bool, default=False)
parser.add_argument('--lr', type=list, default=[1e-4, 2e-4, 1e-4, 2e-4, 2e-4])   # note: [1e-4, 5e-4] for wiki
parser.add_argument('--batch_size', type=int, default=64)      # note: 64 for wiki,  xmedia, nus21, 200 for xmedianet
parser.add_argument('--output_shape', type=int, default=128)   # note: 128 for wiki, xmedia, nus21, 256 for xmedianet
parser.add_argument('--alpha1', type=float, default=1)
parser.add_argument('--beta1', type=float, default=0.5)
parser.add_argument('--alpha2', type=float, default=1)
parser.add_argument('--beta2', type=float, default=0.2)
parser.add_argument('--datasets', type=str, default='wiki')  # xmedia, xmedianet, wiki, nus21 for XMedia, XMediaNet, Wikipedia, NUE-WIDE21
parser.add_argument('--view_id', type=int, default=-1)
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--sample_interval', type=int, default=1)
parser.add_argument('--epochs', type=int, default=50)   # note: 50 for wiki, 200 for xmedia, nus21, xmedianet
from to_seed import to_seed

print("current local time: ", time.asctime(time.localtime(time.time())))
seed = 1000
to_seed(seed=seed)
import os
os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    args = parser.parse_args()
    from MARS import Solver
    solver = Solver(args)

    solver.train()
    exit()

if __name__ == '__main__':
    main()
