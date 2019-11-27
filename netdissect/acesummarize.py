import os, sys, numpy, torch, argparse, skimage, json, shutil
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
import matplotlib

def main():
    parser = argparse.ArgumentParser(description='ACE optimization utility',
            prog='python -m netdissect.aceoptimize')
    parser.add_argument('--classname', type=str, default=None,
                        help='intervention classname')
    parser.add_argument('--layer', type=str, default='layer4',
                        help='layer name')
    parser.add_argument('--l2_lambda', type=float, nargs='+',
                        help='l2 regularizer hyperparameter')
    parser.add_argument('--outdir', type=str, default=None,
                        help='dissection directory')
    parser.add_argument('--variant', type=str, default=None,
                        help='experiment variant')
    args = parser.parse_args()

    if args.variant is None:
        args.variant = 'ace'

    run_command(args)

def run_command(args):
    fig = Figure(figsize=(4.5,3.5))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    for l2_lambda in args.l2_lambda:
        variant = args.variant
        if l2_lambda != 0.01:
            variant += '_reg%g' % l2_lambda

        dirname = os.path.join(args.outdir, args.layer, variant, args.classname)
        snapshots = os.path.join(dirname, 'snapshots')
        try:
            dat = [torch.load(os.path.join(snapshots, 'epoch-%d.pth' % i))
                 for i in range(10)]
        except:
            print('Missing %s snapshots' % dirname)
            return
        print('reg %g' % l2_lambda)
        for i in range(10):
            print(i, dat[i]['avg_loss'],
                  len((dat[i]['ablation'] == 1).nonzero()))

        ax.plot([dat[i]['avg_loss'] for i in range(10)],
            label='reg %g' % l2_lambda)
    ax.set_title('%s %s' % (args.classname, args.variant))
    ax.grid(True)
    ax.legend()
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epochs')
    fig.tight_layout()
    dirname = os.path.join(args.outdir, args.layer,
            args.variant, args.classname)
    fig.savefig(os.path.join(dirname, 'loss-plot.png'))

if __name__ == '__main__':
    main()
