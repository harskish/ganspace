import os, sys, argparse, json, shutil
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
    parser.add_argument('--outdir', type=str, default=None,
                        help='dissection directory')
    parser.add_argument('--metric', type=str, default=None,
                        help='experiment variant')
    args = parser.parse_args()

    if args.metric is None:
        args.metric = 'ace'

    run_command(args)

def run_command(args):
    fig = Figure(figsize=(4.5,3.5))
    FigureCanvas(fig)
    ax = fig.add_subplot(111)
    for metric in [args.metric, 'iou']:
        jsonname = os.path.join(args.outdir, args.layer, 'fullablation',
            '%s-%s.json' % (args.classname, metric))
        with open(jsonname) as f:
            summary = json.load(f)
        baseline = summary['baseline']
        effects = summary['ablation_effects'][:26]
        norm_effects = [0] + [1.0 - e / baseline for e in effects]
        ax.plot(norm_effects, label=
                'Units by ACE' if 'ace' in metric else 'Top units by IoU')
    ax.set_title('Effect of ablating units for %s' % (args.classname))
    ax.grid(True)
    ax.legend()
    ax.set_ylabel('Portion of %s pixels removed' % args.classname)
    ax.set_xlabel('Number of units ablated')
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, 25)
    fig.tight_layout()
    dirname = os.path.join(args.outdir, args.layer, 'fullablation')
    fig.savefig(os.path.join(dirname, 'effect-%s-%s.png' %
        (args.classname, args.metric)))
    fig.savefig(os.path.join(dirname, 'effect-%s-%s.pdf' %
        (args.classname, args.metric)))

if __name__ == '__main__':
    main()
