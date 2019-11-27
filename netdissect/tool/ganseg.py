'''
A simple tool to generate sample of output of a GAN,
and apply semantic segmentation on the output.
'''

import torch, numpy, os, argparse, sys, shutil
from PIL import Image
from torch.utils.data import TensorDataset
from netdissect.zdataset import standard_z_sample, z_dataset_for_model
from netdissect.progress import default_progress, verbose_progress
from netdissect.autoeval import autoimport_eval
from netdissect.workerpool import WorkerBase, WorkerPool
from netdissect.nethook import edit_layers, retain_layers
from netdissect.segviz import segment_visualization
from netdissect.segmenter import UnifiedParsingSegmenter
from scipy.io import savemat

def main():
    parser = argparse.ArgumentParser(description='GAN output segmentation util')
    parser.add_argument('--model', type=str, default=
            'netdissect.proggan.from_pth_file("' + 
            'models/karras/churchoutdoor_lsun.pth")',
            help='constructor for the model to test')
    parser.add_argument('--outdir', type=str, default='images',
            help='directory for image output')
    parser.add_argument('--size', type=int, default=100,
            help='number of images to output')
    parser.add_argument('--seed', type=int, default=1,
            help='seed')
    parser.add_argument('--quiet', action='store_true', default=False,
            help='silences console output')
    #if len(sys.argv) == 1:
    #    parser.print_usage(sys.stderr)
    #    sys.exit(1)
    args = parser.parse_args()
    verbose_progress(not args.quiet)

    # Instantiate the model
    model = autoimport_eval(args.model)

    # Make the standard z
    z_dataset = z_dataset_for_model(model, size=args.size)

    # Make the segmenter
    segmenter = UnifiedParsingSegmenter()

    # Write out text labels
    labels, cats = segmenter.get_label_and_category_names()
    with open(os.path.join(args.outdir, 'labels.txt'), 'w') as f:
        for i, (label, cat) in enumerate(labels):
            f.write('%s %s\n' % (label, cat))

    # Move models to cuda
    model.cuda()

    batch_size = 10
    progress = default_progress()
    dirname = args.outdir

    with torch.no_grad():
        # Pass 2: now generate images
        z_loader = torch.utils.data.DataLoader(z_dataset,
                    batch_size=batch_size, num_workers=2,
                    pin_memory=True)
        for batch_num, [z] in enumerate(progress(z_loader,
                desc='Saving images')):
            z = z.cuda()
            start_index = batch_num * batch_size
            tensor_im = model(z)
            byte_im = ((tensor_im + 1) / 2 * 255).clamp(0, 255).byte().permute(
                    0, 2, 3, 1).cpu()
            seg = segmenter.segment_batch(tensor_im)
            for i in range(len(tensor_im)):
                index = i + start_index
                filename = os.path.join(dirname, '%d_img.jpg' % index)
                Image.fromarray(byte_im[i].numpy()).save(
                        filename, optimize=True, quality=100)
                filename = os.path.join(dirname, '%d_seg.mat' % index)
                savemat(filename, dict(seg=seg[i].cpu().numpy()))
                filename = os.path.join(dirname, '%d_seg.png' % index)
                Image.fromarray(segment_visualization(seg[i].cpu().numpy(),
                    tensor_im.shape[2:])).save(filename)
    srcdir = os.path.realpath(
       os.path.join(os.getcwd(), os.path.dirname(__file__)))
    shutil.copy(os.path.join(srcdir, 'lightbox.html'),
           os.path.join(dirname, '+lightbox.html'))

if __name__ == '__main__':
    main()
