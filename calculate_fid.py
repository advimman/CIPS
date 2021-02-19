import argparse
import os

import torch
import torchvision
from torch_fidelity import calculate_metrics
import numpy as np

import model
from dataset import ImageDataset
from tensor_transforms import convert_to_coord_format


@torch.no_grad()
def calculate_fid(model, fid_dataset, bs, size, num_batches, latent_size, integer_values,
                  save_dir='fid_imgs', device='cuda'):
    coords = convert_to_coord_format(bs, size, size, device, integer_values=integer_values)
    for i in range(num_batches):
        z = torch.randn(bs, latent_size, device=device)
        fake_img, _ = model(coords, [z])
        for j in range(bs):
            torchvision.utils.save_image(fake_img[j, :, :, :],
                                         os.path.join(save_dir, '%s.png' % str(i * bs + j).zfill(5)), range=(-1, 1),
                                         normalize=True)
    metrics_dict = calculate_metrics(save_dir, fid_dataset, cuda=True, isc=False, fid=True, kid=False, verbose=False)
    return metrics_dict


if __name__ == '__main__':

    device = 'cuda'

    # dataset
    parser = argparse.ArgumentParser()
    parser.add_argument('path', type=str)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--coords_size', type=int, default=256)

    # Generator params
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--Generator', type=str, default='ModSIREN')
    parser.add_argument('--coords_integer_values', action='store_true')
    parser.add_argument('--fc_dim', type=int, default=512)
    parser.add_argument('--n_intermediate', type=int, default=9)
    parser.add_argument('--latent', type=int, default=512)
    parser.add_argument('--activation', type=str, default=None)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--to_crop', action='store_true')

    # fid
    parser.add_argument('--generated_path', type=str, default='fid/generated')
    parser.add_argument('--fid_samples', type=int, default=50000)
    parser.add_argument('--batch', type=int, default=2)

    args = parser.parse_args()
    args.n_mlp = 8

    os.makedirs(args.generated_path, exist_ok=True)

    transform_fid = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Lambda(lambda x: x.mul_(255.).byte())])
    dataset = ImageDataset(args.path, transform=transform_fid, resolution=args.coords_size, to_crop=args.to_crop)
    print('initial dataset length', dataset.length)
    dataset.length = args.fid_samples

    Generator = getattr(model, args.Generator)
    generator = Generator(size=args.size, hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                          n_intermediate=args.n_intermediate, activation=args.activation).to(device)

    checkpoint = os.path.join(args.model_path, args.ckpt)
    ckpt = torch.load(checkpoint)
    generator.load_state_dict(ckpt['g_ema'])
    generator.eval()
    cur_metrics = calculate_fid(generator,
                                fid_dataset=dataset,
                                bs=args.batch,
                                size=args.coords_size,
                                num_batches=args.fid_samples//args.batch,
                                latent_size=args.latent,
                                integer_values=args.coords_integer_values,
                                save_dir=args.generated_path)

    print(cur_metrics)
    np.savez(
        os.path.join(args.model_path, 'fid.npz'),
        frechet_inception_distance=cur_metrics['frechet_inception_distance'])
