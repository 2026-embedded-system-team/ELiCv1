"""Encode ELiCv1 geometry using the original checkpoint and stream format."""
import argparse
import os
import re
import time
from glob import glob

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import src.io as io
import src.morton as mt
import src.op as op
from src.runtime import EntropyEncoder, load_arithmetic_coder, parse_bool, require_cuda


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_glob', default='./data/SemanticKITTIExample/*.ply')
    parser.add_argument('--output_folder', default='./data/SemanticKITTI/')
    parser.add_argument('--is_data_pre_quantized', type=parse_bool, nargs='?', const=True, default=False)
    parser.add_argument('--posQ', type=int, choices=(4, 8, 16, 32, 64), default=16)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--ckpt', default='./model/SemanticKITTI/ELiCv1_K5.pt')
    parser.add_argument('--async_entropy', type=parse_bool, nargs='?', const=True, default=False,
                        help='Overlap CPU arithmetic encoding and GPU prediction (optional).')
    parser.add_argument('--csv', default=None, help='Override timing CSV path.')
    args = parser.parse_args()
    files = sorted(glob(args.input_glob, recursive=True))
    if not files:
        parser.error(f'no input point clouds match {args.input_glob}')
    device = require_cuda()
    from torchsparse import SparseTensor
    from torchsparse.nn import functional as spnnF
    from model import ELiCv1

    config = spnnF.conv_config.get_default_conv_config()
    config.kmap_mode = 'hashmap'
    spnnF.conv_config.set_global_conv_config(config)
    op.set_seed(11)
    os.makedirs(args.output_folder, exist_ok=True)
    match = re.search(r'_K(\d+)', args.ckpt)
    if match is None:
        parser.error('checkpoint filename must contain _K<number>')
    K = int(match.group(1))
    net = ELiCv1(args.channels, args.kernel_size, K).to(device)
    net.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=True))
    net.eval()
    ac = load_arithmetic_coder()

    with torch.inference_mode():
        coords = torch.randint(0, 4096, (4096, 3), dtype=torch.int32, device=device)
        net(SparseTensor(coords=torch.cat((coords[:, :1]*0, coords), dim=1),
                         feats=torch.ones((4096, 1), device=device)))
        for encoder in net.BoE:
            encoder.blend_weights.copy_(encoder.blend_weights.softmax(dim=0))
    torch.cuda.synchronize()
    xyz_list = io.read_point_clouds(files)
    rows = []
    with EntropyEncoder(ac.ArithmeticEncoder16(), args.async_entropy) as entropy:
        with torch.inference_mode():
            for path, points in tqdm(zip(files, xyz_list), total=len(files)):
                if len(points) == 0:
                    raise ValueError(f'empty point cloud: {path}')
                if args.is_data_pre_quantized:
                    xyz = torch.tensor(points)
                    if xyz.min().item() < 0:
                        xyz += 131072
                else:
                    xyz = torch.tensor(points / 0.001 + 131072)
                xyz = torch.round(xyz / args.posQ).int()
                if xyz.min() < 0 or xyz.max() >= 2**21:
                    raise ValueError(f'quantized coordinates outside 21-bit Morton range: {path}')
                bxyz = torch.cat((xyz[:, :1]*0, xyz), dim=1).to(device)
                torch.cuda.synchronize()
                started = time.perf_counter()
                hierarchy = mt.build_hierarchy(bxyz)
                indices, propagated = [], None
                for depth, (coords, occ) in enumerate(hierarchy):
                    last = depth + 1 == len(hierarchy)
                    s0, s1 = occ & 15, occ >> 4
                    index = K if depth + 2 <= 6 else net.select_encoder_index(s0, s1)
                    enc = net.BoE[index]
                    x = SparseTensor(coords=coords, feats=torch.ones((len(coords), 1), device=device))
                    x.feats = enc.local_net(x)
                    if propagated is not None:
                        x.feats = enc.blend_weights[0] * x.feats + enc.blend_weights[1] * propagated
                    x = enc.feat_resnet1(enc.feat_resnet0(x))
                    p0 = F.softmax(enc.pred_net0(x.feats), dim=-1)
                    x.feats += enc.prior_net0(s0)
                    x = enc.prior_s0_resnet1(enc.prior_s0_resnet0(x))
                    p1 = F.softmax(enc.pred_net1(x.feats), dim=-1)
                    probs = torch.cat((p0, p1), dim=0)
                    cdf = torch.cat((probs[:, :1]*0, probs.cumsum(dim=-1)), dim=-1)
                    cdf = op._convert_to_int_and_normalize(cdf, True)
                    entropy.encode_chunk(cdf, torch.cat((s0, s1)).to(torch.int16))
                    indices.append(index)
                    if not last:
                        x.feats += enc.prior_net1(s1)
                        x = enc.prior_s1_resnet1(enc.prior_s1_resnet0(x))
                        propagated = mt.upscale_feature(occ, x.feats)
                stream = entropy.finish()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                base = hierarchy[0][0][:, 1:].cpu().numpy()
                out = os.path.join(args.output_folder, os.path.splitext(os.path.basename(path))[0] + '.bin')
                with open(out, 'wb') as handle:
                    handle.write(np.asarray(args.posQ, dtype=np.int16).tobytes())
                    handle.write(np.asarray(len(base), dtype=np.int8).tobytes())
                    handle.write(np.asarray(base, dtype=np.int8).tobytes())
                    handle.write(op.pack_bitstream_and_enc_idx_list(stream, indices))
                rows.append(dict(in_file=path, out_file=out,
                                 bpp=op.get_file_size_in_bits(out)/len(points), enc_time=elapsed))
    testset = 'Ford' if 'Ford' in args.input_glob else 'KITTI' if 'KITTI' in args.input_glob else 'Unknown'
    csv = args.csv or f'{testset}_K{K}_posQ{args.posQ}.csv'
    frame = pd.DataFrame(rows)
    frame.to_csv(csv, index=False)
    print(f'Total: {len(rows)} | Avg. Bpp:{frame.bpp.mean():.4f} | '
          f'Encode time:{frame.enc_time.mean():.6f}s | '
          f'Max GPU Memory:{torch.cuda.max_memory_allocated()/1024**2:.2f}MB')


if __name__ == '__main__':
    main()
