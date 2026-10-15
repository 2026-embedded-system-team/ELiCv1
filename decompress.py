"""Decode original or optimized ELiCv1 streams without changing the model."""
import argparse
import os
import re
import time
from glob import glob

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import src.io as io
import src.morton as mt
import src.op as op
from src.runtime import load_arithmetic_coder, parse_bool, require_cuda


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input_glob', default='./data/SemanticKITTI/*.bin')
    parser.add_argument('--output_folder', default='./data/SemanticKITTI/')
    parser.add_argument('--is_data_pre_quantized', type=parse_bool, nargs='?', const=True, default=False)
    parser.add_argument('--channels', type=int, default=32)
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--ckpt', default='./model/SemanticKITTI/ELiCv1_K5.pt')
    parser.add_argument('--csv', default=None)
    args = parser.parse_args()
    files = sorted(glob(args.input_glob, recursive=True))
    if not files:
        parser.error(f'no compressed streams match {args.input_glob}')
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
    decoder = load_arithmetic_coder().ArithmeticDecoder16()
    with torch.inference_mode():
        coords = torch.randint(0, 4096, (4096, 3), dtype=torch.int32, device=device)
        net(SparseTensor(coords=torch.cat((coords[:, :1]*0, coords), dim=1),
                         feats=torch.ones((4096, 1), device=device)))
        for encoder in net.BoE:
            encoder.blend_weights.copy_(encoder.blend_weights.softmax(dim=0))
    rows, quantizers = [], set()
    with torch.inference_mode():
        for path in tqdm(files):
            with open(path, 'rb') as handle:
                header = handle.read(3)
                if len(header) != 3:
                    raise ValueError(f'truncated header: {path}')
                posQ = int(np.frombuffer(header[:2], dtype=np.int16)[0])
                count = header[2]
                data = handle.read(count * 3)
                if not 1 <= count <= 64 or len(data) != count*3 or posQ not in (4, 8, 16, 32, 64):
                    raise ValueError(f'invalid ELiCv1 header: {path}')
                base = np.frombuffer(data, dtype=np.int8).astype(np.int32).reshape(-1, 3)
                stream, indices = op.unpack_bitstream_and_enc_idx_list(handle.read())
            if not len(indices) or np.any(indices > K):
                raise ValueError(f'invalid encoder indices: {path}')
            coords = torch.tensor(base, device=device)
            coords = torch.cat((coords[:, :1]*0, coords), dim=1)
            decoder.reset(stream)
            torch.cuda.synchronize()
            started = time.perf_counter()
            propagated = None
            for depth, index in enumerate(indices):
                last = depth + 1 == len(indices)
                enc = net.BoE[int(index)]
                x = SparseTensor(coords=coords, feats=torch.ones((len(coords), 1), device=device))
                x.feats = enc.local_net(x)
                if propagated is not None:
                    x.feats = enc.blend_weights[0]*x.feats + enc.blend_weights[1]*propagated
                x = enc.feat_resnet1(enc.feat_resnet0(x))
                probs = enc.pred_net0(x.feats).softmax(dim=-1)
                cdf = torch.cat((probs[:, :1]*0, probs.cumsum(dim=-1)), dim=-1)
                s0 = decoder.decode_chunk(op._convert_to_int_and_normalize(cdf, True).cpu()).to(device)
                x.feats += enc.prior_net0(s0)
                x = enc.prior_s0_resnet1(enc.prior_s0_resnet0(x))
                probs = enc.pred_net1(x.feats).softmax(dim=-1)
                cdf = torch.cat((probs[:, :1]*0, probs.cumsum(dim=-1)), dim=-1)
                s1 = decoder.decode_chunk(op._convert_to_int_and_normalize(cdf, True).cpu()).to(device)
                occ = s1*16 + s0
                if last:
                    coords = mt.upscale_coordinate(coords, occ)
                else:
                    x.feats += enc.prior_net1(s1)
                    x = enc.prior_s1_resnet1(enc.prior_s1_resnet0(x))
                    coords, propagated = mt.upscale_coordinate_feature(coords, occ, x.feats)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            scan = coords[:, 1:]*posQ - 131072
            if not args.is_data_pre_quantized:
                scan = scan * 0.001
            out = os.path.join(args.output_folder, os.path.splitext(os.path.basename(path))[0]+'.ply')
            io.save_ply_ascii_geo(scan.float().cpu().numpy(), out)
            rows.append(dict(out_file=path, decoded_file=out, dec_time=elapsed))
            quantizers.add(posQ)
    frame = pd.DataFrame(rows)
    testset = 'Ford' if 'Ford' in args.input_glob else 'KITTI' if 'KITTI' in args.input_glob else 'Unknown'
    suffix = str(next(iter(quantizers))) if len(quantizers) == 1 else 'mixed'
    csv = args.csv or f'{testset}_K{K}_posQ{suffix}.csv'
    if os.path.exists(csv):
        old = pd.read_csv(csv)
        if 'out_file' in old:
            old = old.drop(columns=['dec_time', 'decoded_file'], errors='ignore')
            frame = old.merge(frame, on='out_file', how='right', validate='one_to_one')
    frame.to_csv(csv, index=False)
    print(f'Total: {len(rows)} | Decode Time:{frame.dec_time.mean():.6f}s | '
          f'Max GPU Memory:{torch.cuda.max_memory_allocated()/1024**2:.2f}MB')


if __name__ == '__main__':
    main()
