"""Local cache-size exploration; final measurements use bench_memory_io.py."""
import functools
import importlib.util
from pathlib import Path
import sys
import time
import statistics
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from src import morton, op

torch.set_num_threads(1)

def morton_batch(xyz, block=0):
    if block and len(xyz) > block:
        out = torch.empty(len(xyz), dtype=torch.int64, device=xyz.device)
        for i in range(0, len(xyz), block):
            out[i:i+block] = morton_batch(xyz[i:i+block])
        return out
    spread = morton._dilate3_u64(xyz)
    x, y, z = spread.unbind(1)
    return x | (y << 1) | (z << 2)

def morton_transpose(xyz, block=0):
    if block and len(xyz) > block:
        out=torch.empty(len(xyz),dtype=torch.int64)
        for i in range(0,len(xyz),block):
            out[i:i+block]=morton_transpose(xyz[i:i+block])
        return out
    spread=xyz.T.to(dtype=torch.int64,copy=True,memory_format=torch.contiguous_format)
    spread.bitwise_and_(0x1fffff)
    for shift,mask in ((32,0x1f00000000ffff),(16,0x1f0000ff0000ff),(8,0x100f00f00f00f00f),(4,0x10c30c30c30c30c3),(2,0x1249249249249249)):
        spread.bitwise_or_(spread << shift).bitwise_and_(mask)
    x,y,z=spread.unbind(0)
    x.bitwise_or_(y.bitwise_left_shift_(1)).bitwise_or_(z.bitwise_left_shift_(2))
    return x

def morton_workspace(xyz,block):
    out=torch.empty(len(xyz),dtype=torch.int64)
    spread=torch.empty((3,min(block,len(xyz))),dtype=torch.int64)
    shifted=torch.empty_like(spread)
    for i in range(0,len(xyz),block):
        n=min(block,len(xyz)-i)
        values=spread[:,:n]
        scratch=shifted[:,:n]
        values.copy_(xyz[i:i+block].T).bitwise_and_(0x1fffff)
        for shift,mask in ((32,0x1f00000000ffff),(16,0x1f0000ff0000ff),(8,0x100f00f00f00f00f),(4,0x10c30c30c30c30c3),(2,0x1249249249249249)):
            torch.bitwise_left_shift(values,shift,out=scratch)
            values.bitwise_or_(scratch).bitwise_and_(mask)
        x,y,z=values.unbind(0)
        x.bitwise_or_(y.bitwise_left_shift_(1)).bitwise_or_(z.bitwise_left_shift_(2))
        out[i:i+block].copy_(x)
    return out

def cdf_batch(cdf, block):
    out = torch.empty_like(cdf, dtype=torch.int16)
    workspace = torch.empty((min(block, len(cdf)), cdf.shape[1]), dtype=cdf.dtype)
    ramp = op._cdf_ramp(cdf.shape[1], cdf.device)
    for i in range(0, len(cdf), block):
        chunk = cdf[i:i+block]
        scratch = workspace[:len(chunk)]
        torch.mul(chunk, 65536-(cdf.shape[1]-1), out=scratch)
        scratch.round_()
        out[i:i+block].copy_(scratch).add_(ramp)
    return out

def cdf_inplace(cdf):
    return cdf.mul(65520).round_().to(torch.int16).add_(op._cdf_ramp(17,cdf.device))

def measure(fns):
    for _ in range(3):
        for fn in fns.values(): fn()
    times={k:[] for k in fns}
    for repeat in range(13):
        keys=list(fns)
        if repeat%2: keys.reverse()
        for k in keys:
            start=time.perf_counter_ns()
            result=fns[k]()
            times[k].append((time.perf_counter_ns()-start)/1e6)
            del result
    return {k:round(statistics.median(v),4) for k,v in times.items()}

with torch.inference_mode():
    for n in (4096,100000,400000):
        bxyz=torch.randint(0,2**18,(n,4),dtype=torch.int32)
        xyz=bxyz[:,1:]
        expected=morton.morton3_code(xyz)
        fns={'before':lambda:morton.morton3_code(xyz)}
        for block in (0,1024,4096,8192,16384,32768):
            fn=functools.partial(morton_batch,xyz,block)
            assert torch.equal(expected,fn())
            fns['block'+str(block)]=fn
        for block in (0,4096,8192,16384,32768):
            fn=functools.partial(morton_transpose,xyz,block)
            assert torch.equal(expected,fn())
            fns['transpose'+str(block)]=fn
        for block in (8192,16384,32768):
            fn=functools.partial(morton_workspace,xyz,block)
            assert torch.equal(expected,fn())
            fns['workspace'+str(block)]=fn
        print('morton',n,measure(fns),flush=True)
        cdf=torch.rand((n,17)).sort(dim=-1).values
        expected=op._convert_to_int_and_normalize(cdf,True)
        fns={'before':lambda:op._convert_to_int_and_normalize(cdf,True),'inplace':lambda:cdf_inplace(cdf)}
        for block in (1024,4096,8192,16384,32768,65536):
            fn=functools.partial(cdf_batch,cdf,block)
            assert torch.equal(expected,fn())
            fns['block'+str(block)]=fn
        print('cdf',n,measure(fns),flush=True)
