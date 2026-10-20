import sys
from pathlib import Path
import statistics
import time
import importlib.util
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('before',Path(__file__).parent/'before/src/morton.py')
before=importlib.util.module_from_spec(spec)
spec.loader.exec_module(before)
torch.set_num_threads(1)

def code_numpy(xyz,block=32768):
    xyz=xyz.numpy()
    code=np.empty(len(xyz),dtype=np.int64)
    for i in range(0,len(xyz),block):
        v=np.array(xyz[i:i+block].T,dtype=np.int64,order='C',copy=True)
        scratch=np.empty_like(v)
        v &= 0x1fffff
        for shift,mask in ((32,0x1f00000000ffff),(16,0x1f0000ff0000ff),(8,0x100f00f00f00f00f),(4,0x10c30c30c30c30c3),(2,0x1249249249249249)):
            np.left_shift(v,shift,out=scratch)
            np.bitwise_or(v,scratch,out=v)
            np.bitwise_and(v,mask,out=v)
        x,y,z=v
        np.left_shift(y,1,out=y)
        np.left_shift(z,2,out=z)
        target=code[i:i+block]
        np.bitwise_or(x,y,out=target)
        np.bitwise_or(target,z,out=target)
    return torch.from_numpy(code)

def cdf_numpy(cdf,block=16384):
    values=cdf.numpy()
    target=np.empty(values.shape,dtype=np.int16)
    scratch=np.empty((min(block,len(values)),17),dtype=values.dtype)
    ramp=np.arange(17,dtype=np.int16)
    for i in range(0,len(values),block):
        v=scratch[:len(values[i:i+block])]
        np.multiply(values[i:i+block],65520,out=v)
        np.rint(v,out=v)
        np.copyto(target[i:i+block],v,casting='unsafe')
        np.add(target[i:i+block],ramp,out=target[i:i+block])
    return torch.from_numpy(target)

def measure(functions):
    data={k:[] for k in functions}
    for _ in range(3):
        for fn in functions.values(): fn()
    for iteration in range(17):
        keys=list(functions)
        if iteration%2:keys.reverse()
        for k in keys:
            start=time.perf_counter_ns()
            result=functions[k]()
            data[k].append((time.perf_counter_ns()-start)/1e6)
            del result
    return {k:round(statistics.median(v),4) for k,v in data.items()}

with torch.inference_mode():
    for n in (64,4096,100000,400000):
        xyz=torch.randint(0,2**21,(n,4),dtype=torch.int32)[:,1:]
        expected=before.morton3_code(xyz)
        assert torch.equal(expected,code_numpy(xyz))
        fns={'old':lambda:before.morton3_code(xyz)}
        for block in (8192,16384,32768,65536):
            fns[str(block)]=lambda block=block:code_numpy(xyz,block)
        print('morton',n,measure(fns),flush=True)
        cdf=torch.rand((n,17))
        expected=cdf.mul(65520).round().to(torch.int16).add_(torch.arange(17,dtype=torch.int16))
        assert torch.equal(expected,cdf_numpy(cdf))
        fns={'old':lambda:cdf.mul(65520).round().to(torch.int16).add_(torch.arange(17,dtype=torch.int16))}
        for block in (8192,16384,32768):
            fns[str(block)]=lambda block=block:cdf_numpy(cdf,block)
        print('cdf',n,measure(fns),flush=True)
