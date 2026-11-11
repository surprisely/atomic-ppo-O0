"""Tensor reproduction of NumPy SeedSequence(pool_size=4) -> PCG64 first draw.

Only nonnegative uint32 entropy *components*, no spawn key, are supported. This
is the exact subset used by the dispatch event keys. No truncation/fallback.
Arithmetic uses base-2**32 limbs so CPU/CUDA int64 wrap yields identical bits.
Algorithm references: NumPy random/bit_generator.pyx and src/pcg64/pcg64.h.
See THIRD_PARTY_RNG_NOTICES.md for attribution and license notices.
"""
import torch

MASK=(1<<32)-1


def _seed_words(keys):
    h=0x43b0d7e5
    def hashmix(value):
        nonlocal h
        value=value^h;h=(h*0x931e8875)&MASK
        value=(value*h)&MASK
        return value^(value>>16)
    def mix(x,y):
        value=(0xca01f9dd*x-0x4973f715*y)&MASK
        return value^(value>>16)
    pool=[hashmix(keys[:,i]) for i in range(4)]
    for src in range(4):
        for dst in range(4):
            if src!=dst:pool[dst]=mix(pool[dst],hashmix(pool[src]))
    for i in range(4,keys.shape[1]):
        for dst in range(4):pool[dst]=mix(pool[dst],hashmix(keys[:,i]))
    h=0x8b51f9dd;words=[]
    for i in range(8):
        value=pool[i%4]^h;h=(h*0x58f38ded)&MASK
        value=(value*h)&MASK
        words.append(value^(value>>16))
    return words


def _validate(keys):
    if keys.dtype!=torch.int64 or keys.ndim!=2 or keys.shape[1]<4:
        raise ValueError('Expected int64 entropy matrix with >=4 uint32 components')
    if bool(((keys<0)|(keys>MASK)).any()):
        raise ValueError('Event entropy components must fit uint32; larger seeds are not silently truncated')


def seed_sequence_state(keys):
    _validate(keys)
    return torch.stack(_seed_words(keys),dim=1)


def _add128(a,b):
    out=[];carry=0
    for x,y in zip(a,b):
        value=x+y+carry;out.append(value&MASK);carry=value>>32
    return out


def _step(state,inc):
    multiplier=(0x9fccf645,0x4385df64,0x1fc65da4,0x2360ed05)
    out=[];carry=0
    for i in range(4):
        low=carry;high=0
        for j in range(i+1):
            product=state[j]*multiplier[i-j]
            low=low+(product&MASK);high=high+((product>>32)&MASK)
        out.append(low&MASK);carry=high+(low>>32)
    return _add128(out,inc)


def pcg64_first_raw_words(keys,validate=True):
    if validate:_validate(keys)
    w=_seed_words(keys)
    state=[w[2],w[3],w[0],w[1]]
    seq=[w[6],w[7],w[4],w[5]]
    inc=[((seq[i]<<1)|(1 if i==0 else seq[i-1]>>31))&MASK for i in range(4)]
    # Seed state=0; advance; add initstate; advance; then advance for first output.
    state=_step(_add128(inc,state),inc)
    state=_step(state,inc)
    low=state[0]^state[2];high=state[1]^state[3]
    rotation=state[3]>>26;swap=rotation>=32
    lo=torch.where(swap,high,low);hi=torch.where(swap,low,high)
    shift=rotation&31
    low=(lo>>shift)|((hi<<(32-shift))&MASK)
    high=(hi>>shift)|((lo<<(32-shift))&MASK)
    return torch.stack((low,high),dim=-1)


def pcg64_first_uniform(keys,validate=True):
    words=pcg64_first_raw_words(keys,validate=validate)
    bits=(words[:,1]<<21)|(words[:,0]>>11)
    return bits.double()*(1./(1<<53))
