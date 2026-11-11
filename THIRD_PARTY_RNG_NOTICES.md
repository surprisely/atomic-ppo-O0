# RNG algorithm attribution

`atomic_dual/event_rng.py` is a Tensor/32-bit-limb compatibility implementation
of the SeedSequence and PCG64 algorithms used by NumPy. Reference sources:

- https://github.com/numpy/numpy/blob/v2.3.0/numpy/random/bit_generator.pyx
- https://github.com/numpy/numpy/blob/v2.3.0/numpy/random/src/pcg64/pcg64.h
- https://github.com/numpy/numpy/blob/v2.3.0/numpy/random/src/pcg64/pcg64.c

SeedSequence attribution: Copyright (c) 2015 Melissa E. O'Neill;
Copyright (c) 2019 NumPy Developers.

PCG attribution: Copyright 2014 Melissa O'Neill; Copyright 2015 Robert Kern.
The NumPy PCG sources state that they were relicensed MIT in May 2019.

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
