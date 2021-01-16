#! /usr/bin/env python3

import time

import numpy as np

from lib_explain_spec import Spec_segmenter

###############################################################################
ti = time.time()
###############################################################################
spec = np.load("test/0_outlier.npy")

segmenter = Spec_segmenter(spec=spec)
idxs = segmenter.slic()

segmenter.plot(idxs=idxs, show=True)

for n, idx in enumerate(idxs):
    if n==0:
        pass
    elif n%2 == 0:
        print(idxs[n]-idxs[n-2]) 
###############################################################################
tf = time.time()
print(f"Runing time: {tf-ti:.2f} s")
