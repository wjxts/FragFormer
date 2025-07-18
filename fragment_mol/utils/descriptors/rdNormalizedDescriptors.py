#  Copyright (c) 2018, Novartis Institutes for BioMedical Research Inc.
#  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#     * Neither the name of Novartis Institutes for BioMedical Research Inc.
#       nor the names of its contributors may be used to endorse or promote
#       products derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
from . import rdDescriptors
from . import dists
from collections import namedtuple
import scipy.stats as st
import numpy as np
import logging
# logging.disable(logging.WARNING) # can remove the no normalization warning

cdfs = {}

for name, (dist, params, minV, maxV, avg, std) in dists.dists.items():
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    dist = getattr(st, dist)
    
    # make the cdf with the parameters
    def cdf(v, dist=dist, arg=arg, loc=loc, scale=scale, minV=minV, maxV=maxV):
        v = dist.cdf(np.clip(v, minV, maxV), loc=loc, scale=scale, *arg) # 映射到cdf的值, 一定服从均匀分布
        return np.clip(v, 0., 1.)
    
    cdfs[name] = cdf

# for name in rdDescriptors.FUNCS:
#     if name not in cdfs:
#         logging.warning("No normalization for %s", name)

# WARNING:root:No normalization for BCUT2D_MWHI
# WARNING:root:No normalization for BCUT2D_MWLOW
# WARNING:root:No normalization for BCUT2D_CHGHI
# WARNING:root:No normalization for BCUT2D_CHGLO
# WARNING:root:No normalization for BCUT2D_LOGPHI
# WARNING:root:No normalization for BCUT2D_LOGPLOW
# WARNING:root:No normalization for BCUT2D_MRHI
# WARNING:root:No normalization for BCUT2D_MRLOW

# 问gpt，这些值的范围似乎都不是[0, 1], 还需要再仔细理解一下

def applyNormalizedFunc(name, m):
    if name not in cdfs:
        return 0.0
    try:
        return cdfs[name](rdDescriptors.applyFunc(name,m))
    except:
        logging.exception("Could not compute %s for molecule", name)
        return 0.0

class RDKit2DNormalized(rdDescriptors.RDKit2D):
    NAME = "RDKit2DNormalized"

    def calculateMol(self, m, smiles, internalParsing=False):
        # 真实的计算函数, process最终调用了这个函数
        # self.columns是RDKIT_PROPS[CURRENT_VERSION], 和返回的数据类型形成的pair
        # print(self.columns)
        res = [ applyNormalizedFunc(name, m) for name, _ in self.columns ]
        return res   
    
RDKit2DNormalized()