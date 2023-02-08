import numpy as np
import sys
from iotbx import mtz
from cctbx.array_family import flex
import glob
import iotbx

def calc_cc(mtz1_name,mtz2_name):
    #Read the 1st mtz file
    mtz1 = mtz.object(mtz1_name)
    
    #Read the 2nd mtz file
    mtz2 = mtz.object(mtz2_name)
    
    #Extract miller arrays by using cctbx
    refl1_ = mtz1.as_miller_arrays()
    refl2_ = mtz2.as_miller_arrays()

    i_obs = refl1_[0]

    ref1 = [x for x in refl1_ if x.info().labels][0]
    ref2 = [x for x in refl2_ if x.info().labels][0]

    ref1=ref1.as_intensity_array()
    ref2=ref2.as_intensity_array()
    # Common reflections
    com1,com2 = ref1.common_sets(ref2,assert_is_similar_symmetry=False)
    ## calculate cc
    difff = flex.linear_correlation(com1.data(),com2.data()).coefficient()

    return difff


# apo mtzs
mtzs = sorted(glob.glob("apo_*.mtz"))

cc_file=open("cc.dat","w")

for index,mtz1 in enumerate(mtzs):
    for index2,mtz2 in enumerate(mtzs[index+1:]):
        difff=calc_cc(mtz1,mtz2)
        cc_file.write("%8.5f\n"%difff)
        print(index,index2,difff)

cc_file.close()
