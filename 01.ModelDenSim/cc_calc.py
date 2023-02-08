import sys
from iotbx import mtz
from cctbx.array_family import flex
import iotbx

#Read the 1st mtz file
mtz1 = mtz.object(sys.argv[1])

#Read the 2nd mtz file
mtz2 = mtz.object(sys.argv[2])

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
with open('over5.txt','a')as f:
    f.write("%12.5e \n" % difff)
print(difff)
