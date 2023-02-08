import numpy as np
import os

def calc_mat(rot_rad,axis):
    if(axis=="x"):
        rot_mat00 = np.cos(rot_rad)
        rot_mat01 = -np.sin(rot_rad)
        rot_mat10 = np.sin(rot_rad)
        rot_mat11 = np.cos(rot_rad)

    elif(axis=="y"):
        rot_mat00 = np.cos(rot_rad)
        rot_mat01 = np.sin(rot_rad)
        rot_mat10 = -np.sin(rot_rad)
        rot_mat11 = np.cos(rot_rad)

    return(rot_mat00,rot_mat01,rot_mat10,rot_mat11)

def run_coms(n_data, axis, mean, sigma):
    for index in range(0,n_data):
        random_angle = np.random.normal(mean,sigma)
        rot_rad = np.radians(random_angle)

        rot_mat00,rot_mat01,rot_mat10,rot_mat11=calc_mat(rot_rad,axis)
    
        # com file
        comname="comfile_%03d"%index
        ofile=open(comname,"w")
        output_pdb = "apo_%s_%03d.pdb" % (axis, index)

        ofile.write("#!/bin/sh\n")
        ofile.write("pdbset xyzin /isilon/users/target/target/Staff/nakayama/rotate_test/apo.pdb xyzout %s <<eof-1\n" % output_pdb)
        ofile.write("transform -\n")
    
        if axis=="x":
            ofile.write("1 0 0  -\n")
            ofile.write("0 %12.8f %12.8f  -\n" % (rot_mat00,rot_mat01))
            ofile.write("0 %12.8f %12.8f  -\n" % (rot_mat10,rot_mat11))
    
        elif axis=="y":
            ofile.write("%12.8f 0 %12.8f  -\n" % (rot_mat00,rot_mat01))
            ofile.write("0 1 0 -\n")
            ofile.write("%12.8f 0 %12.8f  -\n" % (rot_mat10,rot_mat11))
    
        ofile.write("0.0  0.0 0.0\n")
        ofile.write("! can also be done with shift & rotate\n")
        ofile.write("eof-1\n")
        ofile.write("phenix.fmodel %s high_res=1.5\n"%output_pdb)
        ofile.close()
        os.system("sh %s" % comname)

########################################################################################
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

# index
print("Usage mean sigma n_data")
mean = float(sys.argv[1])
sigma = float(sys.argv[2])
n_data = float(sys.argv[3])

nhalf=int(n_data/2)

run_coms(nhalf, "x", mean, sigma)
run_coms(nhalf, "y", mean, sigma)

# apo mtzs
mtzs = sorted(glob.glob("apo_*.mtz"))

cc_file=open("cc.dat","w")

for index,mtz1 in enumerate(mtzs):
    for index2,mtz2 in enumerate(mtzs[index+1:]):
        difff=calc_cc(mtz1,mtz2)
        cc_file.write("%8.5f\n"%difff)
        print(index,index2,difff)

cc_file.close()
