import numpy as np

### random angle (abs(random_angle) < ~ 5.0 deg)
mean=5.0
sigma = 1.0

random_angle = np.random.normal(mean,sigma)
angle = random_angle

def calc_mat(rot_rad):
    rot_mat00 = np.cos(rot_rad)
    rot_mat01 = np.sin(rot_rad)
    rot_mat10 = -np.sin(rot_rad)
    rot_mat11 = np.cos(rot_rad)

    return(rot_mat00,rot_mat01,rot_mat10,rot_mat11)

# index
for index in range(0,100):
    random_angle = np.random.normal(mean,sigma)
    rot_rad = np.radians(random_angle)
    rot_mat00,rot_mat01,rot_mat10,rot_mat11=calc_mat(rot_rad)

    # com file
    comname="comfile_y_%03d"%index
    ofile=open(comname,"w")
    ofile.write("#!/bin/sh\n")
    ofile.write("pdbset xyzin apo.pdb xyzout apo_y_%03d.pdb <<eof-1\n" % index)
    ofile.write("transform -\n")
    ofile.write("%12.8f 0 %12.8f  -\n" % (rot_mat00,rot_mat01))
    ofile.write("0 1 0 -\n")
    ofile.write("%12.8f 0 %12.8f  -\n" % (rot_mat10,rot_mat11))
    ofile.write("0.0  0.0 0.0\n")
    ofile.write("! can also be done with shift & rotate\n")
    ofile.write("eof-1\n")
    ofile.write("\\rm apo_y_%03d.pdb.mtz\n"%index)
    ofile.write("phenix.fmodel ./apo_y_%03d.pdb high_res=1.5\n"%index)

    ofile.close()
    print("sh %s"%comname)
