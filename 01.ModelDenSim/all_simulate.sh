#!/bin/bash
angle_mu_list="0.5 1.0 1.5 2.0 2.5"
angle_sigma_list="0.1 0.5 1.0 2.0 5.0"
n_data_list="50 100 200 500 1000"

curr_dir=`pwd`

PPATH="/isilon/users/target/target/Staff/nakayama/rotate_test"

for angle_mu in $angle_mu_list; do
for angle_sigma in $angle_sigma_list; do
for n_data in $n_data_list; do

dir_name="mu${angle_mu}_sig${angle_sigma}_n${n_data}"
echo $dir_name

mkdir $dir_name
cd $dir_name
yamtbx.python $PPATH/main_control.py $angle_mu $angle_sigma $n_data
cd $curr_dir

done
done
done
