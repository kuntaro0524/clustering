#!/bin/bash

filelist=`ls *_y*pdb.mtz`
for eachfile in $filelist;do
echo $eachfile
yamtbx.python cc_calc.py $eachfile apo.pdb.mtz
done


