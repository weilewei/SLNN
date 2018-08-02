#!/bin/bash
echo "This script is used to run nn_python example"

thr=(1 2 4 8 12 16)

for th in "${thr[@]}"
    do
	export OMP_NUM_THREADS=${th}   
        python3 nn_python.py >> nn_${th}th_itr800_lr0.1_sz_5000 
	echo "done ${th}"	
done
