#!/bin/bash
echo "This script is used to run nn_physl.physl example"

iteration_array=(800)
learning_rate=(0.1)
thr=(10)

for it in "${iteration_array[@]}"
    do
    for lr in "${learning_rate[@]}"
        do
	  for th in "${thr[@]}"
		do
           	 export OMP_NUM_THREADS=1
            		time ~/src/repos/phylanx/build_Release/bin/physl nn_physl.physl ${lr} ${it} --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive --hpx:print-counter=/threads{locality#*/total}/idle-rate >> nnphx_${th}th_itrscs_${it}_${lr}
           	 echo "done ${th}_${it}_${lr}"
       		 done
    	done
   done
