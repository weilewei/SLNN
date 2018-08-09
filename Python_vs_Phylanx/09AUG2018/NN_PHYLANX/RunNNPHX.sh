#!/bin/bash
echo "This script is used to run nn_physl.physl example"

iteration_array=(800)
learning_rate=(0.1)
thr=(1 2 4 8 12 16)
matrix_size=(600 800 1000 1200 1400 1600 1800)

for it in "${iteration_array[@]}"
    do
    for lr in "${learning_rate[@]}"
        do
	for th in "${thr[@]}"
	    do
            for sz in "${matrix_size[@]}"
		do
           	    export OMP_NUM_THREADS=1
                    time ~/src/repos/phylanx/build_Release/bin/physl nn_physl.physl ${lr} ${it} ${sz} --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive --hpx:print-counter=/threads{locality#*/total}/idle-rate >> nnphx_${th}th_itrscs_${it}_${lr}_${sz}
           	 echo "done ${th}_${it}_${lr}_${sz}"
           	 done
             done
         done
     done

