#!/bin/bash
echo "This script is used to run nn_physl.physl VTune"

iteration_array=(800)
learning_rate=(0.1)
thr=(1 8)
matrix_size=(2000)

for it in "${iteration_array[@]}"
    do
    for lr in "${learning_rate[@]}"
        do
	for th in "${thr[@]}"
	    do
            for sz in "${matrix_size[@]}"
		do
                    amplxe-cl --collect hotspots /src/repos/phylanx/build_RelWithDebInfo/bin/physl nn_physl.physl ${lr} ${it} ${sz} --hpx:threads=${th} --hpx:bind=balanced --hpx:numa-sensitive
               	    echo "done ${th}_${it}_${lr}_${sz}"
           	 done
             done
         done
     done

