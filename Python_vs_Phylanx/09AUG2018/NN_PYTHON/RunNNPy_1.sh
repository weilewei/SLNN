#!/bin/bash
echo "This script is used to run nn_python example"


#iteration_array=(10000)
#iteration_array=(1 100 1000 5000 10000)
#row_stop_array=(100 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000)
#row_stop_array=(10000)
thr=(1 2 4 8 12 16)
size=(600 800 1000 1200 1600 1800)

for th in "${thr[@]}"
    do
    for sz in "${size[@]}"
        do
	export OMP_NUM_THREADS=${th}   
        python3 nn_python.py ${sz} >> nn_${th}th_itr800_lr0.1_sz_${sz}
	echo "done ${th}_${sz}"	
    done
done
