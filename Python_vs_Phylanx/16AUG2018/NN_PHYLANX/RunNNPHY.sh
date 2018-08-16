#!/bin/bash
echo "This script is used to run nn_csv_python example"


#iteration_array=(1000)
iteration_array=(800)
row_stop_array=(1000)
#row_stop_array=(1000)
thr=(1)
#hiddenlayer_neurons = (500)

for it in "${iteration_array[@]}"
    do
    for rs in "${row_stop_array[@]}"
        do
	   for th in "${thr[@]}"
	    do
		export OMP_NUM_THREADS=${th}   
        python3 nn_csv_phylanx_v0.py $it 0 $rs 0 $rs 500 >> nnphylanx_${th}th_itrscs_${it}_${rs}_500
		echo "done ${th}_${it}_${rs}_500"	
	    done
        done
    done
