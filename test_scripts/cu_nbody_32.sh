#!/bin/bash
#$ -l mem=4G
#$ -l h_rt=1:00:00
#$ -l cuda=true
#$ -pe single 16
#$ -cwd

for i in {10000..100000..10000}
do
	#echo "starting step $(($i/10000))"
	echo "$i 1" | ./cuda_nbody_v # >>results.txt
	echo "$i 1" | ./cuda_nbody_v # >>results.txt
	echo "$i 1" | ./cuda_nbody_v # >>results.txt
done

for i in {100000..1000000..100000}
do
	#echo "starting step $(($i/10000))"
	echo "$i 1" | ./cuda_nbody_v # >>results.txt
	echo "$i 1" | ./cuda_nbody_v # >>results.txt
	echo "$i 1" | ./cuda_nbody_v # >>results.txt
done
