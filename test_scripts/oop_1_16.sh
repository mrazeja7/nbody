#!/bin/bash
#$ -l mem=1G
#$ -l h_rt=1:00:00
#$ -pe single 16
#$ -cwd

for j in {0..4..1}
do
	for i in {10000..100000..10000}
	do
		#echo "starting step $(($i/10000))"
		echo "$i 1 $((2**j))" | ./oop # >>results.txt
		echo "$i 1 $((2**j))" | ./oop # >>results.txt
		echo "$i 1 $((2**j))" | ./oop # >>results.txt
	done
done