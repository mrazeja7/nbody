#!/bin/bash
#$ -l mem=1G
#$ -l h_rt=4:00:00
#$ -pe single 32
#$ -cwd
for j in {0..5..1}
do
	#echo "starting step $(($i/10000))"
	echo "500000 1 $((2**j))" | ./arr # >>results.txt
done
