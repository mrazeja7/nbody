#!/bin/bash
#$ -l mem=1G
#$ -l h_rt=1:00:00
#$ -pe single 32
#$ -cwd

for i in {100000..1000000..100000}
do
	#echo "starting step $(($i/10000))"
	echo "$i 1 32" | ./arr # >>results.txt
	echo "$i 1 32" | ./arr # >>results.txt
	echo "$i 1 32" | ./arr # >>results.txt
done