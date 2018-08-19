LENGTHS=(16 25 36 49 64 81 100 121 144 169 196 225 256 289 324 361 400 441 484 529 576 625 676 729 784)

# (5 + 1) EA
for l in ${LENGTHS[@]}
do
	for seed in {0..199..1}
	do
		echo -ne "$l    ${seed}\r"
		sum=$((l+seed))
		python solver.py -p isg -m 5 -l 1 -S uniform -a plus -r $sum -e 10000 test_data/isg/isg_${l}_${seed}.txt
	done
done
exit
