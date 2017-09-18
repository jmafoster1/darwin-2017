LENGTHS=(16 25 36 49 64 81 100 121 144 169 196 225 256 289 324 361 400 441 484 529 576 625 676 729 784)

# Random
for l in ${LENGTHS[@]}
do
	for seed in {0..199..1}
	do
		echo -ne "$l    ${seed}\r"
		sum=$((l+seed))
		python solver.py -p onemax -m 10000 -l 1 -S uniform -a plus -r $sum -e 10000 onemax_${l}_${seed}.txt
	done
done
echo -ne "\n"

# (1 + 1) EA
for l in ${LENGTHS[@]}
do
	for seed in {0..199..1}
	do
		echo -ne "$l    ${seed}\r"
		sum=$((l+seed))
		python solver.py -p onemax -m 1 -l 1 -S uniform -a plus -r $sum -e 10000 onemax_${l}_${seed}.txt
	done
done
echo -ne "\n"

# (2 + 1) GA
for l in ${LENGTHS[@]}
do
	for seed in {0..199..1}
	do
		echo -ne "$l    ${seed}\r"
		sum=$((l+seed))
		python solver.py -p onemax -c -m 2 -l 1 -S uniform -a plus -r $sum -e 10000 onemax_${l}_${seed}.txt
	done
done
echo -ne "\n"

# Greedy (2 + 1) GA
for l in ${LENGTHS[@]}
do
	for seed in {0..199..1}
	do
		echo -ne "$l    ${seed}\r"
		sum=$((l+seed))
		python solver.py -p onemax -S uniform -a greedy -m 2 -r $sum -e 10000 onemax_${l}_${seed}.txt
	done
done
echo -ne "\n"

# (5 + 1) EA
for l in ${LENGTHS[@]}
do
	for seed in {0..199..1}
	do
		echo -ne "$l    ${seed}\r"
		sum=$((l+seed))
		python solver.py -p onemax -m 5 -l 1 -S uniform -a plus -r $sum -e 10000 onemax_${l}_${seed}.txt
	done
done
echo -ne "\n"

# (5+1) GA
for l in ${LENGTHS[@]}
do
	for seed in {0..199..1}
	do
		echo -ne "$l    ${seed}\r"
		sum=$((l+seed))
		python solver.py -p onemax -m 5 -l 1 -c -S uniform -a plus -r $sum -e 10000 onemax_${l}_${seed}.txt
	done
done
echo -ne "\n"

# (5+1) GA tournament selection
for l in ${LENGTHS[@]}
do
	for seed in {0..199..1}
	do
		echo -ne "$l    ${seed}\r"
		sum=$((l+seed))
		python solver.py -p onemax -m 5 -l 1 -c -S tournament -s 2 -a plus -r $sum -e 10000 onemax_${l}_${seed}.txt
	done
done
echo -ne "\n"

# (20 + 20) EA
for l in ${LENGTHS[@]}
do
	for seed in {0..199..1}
	do
		echo -ne "$l    ${seed}\r"
		sum=$((l+seed))
		python solver.py -p onemax -m 20 -l 20 -S uniform -a plus -r $sum -e 10000 onemax_${l}_${seed}.txt
	done
done
echo -ne "\n"

# (20 + 20) GA
for l in ${LENGTHS[@]}
do
	for seed in {0..199..1}
	do
		echo -ne "$l    ${seed}\r"
		sum=$((l+seed))
		python solver.py -p onemax -m 20 -l 20 -c -S uniform -a plus -r $sum -e 10000 onemax_${l}_${seed}.txt
	done
done
echo -ne "\n"

# (1 + 1) Fast EA
for l in ${LENGTHS[@]}
do
	for seed in {0..199..1}
	do
		echo -ne "$l    ${seed}\r"
		sum=$((l+seed))
		python solver.py -p onemax -m 1 -l 1 -f -S uniform -a plus -r $sum -e 10000 onemax_${l}_${seed}.txt
	done
done
echo -ne "\n"

# (20 + 20) Fast GA
for l in ${LENGTHS[@]}
do
	for seed in {0..199..1}
	do
		echo -ne "$l    ${seed}\r"
		sum=$((l+seed))
		python solver.py -p onemax -m 20 -l 20 -c -f -S uniform -a plus -r $sum -e 10000 onemax_${l}_${seed}.txt
	done
done
echo -ne "\n"

# (1 + (lambda, lambda)) EA
for l in ${LENGTHS[@]}
do
	for seed in {0..199..1}
	do
		echo -ne "$l    ${seed}\r"
		sum=$((l+seed))
		python solver.py -p onemax -S uniform -a lambdalambda -r $sum -e 10000 onemax_${l}_${seed}.txt
	done
done
echo -ne "\n"

