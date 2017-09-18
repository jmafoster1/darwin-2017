l=128

## Greedy (2 + 1) GA
#(for seed in {0..199..1}
#do
#	>&2 echo -ne "$l    ${seed}\r"
#	sum=$((l+seed))
#	python solver.py -p onemax -S uniform -a greedy -m 2 -r $sum -e 10000 onemax_${l}_${seed}.txt

#done) > results/greedy2+1test.txt
#>&2 echo -ne "\n"

# (1 + (lambda, lambda)) EA
(for seed in {0..999..1}
do
	>&2 echo -ne "$l    ${seed}\r"
	sum=$((l+seed))
	python solver.py -p onemax -S uniform -a lambdalambda -r $sum -e 10000 onemax_${l}_${seed}.txt
done) > results/lambdalambdatest.txt
>&2 echo -ne "\n"
