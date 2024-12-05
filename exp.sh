!#/bin/bash
program=${1:-"gpumatmul"}
optim=${2:-1}
mem_modes=(0 1 2 3 4)
output="experiment-results-$program-$optim.csv"

# write header
for mem_mode in ${mem_modes[@]}
do
    echo -n $mem_mode, >> $output
done
echo "" >> $output

echo "Running $program with optimization=$optim"
for i in {1..13} 
    do
    # power of two
    n=$((2**$i))
    echo "Processing $n"
    for mem_mode in ${mem_modes[@]}
        do
        /usr/bin/time -f '%S,' ./$program $n $optim 0 $mem_mode 1>/dev/null 2> ./tmp ; tr -d "\n" < ./tmp >> $output 
    done
    echo "" >> $output
done
rm ./tmp