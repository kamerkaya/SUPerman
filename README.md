Getting Started
=========
Requirements
----------------
We compiled this program with `nvcc` using `CUDA 11.2`and ran it on ` `. 

Compiling
----------
You can compile the program with the following command:
```
nvcc main.cu -O3 -Xcompiler -fopenmp -o perman
```

Running
----------
While running the program, all the parameters can be given in the command line. Following is an one example for running matrix permanent calculator.
```
./perman -m int/30_0.50_0 -g -p4
```

Parameters
----------
- `-m (required)`: Requires a value which indicates the path for the input matrix.
- `-p (optional)`: Requires a value which indicates the algorithm id of the algorithm for permanent calculation. `Default value is 1`.
- `-t (optional)`: Requires a value which indicates the number of threads to be used in CPU. `Default value is 16`.
- `-s (optional)`: When used, sparse algorithm will be chosen. If not specified, a dense algorithm will be chosen by default.
- `-b (optional)`: When used, matrix will be treated as binary matrix where all values are 1. If not specified, original values of the matrix will be used by default.
- `-g (optional)`: When used, permanent calculations will be run on GPU. If not specified, and `-c` is specified, calculations will be run on CPU.
- `-c (optional)`: When used, permanent calculations will be run on CPU, if and only if `-g` is not specified. If `-c` is not specified, calculations will be run on GPU by default.
- `-a (optional)`: When used, an approximation algorithm will be chosen. If not specified, an exact algorithm will be chosen by default.
- `-x (optional)`: Requires a value which indicates number of trials for an approximation algorithm. `Default value is 100000`.
- `-y (optional)`: Requires a value which indicates scale intervals for an scaling approximation algorithm. `Default value is 4`.
- `-z (optional)`: Requires a value which indicates number of time to scale for an scaling approximation algorithm. `Default value is 5`.

Algorithms
=========
##### gpu_perman64_xlocal
`Algorithm ID:1`
Ryser algorithm is implemented in GPU where every thread has its own X vector on registers. Here is an example usage for this algorithm:
```
./perman -m int/30_0.50_0 -g -p1
```
##### gpu_perman64_xshared
`Algorithm ID:2`
Ryser algorithm is implemented in GPU where every thread has its own X vector on shared memory for each block. Here is an example usage for this algorithm:
```
./perman -m int/30_0.50_0 -g -p2
```
##### gpu_perman64_xshared_coalescing
`Algorithm ID:3`
Ryser algorithm is implemented in GPU where every thread has its own X vector on shared memory for each block. Access pattern to this shared memory is coalesced among the threads in a block. Here is an example usage for this algorithm:
```
./perman -m int/30_0.50_0 -g -p3
```
##### gpu_perman64_xshared_coalescing_mshared
`Algorithm ID:4`
Ryser algorithm is implemented in GPU where every thread has its own X vector on shared memory for each block. Access pattern to this shared memory is coalesced among the threads in a block. Matrix itself is also created in shared memory for a faster access. Here is an example usage for this algorithm:
```
./perman -m int/30_0.50_0 -g -p4
```
##### cpu_parallel_perman64
`Algorithm ID:1`
Ryser algorithm is implemented in CPU where every thread has its own X vector. Wrokload is distributed among threads on CPU. Here is an example usage for this algorithm:
```
./perman -m int/30_0.50_0 -c -t16
```
##### approximation_rasmussen
`Algorithm ID:1`
Rasmussen algorithm is implemented both in CPU and GPU to run many times and obtain the average of trials as the approximation value. `This algoritm only works for binary matrices.` Here are the two example to run this algorithm on CPU and GPU respectively.
```
./perman -m int/40_0.50_0 -b -c -t16 -a -p1 -x400000
./perman -m int/40_0.50_0 -b -g -a -p1 -x2000000
```
##### approximation_scaling
`Algorithm ID:2`
Scaling approximation algorithm is implemented both in CPU and GPU to run many times and obtain the average of trials as the approximation value. In each trial, it iterates many times as matrix dimension. Where it scales matrix at every scale interval whose value is 4 by default. In each scaling, it takes as many rounds as scale times whose value is 5 by default. `This algoritm only works for binary matrices.` Here are the two example to run this algorithm on CPU and GPU respectively.
```
./perman -m int/40_0.50_0 -b -c -t16 -a -p2 -x400000 -y4 -z10
./perman -m int/40_0.50_0 -b -g -a -p2 -x2000000 -y2 -z5
```



