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
- `-f, --file (required)`: Requires a value which indicates the path for the input matrix.
- `-p, --perman (optional)`: Requires a value which indicates the algorithm id of the algorithm for permanent calculation. `Default value is 1`.
- `-t, --threads (optional)`: Requires a value which indicates the number of threads to be used in CPU. `Default value is 16`.
- `-s, --sparse (optional)`: When used, sparse algorithm will be chosen. If not specified, a dense algorithm will be chosen by default.
- `-b, --binary (optional)`: When used, matrix will be treated as binary matrix where all values are 1. If not specified, original values of the matrix will be used by default.
- `-g, --gpu (optional)`: When used, permanent calculations will be run on GPU. If not specified, and `-c` is specified, calculations will be run on CPU.
- `-d, --device (optional)`: Requires a value which indicates the number of devices to be used in a multigpu algorithm. `Default value is 2`.
- `-c, --cpu (optional)`: When used, permanent calculations will be run on CPU, if and only if `-g` is not specified. If `-c` is not specified, calculations will be run on GPU by default.
- `-a, --approximation (optional)`: When used, an approximation algorithm will be chosen. If not specified, an exact algorithm will be chosen by default.
- `-x, --numOfTimes (optional)`: Requires a value which indicates number of trials for an approximation algorithm. `Default value is 100000`.
- `-y, --scaleIntervals (optional)`: Requires a value which indicates scale intervals for an scaling approximation algorithm. `Default value is 4`.
- `-z, --scaleTimes (optional)`: Requires a value which indicates number of time to scale for an scaling approximation algorithm. `Default value is 5`.
- `-r, --preprocessing (optional)`: Requires a value which indicates the preprocessing to be applied. If `1` is specified, `SortOrder` is applied. If `2` is specified, `SkipOrder` is applied. If not specified, there will be no preprocessing.
- `-i, --grid (optional)`: When used, a grid graph will be created using `--gridm` and `gridn` dimensions, and a sparse approximation algorithm will be chosen by `--perman`.
- `-m, --gridm (optional)`: Requires a value which indicates the first dimension of the grid graph. `Default value is 36`.
- `-n, --gridn (optional)`: Requires a value which indicates the second dimension of the grid graph. `Default value is 36`.

Algorithms
=========
##### gpu_perman64_xglobal
`Algorithm ID:0`
Ryser algorithm is implemented in GPU where every thread accesses its own X vector using global memory accesses. Here is an example usage for this algorithm:
```
./perman -f int/30_0.50_0 -g -p0
```
##### gpu_perman64_xlocal
`Algorithm ID:1`
Ryser algorithm is implemented in GPU where every thread has its own X vector on registers. Sparyser version is available using `-s(--sparse)` tag. Here is an example usage for this algorithm:
```
./perman -f int/30_0.50_0 -g -p1
./perman -f int/30_0.20_0 -g -p1 -s -r1
```
##### gpu_perman64_xshared
`Algorithm ID:2`
Ryser algorithm is implemented in GPU where every thread has its own X vector on shared memory for each block. Sparyser version is available using `-s(--sparse)` tag. Here is an example usage for this algorithm:
```
./perman -f int/30_0.50_0 -g -p2
./perman -f int/30_0.20_0 -g -p2 -s -r1
```
##### gpu_perman64_xshared_coalescing
`Algorithm ID:3`
Ryser algorithm is implemented in GPU where every thread has its own X vector on shared memory for each block. Access pattern to this shared memory is coalesced among the threads in a block. Sparyser version is available using `-s(--sparse)` tag. Here is an example usage for this algorithm:
```
./perman -f int/30_0.50_0 -g -p3
./perman -f int/30_0.20_0 -g -p3 -s -r1
```
##### gpu_perman64_xshared_coalescing_mshared
`Algorithm ID:4`
Ryser algorithm is implemented in GPU where every thread has its own X vector on shared memory for each block. Access pattern to this shared memory is coalesced among the threads in a block. Matrix itself is also created in shared memory for a faster access. Sparyser version is available using `-s(--sparse)` tag. Here is an example usage for this algorithm:
```
./perman -f int/30_0.50_0 -g -p4
./perman -f int/30_0.20_0 -g -p4 -s -r1
```
##### gpu_perman64_xshared_coalescing_mshared_multigpu
`Algorithm ID:5`
Static multigpu implementation of `gpu_perman64_xshared_coalescing_mshared`. Sparyser version is available using `-s(--sparse)` tag. Here is an example usage for this algorithm:
```
./perman -f int/32_0.50_0 -g -p5 -d2
./perman -f int/32_0.20_0 -g -p5 -d2 -s -r1
```
##### gpu_perman64_xshared_coalescing_mshared_multigpu_chunks
`Algorithm ID:6`
Dynamic hybrid implementation of `gpu_perman64_xshared_coalescing_mshared`. Sparyser version is available using `-s(--sparse)` tag. Here is an example usage for this algorithm:
```
./perman -f int/32_0.50_0 -g -p6 -d2
./perman -f int/32_0.50_0 -g -p6 -d4 -c -t8
./perman -f int/32_0.20_0 -g -p6 -d2 -s -r1
```
##### gpu_perman64_xshared_coalescing_mshared_skipper
`Algorithm ID:7`
Skipper implementation of `gpu_perman64_xshared_coalescing_mshared`. `-s(--sparse)` tag should be used since this is a sparse implementation. Here is an example usage for this algorithm:
```
./perman -f int/30_0.20_0 -g -p7 -s -r2
```
##### gpu_perman64_xshared_coalescing_mshared_multigpu_chunks_skipper
`Algorithm ID:8`
Dynamic hybrid implementation of `gpu_perman64_xshared_coalescing_mshared_skipper`. `-s(--sparse)` tag should be used since this is a sparse implementation. Here is an example usage for this algorithm:
```
./perman -f int/32_0.20_0 -g -p8 -d4 -s -r2
```
##### cpu_parallel_perman64
`Algorithm ID:1`
Ryser algorithm is implemented in CPU where every thread has its own X vector. Workload is distributed among threads on CPU. Sparyser version is available using `-s(--sparse)` tag. Here is an example usage for this algorithm:
```
./perman -f int/30_0.50_0 -c -t16
./perman -f int/30_0.20_0 -c -t16 -s -r1
```
##### cpu_parallel_skipper_balanced
`Algorithm ID:3`
Skipper algorithm is implemented in CPU with dynamic scheduling. Here is an example usage for this algorithm:
```
./a.out -f int/30_0.20_0 -c -t16 -p3 -s -r2
```
##### approximation_rasmussen
`Algorithm ID:1`
Rasmussen algorithm is implemented both in CPU and GPU to run many times and obtain the average of trials as the approximation value. `This algoritm only works for binary matrices.` For the sparse implementation of this algorithm, `-s(--sparse)` tag should be used. Here are the two example to run this algorithm on CPU and GPU respectively.
```
./perman -f int/40_0.50_0 -b -c -t16 -a -p1 -x400000
./perman -f int/40_0.50_0 -b -g -a -p1 -x2000000
```
##### approximation_scaling
`Algorithm ID:2`
Scaling approximation algorithm is implemented both in CPU and GPU to run many times and obtain the average of trials as the approximation value. In each trial, it iterates many times as matrix dimension. Where it scales matrix at every scale interval whose value is 4 by default. In each scaling, it takes as many rounds as scale times whose value is 5 by default. `This algoritm only works for binary matrices.` Here are the two example to run this algorithm on CPU and GPU respectively.
```
./perman -f int/40_0.50_0 -b -c -t16 -a -p2 -x400000 -y4 -z10
./perman -f int/40_0.50_0 -b -g -a -p2 -x2000000 -y2 -z5
```
##### approximation_rasmussen_multigpu
`Algorithm ID:3`
Multigpu implementation of `approximation_rasmussen`. `This algoritm only works for binary matrices.` For the sparse implementation of this algorithm, `-s(--sparse)` tag should be used. Here is an example usage for this algorithm:
```
./perman -f int/40_0.50_0 -b -g -a -p3 -x10000000
./perman -f int/40_0.20_0 -b -g -a -p3 -x10000000 -s
```
##### approximation_scaling_multigpu
`Algorithm ID:4`
Multigpu implementation of `approximation_scaling`. `This algoritm only works for binary matrices.` For the sparse implementation of this algorithm, `-s(--sparse)` tag should be used. Here is an example usage for this algorithm:
```
./perman -f int/40_0.50_0 -b -g -a -p4 -x10000000 -y5 -z5
./perman -f int/40_0.20_0 -b -g -a -p4 -x10000000 -y5 -z5 -s
```
