all:
	nvcc -o perman main.cu -Xcompiler -fopenmp -O3
