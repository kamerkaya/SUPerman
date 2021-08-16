main:
	nvcc -c gpu_exact_dense.cu
	echo "GPU Dense Exact object.. OK"
	nvcc -o r_perman gpu_exact_dense.o main.cpp -Xcompiler -fopenmp -O3 -std=c++11