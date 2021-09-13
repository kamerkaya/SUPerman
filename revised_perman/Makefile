main:
	nvcc -c gpu_exact_dense.cu
	echo "GPU Dense Exact object.. OK"
	nvcc -c gpu_exact_sparse.cu
	echo "Gpu Sparse Exact object.. OK"
	nvcc -c gpu_approximation_dense.cu
	echo "Gpu Dense Approximation object.. OK"
	nvcc -c gpu_approximation_sparse.cu
	echo "Gpu Sparse Approximation object.. OK"
	g++ -c main.cpp mmio.c -fopenmp -O3 -std=c++11 -lcudart -g 
	echo "Main cpp object.. OK"
	nvcc -o r_perman main.o gpu_exact_dense.o gpu_exact_sparse.o gpu_approximation_dense.o gpu_approximation_sparse.o -Xcompiler -fopenmp -O3 -std=c++11 -g

clean:
	rm r_perman
