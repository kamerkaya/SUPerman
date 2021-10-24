main:
	nvcc -c gpu_exact_dense.cu
	echo "GPU Dense Exact object.. OK"
	nvcc -c gpu_exact_sparse.cu
	echo "Gpu Sparse Exact object.. OK"
	nvcc -c gpu_approximation_dense.cu
	echo "Gpu Dense Approximation object.. OK"
	nvcc -c gpu_approximation_sparse.cu
	echo "Gpu Sparse Approximation object.. OK"
	g++ -c main.cpp mmio.c -fopenmp -O3 -std=c++11 -lcudart
	echo "Main cpp object.. OK"
	nvcc -o gpu_perman main.o gpu_exact_dense.o gpu_exact_sparse.o gpu_approximation_dense.o gpu_approximation_sparse.o -Xcompiler -fopenmp -O3

debug:
	nvcc -c gpu_exact_dense.cu -g -G
	echo "DEBUG GPU Dense Exact object.. OK"
	nvcc -c gpu_exact_sparse.cu -g -G
	echo "DEBUG Gpu Sparse Exact object.. OK"
	nvcc -c gpu_approximation_dense.cu -g -G
	echo "DEBUG Gpu Dense Approximation object.. OK"
	nvcc -c gpu_approximation_sparse.cu -g -G
	echo "DEBUG Gpu Sparse Approximation object.. OK"
	g++ -c main.cpp mmio.c -fopenmp -O3 -std=c++11 -lcudart -g
	echo "DEBUG Main cpp object.. OK"
	nvcc -o debug_perman main.o gpu_exact_dense.o gpu_exact_sparse.o gpu_approximation_dense.o gpu_approximation_sparse.o -Xcompiler -fopenmp -O3 -g -G

cpu:
	g++ -o cpu_perman main.cpp mmio.c -fopenmp -O3 -std=c++11 -DONLYCPU

clean:
	rm gpu_perman
	rm debug_perman
