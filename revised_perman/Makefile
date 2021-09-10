main:
	nvcc -c gpu_exact_dense.cu
	echo "GPU Dense Exact object.. OK"
	nvcc -c gpu_exact_sparse.cu
	echo "Gpu Sparse Exact object.. OK"
	g++ -c main.cpp mmio.c -fopenmp -O3 -std=c++11 -lcudart -g 
	echo "Main cpp object.. OK"
	nvcc -o r_perman main.o gpu_exact_dense.o gpu_exact_sparse.o -Xcompiler -fopenmp -O3 -std=c++11 -g

clean:
	rm r_perman
