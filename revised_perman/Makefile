main:
	nvcc -c gpu_exact_dense.cu
	echo "GPU Dense Exact object.. OK"
	g++ -c main.cpp -fopenmp -O3 -std=c++11
	echo "Main cpp object.. OK"
	nvcc -o r_perman main.o gpu_exact_dense.o -Xcompiler -fopenmp -O3 -std=c++11