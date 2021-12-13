main:
	nvcc -c gpu_exact_dense.cu -Xcompiler -fopenmp
	echo "GPU Dense Exact object.. OK"
	nvcc -c gpu_exact_sparse.cu -Xcompiler -fopenmp
	echo "Gpu Sparse Exact object.. OK"
	nvcc -c gpu_approximation_dense.cu -Xcompiler -fopenmp
	echo "Gpu Dense Approximation object.. OK"
	nvcc -c gpu_approximation_sparse.cu -Xcompiler -fopenmp
	echo "Gpu Sparse Approximation object.. OK"
	g++ -c main.cpp mmio.c -fopenmp -O3 -std=c++11 -lcudart
	echo "Main cpp object.. OK"
	nvcc -o gpu_perman main.o gpu_exact_dense.o gpu_exact_sparse.o gpu_approximation_dense.o gpu_approximation_sparse.o -Xcompiler -fopenmp -O3
	rm *.o

debug:
	nvcc -c gpu_exact_dense.cu -g -G
	echo "DEBUG GPU Dense Exact object.. OK"
	nvcc -c gpu_exact_sparse.cu -g -G 
	echo "DEBUG Gpu Sparse Exact object.. OK"
	nvcc -c gpu_approximation_dense.cu -g -G
	echo "DEBUG Gpu Dense Approximation object.. OK"
	nvcc -c gpu_approximation_sparse.cu -g -G
	echo "DEBUG Gpu Sparse Approximation object.. OK"
	g++ -c main.cpp mmio.c -fopenmp -O3 -std=c++11 -lcudart -g -DDEBUG
	echo "DEBUG Main cpp object.. OK"
	nvcc -o debug_perman main.o gpu_exact_dense.o gpu_exact_sparse.o gpu_approximation_dense.o gpu_approximation_sparse.o -Xcompiler -fopenmp -O3 -g -G 
	rm *.o

profile:
	nvcc -c gpu_exact_dense.cu -arch=sm_70 -lineinfo
	echo "GPU Dense Exact object.. OK" 
	nvcc -c gpu_exact_sparse.cu -arch=sm_70 -lineinfo
	echo "Gpu Sparse Exact object.. OK"
	nvcc -c gpu_approximation_dense.cu -arch=sm_70 -lineinfo
	echo "Gpu Dense Approximation object.. OK"
	nvcc -c gpu_approximation_sparse.cu -arch=sm_70 -lineinfo 
	echo "Gpu Sparse Approximation object.. OK"
	g++ -c main.cpp mmio.c -fopenmp -O3 -std=c++11 -lcudart 
	echo "Main cpp object.. OK"
	nvcc -o profile_perman main.o gpu_exact_dense.o gpu_exact_sparse.o gpu_approximation_dense.o gpu_approximation_sparse.o -Xcompiler -fopenmp -O3 -arch=sm_70 
	rm *.o

cpu:
	g++ -o cpu_perman main.cpp mmio.c -fopenmp -O3 -std=c++11 -DONLYCPU

clean:
	rm gpu_perman
	rm debug_perman

#mainfpc:
#	nvcc-fpchecker -c gpu_exact_dense.cu -arch=sm_70
#	echo "GPU Dense Exact object.. OK"
#	nvcc-fpchecker -c gpu_exact_sparse.cu -arch=sm_70
#	echo "Gpu Sparse Exact object.. OK"
#	nvcc-fpchecker -c gpu_approximation_dense.cu -arch=sm_70
#	echo "Gpu Dense Approximation object.. OK"
#	nvcc-fpchecker -c gpu_approximation_sparse.cu -arch=sm_70
#	echo "Gpu Sparse Approximation object.. OK"
#	clang++-12fpchecker -c main.cpp mmio.c -fopenmp -O3 -std=c++11 -lcudart
#	echo "Main cpp object.. OK"
#	nvcc-fpchecker -o gpu_perman main.o gpu_exact_dense.o gpu_exact_sparse.o gpu_approximation_dense.o gpu_approximation_sparse.o -Xcompiler -fopenmp -O3 -arch=sm_70
#	rm *.o

#fpcpu:
#	clang++-12fpchecker -o fpc_perman main.cpp mmio.c -fopenmp -O3 -std=c++11 -DONLYCPU=\"ONLYCPU\"
