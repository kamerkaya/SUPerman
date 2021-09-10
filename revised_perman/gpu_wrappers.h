// This function is exists in gpu_exact_dense.cu, but only called within file therefore no need to have a signature. 
//template <class T>
//extern double cpu_perman64(T *mat_t, double x[], int nov, long long start, long long end, int threads); //This is a CPU helper for hybrid setting

#ifndef GPU_WRAPPERS_H
#define GPU_WRAPPERS_H

#include "flags.h"
//#include "gpu_exact_dense.cu"
//#include <stdio.h>
//#include <cuda_runtime.h>

template <class T>
extern double gpu_perman64_xglobal(DenseMatrix<T>* densemat, flags flags);
//extern double gpu_perman64_xglobal(DenseMatrix<T>* densemat, flags flags);

template <class T>
extern double gpu_perman64_xlocal(T* mat, int nov, flags flags);

template <class T>
extern double gpu_perman64_xshared(T* mat, int nov, flags flags);

template <class T>
extern double gpu_perman64_xshared_coalescing(T* mat, int nov, flags flags);

template <class T>
extern double gpu_perman64_xshared_coalescing_mshared(T* mat, int nov, flags flags);

template <class T>
extern double gpu_perman64_xshared_coalescing_mshared_multigpu(T* mat, int nov, flags flags);

//template <class T>
//extern double gpu_perman64_xshared_coalescing_mshared_multigpu_chunks(T* mat, int nov, flags flags);
//I don't even know if this function exist
//Look to that from the docs

template <class T>
extern double gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks(T* mat, int nov, flags flags);

//Same goes for that one
//template <class T>
//extern double gpu_perman64_xshared_coalescing_mshared_manual_distribution(T* mat, int nov, flags flags);

template <class T>
extern double gpu_perman64_xshared_coalescing_mshared_multigpu_manual_distribution(T* mat, int nov, flags flags);

#endif
