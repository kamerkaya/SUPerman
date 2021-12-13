// This function is exists in gpu_exact_dense.cu, but only called within file therefore no need to have a signature. 
//template <class T>
//extern double cpu_perman64(T *mat_t, double x[], int nov, long long start, long long end, int threads); //This is a CPU helper for hybrid setting

#ifndef GPU_WRAPPERS_H
#define GPU_WRAPPERS_H

#include "flags.h"

//##############~~#####//FUNCTIONS FROM: gpu_exact_dense.cu//#####~~##############//
template <class C, class S>
extern Result gpu_perman64_xglobal(DenseMatrix<S>* densemat, flags flags);

template <class C, class S>
extern Result gpu_perman64_xlocal(DenseMatrix<S>* densemat, flags flags);

template <class C, class S>
extern Result gpu_perman64_xshared(DenseMatrix<S>* densemat, flags flags);

template <class C, class S>
extern Result gpu_perman64_xshared_coalescing(DenseMatrix<S>* densemat, flags flags);

template <class C, class S>
extern Result gpu_perman64_xshared_coalescing_mshared(DenseMatrix<S>* densemat, flags flags);

template <class T>
extern double gpu_perman64_xshared_coalescing_mshared_multigpu(DenseMatrix<T>* densemat, flags flags);

template <class T>
extern double gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks(DenseMatrix<T>* densemat, flags flags);

template <class T>
extern double gpu_perman64_xshared_coalescing_mshared_multigpu_manual_distribution(DenseMatrix<T>* densemat, flags flags);
//##############~~#####//FUNCTIONS FROM: gpu_exact_dense.cu//#####~~##############//



//##############~~#####//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//#####~~##############//
//##############~~#####//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//#####~~##############//


//##############~~#####//FUNCTIONS FROM: gpu_exact_sparse.cu//#####~~##############//

template <class C, class S>
extern Result gpu_perman64_xlocal_sparse(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags flags);

template <class C, class S>
extern Result gpu_perman64_xshared_sparse(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags flags);

template <class C, class S>
extern Result gpu_perman64_xshared_coalescing_sparse(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags flags);

template <class C, class S>
extern Result gpu_perman64_xshared_coalescing_mshared_sparse(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags flags);

template <class C, class S>
extern Result gpu_perman64_xshared_coalescing_mshared_multigpu_sparse(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags flags);

template <class T>
extern double gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse(DenseMatrix<T>* densemat, SparseMatrix<T>* sparsemat, flags flags);

template <class C, class S>
extern Result gpu_perman64_xshared_coalescing_mshared_skipper(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags flags);

template <class T>
extern double gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_skipper(DenseMatrix<T>* densemat, SparseMatrix<T>* sparsemat, flags flags);

template <class T>
extern double gpu_perman64_xshared_coalescing_mshared_multigpu_sparse_manual_distribution(DenseMatrix<T>* densemat, SparseMatrix<T>* sparsemat, flags flags);
//##############~~#####//FUNCTIONS FROM: gpu_exact_sparse.cu//#####~~##############//

//##############~~#####//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//#####~~##############//
//##############~~#####//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//#####~~##############//

//##############~~#####//FUNCTIONS FROM: gpu_approximation_dense.cu//#####~~##############//

template <class C, class S>
extern Result gpu_perman64_rasmussen(DenseMatrix<S>* densemat, flags flags);

template <class T>
extern double gpu_perman64_rasmussen_multigpucpu_chunks(DenseMatrix<T>* densemat, flags flags);

template <class C, class S>
extern Result gpu_perman64_approximation(DenseMatrix<S>* densemat, flags flags);

template <class T>
extern double gpu_perman64_approximation_multigpucpu_chunks(DenseMatrix<T>* densemat, flags flags);

//##############~~#####//FUNCTIONS FROM: gpu_approximation_dense.cu//#####~~##############//

//##############~~#####//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//#####~~##############//
//##############~~#####//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//#####~~##############//

//##############~~#####//FUNCTIONS FROM: gpu_approximation_sparse.cu//#####~~##############//

template <class C, class S>
extern Result gpu_perman64_rasmussen_sparse(SparseMatrix<S>* sparsemat, flags flags);

template <class T>
extern double gpu_perman64_rasmussen_multigpucpu_chunks_sparse(SparseMatrix<T>* sparsemat, flags flags);

template <class C, class S>
extern Result gpu_perman64_approximation_sparse(SparseMatrix<S>* sparsemat, flags flags);

template <class T>
extern double gpu_perman64_approximation_multigpucpu_chunks_sparse(SparseMatrix<T>* sparsemat, flags flags);

//##############~~#####//FUNCTIONS FROM: gpu_approximation_sparse.cu//#####~~##############//




#endif
