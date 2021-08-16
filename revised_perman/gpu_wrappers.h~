template <class T>
extern double cpu_perman64(T *mat_t, double x[], int nov, long long start, long long end, int threads); //This is a CPU helper for hybrid setting

template <class T>
extern double gpu_perman64_xglobal(T* mat, int nov, int grid_dim, int block_dim);

template <class T>
extern double gpu_perman64_xlocal(T* mat, int nov, int grid_dim, int block_dim);

template <class T>
extern double gpu_perman64_xshared(T* mat, int nov, int grid_dim, int block_dim);

template <class T>
extern double gpu_perman64_xshared_coalescing(T* mat, int nov, int grid_dim, int block_dim);

template <class T>
extern double gpu_perman64_xshared_coalescing_mshared(T* mat, int nov, int grid_dim, int block_dim);

template <class T>
extern double gpu_perman64_xshared_coalescing_mshared_multigpu(T* mat, int nov, int grid_dim, int block_dim);

template <class T>
extern double gpu_perman64_xshared_coalescing_mshared_multigpu_chunks(T* mat, int nov, int gpu_num, bool cpu, int threads, int grid_dim, int block_dim);

template <class T>
extern double gpu_perman64_xshared_coalescing_mshared_manual_distribution(T* mat, int nov, int grid_dim, int block_dim);








