#include <omp.h>
#include <stdio.h>
using namespace std;


template <class T>
double cpu_perman64(T* mat_t, double x[], int nov, long long start, long long end, int threads) {
  double p = 0; //product of the elements in vector 'x'
  long long one = 1;
  long long chunk_size = (end - start) / threads + 1;
  omp_set_num_threads(threads);

  #pragma omp parallel
  { 
    double my_x[nov];
    for (int i = 0; i < nov; i++) {
      my_x[i] = x[i];
    }
    int tid = omp_get_thread_num();
    long long my_start = start + tid * chunk_size;
    long long my_end = min(start + ((tid+1) * chunk_size), end);
    
    double *xptr; 
    int s;  //+1 or -1 
    double prod; //product of the elements in vector 'x'
    double my_p = 0;
    long long i = my_start;
    long long gray = (i-1) ^ ((i-1) >> 1);

    for (int k = 0; k < (nov-1); k++) {
      if ((gray >> k) & 1LL) { // whether kth column should be added to x vector or not
        xptr = (double*)my_x;
        for (int j = 0; j < nov; j++) {
          *xptr += mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
          xptr++;
        }
      }
    }
    int k;

    int prodSign = 1;
    if(i & 1LL) {
      prodSign = -1;
    }
    while (i < my_end) {
      //compute the gray code
      k = __builtin_ctzll(i);
      gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
      //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
      s = ((one << k) & gray) ? 1 : -1;
      
      prod = 1.0;
      xptr = (double*)my_x;
      for (int j = 0; j < nov; j++) {
        *xptr += s * mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
        prod *= *xptr++;  //product of the elements in vector 'x'
      }

      my_p += prodSign * prod; 
      prodSign *= -1;
      i++;
    }

    #pragma omp atomic
      p += my_p;
  }

  return p;
}

template <class T>
__global__ void kernel_xglobal(T* mat_t, float* x, double* p, int nov) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  
  long long number_of_threads = blockDim.x * gridDim.x;

  long long one = 1;
  long long start = 1;
  long long end = (1LL << (nov-1));
  
  long long chunk_size = end / number_of_threads + 1;

  long long my_start = start + tid * chunk_size;
  long long my_end = min(start + ((tid+1) * chunk_size), end);
     
  int s;  //+1 or -1 
  double prod; //product of the elements in vector 'x'
  double my_p = 0;
  long long i = my_start;
  long long gray = (i-1) ^ ((i-1) >> 1);

  for (int k = 0; k < (nov-1); k++) {
    if ((gray >> k) & 1LL) { // whether kth column should be added to x vector or not
      for (int j = 0; j < nov; j++) {
        x[tid*nov + j] += mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      }
    }
  }
    
  long long gray_diff;
  int k;

  int prodSign = 1;
  if(i & 1LL) {
    prodSign = -1;
  }

  while (i < my_end) {
    gray_diff = (i ^ (i >> 1)) ^ gray;
    k = __ffsll(gray_diff) - 1;
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
      
    prod = 1.0;
    for (int j = 0; j < nov; j++) {
      x[tid*nov + j] += s * mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      prod *= x[tid*nov + j];  //product of the elements in vector 'x'
    }

    my_p += prodSign * prod; 
    prodSign *= -1;
    i++;
  }

  p[tid] = my_p;
}

template <class T>
__global__ void kernel_xlocal(T* mat_t, double* x, double* p, int nov) {
  float my_x[40];
  for (int k = 0; k < nov; k++) {
    my_x[k] = x[k];
  }
  
  long long number_of_threads = blockDim.x * gridDim.x;

  long long one = 1;
  long long start = 1;
  long long end = (1LL << (nov-1));
  
  long long chunk_size = end / number_of_threads + 1;

  
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  long long my_start = start + tid * chunk_size;
  long long my_end = min(start + ((tid+1) * chunk_size), end);
    
  float *xptr; 
  int s;  //+1 or -1 
  double prod; //product of the elements in vector 'x'
  double my_p = 0;
  long long i = my_start;
  long long gray = (i-1) ^ ((i-1) >> 1);

  for (int k = 0; k < (nov-1); k++) {
    if ((gray >> k) & 1LL) { // whether kth column should be added to x vector or not
      xptr = (float*)my_x;
      for (int j = 0; j < nov; j++) {
        *xptr += mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
        xptr++;
      }
    }
  }
    
  long long gray_diff;
  int k;

  int prodSign = 1;
  if(i & 1LL) {
    prodSign = -1;
  }

  while (i < my_end) {
    gray_diff = (i ^ (i >> 1)) ^ gray;
    k = __ffsll(gray_diff) - 1;
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
      
    prod = 1.0;
    xptr = (float*)my_x;
    for (int j = 0; j < nov; j++) {
      *xptr += s * mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      prod *= *xptr++;  //product of the elements in vector 'x'
    }

    my_p += prodSign * prod; 
    prodSign *= -1;
    i++;
  }

  p[tid] = my_p;
}

template <class T>
__global__ void kernel_xshared(T* mat_t, double* x, double* p, int nov) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;

  extern __shared__ float shared_mem[]; 
  float *my_x = shared_mem; // size = nov * BLOCK_SIZE

  for (int k = 0; k < nov; k++) {
    my_x[thread_id*nov + k] = x[k];
  }
  
  long long number_of_threads = blockDim.x * gridDim.x;

  long long one = 1;
  long long start = 1;
  long long end = (1LL << (nov-1));
  
  long long chunk_size = end / number_of_threads + 1;

  long long my_start = start + tid * chunk_size;
  long long my_end = min(start + ((tid+1) * chunk_size), end);
     
  int s;  //+1 or -1 
  double prod; //product of the elements in vector 'x'
  double my_p = 0;
  long long i = my_start;
  long long gray = (i-1) ^ ((i-1) >> 1);

  for (int k = 0; k < (nov-1); k++) {
    if ((gray >> k) & 1LL) { // whether kth column should be added to x vector or not
      for (int j = 0; j < nov; j++) {
        my_x[thread_id*nov + j] += mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      }
    }
  }
    
  long long gray_diff;
  int k;

  int prodSign = 1;
  if(i & 1LL) {
    prodSign = -1;
  }

  while (i < my_end) {
    gray_diff = (i ^ (i >> 1)) ^ gray;
    k = __ffsll(gray_diff) - 1;
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
      
    prod = 1.0;
    for (int j = 0; j < nov; j++) {
      my_x[thread_id*nov + j] += s * mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      prod *= my_x[thread_id*nov + j];  //product of the elements in vector 'x'
    }

    my_p += prodSign * prod; 
    prodSign *= -1;
    i++;
  }

  p[tid] = my_p;
}

template <class T>
__global__ void kernel_xshared_coalescing(T* mat_t, double* x, double* p, int nov) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ float shared_mem[]; 
  float *my_x = shared_mem; // size = nov * BLOCK_SIZE

  for (int k = 0; k < nov; k++) {
    my_x[block_dim*k + thread_id] = x[k];
  }

  long long number_of_threads = blockDim.x * gridDim.x;

  long long one = 1;
  long long start = 1;
  long long end = (1LL << (nov-1));
  
  long long chunk_size = end / number_of_threads + 1;

  long long my_start = start + tid * chunk_size;
  long long my_end = min(start + ((tid+1) * chunk_size), end);
  
  int s;  //+1 or -1 
  double prod; //product of the elements in vector 'x'
  double my_p = 0;
  long long i = my_start;
  long long gray = (i-1) ^ ((i-1) >> 1);

  for (int k = 0; k < (nov-1); k++) {
    if ((gray >> k) & 1LL) { // whether kth column should be added to x vector or not
      for (int j = 0; j < nov; j++) {
        my_x[block_dim*j + thread_id] += mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      }
    }
  }
  
  long long gray_diff;
  int k;

  int prodSign = 1;
  if(i & 1LL) {
    prodSign = -1;
  }

  while (i < my_end) {
    gray_diff = (i ^ (i >> 1)) ^ gray;
    k = __ffsll(gray_diff) - 1;
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
      
    prod = 1.0;
    for (int j = 0; j < nov; j++) {
      my_x[block_dim*j + thread_id] += s * mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      prod *= my_x[block_dim*j + thread_id];  //product of the elements in vector 'x'
    }

    my_p += prodSign * prod; 
    prodSign *= -1;
    i++;
  }

  p[tid] = my_p;
}

template <class T>
__global__ void kernel_xshared_coalescing_mshared(T* mat_t, double* x, double* p, int nov, long long start, long long end) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ float shared_mem[]; 
  float *my_x = shared_mem; // size = nov * BLOCK_SIZE
  T *shared_mat_t = (T*) &my_x[nov * block_dim]; // size = nov * nov

  for (int k = 0; k < nov; k++) {
    my_x[block_dim*k + thread_id] = x[k];
  }

  for (int k = 0; k < ((nov*nov)/block_dim + 1); k++) {
    if ((block_dim * k + thread_id) < (nov * nov))
    shared_mat_t[block_dim * k + thread_id] = mat_t[block_dim * k + thread_id];
  }

  __syncthreads();

  long long number_of_threads = blockDim.x * gridDim.x;

  long long one = 1;
  
  long long chunk_size = (end - start) / number_of_threads + 1;

  long long my_start = start + tid * chunk_size;
  long long my_end = min(start + ((tid+1) * chunk_size), end);
  
  int s;  //+1 or -1 
  double prod; //product of the elements in vector 'x'
  double my_p = 0;
  long long i = my_start;
  long long gray = (i-1) ^ ((i-1) >> 1);

  for (int k = 0; k < (nov-1); k++) {
    if ((gray >> k) & 1LL) { // whether kth column should be added to x vector or not
      for (int j = 0; j < nov; j++) {
        my_x[block_dim*j + thread_id] += shared_mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      }
    }
  }
    
  long long gray_diff;
  int k;

  int prodSign = 1;
  if (i & 1LL) {
    prodSign = -1;
  }
  while (i < my_end) {
    gray_diff = (i ^ (i >> 1)) ^ gray;
    k = __ffsll(gray_diff) - 1;
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
      
    prod = 1.0;
    for (int j = 0; j < nov; j++) {
      my_x[block_dim*j + thread_id] += s * shared_mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      prod *= my_x[block_dim*j + thread_id];  //product of the elements in vector 'x'
    }

    my_p += prodSign * prod; 
    prodSign *= -1;
    i++;
  }

  p[tid] = my_p;
}

template <class T>
double gpu_perman64_xglobal(T* mat, int nov, int grid_dim, int block_dim) {
  float x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      rs += mat[(j * nov) + k];  // sum of row j
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  //create the transpose of the matrix
  T* mat_t = new T[nov * nov];
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(i * nov) + j] = mat[(j * nov) + i];
    }
  }

  float *h_x = new float[nov*grid_dim*block_dim];
  for (int i = 0; i < nov*grid_dim*block_dim; i++) {
    h_x[i] = x[i%nov];
  }

  cudaSetDevice(1);
  T *d_mat_t;
  float *d_x;
  double *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov*grid_dim*block_dim) * sizeof(float));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

  cudaMemcpy( d_x, h_x, (nov*grid_dim*block_dim) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xglobal<<< grid_dim , block_dim >>> (d_mat_t, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_mat_t);
  cudaFree(d_x);
  cudaFree(d_p);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete [] mat_t;
  delete [] h_x;
  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xlocal(T* mat, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      rs += mat[(j * nov) + k];  // sum of row j
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  //create the transpose of the matrix
  T* mat_t = new T[nov * nov];
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(i * nov) + j] = mat[(j * nov) + i];
    }
  }

  cudaSetDevice(1);
  T *d_mat_t;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xlocal<<< grid_dim , block_dim >>> (d_mat_t, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_mat_t);
  cudaFree(d_x);
  cudaFree(d_p);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete [] mat_t;
  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared(T* mat, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      rs += mat[(j * nov) + k];  // sum of row j
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  //create the transpose of the matrix
  T* mat_t = new T[nov * nov];
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(i * nov) + j] = mat[(j * nov) + i];
    }
  }

  cudaSetDevice(1);
  T *d_mat_t;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared<<< grid_dim , block_dim , nov*block_dim*sizeof(float) >>> (d_mat_t, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_mat_t);
  cudaFree(d_x);
  cudaFree(d_p);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete [] mat_t;
  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared_coalescing(T* mat, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      rs += mat[(j * nov) + k];  // sum of row j
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  //create the transpose of the matrix
  T* mat_t = new T[nov * nov];
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(i * nov) + j] = mat[(j * nov) + i];
    }
  }

  cudaSetDevice(1);
  T *d_mat_t;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared_coalescing<<< grid_dim , block_dim , nov*block_dim*sizeof(float) >>> (d_mat_t, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_mat_t);
  cudaFree(d_x);
  cudaFree(d_p);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete [] mat_t;
  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared_coalescing_mshared(T* mat, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      rs += mat[(j * nov) + k];  // sum of row j
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  //create the transpose of the matrix
  T* mat_t = new T[nov * nov];
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(i * nov) + j] = mat[(j * nov) + i];
    }
  }

  cudaSetDevice(1);
  T *d_mat_t;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

  long long start = 1;
  long long end = (1LL << (nov-1));

  double stt = omp_get_wtime();
  kernel_xshared_coalescing_mshared<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + nov*nov*sizeof(T)) >>> (d_mat_t, d_x, d_p, nov, start, end);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_mat_t);
  cudaFree(d_x);
  cudaFree(d_p);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete [] mat_t;
  delete[] h_p;

  return((4*(nov&1)-2) * p);
} 

template <class T>
double gpu_perman64_xshared_coalescing_mshared_multigpu(T* mat, int nov, int gpu_num, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  double p_partial[gpu_num];
  for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
    p_partial[gpu_id] = 0;
  }
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      rs += mat[(j * nov) + k];  // sum of row j
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  //create the transpose of the matrix
  T* mat_t = new T[nov * nov];
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(i * nov) + j] = mat[(j * nov) + i];
    }
  }

  long long start = 1;
  long long end = (1LL << (nov-1));
  long long offset = (end - start) / gpu_num;

  #pragma omp parallel for num_threads(gpu_num)
    for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
      cudaSetDevice(gpu_id);
      T *d_mat_t;
      double *d_x, *d_p;
      double *h_p = new double[grid_dim * block_dim];

      cudaMalloc( &d_x, (nov) * sizeof(double));
      cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
      cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

      cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

      double stt = omp_get_wtime();
      if (gpu_id == gpu_num-1) {
        kernel_xshared_coalescing_mshared<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + nov*nov*sizeof(T)) >>> (d_mat_t, d_x, d_p, nov, (start + gpu_id*offset), end);
      } else {
        kernel_xshared_coalescing_mshared<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + nov*nov*sizeof(T)) >>> (d_mat_t, d_x, d_p, nov, (start + gpu_id*offset), (start + (gpu_id+1)*offset));
      }
      cudaDeviceSynchronize();
      double enn = omp_get_wtime();
      cout << "kernel" << gpu_id << " in " << (enn - stt) << endl;
        
      cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

      cudaFree(d_mat_t);
      cudaFree(d_x);
      cudaFree(d_p);
      for (int i = 0; i < grid_dim * block_dim; i++) {
        p_partial[gpu_id] += h_p[i];
      }
      delete[] h_p;
    }

  delete [] mat_t;
  for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
    p += p_partial[gpu_id];
  }

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks(T* mat, int nov, int gpu_num, bool cpu, int threads, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  double p_partial[gpu_num+1];
  for (int id = 0; id < gpu_num+1; id++) {
    p_partial[id] = 0;
  }

  int number_of_chunks = 1;
  int init = 29;
  if (cpu) {
    init = 28;
  }
  for (int i = init; i < nov; i++) {
    number_of_chunks *= 2;
  }
  int chunk_id = 0;
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      rs += mat[(j * nov) + k];  // sum of row j
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  //create the transpose of the matrix
  T* mat_t = new T[nov * nov];
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(i * nov) + j] = mat[(j * nov) + i];
    }
  }

  long long start = 1;
  long long end = (1LL << (nov-1));
  long long offset = (end - start) / number_of_chunks;

  omp_set_nested(1);
  omp_set_dynamic(0);
  #pragma omp parallel for num_threads(gpu_num+1)
    for (int id = 0; id < gpu_num+1; id++) {
      if (id == gpu_num) {
        if (cpu) {
          int curr_chunk_id;
          #pragma omp critical 
          {
            curr_chunk_id = chunk_id;
            chunk_id++;
          }
          while (curr_chunk_id < number_of_chunks) {
            double stt = omp_get_wtime();
            if (curr_chunk_id == number_of_chunks - 1) {
              p_partial[id] += cpu_perman64(mat_t, x, nov, (start + curr_chunk_id*offset), end, threads);
            } else {
              p_partial[id] += cpu_perman64(mat_t, x, nov, (start + curr_chunk_id*offset), (start + (curr_chunk_id+1)*offset), threads);
            }
            double enn = omp_get_wtime();
            cout << "ChunkID " << curr_chunk_id << "is DONE by CPU" << " in " << (enn - stt) << endl;
            #pragma omp critical 
            {
              curr_chunk_id = chunk_id;
              chunk_id++;
            }
          }
        }
      } else {
        cudaSetDevice(id);
        
        T *d_mat_t;
        double *d_x, *d_p;
        double *h_p = new double[grid_dim * block_dim];

        cudaMalloc( &d_x, (nov) * sizeof(double));
        cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
        cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

        cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

        int curr_chunk_id;
            
        #pragma omp critical 
        {
          curr_chunk_id = chunk_id;
          chunk_id++;
        }
        while (curr_chunk_id < number_of_chunks) {
          double stt = omp_get_wtime();
          if (curr_chunk_id == number_of_chunks - 1) {
            kernel_xshared_coalescing_mshared<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + nov*nov*sizeof(T)) >>> (d_mat_t, d_x, d_p, nov, (start + curr_chunk_id*offset), end);
          } else {
            kernel_xshared_coalescing_mshared<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + nov*nov*sizeof(T)) >>> (d_mat_t, d_x, d_p, nov, (start + curr_chunk_id*offset), (start + (curr_chunk_id+1)*offset));
          }
          cudaDeviceSynchronize();
          double enn = omp_get_wtime();
          cout << "ChunkID " << curr_chunk_id << "is DONE by kernel" << id << " in " << (enn - stt) << endl;
                
          cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);
              
          for (int i = 0; i < grid_dim * block_dim; i++) {
            p_partial[id] += h_p[i];
          }
              
          #pragma omp critical 
          {
            curr_chunk_id = chunk_id;
            chunk_id++;
          }
        }

        cudaFree(d_mat_t);
        cudaFree(d_x);
        cudaFree(d_p);
        delete[] h_p;
      }
    }

  delete [] mat_t;
  for (int id = 0; id < gpu_num+1; id++) {
    p += p_partial[id];
  }

  return((4*(nov&1)-2) * p);
}








template <class T>
double gpu_perman64_xshared_coalescing_mshared_multigpu_manual_distribution(T* mat, int nov, int gpu_num, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  double p_partial[gpu_num];
  for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
    p_partial[gpu_id] = 0;
  }
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      rs += mat[(j * nov) + k];  // sum of row j
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  //create the transpose of the matrix
  T* mat_t = new T[nov * nov];
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(i * nov) + j] = mat[(j * nov) + i];
    }
  }

  long long start = 1;
  long long end = (1LL << (nov-1));
  long long offset = (end - start) / 8;

  #pragma omp parallel for num_threads(gpu_num)
    for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
      cudaSetDevice(gpu_id);
      T *d_mat_t;
      double *d_x, *d_p;
      double *h_p = new double[grid_dim * block_dim];

      cudaMalloc( &d_x, (nov) * sizeof(double));
      cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
      cudaMalloc( &d_mat_t, (nov * nov) * sizeof(T));

      cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy( d_mat_t, mat_t, (nov * nov) * sizeof(T), cudaMemcpyHostToDevice);

      double stt = omp_get_wtime();
      if (gpu_id == 0) {
        kernel_xshared_coalescing_mshared<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + nov*nov*sizeof(T)) >>> (d_mat_t, d_x, d_p, nov, start, start + 3*offset);
      } else if (gpu_id == 1) {
        kernel_xshared_coalescing_mshared<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + nov*nov*sizeof(T)) >>> (d_mat_t, d_x, d_p, nov, start + 3*offset, start + 6*offset);
      } else if (gpu_id == 2) {
        kernel_xshared_coalescing_mshared<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + nov*nov*sizeof(T)) >>> (d_mat_t, d_x, d_p, nov, start + 6*offset, start + 7*offset);
      } else if (gpu_id == 3) {
        kernel_xshared_coalescing_mshared<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + nov*nov*sizeof(T)) >>> (d_mat_t, d_x, d_p, nov, start + 7*offset, end);
      }
      cudaDeviceSynchronize();
      double enn = omp_get_wtime();
      cout << "kernel" << gpu_id << " in " << (enn - stt) << endl;
        
      cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

      cudaFree(d_mat_t);
      cudaFree(d_x);
      cudaFree(d_p);
      for (int i = 0; i < grid_dim * block_dim; i++) {
        p_partial[gpu_id] += h_p[i];
      }
      delete[] h_p;
    }

  delete [] mat_t;
  for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
    p += p_partial[gpu_id];
  }

  return((4*(nov&1)-2) * p);
}