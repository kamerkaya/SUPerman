#include <omp.h>
#include <stdio.h>
using namespace std;


template <class T>
double cpu_perman64_sparse(int* cptrs, int* rows, T* cvals, double x[], int nov, long long start, long long end, int threads) {
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
    
    int s;  //+1 or -1 
    double prod; //product of the elements in vector 'x'
    double my_p = 0;
    long long i = my_start;
    long long gray = (i-1) ^ ((i-1) >> 1);

    for (int k = 0; k < (nov-1); k++) {
      if ((gray >> k) & 1LL) { // whether kth column should be added to x vector or not
        for (int j = cptrs[k]; j < cptrs[k+1]; j++) {
          my_x[rows[j]] += cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        }
      }
    }

    prod = 1.0;
    int zero_num = 0;
    for (int j = 0; j < nov; j++) {
      if (my_x[j] == 0) {
        zero_num++;
      } else {
        prod *= my_x[j];  //product of the elements in vector 'x'
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
      
      for (int j = cptrs[k]; j < cptrs[k+1]; j++) {
        if (my_x[rows[j]] == 0) {
          zero_num--;
          my_x[rows[j]] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
          prod *= my_x[rows[j]];  //product of the elements in vector 'x'
        } else {
          prod /= my_x[rows[j]];
          my_x[rows[j]] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
          if (my_x[rows[j]] == 0) {
            zero_num++;
          } else {
            prod *= my_x[rows[j]];  //product of the elements in vector 'x'
          }
        }
      }

      if(zero_num == 0) {
        my_p += prodSign * prod; 
      }
      prodSign *= -1;
      i++;
    }

    #pragma omp atomic
      p += my_p;
  }

  return p;
}

template <class T>
double cpu_perman64_skipper(int *rptrs, int *cols, int* cptrs, int* rows, T* cvals, double x[], int nov, long long start, long long end, int threads) {
  //first initialize the vector then we will copy it to ourselves
  double p;
  int j, ptr;
  unsigned long long ci, chunk_size, change_j;

  p = 0;

  int no_chunks = 512;
  chunk_size = (end - start + 1) / no_chunks + 1;

  #pragma omp parallel num_threads(threads) private(j, ci, change_j) 
  {
    double my_x[nov];
    
    #pragma omp for schedule(dynamic, 1)
      for(int cid = 0; cid < no_chunks; cid++) {
      //    int tid = omp_get_thread_num();
        unsigned long long my_start = start + cid * chunk_size;
        unsigned long long my_end = min(start + ((cid+1) * chunk_size), end);
      
        //update if neccessary
        double my_p = 0;
        
        unsigned long long my_gray;    
        unsigned long long my_prev_gray = 0;
        memcpy(my_x, x, sizeof(double) * nov);

        int ptr, last_zero;
        unsigned long long period, steps, step_start;
        
        unsigned long long i = my_start;
        
        while (i < my_end) {
          //k = __builtin_ctzll(i + 1);
          my_gray = i ^ (i >> 1);
          
          unsigned long long gray_diff = my_prev_gray ^ my_gray;
          
          j = 0;
          while(gray_diff > 0) { // this contains the bit to be updated
            unsigned long long onej = 1ULL << j;
            if(gray_diff & onej) { // if bit l is changed 
              gray_diff ^= onej;   // unset bit
              if(my_gray & onej) {    // do the update
                for (ptr = cptrs[j]; ptr < cptrs[j + 1]; ptr++) {
                  my_x[rows[ptr]] += cvals[ptr];
                }
              }
              else {
                for (ptr = cptrs[j]; ptr < cptrs[j + 1]; ptr++) {
                  my_x[rows[ptr]] -= cvals[ptr];
                }
              }
            }
            j++;
          }
          //counter++;
          my_prev_gray = my_gray;
          last_zero = -1;
          double my_prod = 1; 
          for(j = nov - 1; j >= 0; j--) {
            my_prod *= my_x[j];
            if(my_x[j] == 0) {
              last_zero = j;
              break;
            }
          }
  
          if(my_prod != 0) {
            my_p += ((i&1ULL)? -1.0:1.0) * my_prod;
            i++;
          } 
          else {
            change_j = -1;
            for (ptr = rptrs[last_zero]; ptr < rptrs[last_zero + 1]; ptr++) {
              step_start = 1ULL << cols[ptr]; 
              period = step_start << 1; 
              ci = step_start;
              if(i >= step_start) {
                steps = (i - step_start) / period;
                ci = step_start + ((steps + 1) * period);
              }
              if(ci < change_j) {
                change_j = ci;
              }
            }
      
            i++;
            if(change_j > i) {
              i = change_j;
            } 
          }
        }
      
        #pragma omp critical
          p += my_p;
      }
  }
    
  return p;
}

template <class T>
__global__ void kernel_xlocal_sparse(int* cptrs, int* rows, T* cvals, double* x, double* p, int nov) {
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
  
  int s;  //+1 or -1 
  double prod; //product of the elements in vector 'x'
  double my_p = 0;
  long long i = my_start;
  long long gray = (i-1) ^ ((i-1) >> 1);

  for (int k = 0; k < (nov-1); k++) {
    if ((gray >> k) & 1LL) { // whether kth column should be added to x vector or not
      for (int j = cptrs[k]; j < cptrs[k+1]; j++) {
        my_x[rows[j]] += cvals[j]; // see Nijenhuis and Wilf - update x vector entries
      }
    }
  }

  prod = 1.0;
  int zero_num = 0;
  for (int j = 0; j < nov; j++) {
    if (my_x[j] == 0) {
      zero_num++;
    } else {
      prod *= my_x[j];  //product of the elements in vector 'x'
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
      
    for (int j = cptrs[k]; j < cptrs[k+1]; j++) {
      if (my_x[rows[j]] == 0) {
        zero_num--;
        my_x[rows[j]] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        prod *= my_x[rows[j]];  //product of the elements in vector 'x'
      } else {
        prod /= my_x[rows[j]];
        my_x[rows[j]] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        if (my_x[rows[j]] == 0) {
          zero_num++;
        } else {
          prod *= my_x[rows[j]];  //product of the elements in vector 'x'
        }
      }
    }

    if(zero_num == 0) {
      my_p += prodSign * prod; 
    }
    prodSign *= -1;
    i++;
  }

  p[tid] = my_p;
}

template <class T>
__global__ void kernel_xshared_sparse(int* cptrs, int* rows, T* cvals, double* x, double* p, int nov) {
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
      for (int j = cptrs[k]; j < cptrs[k+1]; j++) {
        my_x[thread_id*nov + rows[j]] += cvals[j]; // see Nijenhuis and Wilf - update x vector entries
      }
    }
  }

  prod = 1.0;
  int zero_num = 0;
  for (int j = 0; j < nov; j++) {
    if (my_x[thread_id*nov + j] == 0) {
      zero_num++;
    } else {
      prod *= my_x[thread_id*nov + j];  //product of the elements in vector 'x'
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

    for (int j = cptrs[k]; j < cptrs[k+1]; j++) {
      if (my_x[thread_id*nov + rows[j]] == 0) {
        zero_num--;
        my_x[thread_id*nov + rows[j]] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        prod *= my_x[thread_id*nov + rows[j]];  //product of the elements in vector 'x'
      } else {
        prod /= my_x[thread_id*nov + rows[j]];
        my_x[thread_id*nov + rows[j]] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        if (my_x[thread_id*nov + rows[j]] == 0) {
          zero_num++;
        } else {
          prod *= my_x[thread_id*nov + rows[j]];  //product of the elements in vector 'x'
        }
      }
    }

    if(zero_num == 0) {
      my_p += prodSign * prod; 
    }
    prodSign *= -1;
    i++;
  }

  p[tid] = my_p;
}

template <class T>
__global__ void kernel_xshared_coalescing_sparse(int* cptrs, int* rows, T* cvals, double* x, double* p, int nov) {
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
      for (int j = cptrs[k]; j < cptrs[k+1]; j++) {
        my_x[block_dim*rows[j] + thread_id] += cvals[j]; // see Nijenhuis and Wilf - update x vector entries
      }
    }
  }

  prod = 1.0;
  int zero_num = 0;
  for (int j = 0; j < nov; j++) {
    if (my_x[block_dim*j + thread_id] == 0) {
      zero_num++;
    } else {
      prod *= my_x[block_dim*j + thread_id];  //product of the elements in vector 'x'
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
      
    for (int j = cptrs[k]; j < cptrs[k+1]; j++) {
      if (my_x[block_dim*rows[j] + thread_id] == 0) {
        zero_num--;
        my_x[block_dim*rows[j] + thread_id] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        prod *= my_x[block_dim*rows[j] + thread_id];  //product of the elements in vector 'x'
      } else {
        prod /= my_x[block_dim*rows[j] + thread_id];
        my_x[block_dim*rows[j] + thread_id] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        if (my_x[block_dim*rows[j] + thread_id] == 0) {
          zero_num++;
        } else {
          prod *= my_x[block_dim*rows[j] + thread_id];  //product of the elements in vector 'x'
        }
      }
    }

    if(zero_num == 0) {
      my_p += prodSign * prod; 
    }
    prodSign *= -1;
    i++;
  }

  p[tid] = my_p;
}

template <class T>
__global__ void kernel_xshared_coalescing_mshared_sparse(int* cptrs, int* rows, T* cvals, double* x, double* p, int nov, int total, long long start, long long end) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ float shared_mem[]; 
  float *my_x = shared_mem; // size = nov * BLOCK_SIZE
  int *shared_cptrs = (int*) &my_x[nov * block_dim]; // size = nov + 1
  int *shared_rows = (int*) &shared_cptrs[nov + 1]; // size = total num of elts
  T *shared_cvals = (T*) &shared_rows[total]; // size = total num of elts

  for (int k = 0; k < nov; k++) {
    my_x[block_dim*k + thread_id] = x[k];
    shared_cptrs[k] = cptrs[k];
  }
  shared_cptrs[nov] = cptrs[nov];
  
  for (int k = 0; k < total; k++) {
    shared_rows[k] = rows[k];
    shared_cvals[k] = cvals[k];
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
      for (int j = shared_cptrs[k]; j < shared_cptrs[k+1]; j++) {
        my_x[block_dim*shared_rows[j] + thread_id] += shared_cvals[j]; // see Nijenhuis and Wilf - update x vector entries
      }
    }
  }

  prod = 1.0;
  int zero_num = 0;
  for (int j = 0; j < nov; j++) {
    if (my_x[block_dim*j + thread_id] == 0) {
      zero_num++;
    } else {
      prod *= my_x[block_dim*j + thread_id];  //product of the elements in vector 'x'
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
      
    for (int j = shared_cptrs[k]; j < shared_cptrs[k+1]; j++) {
      if (my_x[block_dim*shared_rows[j] + thread_id] == 0) {
        zero_num--;
        my_x[block_dim*shared_rows[j] + thread_id] += s * shared_cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        prod *= my_x[block_dim*shared_rows[j] + thread_id];  //product of the elements in vector 'x'
      } else {
        prod /= my_x[block_dim*shared_rows[j] + thread_id];
        my_x[block_dim*shared_rows[j] + thread_id] += s * shared_cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        if (my_x[block_dim*shared_rows[j] + thread_id] == 0) {
          zero_num++;
        } else {
          prod *= my_x[block_dim*shared_rows[j] + thread_id];  //product of the elements in vector 'x'
        }
      }
    }

    if(zero_num == 0) {
      my_p += prodSign * prod; 
    }
    prodSign *= -1;
    i++;
  }

  p[tid] = my_p;
}


template <class T>
__global__ void kernel_xshared_coalescing_mshared_skipper(int* rptrs, int* cols, int* cptrs, int* rows, T* cvals, double* x, double* p, int nov, int total, long long start, long long end) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ float shared_mem[]; 
  float *my_x = shared_mem; // size = nov * BLOCK_SIZE
  int *shared_rptrs = (int*) &my_x[nov * block_dim]; // size = nov + 1
  int *shared_cols = (int*) &shared_rptrs[nov + 1]; // size = total num of elts
  int *shared_cptrs = (int*) &shared_cols[total]; // size = nov + 1
  int *shared_rows = (int*) &shared_cptrs[nov + 1]; // size = total num of elts
  T *shared_cvals = (T*) &shared_rows[total]; // size = total num of elts

  for (int k = 0; k < nov; k++) {
    my_x[block_dim*k + thread_id] = x[k];
    shared_rptrs[k] = rptrs[k];
    shared_cptrs[k] = cptrs[k];
  }
  shared_rptrs[nov] = rptrs[nov];
  shared_cptrs[nov] = cptrs[nov];
  
  for (int k = 0; k < total; k++) {
    shared_cols[k] = cols[k];
    shared_rows[k] = rows[k];
    shared_cvals[k] = cvals[k];
  }

  __syncthreads();

  long long number_of_threads = blockDim.x * gridDim.x;

  long long chunk_size = (end - start) / number_of_threads + 1;

  long long my_start = start + tid * chunk_size;
  long long my_end = min(start + ((tid+1) * chunk_size), end);
  
  double prod; //product of the elements in vector 'x'
  double my_p = 0;
  long long i = my_start;
  long long prev_gray = 0;
  long long gray;

  int zero_num = 0;
  for (int j = 0; j < nov; j++) {
    if (my_x[block_dim*j + thread_id] == 0) {
      zero_num++;
    } else {
      prod *= my_x[block_dim*j + thread_id];  //product of the elements in vector 'x'
    }
  }
    
  long long gray_diff;
  unsigned long long change_j, ci, period, steps, step_start;
  int j = 0;
  while (i < my_end) {
    gray = i ^ (i >> 1);
    gray_diff = prev_gray ^ gray;

    j = 0;
    while(gray_diff > 0) { // this contains the bit to be updated
      long long onej = 1LL << j;
      if(gray_diff & onej) { // if bit l is changed 
        gray_diff ^= onej;   // unset bit
        if(gray & onej) {    // do the update
          for (int ptr = shared_cptrs[j]; ptr < shared_cptrs[j + 1]; ptr++) {
            my_x[block_dim*shared_rows[ptr] + thread_id] += shared_cvals[ptr];
          }
        }
        else {
          for (int ptr = shared_cptrs[j]; ptr < shared_cptrs[j + 1]; ptr++) {
            my_x[block_dim*shared_rows[ptr] + thread_id] -= shared_cvals[ptr];
          }
        }
      }
      j++;
    }
    
    prev_gray = gray;
    int last_zero = -1;
    prod = 1.0; 
    for(j = nov - 1; j >= 0; j--) {
      prod *= my_x[block_dim*j + thread_id];
      if(my_x[block_dim*j + thread_id] == 0) {
        last_zero = j;
        break;
      }
    }

    if(prod != 0) {
      my_p += ((i&1LL)? -1.0:1.0) * prod;
      i++;
    }
    else {
      change_j = -1;
      for (int ptr = shared_rptrs[last_zero]; ptr < shared_rptrs[last_zero + 1]; ptr++) {
        step_start = 1ULL << shared_cols[ptr]; 
        period = step_start << 1; 
        ci = step_start;
        if(i >= step_start) {
          steps = (i - step_start) / period;
          ci = step_start + ((steps + 1) * period);
        }
        if(ci < change_j) {
          change_j = ci;
        }
      }
      i++;
      if(change_j > i) {
        i = change_j;
      } 
    }
  }

  p[tid] = my_p;
}


template <class T>
double gpu_perman64_xlocal_sparse(T* mat, int* cptrs, int* rows, T* cvals, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  int total = 0;
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      if (mat[(j * nov) + k] != 0) {
        total++;
        rs += mat[(j * nov) + k];  // sum of row j
      }
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  cudaSetDevice(1);
  T *d_cvals;
  int *d_cptrs, *d_rows;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xlocal_sparse<<< grid_dim , block_dim >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared_sparse(T* mat, int* cptrs, int* rows, T* cvals, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  int total = 0;
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      if (mat[(j * nov) + k] != 0) {
        total++;
        rs += mat[(j * nov) + k];  // sum of row j
      }
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  cudaSetDevice(1);
  T *d_cvals;
  int *d_cptrs, *d_rows;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared_sparse<<< grid_dim , block_dim , nov*block_dim*sizeof(float) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared_coalescing_sparse(T* mat, int* cptrs, int* rows, T* cvals, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  int total = 0;
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      if (mat[(j * nov) + k] != 0) {
        total++;
        rs += mat[(j * nov) + k];  // sum of row j
      }
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  cudaSetDevice(1);
  T *d_cvals;
  int *d_cptrs, *d_rows;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

  double stt = omp_get_wtime();
  kernel_xshared_coalescing_sparse<<< grid_dim , block_dim , nov*block_dim*sizeof(float) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared_coalescing_mshared_sparse(T* mat, int* cptrs, int* rows, T* cvals, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  int total = 0;
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      if (mat[(j * nov) + k] != 0) {
        total++;
        rs += mat[(j * nov) + k];  // sum of row j
      }
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  cudaSetDevice(1);
  T *d_cvals;
  int *d_cptrs, *d_rows;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

  long long start = 1;
  long long end = (1LL << (nov-1));

  double stt = omp_get_wtime();
  kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, start, end);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared_coalescing_mshared_multigpu_sparse(T* mat, int* cptrs, int* rows, T* cvals, int nov, int gpu_num, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  double p_partial[gpu_num];
  for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
    p_partial[gpu_id] = 0;
  }
  int total = 0;
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      if (mat[(j * nov) + k] != 0) {
        total++;
        rs += mat[(j * nov) + k];  // sum of row j
      }
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  long long start = 1;
  long long end = (1LL << (nov-1));
  long long offset = (end - start) / gpu_num;

  #pragma omp parallel for num_threads(gpu_num)
    for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
      cudaSetDevice(gpu_id);
      T *d_cvals;
      int *d_cptrs, *d_rows;
      double *d_x, *d_p;
      double *h_p = new double[grid_dim * block_dim];

      cudaMalloc( &d_x, (nov) * sizeof(double));
      cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
      cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
      cudaMalloc( &d_rows, (total) * sizeof(int));
      cudaMalloc( &d_cvals, (total) * sizeof(T));

      cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

      double stt = omp_get_wtime();
      if (gpu_id == gpu_num-1) {
        kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, (start + gpu_id*offset), end);
      } else {
        kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, (start + gpu_id*offset), (start + (gpu_id+1)*offset));
      }
      cudaDeviceSynchronize();
      double enn = omp_get_wtime();
      cout << "kernel" << gpu_id << " in " << (enn - stt) << endl;
        
      cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

      cudaFree(d_x);
      cudaFree(d_p);
      cudaFree(d_cptrs);
      cudaFree(d_rows);
      cudaFree(d_cvals);

      for (int i = 0; i < grid_dim * block_dim; i++) {
        p_partial[gpu_id] += h_p[i];
      }

      delete[] h_p;
    }

  for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
    p += p_partial[gpu_id];
  }

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse(T* mat, int* cptrs, int* rows, T* cvals, int nov, int gpu_num, bool cpu, int threads, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  double p_partial[gpu_num+1];
  for (int id = 0; id < gpu_num+1; id++) {
    p_partial[id] = 0;
  }

  int number_of_chunks = 1;
  for (int i = 30; i < nov; i++) {
    number_of_chunks *= 2;
  }
  int chunk_id = 0;
  
  int total = 0;
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      if (mat[(j * nov) + k] != 0) {
        total++;
        rs += mat[(j * nov) + k];  // sum of row j
      }
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
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
              p_partial[id] += cpu_perman64_sparse(cptrs, rows, cvals, x, nov, (start + curr_chunk_id*offset), end, threads);
            } else {
              p_partial[id] += cpu_perman64_sparse(cptrs, rows, cvals, x, nov, (start + curr_chunk_id*offset), (start + (curr_chunk_id+1)*offset), threads);
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

        T *d_cvals;
        int *d_cptrs, *d_rows;
        double *d_x, *d_p;
        double *h_p = new double[grid_dim * block_dim];

        cudaMalloc( &d_x, (nov) * sizeof(double));
        cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
        cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
        cudaMalloc( &d_rows, (total) * sizeof(int));
        cudaMalloc( &d_cvals, (total) * sizeof(T));

        cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

        int curr_chunk_id;
            
        #pragma omp critical 
        {
          curr_chunk_id = chunk_id;
          chunk_id++;
        }
        while (curr_chunk_id < number_of_chunks) {
          double stt = omp_get_wtime();
          if (curr_chunk_id == number_of_chunks - 1) {
            kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, (start + curr_chunk_id*offset), end);
          } else {
            kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, (start + curr_chunk_id*offset), (start + (curr_chunk_id+1)*offset));
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

        cudaFree(d_x);
        cudaFree(d_p);
        cudaFree(d_cptrs);
        cudaFree(d_rows);
        cudaFree(d_cvals);
        delete[] h_p;
      }
    }

  for (int id = 0; id < gpu_num+1; id++) {
    p += p_partial[id];
  }

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared_coalescing_mshared_skipper(T* mat, int* rptrs, int* cols, int* cptrs, int* rows, T* cvals, int nov, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  int total = 0;
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      if (mat[(j * nov) + k] != 0) {
        total++;
        rs += mat[(j * nov) + k];  // sum of row j
      }
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  cudaSetDevice(1);
  T *d_cvals;
  int *d_rptrs, *d_cols, *d_cptrs, *d_rows;
  double *d_x, *d_p;
  double *h_p = new double[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(double));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
  cudaMalloc( &d_rptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_cols, (total) * sizeof(int));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(T));

  cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rptrs, rptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cols, cols, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

  long long start = 1;
  long long end = (1LL << (nov-1));

  double stt = omp_get_wtime();
  kernel_xshared_coalescing_mshared_skipper<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + 2*(nov+1)*sizeof(int) + 2*total*sizeof(int) + total*sizeof(T)) >>> (d_rptrs, d_cols, d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, start, end);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  cout << "kernel" << " in " << (enn - stt) << endl;
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_rptrs);
  cudaFree(d_cols);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  for (int i = 0; i < grid_dim * block_dim; i++) {
    p += h_p[i];
  }

  delete[] h_p;

  return((4*(nov&1)-2) * p);
}

template <class T>
double gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_skipper(T* mat, int* rptrs, int* cols, int* cptrs, int* rows, T* cvals, int nov, int gpu_num, bool cpu, int threads, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  double p_partial[gpu_num+1];
  for (int id = 0; id < gpu_num+1; id++) {
    p_partial[id] = 0;
  }

  int number_of_chunks = 1;
  for (int i = 30; i < nov; i++) {
    number_of_chunks *= 2;
  }
  int chunk_id = 0;
  
  int total = 0;
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      if (mat[(j * nov) + k] != 0) {
        total++;
        rs += mat[(j * nov) + k];  // sum of row j
      }
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
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
              p_partial[id] += cpu_perman64_skipper(rptrs, cols, cptrs, rows, cvals, x, nov, (start + curr_chunk_id*offset), end, threads);
            } else {
              p_partial[id] += cpu_perman64_skipper(rptrs, cols, cptrs, rows, cvals, x, nov, (start + curr_chunk_id*offset), (start + (curr_chunk_id+1)*offset), threads);
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

        T *d_cvals;
        int *d_rptrs, *d_cols, *d_cptrs, *d_rows;
        double *d_x, *d_p;
        double *h_p = new double[grid_dim * block_dim];

        cudaMalloc( &d_x, (nov) * sizeof(double));
        cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
        cudaMalloc( &d_rptrs, (nov + 1) * sizeof(int));
        cudaMalloc( &d_cols, (total) * sizeof(int));
        cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
        cudaMalloc( &d_rows, (total) * sizeof(int));
        cudaMalloc( &d_cvals, (total) * sizeof(T));

        cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy( d_rptrs, rptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy( d_cols, cols, (total) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

        int curr_chunk_id;
            
        #pragma omp critical 
        {
          curr_chunk_id = chunk_id;
          chunk_id++;
        }
        while (curr_chunk_id < number_of_chunks) {
          double stt = omp_get_wtime();
          if (curr_chunk_id == number_of_chunks - 1) {
            kernel_xshared_coalescing_mshared_skipper<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + 2*(nov+1)*sizeof(int) + 2*total*sizeof(int) + total*sizeof(T)) >>> (d_rptrs, d_cols, d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, (start + curr_chunk_id*offset), end);
          } else {
            kernel_xshared_coalescing_mshared_skipper<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + 2*(nov+1)*sizeof(int) + 2*total*sizeof(int) + total*sizeof(T)) >>> (d_rptrs, d_cols, d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, (start + curr_chunk_id*offset), (start + (curr_chunk_id+1)*offset));
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

        cudaFree(d_x);
        cudaFree(d_p);
        cudaFree(d_rptrs);
        cudaFree(d_cols);
        cudaFree(d_cptrs);
        cudaFree(d_rows);
        cudaFree(d_cvals);
        delete[] h_p;
      }
    }

  for (int id = 0; id < gpu_num+1; id++) {
    p += p_partial[id];
  }

  return((4*(nov&1)-2) * p);
}


template <class T>
double gpu_perman64_xshared_coalescing_mshared_multigpu_sparse_manual_distribution(T* mat, int* cptrs, int* rows, T* cvals, int nov, int gpu_num, int grid_dim, int block_dim) {
  double x[nov]; 
  double rs; //row sum
  double p = 1; //product of the elements in vector 'x'
  double p_partial[gpu_num];
  for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
    p_partial[gpu_id] = 0;
  }
  int total = 0;
  
  //create the x vector and initiate the permanent
  for (int j = 0; j < nov; j++) {
    rs = .0f;
    for (int k = 0; k < nov; k++) {
      if (mat[(j * nov) + k] != 0) {
        total++;
        rs += mat[(j * nov) + k];  // sum of row j
      }
    }
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  long long start = 1;
  long long end = (1LL << (nov-1));
  long long offset = (end - start) / 8;

  #pragma omp parallel for num_threads(gpu_num)
    for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
      cudaSetDevice(gpu_id);
      T *d_cvals;
      int *d_cptrs, *d_rows;
      double *d_x, *d_p;
      double *h_p = new double[grid_dim * block_dim];

      cudaMalloc( &d_x, (nov) * sizeof(double));
      cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(double));
      cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
      cudaMalloc( &d_rows, (total) * sizeof(int));
      cudaMalloc( &d_cvals, (total) * sizeof(T));

      cudaMemcpy( d_x, x, (nov) * sizeof(double), cudaMemcpyHostToDevice);
      cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

      double stt = omp_get_wtime();
      if (gpu_id == 0) {
        kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, start, start + 3*offset);
      } else if (gpu_id == 1) {
        kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, start + 3*offset, start + 6*offset);
      } else if (gpu_id == 2) {
        kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, start + 6*offset, start + 7*offset);
      } else if (gpu_id == 3) {
        kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, start + 7*offset, end);
      }
      cudaDeviceSynchronize();
      double enn = omp_get_wtime();
      cout << "kernel" << gpu_id << " in " << (enn - stt) << endl;
        
      cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);

      cudaFree(d_x);
      cudaFree(d_p);
      cudaFree(d_cptrs);
      cudaFree(d_rows);
      cudaFree(d_cvals);

      for (int i = 0; i < grid_dim * block_dim; i++) {
        p_partial[gpu_id] += h_p[i];
      }

      delete[] h_p;
    }

  for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
    p += p_partial[gpu_id];
  }

  return((4*(nov&1)-2) * p);
}