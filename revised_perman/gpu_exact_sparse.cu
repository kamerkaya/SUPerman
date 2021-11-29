#include <omp.h>

#include <stdio.h>
#include "flags.h"
#include "gpu_wrappers.h"

int glob_nov;
int glob_total;
int glob_sizeof_c; //Size of type used for calculation
int glob_sizeof_s; //Size of type used for storage

template <class T>
double cpu_perman64_sparse(int* cptrs,
			   int* rows,
			   T* cvals,
			   T x[],
			   int nov,
			   long long start,
			   long long end,
			   int threads) {
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
double cpu_perman64_skipper(int *rptrs,
			    int *cols,
			    int* cptrs,
			    int* rows,
			    T* cvals,
			    double x[],
			    int nov,
			    long long start,
			    long long end,
			    int threads) {
  //first initialize the vector then we will copy it to ourselves
  double p;
  int j, ptr;
  unsigned long long ci, chunk_size;
  double change_j;
  
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

//Unary functions for cudaOccupancyMaxPotentialBlockSizeVariableSmem

int xshared_sparse_sharedmem(int b){
  return glob_nov*b*glob_sizeof_c;
}

int xshared_coalescing_sparse_sharedmem(int b){ //Actually the same but no need to confusion
  return glob_nov*b*glob_sizeof_c;
}

int xshared_coalescing_mshared_sparse_sharedmem(int b){
  return glob_nov*b*glob_sizeof_c + (glob_nov+1)*sizeof(int) + glob_total*sizeof(int)  + glob_total*glob_sizeof_s;
  //////////////for my_x////////////////////for d_cptrs//////////for d_rows///////////////////for d_cvals////////////
  //Note that d_x is not resides at the shared memory, in contrary, we copy it to d_p at the very beginning
}

int xshared_coalescing_mshared_skipper_sharedmem(int b){
  return glob_nov*b*glob_sizeof_c + 2*(glob_nov+1)*sizeof(int) + 2*glob_total*sizeof(int) + glob_total*glob_sizeof_s;
}

template <class C, class S>
__global__ void kernel_xlocal_sparse(int* cptrs,
				     int* rows,
				     S* cvals,
				     C* x,
				     C* p,
				     int nov) {
  C my_x[40]; //That should not be named as local, should be names as *register*
  //And also, there should be controversy about it's size
  //What if number of registers vary with GPU -> register spilling = BAD
  for (int k = 0; k < nov; k++) {
    my_x[k] = x[k];
  }
  
  long long number_of_threads = blockDim.x * gridDim.x;

  long long one = 1;
  long long start = 1;
  long long end = (1LL << (nov-1));
  
  long long chunk_size = (end - start) / number_of_threads + 1;

  
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  long long my_start = start + tid * chunk_size;
  long long my_end = min(start + ((tid+1) * chunk_size), end);
  
  C s;  //+1 or -1 
  C prod; //product of the elements in vector 'x'
  C my_p = 0;
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

template <class C, class S>
__global__ void kernel_xshared_sparse(int* cptrs,
				      int* rows,
				      S* cvals,
				      C* x,
				      C* p,
				      int nov) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  
  extern __shared__ double shared_mem[]; 
  C *my_x = (C*)shared_mem; // size = nov * BLOCK_SIZE
  
  for (int k = 0; k < nov; k++) {
    my_x[thread_id*nov + k] = x[k];
  }
  
  long long number_of_threads = blockDim.x * gridDim.x;

  long long one = 1;
  long long start = 1;
  long long end = (1LL << (nov-1));
  
  long long chunk_size = (end - start) / number_of_threads + 1;

  long long my_start = start + tid * chunk_size;
  long long my_end = min(start + ((tid+1) * chunk_size), end);
     
  C s;  //+1 or -1 
  C prod; //product of the elements in vector 'x'
  C my_p = 0;
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

  C prodSign = 1;  //Optimization point
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

template <class C, class S>
__global__ void kernel_xshared_coalescing_sparse(int* cptrs,
						 int* rows,
						 S* cvals,
						 C* x,
						 C* p,
						 int nov) {
  
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ double shared_mem[]; 
  C *my_x = (C*)shared_mem; // size = nov * BLOCK_SIZE

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
  
  C s;  //+1 or -1 
  C prod; //product of the elements in vector 'x'
  C my_p = 0;
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

template <class C, class S>
  __global__ void kernel_xshared_coalescing_mshared_sparse(int* cptrs,
							   int* rows,
							   S* cvals,
							   C* x,
							   C* p,
							   int nov,
							   int total,
							   long long start,
							   long long end) {
  
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);

  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;
  //int block_dim = 2;

  //(nov*block_dim*sizeof(float) + (nov+1)*sizeof(int) + 2*total*sizeof(T)) 
  extern __shared__ double shared_mem[]; 
  C *my_x = (C*)shared_mem; // size = nov * BLOCK_SIZE
  int *shared_cptrs = (int*) &my_x[nov * block_dim]; // size = nov + 1
  int *shared_rows = (int*) &shared_cptrs[nov + 1];  // size = total num of elts
  S *shared_cvals;
  
  if((nov * block_dim + nov + 1 + total % 2) == 0 && sizeof(S) == 4) {
    shared_cvals = (S*) &shared_rows[total]; // size = total num of elts -- Misaligned address
  } else if(sizeof(S) == 4){
    shared_cvals = (S*) (&shared_rows[total+1]); // size = total num of elts -- Misaligned address
  }
  
  if((nov * block_dim + nov + 1 + total % 8) == 0 && sizeof(S) == 8) {
    shared_cvals = (S*) &shared_rows[total]; // size = total num of elts -- Misaligned address
  } else if(sizeof(S) == 8) {
    unsigned int offset = 8 - ((nov * block_dim + nov + 1 + total) % 8);
    shared_cvals = (S*) (&shared_rows[total+offset]); // size = total num of elts -- Misaligned address
  }
  
  /*
  if(tid == 0){
  printf("size of T: %d \n" , sizeof(T));
    
    printf("%2: %ul \n", nov * block_dim + nov + 1 + total % 2);
    printf("%4: %ul \n", nov * block_dim + nov + 1 + total % 4);
    printf("%8: %ul \n", nov * block_dim + nov + 1 + total % 8);
    
    printf("%2: %ul \n", (nov * block_dim + nov + 1 + total) % 2);
    printf("%4: %ul \n", (nov * block_dim + nov + 1 + total) % 4);
    printf("%8: %ul \n", (nov * block_dim + nov + 1 + total) % 8);
  }
  */
  
  __syncthreads();

  for (int k = 0; k < nov; k++) {
    my_x[block_dim*k + thread_id] = x[k];
    shared_cptrs[k] = cptrs[k];
  }
  shared_cptrs[nov] = cptrs[nov];
  
  for (int k = 0; k < total; k++) {
    shared_rows[k] = rows[k];
    //printf("Adress of misaligned: %p \n", &shared_cvals[k]);
    shared_cvals[k] = cvals[k];
  }

  __syncthreads();

  long long number_of_threads = blockDim.x * gridDim.x;

  long long one = 1;
  
  long long chunk_size = (end - start) / number_of_threads + 1;

  long long my_start = start + tid * chunk_size;
  long long my_end = min(start + ((tid+1) * chunk_size), end);
  
  C s;  //+1 or -1 
  C prod; //product of the elements in vector 'x'
  C my_p = 0;
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


template <class C, class S>
__global__ void kernel_xshared_coalescing_mshared_skipper(int* rptrs,
							  int* cols,
							  int* cptrs,
							  int* rows,
							  S* cvals,
							  C* x,
							  C* p,
							  int nov,
							  int total,
							  long long start,
							  long long end) {
  int tid = threadIdx.x + (blockIdx.x * blockDim.x);
  int thread_id = threadIdx.x;
  int block_dim = blockDim.x;

  extern __shared__ double shared_mem[]; 
  C *my_x = (C*) shared_mem; // size = nov * BLOCK_SIZE
  int *shared_rptrs = (int*) &my_x[nov * block_dim]; // size = nov + 1
  int *shared_cols = (int*) &shared_rptrs[nov + 1]; // size = total num of elts
  int *shared_cptrs = (int*) &shared_cols[total]; // size = nov + 1
  int *shared_rows = (int*) &shared_cptrs[nov + 1]; // size = total num of elts
  //printf("Working total: %d \n" , total);
  S *shared_cvals = (S*) &shared_rows[total]; // size = total num of elts

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
  
  C prod; //product of the elements in vector 'x'
  C my_p = 0;
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
  unsigned long long ci, period, steps, step_start;
  double change_j;
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


template <class C, class S>
  extern Result gpu_perman64_xlocal_sparse(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags flags){

  //Pack parameters
  S* mat = densemat->mat;
  int* cptrs = sparsemat->cptrs;
  int* rows = sparsemat->rows;
  S* cvals = sparsemat->cvals;
  int nov = sparsemat->nov;
  //Pack parameters

  //Pack flags
  int grid_dim = flags.grid_dim;
  int block_dim = flags.block_dim;
  int device_id = flags.device_id;
  int grid_dim_multip = flags.grid_multip;
  //Pack flags
  
  cudaSetDevice(device_id);
  cudaDeviceSynchronize();
  double starttime = omp_get_wtime();
  
  C x[nov]; 
  C rs; //row sum
  C p = 1; //product of the elements in vector 'x'
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

  cudaOccupancyMaxPotentialBlockSize(&grid_dim,
				     &block_dim,
				     &kernel_xlocal_sparse<C,S>,
				     0,
				     0);

  printf("==SC== No Shared memory is used for the kernel..\n");
  printf("==SC== Grid dim is set to : %d \n", grid_dim);
  printf("==SC== Block dim is set to : %d \n", block_dim);
  
  if(grid_dim_multip != 1){
    grid_dim*=grid_dim_multip;
    printf("==SC== Grid dim is re-set to : %d \n", grid_dim);
  }
    
  S *d_cvals;
  int *d_cptrs, *d_rows;
  C *d_x, *d_p;
  C *h_p = new C[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(C));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(C));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(S));

  cudaMemcpy( d_x, x, (nov) * sizeof(C), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(S), cudaMemcpyHostToDevice);

  //double stt = omp_get_wtime();
  kernel_xlocal_sparse<C,S><<<grid_dim , block_dim>>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov);
  cudaDeviceSynchronize();
  //double enn = omp_get_wtime();
  //cout << "kernel" << " in " << (enn - stt) << endl;
  //printf("kernel in %f \n", enn - stt);
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(C), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  double return_p = p;

  for (int i = 0; i < grid_dim * block_dim; i++) {
    return_p += (double)h_p[i];
  }

  delete[] h_p;

  double perman = (4*(nov&1)-2) * return_p;
  double duration = omp_get_wtime() - starttime;
  
  
  //return((4*(nov&1)-2) * return_p)
  Result result(perman, duration);
  return result;
}

template <class C, class S>
extern Result gpu_perman64_xshared_sparse(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags flags) {

  
  //Pack parameters
  S* mat = densemat->mat;
  int* cptrs = sparsemat->cptrs;
  int* rows = sparsemat->rows;
  S* cvals = sparsemat->cvals;
  int nov = sparsemat->nov;
  //Pack parameters

  //Pack flags
  int grid_dim = flags.grid_dim;
  int block_dim = flags.block_dim;
  int device_id = flags.device_id;
  int grid_dim_multip = flags.grid_multip;
  //Pack flags

  cudaSetDevice(device_id);
  cudaDeviceSynchronize();
  double starttime = omp_get_wtime();
  
  C x[nov]; 
  C rs; //row sum
  C p = 1; //product of the elements in vector 'x'
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

  //For variable smem
  glob_nov = nov;
  glob_total = total;
  glob_sizeof_c = sizeof(C);
  glob_sizeof_s = sizeof(S);
  //For variable smem
  
  cudaOccupancyMaxPotentialBlockSizeVariableSMem(&grid_dim,
						 &block_dim,
						 &kernel_xshared_sparse<C,S>,
						 xshared_sparse_sharedmem,
						 0);

  //grid_dim = 160;
  //block_dim = 160;
  
  size_t size = nov*block_dim*sizeof(C);
  
  printf("==SC== Shared memory per block is set to : %zu \n", size);
  printf("==SC== Grid dim is set to : %d \n", grid_dim);
  printf("==SC== Block dim is set to : %d \n", block_dim);
  
  if(grid_dim_multip != 1){
    grid_dim*=grid_dim_multip;
    printf("==SC== Grid dim is re-set to : %d \n", grid_dim);
  }
  
  S *d_cvals;
  int *d_cptrs, *d_rows;
  C *d_x, *d_p;
  C *h_p = new C[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(C));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(C));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(S));

  cudaMemcpy( d_x, x, (nov) * sizeof(C), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(S), cudaMemcpyHostToDevice);
  
  
  //double stt = omp_get_wtime();
  kernel_xshared_sparse<C,S><<< grid_dim , block_dim , size>>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov);
  cudaDeviceSynchronize();
  //double enn = omp_get_wtime();
  //cout << "kernel" << " in " << (enn - stt) << endl;
  //printf("kernel in %f \n", enn - stt);
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(C), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  double return_p = p;

  for (int i = 0; i < grid_dim * block_dim; i++) {
    //printf("p: %e || i: %d hp[i]: %e |||", return_p, i, h_p[i]);
    return_p += (double)h_p[i];
    //printf("%e %e \n", (double)h_p[i], return_p);
    //printf("--->p: %e \n", return_p);
  }

  delete[] h_p;

  double perman = (4*(nov&1)-2) * return_p;
  double duration = omp_get_wtime() - starttime;
  Result result(perman, duration);
  return result;

  //return((4*(nov&1)-2) * return_p);
}

template <class C, class S>
  extern Result gpu_perman64_xshared_coalescing_sparse(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags flags) {

  //Pack parameters
  S* mat = densemat->mat;
  int* cptrs = sparsemat->cptrs;
  int* rows = sparsemat->rows;
  S* cvals = sparsemat->cvals;
  int nov = sparsemat->nov;
  //Pack parameters

  //Pack flags
  int grid_dim = flags.grid_dim;
  int block_dim = flags.block_dim;
  int device_id = flags.device_id;
  int grid_dim_multip = flags.grid_multip;
  //Pack flags

  cudaSetDevice(device_id);
  cudaDeviceSynchronize();

  double starttime = omp_get_wtime();
  
  C x[nov]; 
  C rs; //row sum
  C p = 1; //product of the elements in vector 'x'
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

  //For variable smem
  glob_nov = nov;
  glob_total = total;
  glob_sizeof_c = sizeof(C);
  glob_sizeof_s = sizeof(S);
  //For variable smem
  
  cudaOccupancyMaxPotentialBlockSizeVariableSMem(&grid_dim,
						 &block_dim,
						 &kernel_xshared_coalescing_sparse<C,S>,
						 xshared_coalescing_sparse_sharedmem,
						 0);
  
  size_t size = nov*block_dim*sizeof(C);
  
  printf("==SC== Shared memory per block is set to : %zu \n", size);
  printf("==SC== Grid dim is set to : %d \n", grid_dim);
  printf("==SC== Block dim is set to : %d \n", block_dim);
  
  if(grid_dim_multip != 1){
    grid_dim*=grid_dim_multip;
    printf("==SC== Grid dim is re-set to : %d \n", grid_dim);
  }

  cudaSetDevice(device_id);
  S *d_cvals;
  int *d_cptrs, *d_rows;
  C *d_x, *d_p;
  C *h_p = new C[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(C));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(C));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(S));

  cudaMemcpy( d_x, x, (nov) * sizeof(C), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(S), cudaMemcpyHostToDevice);

    
  //double stt = omp_get_wtime();
  kernel_xshared_coalescing_sparse<C,S><<<grid_dim , block_dim , size>>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov);
  cudaDeviceSynchronize();
  //double enn = omp_get_wtime();
  //cout << "kernel" << " in " << (enn - stt) << endl;
  //printf("kernel in %f \n", enn - stt);
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(C), cudaMemcpyDeviceToHost);
    
  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  double return_p = p;

  for (int i = 0; i < grid_dim * block_dim; i++) {
    return_p += (double)h_p[i];
  }

  delete[] h_p;

  double perman = (4*(nov&1)-2) * return_p;
  double duration = omp_get_wtime() - starttime;
  Result result(perman, duration);
  return result;
  
  //return((4*(nov&1)-2) * return_p);
}

template <class C, class S>
  extern Result gpu_perman64_xshared_coalescing_mshared_sparse(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags flags) {

  //Pack parameters
  S* mat = densemat->mat;
  int* cptrs = sparsemat->cptrs;
  int* rows = sparsemat->rows;
  S* cvals = sparsemat->cvals;
  int nov = sparsemat->nov;
  //Pack parameters

  //Pack flags
  int grid_dim = flags.grid_dim;
  int block_dim = flags.block_dim;
  int device_id = flags.device_id;
  int grid_dim_multip = flags.grid_multip;
  //Pack flags
  
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, device_id);

  size_t max_shared_per_block = prop.sharedMemPerBlock;

  cudaSetDevice(device_id);
  cudaDeviceSynchronize();
  
  double starttime = omp_get_wtime();
  
  C x[nov]; 
  C rs; //row sum
  C p = 1; //product of the elements in vector 'x'
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
  
  //For variable smem
  glob_nov = nov;
  glob_total = total;
  glob_sizeof_c = sizeof(C);
  glob_sizeof_s = sizeof(S);
  //For variable smem
  
  cudaOccupancyMaxPotentialBlockSizeVariableSMem(&grid_dim,
						 &block_dim,
						 &kernel_xshared_coalescing_mshared_sparse<C,S>,
						 xshared_coalescing_mshared_sparse_sharedmem,
						 (int)max_shared_per_block);

  size_t size = nov*block_dim*sizeof(C) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(S);

  printf("==SC== Maximum Shared memory per block : %zu \n", max_shared_per_block);
  printf("==SC== Shared memory per block is set to : %zu \n", size);
  printf("==SC== Grid dim is set to : %d \n", grid_dim);
  printf("==SC== Block dim is set to : %d \n", block_dim);

  if(grid_dim_multip != 1){
    grid_dim*=grid_dim_multip;
    printf("==SC== Grid dim is re-set to : %d \n", grid_dim);
  }
  
  S *d_cvals;
  int *d_cptrs, *d_rows;
  C *d_x, *d_p;
  C *h_p = new C[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(C));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(C));  
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(S));

  cudaMemcpy( d_x, x, (nov) * sizeof(C), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(S), cudaMemcpyHostToDevice);

  long long start = 1;
  long long end = (1LL << (nov-1));
  
  double stt = omp_get_wtime();
  kernel_xshared_coalescing_mshared_sparse<C,S><<<grid_dim , block_dim , size>>>(d_cptrs,
										 d_rows,
										 d_cvals,
										 d_x,
										 d_p,
										 nov,
										 total,
										 start,
										 end);
  cudaDeviceSynchronize();
  double enn = omp_get_wtime();
  //printf("kernel in %f \n", enn - stt);
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(C), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  double return_p = p;
  
  for (int i = 0; i < grid_dim * block_dim; i++) {
    return_p += (double)h_p[i];
  }

  delete[] h_p;

  double perman = (4*(nov&1)-2) * return_p;
  double duration = omp_get_wtime() - starttime;
  Result result(perman, duration);
  return result;
  
  //return((4*(nov&1)-2) * return_p);
}

template <class T>
extern double gpu_perman64_xshared_coalescing_mshared_multigpu_sparse(DenseMatrix<T>* densemat, SparseMatrix<T>* sparsemat, flags flags) {
  
  //Pack parameters
  T* mat = densemat->mat;
  int* cptrs = sparsemat->cptrs;
  int* rows = sparsemat->rows;
  T* cvals = sparsemat->cvals;
  int nov = sparsemat->nov;
  //Pack parameters
  
  //Pack flags
  int grid_dim = flags.grid_dim;
  int block_dim = flags.block_dim;
  int gpu_num = flags.gpu_num;
  //Pack flags
  
  T x[nov]; 
  T rs; //row sum
  T p = 1; //product of the elements in vector 'x'
  
  double p_partial[gpu_num]; //This is used only once and while computing return value
  //So it's double in order not to lose further precision while summing partial
  //results up
  
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
    T *d_x, *d_p;
    T *h_p = new T[grid_dim * block_dim];
    
    cudaMalloc( &d_x, (nov) * sizeof(T)); //Why is this exist ? 
    cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(T));
    cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
    cudaMalloc( &d_rows, (total) * sizeof(int));
    cudaMalloc( &d_cvals, (total) * sizeof(T));

    cudaMemcpy( d_x, x, (nov) * sizeof(T), cudaMemcpyHostToDevice);
    cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);
    
    size_t size = nov*block_dim*sizeof(T) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T);
    
    int x;
    double stt = omp_get_wtime();
    if (gpu_id == gpu_num-1) {
      //kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , size >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, (start + gpu_id*offset), end);
      x = 1;
    } else {
      //kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , size >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, (start + gpu_id*offset), (start + (gpu_id+1)*offset));
      x = 2;
    }
    cudaDeviceSynchronize();
    double enn = omp_get_wtime();
    //cout << "kernel" << gpu_id << " in " << (enn - stt) << endl;
    printf("kernel %d in %f \n", gpu_id, enn - stt);
    
    cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(T), cudaMemcpyDeviceToHost);
    
    cudaFree(d_x);
    cudaFree(d_p);
    cudaFree(d_cptrs);
    cudaFree(d_rows);
    cudaFree(d_cvals);
    
    for (int i = 0; i < grid_dim * block_dim; i++) {
      p_partial[gpu_id] += (double)h_p[i];
    }
    
    delete[] h_p;
  }
  
  double return_p = p;
  
  for (int gpu_id = 0; gpu_id < gpu_num; gpu_id++) {
    return_p += p_partial[gpu_id];
  }
  
  return((4*(nov&1)-2) * return_p);
}


//Returning double, is another problem ?
//What if we have a really big integer ?
//Moreover, hybrid approach designed really very bad, need to restucture this function
template <class T>
extern double gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse(DenseMatrix<T>* densemat, SparseMatrix<T>* sparsemat, flags flags) {

  //Pack parameters
  T* mat = densemat->mat;
  int* cptrs = sparsemat->cptrs;
  int* rows = sparsemat->rows;
  T* cvals = sparsemat->cvals;
  int nov = sparsemat->nov;
  //Pack parameters

  //Pack flags
  int grid_dim = flags.grid_dim;
  int block_dim = flags.block_dim;  
  int gpu_num = flags.gpu_num;
  bool cpu = flags.cpu;
  int threads = flags.threads;
  //Pack flags
 
  
  T x[nov]; 
  T rs; //row sum
  T p = 1; //product of the elements in vector 'x'
  
  double p_partial[gpu_num+1]; //This is only used while calculating return value
  
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
	  //cout << "ChunkID " << curr_chunk_id << "is DONE by CPU" << " in " << (enn - stt) << endl;
	  printf("ChunkID %d is DONE by CPU in %f \n", curr_chunk_id, enn-stt);
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
      T *d_x, *d_p;
      T *h_p = new T[grid_dim * block_dim];
      
      cudaMalloc( &d_x, (nov) * sizeof(T));
      cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(T));
      cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
      cudaMalloc( &d_rows, (total) * sizeof(int));
      cudaMalloc( &d_cvals, (total) * sizeof(T));
      
      cudaMemcpy( d_x, x, (nov) * sizeof(T), cudaMemcpyHostToDevice);
      cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);
      
      int curr_chunk_id;
      
#pragma omp critical 
      {
	curr_chunk_id = chunk_id;
	chunk_id++;
      }
      
      int x;
      while (curr_chunk_id < number_of_chunks) {
	double stt = omp_get_wtime();
	if (curr_chunk_id == number_of_chunks - 1) {
	  //kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, (start + curr_chunk_id*offset), end);
	  x = 1;
	} else {
	  //kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, (start + curr_chunk_id*offset), (start + (curr_chunk_id+1)*offset));
	  x = 2;
	}
	cudaDeviceSynchronize();
	double enn = omp_get_wtime();
	//cout << "ChunkID " << curr_chunk_id << "is DONE by kernel" << id << " in " << (enn - stt) << endl;
	//printf("ChunkID %d is DONE by kernel %d in %f \n", curr_chunk_id, id, enn-stt);
        
	cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);
        
	for (int i = 0; i < grid_dim * block_dim; i++) {
	  p_partial[id] += (double)h_p[i];
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
  
  double return_p = p;
  
  for (int id = 0; id < gpu_num+1; id++) {
    return_p += p_partial[id];
  }
  
  return((4*(nov&1)-2) * return_p);
}

template <class C, class S>
extern Result gpu_perman64_xshared_coalescing_mshared_skipper(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags flags) {

  //Pack parameters
  S* mat = densemat->mat;
  int* cptrs = sparsemat->cptrs;
  int* rows = sparsemat->rows;
  int* rptrs = sparsemat->rptrs;
  int* cols = sparsemat->cols;
  S* cvals = sparsemat->cvals;
  int nov = sparsemat->nov;
  //Pack parameters

  //Pack flags
  int grid_dim = flags.grid_dim;
  int block_dim = flags.block_dim;
  int device_id = flags.device_id;
  int grid_dim_multip = flags.grid_multip;
  //Pack flags

  //This is where all will be unified, set device, and then start timing
  cudaSetDevice(device_id);
  cudaDeviceSynchronize();

  double starttime = omp_get_wtime();
  
  C x[nov]; 
  C rs; //row sum
  C p = 1; //product of the elements in vector 'x'
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

  //For variable smem
  glob_nov = nov;
  glob_total = total;
  glob_sizeof_c = sizeof(C);
  glob_sizeof_s = sizeof(S);
  //For variable smem
  
  cudaOccupancyMaxPotentialBlockSizeVariableSMem(&grid_dim,
						 &block_dim,
						 &kernel_xshared_coalescing_mshared_skipper<C,S>,
						 xshared_coalescing_mshared_skipper_sharedmem,
						 0);
  
  size_t size = nov*block_dim*sizeof(C) + 2*(nov+1)*sizeof(int) + 2*total*sizeof(int) + total*sizeof(S);
  
  printf("==SC== Shared memory per block is set to : %zu \n", size);
  printf("==SC== Grid dim is set to : %d \n", grid_dim);
  printf("==SC== Block dim is set to : %d \n", block_dim);
  
  if(grid_dim_multip != 1){
    grid_dim*=grid_dim_multip;
    printf("==SC== Grid dim is re-set to : %d \n", grid_dim);
  }
  
  S *d_cvals;
  int *d_rptrs, *d_cols, *d_cptrs, *d_rows;
  C *d_x, *d_p;
  C *h_p = new C[grid_dim * block_dim];

  cudaMalloc( &d_x, (nov) * sizeof(C));
  cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(C));
  cudaMalloc( &d_rptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_cols, (total) * sizeof(int));
  cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int));
  cudaMalloc( &d_rows, (total) * sizeof(int));
  cudaMalloc( &d_cvals, (total) * sizeof(S));

  cudaMemcpy( d_x, x, (nov) * sizeof(C), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rptrs, rptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cols, cols, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy( d_cvals, cvals, (total) * sizeof(S), cudaMemcpyHostToDevice);

  long long start = 1;
  long long end = (1LL << (nov-1));

  //double stt = omp_get_wtime();
  kernel_xshared_coalescing_mshared_skipper<C,S><<<grid_dim , block_dim , size>>>(d_rptrs, d_cols, d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, start, end);
  cudaDeviceSynchronize();
  //double enn = omp_get_wtime();
  //cout << "kernel" << " in " << (enn - stt) << endl;
  //printf("kernel in %f \n", enn - stt);
  
  cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(C), cudaMemcpyDeviceToHost);

  cudaFree(d_x);
  cudaFree(d_p);
  cudaFree(d_rptrs);
  cudaFree(d_cols);
  cudaFree(d_cptrs);
  cudaFree(d_rows);
  cudaFree(d_cvals);

  double return_p = p;

  for (int i = 0; i < grid_dim * block_dim; i++) {
    return_p += (double)h_p[i];
  }

  delete[] h_p;

  double perman = (4*(nov&1)-2) * return_p;
  double duration = omp_get_wtime() - starttime;
  Result result(perman, duration);
  return result;

  //return((4*(nov&1)-2) * return_p);
}

template <class T>
extern double gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_skipper(DenseMatrix<T>* densemat, SparseMatrix<T>* sparsemat, flags flags) {

  //Pack parameters
  T* mat = densemat->mat;
  int* cptrs = sparsemat->cptrs;
  int* rows = sparsemat->rows;
  int* rptrs = sparsemat->rptrs;
  int* cols = sparsemat->cols;
  T* cvals = sparsemat->cvals;
  int nov = sparsemat->nov;
  //Pack parameters

  //Pack flags
  int gpu_num = flags.gpu_num;
  bool cpu = flags.cpu;
  int threads = flags.threads;
  int grid_dim = flags.grid_dim;
  int block_dim = flags.block_dim;
  //Pack flags
  
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
            //cout << "ChunkID " << curr_chunk_id << "is DONE by CPU" << " in " << (enn - stt) << endl;
	    printf("ChunkID %d is DONE by CPU in %f \n", curr_chunk_id, enn-stt);
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

	int x;
	
        while (curr_chunk_id < number_of_chunks) {
          double stt = omp_get_wtime();
          if (curr_chunk_id == number_of_chunks - 1) {
            //kernel_xshared_coalescing_mshared_skipper<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + 2*(nov+1)*sizeof(int) + 2*total*sizeof(int) + total*sizeof(T)) >>> (d_rptrs, d_cols, d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, (start + curr_chunk_id*offset), end);
	    x = 1;
          } else {
            kernel_xshared_coalescing_mshared_skipper<<< grid_dim , block_dim , (nov*block_dim*sizeof(float) + 2*(nov+1)*sizeof(int) + 2*total*sizeof(int) + total*sizeof(T)) >>> (d_rptrs, d_cols, d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, (start + curr_chunk_id*offset), (start + (curr_chunk_id+1)*offset));
	    x = 2;
          }
          cudaDeviceSynchronize();
          double enn = omp_get_wtime();
          //cout << "ChunkID " << curr_chunk_id << "is DONE by kernel" << id << " in " << (enn - stt) << endl;
	  printf("ChunkID %d is DONE by kernel %d in %f \n", curr_chunk_id, id, enn-stt);
                
          cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(double), cudaMemcpyDeviceToHost);
              
          for (int i = 0; i < grid_dim * block_dim; i++) {
            p_partial[id] += (double)h_p[i];
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

    double return_p = p;
    
  for (int id = 0; id < gpu_num+1; id++) {
    return_p += p_partial[id];
  }

  return((4*(nov&1)-2) * return_p);
}


template <class T>
double gpu_perman64_xshared_coalescing_mshared_multigpu_sparse_manual_distribution(DenseMatrix<T>* densemat, SparseMatrix<T>* sparsemat, flags flags) {

  //Pack parameters
  T* mat = densemat->mat;
  int* cptrs = sparsemat->cptrs;
  int* rows = sparsemat->rows;
  T* cvals = sparsemat->cvals;
  int nov = sparsemat->nov;
  //Pack parameters

  //Pack flags
  int gpu_num = flags.gpu_num;
  int grid_dim = flags.grid_dim;
  int block_dim = flags.block_dim;
  //Pack flags
  
  
  T x[nov]; 
  T rs; //row sum
  T p = 1; //product of the elements in vector 'x'
  double p_partial[gpu_num]; //This is only used while 
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
      T *d_x, *d_p;
      T *h_p = new T[grid_dim * block_dim];

      cudaMalloc( &d_x, (nov) * sizeof(T) );
      cudaMalloc( &d_p, (grid_dim * block_dim) * sizeof(T) );
      cudaMalloc( &d_cptrs, (nov + 1) * sizeof(int) );
      cudaMalloc( &d_rows, (total) * sizeof(int) );
      cudaMalloc( &d_cvals, (total) * sizeof(T) );

      cudaMemcpy( d_x, x, (nov) * sizeof(T), cudaMemcpyHostToDevice);
      cudaMemcpy( d_cptrs, cptrs, (nov + 1) * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy( d_rows, rows, (total) * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy( d_cvals, cvals, (total) * sizeof(T), cudaMemcpyHostToDevice);

      int x;
      double stt = omp_get_wtime();
      if (gpu_id == 0) {
        //kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , (nov*block_dim*sizeof(T) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, start, start + 3*offset);
	x = 1;
      } else if (gpu_id == 1) {
        //kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , (nov*block_dim*sizeof(T) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, start + 3*offset, start + 6*offset);
	x = 2;
      } else if (gpu_id == 2) {
        //kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , (nov*block_dim*sizeof(T) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, start + 6*offset, start + 7*offset);
	x = 3;
      } else if (gpu_id == 3) {
        //kernel_xshared_coalescing_mshared_sparse<<< grid_dim , block_dim , (nov*block_dim*sizeof(T) + (nov+1)*sizeof(int) + total*sizeof(int) + total*sizeof(T)) >>> (d_cptrs, d_rows, d_cvals, d_x, d_p, nov, total, start + 7*offset, end);
	x = 4;
      }
      cudaDeviceSynchronize();
      double enn = omp_get_wtime();

      //cout << "kernel" << gpu_id << " in " << (enn - stt) << endl;
      printf("kernel %d in %f \n", gpu_id, enn - stt);
        
      cudaMemcpy( h_p, d_p, grid_dim * block_dim * sizeof(T), cudaMemcpyDeviceToHost);

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


//Explicit instantiations required for separate compilation//

/////
template extern Result gpu_perman64_xlocal_sparse<float, int>(DenseMatrix<int>* densemat, SparseMatrix<int>* sparsemat, flags flags);
template extern Result gpu_perman64_xlocal_sparse<double, int>(DenseMatrix<int>* densemat, SparseMatrix<int>* sparsemat, flags flags);
template extern Result gpu_perman64_xlocal_sparse<float, float>(DenseMatrix<float>* densemat, SparseMatrix<float>* sparsemat, flags flags);
template extern Result gpu_perman64_xlocal_sparse<double, float>(DenseMatrix<float>* densemat, SparseMatrix<float>* sparsemat, flags flags);
template extern Result gpu_perman64_xlocal_sparse<float, double>(DenseMatrix<double>* densemat, SparseMatrix<double>* sparsemat, flags flags);
template extern Result gpu_perman64_xlocal_sparse<double, double>(DenseMatrix<double>* densemat, SparseMatrix<double>* sparsemat, flags flags);
/////

/////
template extern Result gpu_perman64_xshared_sparse<float, int>(DenseMatrix<int>* densemat, SparseMatrix<int>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_sparse<double, int>(DenseMatrix<int>* densemat, SparseMatrix<int>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_sparse<float, float>(DenseMatrix<float>* densemat, SparseMatrix<float>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_sparse<double, float>(DenseMatrix<float>* densemat, SparseMatrix<float>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_sparse<float, double>(DenseMatrix<double>* densemat, SparseMatrix<double>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_sparse<double, double>(DenseMatrix<double>* densemat, SparseMatrix<double>* sparsemat, flags flags);
/////

/////
template extern Result gpu_perman64_xshared_coalescing_sparse<float, int>(DenseMatrix<int>* densemat, SparseMatrix<int>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_coalescing_sparse<double, int>(DenseMatrix<int>* densemat, SparseMatrix<int>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_coalescing_sparse<float, float>(DenseMatrix<float>* densemat, SparseMatrix<float>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_coalescing_sparse<double, float>(DenseMatrix<float>* densemat, SparseMatrix<float>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_coalescing_sparse<float, double>(DenseMatrix<double>* densemat, SparseMatrix<double>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_coalescing_sparse<double, double>(DenseMatrix<double>* densemat, SparseMatrix<double>* sparsemat, flags flags);
/////

/////
template extern Result gpu_perman64_xshared_coalescing_mshared_sparse<float, int>(DenseMatrix<int>* densemat, SparseMatrix<int>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_coalescing_mshared_sparse<double, int>(DenseMatrix<int>* densemat, SparseMatrix<int>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_coalescing_mshared_sparse<float, float>(DenseMatrix<float>* densemat, SparseMatrix<float>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_coalescing_mshared_sparse<double, float>(DenseMatrix<float>* densemat, SparseMatrix<float>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_coalescing_mshared_sparse<float, double>(DenseMatrix<double>* densemat, SparseMatrix<double>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_coalescing_mshared_sparse<double, double>(DenseMatrix<double>* densemat, SparseMatrix<double>* sparsemat, flags flags);
/////

/////
template extern Result gpu_perman64_xshared_coalescing_mshared_skipper<float, int>(DenseMatrix<int>* densemat, SparseMatrix<int>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_coalescing_mshared_skipper<double, int>(DenseMatrix<int>* densemat, SparseMatrix<int>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_coalescing_mshared_skipper<float, float>(DenseMatrix<float>* densemat, SparseMatrix<float>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_coalescing_mshared_skipper<double, float>(DenseMatrix<float>* densemat, SparseMatrix<float>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_coalescing_mshared_skipper<float, double>(DenseMatrix<double>* densemat, SparseMatrix<double>* sparsemat, flags flags);
template extern Result gpu_perman64_xshared_coalescing_mshared_skipper<double, double>(DenseMatrix<double>* densemat, SparseMatrix<double>* sparsemat, flags flags);
/////

template extern double gpu_perman64_xshared_coalescing_mshared_multigpu_sparse<int>(DenseMatrix<int>* densemat, SparseMatrix<int>* sparsemat, flags flags);
template extern double gpu_perman64_xshared_coalescing_mshared_multigpu_sparse<float>(DenseMatrix<float>* densemat, SparseMatrix<float>* sparsemat, flags flags);
template extern double gpu_perman64_xshared_coalescing_mshared_multigpu_sparse<double>(DenseMatrix<double>* densemat, SparseMatrix<double>* sparsemat, flags flags);

template extern double gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse<int>(DenseMatrix<int>* densemat, SparseMatrix<int>* sparsemat, flags flags);
template extern double gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse<float>(DenseMatrix<float>* densemat, SparseMatrix<float>* sparsemat, flags flags);
template extern double gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse<double>(DenseMatrix<double>* densemat, SparseMatrix<double>* sparsemat, flags flags);

template extern double gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_skipper<int>(DenseMatrix<int>* densemat, SparseMatrix<int>* sparsemat, flags flags);
template extern double gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_skipper<float>(DenseMatrix<float>* densemat, SparseMatrix<float>* sparsemat, flags flags);
template extern double gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_skipper<double>(DenseMatrix<double>* densemat, SparseMatrix<double>* sparsemat, flags flags);

template extern double gpu_perman64_xshared_coalescing_mshared_multigpu_sparse_manual_distribution<int>(DenseMatrix<int>* densemat, SparseMatrix<int>* sparsemat, flags flags);
template extern double gpu_perman64_xshared_coalescing_mshared_multigpu_sparse_manual_distribution<float>(DenseMatrix<float>* densemat, SparseMatrix<float>* sparsemat, flags flags);
template extern double gpu_perman64_xshared_coalescing_mshared_multigpu_sparse_manual_distribution<double>(DenseMatrix<double>* densemat, SparseMatrix<double>* sparsemat, flags flags);
//Explicit instantiations required for separate compilation//
