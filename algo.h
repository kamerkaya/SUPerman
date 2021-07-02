#ifndef ALGO
#define ALGO

#include <iostream>
#include <bitset>
#include <omp.h>
#include <string.h>
#include "util.h"
using namespace std;


template <class T>
double greedy(T* mat, int nov, int number_of_times) {
  T* mat_t = new T[nov * nov];
  
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(j * nov) + i] = mat[(i * nov) + j];
    }
  }

  srand(time(0));

  int nt = omp_get_max_threads();

  double sum_perm = 0;
  double sum_zeros = 0;
  
  #pragma omp parallel for num_threads(nt) reduction(+:sum_perm) reduction(+:sum_zeros)
    for (int time = 0; time < number_of_times; time++) {
      int row_nnz[nov];
      int col_nnz[nov];
      bool row_extracted[nov];
      bool col_extracted[nov];
      
      for (int i = 0; i < nov; i++) {
        row_nnz[i] = 0;
        col_nnz[i] = 0;
        row_extracted[i] = false;
        col_extracted[i] = false;
        for (int j = 0; j < nov; j++) {
          if (mat[(i * nov) + j] != 0) {
            row_nnz[i] += 1;
          }
          if (mat_t[(i * nov) + j] != 0) {
            col_nnz[i] += 1;
          }
        }
      }
      
      double perm = 1;
      int row, col, deg;
      float sum_pk, pk, random;
      
      for (int i = 0; i < nov; i++) {
        // choose the row to be extracted
        deg = nov+1;
        int n = 0;
        for (int l = 0; l < nov; l++) {
          if (!row_extracted[l]) {
            if (row_nnz[l] < deg) {
              n = 1;
              row = l;
              deg = row_nnz[l];
            } else if (row_nnz[l] == deg) {
              n++;
            }
          }
        }
        if (n > 1) {
          random = rand() % n;
          for (int l = 0; l < nov; l++) {
            if (!row_extracted[l] && row_nnz[l] == deg) {
              if (random == 0) {
                row = l; break;
              }
              random--;
            }
          }
        }

        // compute sum of probabilities, when finding deg = 1, set col
        sum_pk = 0;
        int sum_ones = 0;
        for (int k = 0; k < nov; k++) {
          if (!col_extracted[k] && mat[row * nov + k] != 0) {
            sum_pk += 1 / float(col_nnz[k]);
          }
        }

        // choose the col to be extracted if not chosen already
        if (sum_ones == 0) {
          random = (float(rand()) / RAND_MAX) * sum_pk;
          sum_pk = 0;
          for (int k = 0; k < nov; k++) {
            if (!col_extracted[k] && mat[row * nov + k] != 0) {
              sum_pk += 1 / float(col_nnz[k]);
              if (random <= sum_pk) {
                col = k;
                pk = 1 / float(col_nnz[k]);
                break;
              }
            }
          }
        } else {
          random = rand() % sum_ones;
          for (int k = 0; k < nov; k++) {
            if (!col_extracted[k] && mat[row * nov + k] != 0 && col_nnz[k] == 1) {
              if (random == 0) {
                col = k;
                pk = 1;
                break;
              }
              random--;
            }
          }
        }
        
        // multiply permanent with 1 / pk
        perm /= pk;

        // extract row and col
        row_extracted[row] = true;
        col_extracted[col] = true;

        // update number of nonzeros of the rows and cols after extraction
        bool zero_row = false;
        for (int r = 0; r < nov; r++) {
          if (!row_extracted[r] && mat_t[col * nov + r] != 0) {
            row_nnz[r]--;
            if (row_nnz[r] == 0) {
              zero_row = true;
              break;
            }
          }
        }

        if (zero_row) {
          perm = 0;
          sum_zeros += 1;
          break;
        }

        for (int c = 0; c < nov; c++) {
          if (!col_extracted[c] && mat[row * nov + c] != 0) {
            col_nnz[c]--;
            if (col_nnz[c] == 0) {
              zero_row = true;
              break;
            }
          }
        }

        if (zero_row) {
          perm = 0;
          sum_zeros += 1;
          break;
        }
      }

      sum_perm += perm;
    }

  delete[] mat_t;

  cout << "number of zeros: " << sum_zeros << endl;
  
  return (sum_perm / number_of_times);
}

template <class T>
double rasmussen_sparse(T *mat, int *rptrs, int *cols, int nov, int number_of_times, int threads) {
  T* mat_t = new T[nov * nov];
  
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(j * nov) + i] = mat[(i * nov) + j];
    }
  }

  srand(time(0));

  double sum_perm = 0;
  double sum_zeros = 0;
    
  #pragma omp parallel for num_threads(threads) reduction(+:sum_perm) reduction(+:sum_zeros)
    for (int time = 0; time < number_of_times; time++) {
      int row_nnz[nov];
      int col_extracted[21];
      int row_extracted[21];
      for (int i = 0; i < 21; i++) {
        col_extracted[i]=0;
        row_extracted[i]=0;
      }

      int row;
      int min=nov+1;
      
      for (int i = 0; i < nov; i++) {
        row_nnz[i] = rptrs[i+1] - rptrs[i];
        if (min > row_nnz[i]) {
          min = row_nnz[i];
          row = i;
        }
      }
      
      double perm = 1;
      
      for (int k = 0; k < nov; k++) {
        // multiply permanent with number of nonzeros in the current row
        perm *= row_nnz[row];

        // choose the column to be extracted randomly
        int random = rand() % row_nnz[row];
        int col;
        for (int i = rptrs[row]; i < rptrs[row+1]; i++) {
          int c = cols[i];
          if (!((col_extracted[c / 32] >> (c % 32)) & 1)) {
            if (random == 0) {
              col = c;
              break;
            } else {
              random--;
            }        
          }
        }

        // exract the column
        col_extracted[col / 32] |= (1 << (col % 32));
        row_extracted[row / 32] |= (1 << (row % 32));

        min = nov+1;

        // update number of nonzeros of the rows after extracting the column
        bool zero_row = false;
        for (int r = 0; r < nov; r++) {
          if (!((row_extracted[r / 32] >> (r % 32)) & 1)) {
            if (mat_t[col * nov + r] != 0) {
              row_nnz[r]--;
              if (row_nnz[r] == 0) {
                zero_row = true;
                break;
              }
            }
            if (min > row_nnz[r]) {
              min = row_nnz[r];
              row = r;
            }
          }
        }

        if (zero_row) {
          perm = 0;
          sum_zeros += 1;
          break;
        }
      }

      sum_perm += perm;
    }

  delete[] mat_t;

  cout << "number of zeros: " << sum_zeros << endl;
  
  return (sum_perm / number_of_times);
}

template <class T>
double rasmussen(T* mat, int nov, int number_of_times, int threads) {
  T* mat_t = new T[nov * nov];
  
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(j * nov) + i] = mat[(i * nov) + j];
    }
  }

  srand(time(0));

  double sum_perm = 0;
  double sum_zeros = 0;
  
  #pragma omp parallel for num_threads(threads) reduction(+:sum_perm) reduction(+:sum_zeros)
    for (int time = 0; time < number_of_times; time++) {
      int row_nnz[nov];
      long col_extracted = 0;
      long row_extracted = 0;
      
      int row;
      int min=nov+1;

      for (int i = 0; i < nov; i++) {
        row_nnz[i] = 0;
        for (int j = 0; j < nov; j++) {
          if (mat[(i * nov) + j] != 0) {
            row_nnz[i] += 1;
          }
        }
        if (min > row_nnz[i]) {
          min = row_nnz[i];
          row = i;
        }
      }
      
      double perm = 1;
      
      for (int i = 0; i < nov; i++) {
        // multiply permanent with number of nonzeros in the current row
        perm *= row_nnz[row];

        // choose the column to be extracted randomly
        int random = rand() % row_nnz[row];
        int col;
        for (int c = 0; c < nov; c++) {
          if (!((col_extracted >> c) & 1L) && mat[row * nov + c] != 0) {
            if (random == 0) {
              col = c;
              break;
            } else {
              random--;
            }        
          }
        }

        // exract the column
        col_extracted |= (1L << col);
        row_extracted |= (1L << row);

        min = nov+1;
        // update number of nonzeros of the rows after extracting the column
        bool zero_row = false;
        for (int r = 0; r < nov; r++) {
          if (!((row_extracted >> r) & 1L)) {
            if (mat_t[col * nov + r] != 0) {
              row_nnz[r]--;
              if (row_nnz[r] == 0) {
                zero_row = true;
                break;
              }
            }
            if (min > row_nnz[r]) {
              min = row_nnz[r];
              row = r;
            }
          }
        }

        if (zero_row) {
          perm = 0;
          sum_zeros += 1;
          break;
        }
      }

      sum_perm += perm;
    }

  delete[] mat_t;

  cout << "number of zeros: " << sum_zeros << endl;
  
  return (sum_perm / number_of_times);
}

double approximation_perman64_sparse(int *cptrs, int *rows, int *rptrs, int *cols, int nov, int number_of_times, int scale_intervals, int scale_times, int threads) {

  srand(time(0));

  double sum_perm = 0;
  double sum_zeros = 0;
    
  #pragma omp parallel for num_threads(threads) reduction(+:sum_perm) reduction(+:sum_zeros)
    for (int time = 0; time < number_of_times; time++) {
      int col_extracted[21];
      int row_extracted[21];
      for (int i = 0; i < 21; i++) {
        col_extracted[i]=0;
        row_extracted[i]=0;
      }

      double Xa = 1;
      double d_r[nov];
      double d_c[nov];
      for (int i = 0; i < nov; i++) {
        d_r[i] = 1;
        d_c[i] = 1;
      }

      int row;
      int min;
      int nnz;

      for (int k = 0; k < nov; k++) {
        min=nov+1;
        for (int i = 0; i < nov; i++) {
          if (!((row_extracted[i / 32] >> (i % 32)) & 1)) {
            nnz = 0;
            for (int j = rptrs[i]; j < rptrs[i+1]; j++) {
              int c = cols[j];
              if (!((col_extracted[c / 32] >> (c % 32)) & 1)) {
                nnz++;
              }
            }
            if (min > nnz) {
              min = nnz;
              row = i;
            }
          }
        }

        // Scale part
        if (time % scale_intervals == 0) {
          bool success = ScaleMatrix_sparse(cptrs, rows, rptrs, cols, nov, row_extracted, col_extracted, d_r, d_c, scale_times);
          if (!success) {
            Xa = 0;
            sum_zeros++;
            break;
          }
        }

        // use scaled matrix for pj
        double sum_row_of_S = 0;
        for (int i = rptrs[row]; i < rptrs[row+1]; i++) {
          int c = cols[i];
          if (!((col_extracted[c / 32] >> (c % 32)) & 1)) {
            sum_row_of_S += d_r[row] * d_c[c];
          }
        }
        if (sum_row_of_S == 0) {
          Xa = 0;
          sum_zeros++;
          break;
        }

        double random = (double(rand()) / RAND_MAX) * sum_row_of_S;
        double temp = 0;
        double s, pj;
        int col;
        for (int i = rptrs[row]; i < rptrs[row+1]; i++) {
          int c = cols[i];
          if (!((col_extracted[c / 32] >> (c % 32)) & 1)) {
            s = d_r[row] * d_c[c];
            temp += s;
            if (random <= temp) {
              col = c;
              pj = s / sum_row_of_S;
              break;
            }
          }
        }

        // update Xa
        Xa /= pj;
        
        // exract the column
        col_extracted[col / 32] |= (1 << (col % 32));
        // exract the row
        row_extracted[row / 32] |= (1 << (row % 32));

      }

      sum_perm += Xa;
    }
  
  cout << "number of zeros: " << sum_zeros << endl;
  
  return (sum_perm / number_of_times);
}

template <class T>
double approximation_perman64(T* mat, int nov, int number_of_times, int scale_intervals, int scale_times, int threads) {
  srand(time(0));

  double sum_perm = 0;
  double sum_zeros = 0;
    
  #pragma omp parallel for num_threads(threads) reduction(+:sum_perm) reduction(+:sum_zeros)
    for (int time = 0; time < number_of_times; time++) {
      long col_extracted = 0;
      long row_extracted = 0;

      double Xa = 1;
      double d_r[nov];
      double d_c[nov];
      for (int i = 0; i < nov; i++) {
        d_r[i] = 1;
        d_c[i] = 1;
      }

      int row;
      int min;
      int nnz;

      for (int k = 0; k < nov; k++) {
        min=nov+1;
        for (int i = 0; i < nov; i++) {
          if (!((row_extracted >> i) & 1L)) {
            nnz = 0;
            for (int j = 0; j < nov; j++) {
              if (!((col_extracted >> j) & 1L) && mat[(i * nov) + j] != 0) {
                nnz++;
              }
            }
            if (min > nnz) {
              min = nnz;
              row = i;
            }
          }
        }
        // Scale part
        if (time % scale_intervals == 0) {
          bool success = ScaleMatrix(mat, nov, row_extracted, col_extracted, d_r, d_c, scale_times);
          if (!success) {
            Xa = 0;
            sum_zeros++;
            break;
          }
        }
        
        // use scaled matrix for pj
        double sum_row_of_S = 0;
        for (int j = 0; j < nov; j++) {
          if (!((col_extracted >> j) & 1L) && mat[(row * nov) + j] != 0) {
            sum_row_of_S += d_r[row] * mat[(row * nov) + j] * d_c[j];
          }
        }
        if (sum_row_of_S == 0) {
          Xa = 0;
          sum_zeros++;
          break;
        }

        double random = (double(rand()) / RAND_MAX) * sum_row_of_S;
        double temp = 0;
        double s, pj;
        int col;
        for (int j = 0; j < nov; j++) {
          if (!((col_extracted >> j) & 1L) && mat[(row * nov) + j] != 0) {
            s = d_r[row] * mat[(row * nov) + j] * d_c[j];
            temp += s;
            if (random <= temp) {
              col = j;
              pj = s / sum_row_of_S;
              break;
            }
          }
        }

        // update Xa
        Xa /= pj;
        
        // exract the column
        col_extracted |= (1L << col);
        // exract the row
        row_extracted |= (1L << row);

      }

      sum_perm += Xa;
    }
  
  cout << "number of zeros: " << sum_zeros << endl;
  
  return (sum_perm / number_of_times);
}

template <class T>
double parallel_perman64_sparse(T* mat, int* cptrs, int* rows, T* cvals, int nov, int threads) {
  float x[nov];   
  float rs; //row sum
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

  long long one = 1;
  long long start = 1;
  long long end = (1LL << (nov-1));
  
  long long chunk_size = end / threads + 1;

  #pragma omp parallel num_threads(threads) firstprivate(x)
  { 
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
          x[rows[j]] += cvals[j]; // see Nijenhuis and Wilf - update x vector entries
        }
      }
    }

    prod = 1.0;
    int zero_num = 0;
    for (int j = 0; j < nov; j++) {
      if (x[j] == 0) {
        zero_num++;
      } else {
        prod *= x[j];  //product of the elements in vector 'x'
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
        if (x[rows[j]] == 0) {
          zero_num--;
          x[rows[j]] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
          prod *= x[rows[j]];  //product of the elements in vector 'x'
        } else {
          prod /= x[rows[j]];
          x[rows[j]] += s * cvals[j]; // see Nijenhuis and Wilf - update x vector entries
          if (x[rows[j]] == 0) {
            zero_num++;
          } else {
            prod *= x[rows[j]];  //product of the elements in vector 'x'
          }
        }
      }

      if(zero_num == 0) {
        my_p += prodSign * prod; 
      }
      prodSign *= -1;
      i++;
    }

    #pragma omp critical
      p += my_p;
  }

  return((4*(nov&1)-2) * p);
}

template <class T>
double parallel_perman64(T* mat, int nov, int threads) {
  float x[nov];   
  float rs; //row sum
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

  long long one = 1;
  long long start = 1;
  long long end = (1LL << (nov-1));
  
  long long chunk_size = end / threads + 1;

  #pragma omp parallel num_threads(threads) firstprivate(x)
  { 
    int tid = omp_get_thread_num();
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
        xptr = (float*)x;
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
      xptr = (float*)x;
      for (int j = 0; j < nov; j++) {
        *xptr += s * mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
        prod *= *xptr++;  //product of the elements in vector 'x'
      }

      my_p += prodSign * prod; 
      prodSign *= -1;
      i++;
    }

    #pragma omp critical
      p += my_p;
  }

  delete [] mat_t;

  return((4*(nov&1)-2) * p);
}

template <class T>
double parallel_skip_perman64_w(int *rptrs, int *cols, T *rvals, int *cptrs, int *rows, T *cvals, int nov, int threads) {
  //first initialize the vector then we will copy it to ourselves
  std::cout << "I'm here " << std::endl;
  double rs, x[64], p;
  int j, ptr;
  unsigned long long ci, start, end, chunk_size, change_j;

  //initialize the vector entries                                                                                        
  for (j = 0; j < nov; j++) {
    rs = .0f; 
    for(ptr = rptrs[j]; ptr < rptrs[j+1]; ptr++) 
      rs += rvals[ptr];
    x[j] = -rs/(2.0f);
  }
  
  for(ptr = cptrs[nov-1]; ptr < cptrs[nov]; ptr++) {
    x[rows[ptr]] += cvals[ptr];
  }

  //update perman with initial x
  double prod = 1;
  for(j = 0; j < nov; j++) {
    prod *= x[j];
  }
  p = prod;

  //find start location        
  start = 1;
  for(int j = 0; j < nov; j++) {
    if(x[j] == 0) {
      change_j = -1;
      for (ptr = rptrs[j]; ptr < rptrs[j + 1]; ptr++) {
        ci = 1ULL << cols[ptr]; 
        if(ci < change_j) {
          change_j = ci;
        }
      }
      if(change_j > start) {
        start = change_j;
      }
    }
  }

  end = (1ULL << (nov-1));
  
  chunk_size = (end - start + 1) / threads + 1;

  #pragma omp parallel num_threads(threads) private(j, ci, change_j) 
  {
    double my_x[64];
    memcpy(my_x, x, sizeof(double) * 64);
    
    int tid = omp_get_thread_num();
    unsigned long long my_start = start + tid * chunk_size;
    unsigned long long my_end = min(start + ((tid+1) * chunk_size), end);
    
    //update if neccessary
    double my_p = 0;

    unsigned long long my_gray;    
    unsigned long long my_prev_gray = 0;

    int ptr, last_zero;
    unsigned long long period, steps, step_start;

    unsigned long long i = my_start;

    while (i < my_end) {
      //k = __builtin_ctzll(i + 1);
      my_gray = i ^ (i >> 1);

      unsigned long long gray_diff = my_prev_gray ^ my_gray;
      //cout << "X: " << gray_diff << endl;
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
  return ((4*(nov&1)-2) * p);
}


template <class T>
double parallel_skip_perman64_w_balanced(int *rptrs, int *cols, T *rvals, int *cptrs, int *rows, T *cvals, int nov, int threads) {
  //first initialize the vector then we will copy it to ourselves
  double rs, x[nov], p;
  int j, ptr;
  unsigned long long ci, start, end, chunk_size, change_j;

  //initialize the vector entries                                                                                        
  for (j = 0; j < nov; j++) {
    rs = .0f; 
    for(ptr = rptrs[j]; ptr < rptrs[j+1]; ptr++) 
      rs += rvals[ptr];
    x[j] = -rs/(2.0f);
  }
  
  for(ptr = cptrs[nov-1]; ptr < cptrs[nov]; ptr++) {
    x[rows[ptr]] += cvals[ptr];
  }

  //update perman with initial x
  double prod = 1;
  for(j = 0; j < nov; j++) {
    prod *= x[j];
  }
  p = prod;

  //find start location        
  start = 1;
  for(int j = 0; j < nov; j++) {
    if(x[j] == 0) {
      change_j = -1;
      for (ptr = rptrs[j]; ptr < rptrs[j + 1]; ptr++) {
        ci = 1ULL << cols[ptr]; 
        if(ci < change_j) {
          change_j = ci;
        }
      }
      if(change_j > start) {
        start = change_j;
      }
    }
  }

  end = (1ULL << (nov-1));

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
    
  return ((4*(nov&1)-2) * p);
}







template <class T>
double perman64(T* mat, int nov) {
  double x[64];   
  double rs; //row sum
  double s;  //+1 or -1 
  double prod; //product of the elements in vector 'x'
  double p = 1; //product of the elements in vector 'x'
  double *xptr; 
  int j, k;
  unsigned long long i, tn11 = (1ULL << (nov-1)) - 1ULL;
  unsigned long long int gray;
  
  //create the x vector and initiate the permanent
  for (j = 0; j < nov; j++) {
    rs = .0f;
    for (k = 0; k < nov; k++)
      rs += mat[(j * nov) + k];  // sum of row j
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

  gray = 0;
  unsigned long long one = 1;

  unsigned long long counter = 0;

  double t_start = omp_get_wtime();
  for (i = 1; i <= tn11; i++) {

    //compute the gray code
    k = __builtin_ctzll(i);
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
    
    counter++;
    prod = 1.0;
    xptr = (double*)x;
    for (j = 0; j < nov; j++) {
      *xptr += s * mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      prod *= *xptr++;  //product of the elements in vector 'x'
    }

    p += ((i&1ULL)? -1.0:1.0) * prod; 
  }

  delete [] mat_t;

  return((4*(nov&1)-2) * p);
}


template <class T>
double brute_w(int *xadj, int *adj, T* val, int nov) {
  double perman = 0;
  double prod;

  int* matched = new int[nov];
  for(int i = 0; i < nov; i++) matched[i] = 0;
  
  int h_nov = nov/2;
  int* ptrs = new int[h_nov];
  for(int i = 0; i < h_nov; i++) {ptrs[i] = xadj[i];}

  matched[0] = adj[0];
  matched[adj[0]] = 1;
  ptrs[0] = 1;
  prod = val[0];

  int curr = 1;
  while(curr >= 0) {
    //clear the existing matching on current
    if(matched[curr] != 0) {
      prod /= val[ptrs[curr] - 1];
      matched[matched[curr]] = 0;
      matched[curr] = 0;
    }

    //check if we can increase the matching by matching curr
    int ptr = ptrs[curr];
    int partner;
    for(; ptr < xadj[curr + 1]; ptr++) {
      if(matched[adj[ptr]] == 0) {
  partner = adj[ptr];
  ptrs[curr] = ptr + 1;
  prod *= val[ptr];
  break;
      }
    }

    if(ptr < xadj[curr + 1]) { //we can extend matching
      if(curr == h_nov - 1) {
  perman += prod;
  prod /= val[ptr];   
  ptrs[curr] = xadj[curr];
  curr--;
      } else {
  matched[curr] = partner;
  matched[partner] = 1;
  curr++;
      }
    } else {
      ptrs[curr] = xadj[curr];
      curr--;
    }
  }
  return perman;
}




#endif
