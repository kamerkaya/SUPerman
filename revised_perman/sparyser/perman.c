
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
/*#include "omp.h"*/

int nnz(double* a, int m) {
  int nnzcnt = 0;
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < m; j++) {
      if(a[i*m + j] != 0) {
        nnzcnt++;
      }
    }
  }
  return nnzcnt;
}

void print_mat(double* a, int m) {
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < m; j++) {
      if(a[i*m + j] == 0) printf("_.__   "); else printf("%.2f   ", a[i*m + j]);
    }
    puts("");
  }
  puts("");
}

double perman (double *a, int m) {
  double x[32];// temporary vector as used by Nijenhuis and Wilf
  double rs;   // row sum of matrix
  double s;    // +1 or -1
  double prod; // product of the elements in vector 'x'
  double p=1.0;  // many results accumulate here, MAY need extra precision
  double *xptr, *aptr; 
  int j, k;
  unsigned long int i, tn11 = (1UL<<(m-1))-1;  // tn11 = 2^(n-1)-1
  unsigned long int gray, prevgray=0, two_to_k;
  
  xptr = (double *)x;
  aptr = &a[(m-1)*m];
  for (j=0; j<m; j++) {
    rs = 0.0;
    for (k=0; k<m; k++){
      rs += a[j + k*m];    // sum of row j
    }
    *xptr = *aptr++ - rs/2;  // see Nijenhuis and Wilf
    p *= *xptr++;   // product of the elements in vector 'x'
  }
  
  for (i=1; i<=tn11; i++) {
    gray=i^(i>>1); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    
    two_to_k=1;    // two_to_k = 2 raised to the k power (2^k)
    k=0;
    while (two_to_k < (gray^prevgray)) {
      two_to_k<<=1;  // two_to_k is a bitmask to find location of 1
      k++;
    }
    s = (two_to_k & gray) ? +1.0 : -1.0;
    prevgray = gray;        
    
    prod = 1.0;
    xptr = (double *)x;
    aptr = &a[k*m];
    for (j=0; j<m; j++) {
      *xptr += s * *aptr++;  // see Nijenhuis and Wilf
      prod *= *xptr++;  // product of the elements in vector 'x'
    }
    // Keep the summing below in the loop, moving it loses important resolution on x87
    p += ((i&1)? -1.0:1.0) * prod; 
  }
  return (double)(4*(m&1)-2) * (double)p;          
}

void get_mins_tb(double* a, int m, int* rdegs, int* cdegs, int* pmindeg, int* pminindex) {
  memset(rdegs, 0, sizeof(int) * m);
  memset(cdegs, 0, sizeof(int) * m);
  
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < m; j++) {
      if(a[i*m + j] != 0) {
	rdegs[i]++;
	cdegs[j]++;
      }
    }
  }
  
  int mindeg = m+1;
  int minindex;
  for(int i = 0; i < m; i++) {
    if(rdegs[i] < mindeg) {
      mindeg = rdegs[i];
      minindex = i+1;
    }
  }
  
  for(int i = 0; i < m; i++) {
    if(cdegs[i] < mindeg) {
      mindeg = cdegs[i];
      minindex = -(i+1);
    }
  }
  *pmindeg = mindeg;
  *pminindex = minindex;
}

void get_mins(double* a, int m, int* rdegs, int* cdegs, int* pmindeg, int* pminindex) {
  memset(rdegs, 0, sizeof(int) * m);
  memset(cdegs, 0, sizeof(int) * m);
  
  for(int i = 0; i < m; i++) {
    for(int j = 0; j < m; j++) {
      if(a[i*m + j] != 0) {
	rdegs[i]++;
	cdegs[j]++;
      }
    }
  }
  
  int mindeg = m+1;
  int minindex;
  for(int i = 0; i < m; i++) {
    if(rdegs[i] < mindeg) {
      mindeg = rdegs[i];
      minindex = i+1;
    }
  }
  
  for(int i = 0; i < m; i++) {
    if(cdegs[i] < mindeg) {
      mindeg = cdegs[i];
      minindex = -(i+1);
    }
  }
  *pmindeg = mindeg;
  *pminindex = minindex;
}

void get_cols(double* a, int m, int minindex, int cols[2]) {
  cols[0] = cols[1] = -1;
  
  int count = 0;
  for(int i = 0; i < m; i++) {  
    if(a[minindex*m + i] != 0) {
      cols[count++] = i;
      if(count == 2) {
	break;
      }
    }
  }
  
  if(count < 2) {
    if(cols[0] != 0) {
      cols[1] = 0;
    } else {
      cols[1] = 1;
    }
  }
}

void get_rows(double* a, int m, int minindex, int rows[2]) {
  rows[0] = rows[1] = -1;
  
  int count = 0;
  for(int i = 0; i < m; i++) {  
    if(a[i*m + minindex] != 0) {
      rows[count++] = i;
      if(count == 2) {
	break;
      }
    }
  }
  
  if(count < 2) {
    if(rows[0] != 0) {
      rows[1] = 0;
    } else {
      rows[1] = 1;
    }
  }
}

void get_A1(double* a, int m, int minindex, double* a1, double* pval) {
  if(minindex > 0) { //row                                                                                                                
    minindex = minindex - 1; //real row id

    int cols[2]; 
    get_cols(a, m, minindex, cols);
    *pval = a[minindex * m + cols[0]];

#ifdef DEBUG
    printf("column is %d value is %f\n", cols[0], *pval);
#endif
    
    int rc = 0;
    for(int i = 0; i < m; i++) {
      if(i == minindex) continue;
      int cc = 0;
      for(int j = 0; j < m; j++) {
	if(j == cols[0]) continue; 
	a1[rc * (m-1) + cc] = a[i * m + j];
	cc++;
      }
      rc++;
    }
  } else if(minindex < 0) { //col                                                                                                                                                                                                                                         
    minindex = (-minindex) - 1;

    int rows[2]; 
    get_rows(a, m, minindex, rows);
    *pval = a[rows[0] * m + minindex];

#ifdef DEBUG
    printf("row is %d, value is %f\n", rows[0], *pval);
#endif

    int rc = 0;
    for(int i = 0; i < m; i++) {
      if(i == rows[0]) continue; 
      int cc = 0;
      for(int j = 0; j < m; j++) {
	if(j == minindex) continue;
	a1[rc * (m-1) + cc] = a[i * m + j];
	cc++;
      }
      rc++;
    }
  } else {
    printf("minindex = 0: error\n");
    exit(0);
  }
}


void get_A1_A2(double* a, int m, int minindex, double* a1, double* a2) {
  if(minindex > 0) { //row                                                                                                                
    minindex = minindex - 1; //real row id
    int cols[2]; 
    get_cols(a, m, minindex, cols);
#ifdef DEBUG
    printf("columns are %d %d\n", cols[0], cols[1]);
#endif
    memcpy(a1, a, sizeof(double) * m * m);
    a1[minindex * m + cols[0]] = a1[minindex * m + cols[1]] = 0;

    int rc = 0;
    for(int i = 0; i < m; i++) {
      if(i == minindex) continue;
      int cc = 1;
      for(int j = 0; j < m; j++) {
	if(j == cols[0] || j == cols[1]) continue; 
	a2[rc * (m-1) + cc] = a[i * m + j];
	cc++;
      }
      rc++;
    }
    
    double v0 = a[minindex * m + cols[0]];
    double v1 = a[minindex * m + cols[1]];
    rc = 0;
    for(int i = 0; i < m; i++) {
      if(i != minindex) {
	 a2[(rc++) * (m-1)] = v0 * a[i*m + cols[1]] + v1 * a[i*m + cols[0]]; 
      }
    }
  } else if(minindex < 0) { //col                                                                                                                                                                                                                                                       
    minindex = (-minindex) - 1;

    int rows[2]; 
    get_rows(a, m, minindex, rows);
#ifdef DEBUG
    printf("rows are %d %d\n", rows[0], rows[1]);
#endif
    memcpy(a1, a, sizeof(double) * m * m);
    a1[rows[0] * m + minindex] = a1[rows[1] * m + minindex] = 0;

    int rc = 1;
    for(int i = 0; i < m; i++) {
      if(i == rows[0] || i == rows[1]) continue; 
      int cc = 0;
      for(int j = 0; j < m; j++) {
	if(j == minindex) continue;
	a2[rc * (m-1) + cc] = a[i * m + j];
	cc++;
      }
      rc++;
    }
    
    double v0 = a[rows[0] * m + minindex];
    double v1 = a[rows[1] * m + minindex];
    int cc = 0;
    for(int i = 0; i < m; i++) {
      if(i != minindex) {
	a2[cc++] = v0 * a[rows[1] * m + i] + v1 * a[rows[0] * m + i]; 
      }
    }
  } else {
    printf("minindex = 0: error\n");
    exit(0);
  }
}

double perman_hybrid_naive(double *a, int m, int* rdegs, int* cdegs) {
#ifdef DEBUG
  printf("A is: \n");
  print_mat(a, m);
#endif  
  int mindeg, minindex;
  if(m > 2) {
    get_mins(a, m, rdegs, cdegs, &mindeg, &minindex);
#ifdef DEBUG
    if(minindex < 0) {
      printf("mindeg %d -- col id %d\n", mindeg, (-minindex) - 1);
    } else {
      printf("mindeg %d -- row id %d\n", mindeg, minindex-1);
    }
#endif
    if(mindeg == 0) return 0;

    if(mindeg < 5) { 
      double* a1 = (double*) malloc(m * m * sizeof(double));
      double* a2 = (double*) malloc((m-1) * (m-1) * sizeof(double));
      //      printf("--- minindex %d\n", minindex);

      get_A1_A2(a, m, minindex, a1, a2);

#ifdef DEBUG
      printf("A1 and A2 is:\n");
      print_mat(a1, m);
      print_mat(a2, m-1);
      puts("----------------------------------------------------------");
#endif      

      double perman_1 = perman_hybrid_naive(a1, m, rdegs, cdegs);
      double perman_2 = perman_hybrid_naive(a2, m-1, rdegs, cdegs);

      free(a1);
      free(a2);
      return perman_1 + perman_2;
      
    } else {
      return perman(a, m);
    } 
  } else {
    return perman(a, m);
  }
}

double perman_hybrid_d1(double *a, int m, int* rdegs, int* cdegs) {
#ifdef DEBUG
  printf("A is: \n");
  print_mat(a, m);
#endif  
  int mindeg, minindex;
  if(m > 2) {
    get_mins(a, m, rdegs, cdegs, &mindeg, &minindex);
#ifdef DEBUG
    if(minindex < 0) {
      printf("mindeg %d -- col id %d\n", mindeg, (-minindex) - 1);
    } else {
      printf("mindeg %d -- row id %d\n", mindeg, minindex-1);
    }
#endif
    if(mindeg == 0) return 0;
    
    if(mindeg < 5) { 
      if(mindeg == 1) {
#ifdef DEBUG
	puts("minimum degree is 1");
#endif
	double* a1 = (double*) malloc((m-1) * (m-1) * sizeof(double));
	double val;

	get_A1(a, m, minindex, a1, &val);
#ifdef DEBUG
        printf("removed A1 is:\n");
        print_mat(a1, m-1);
        puts("----------------------------------------------------------");
#endif
        double perman_1 = perman_hybrid_naive(a1, m-1, rdegs, cdegs);
	return perman_1 * val;
	
      }	else {
	double* a1 = (double*) malloc(m * m * sizeof(double));
	double* a2 = (double*) malloc((m-1) * (m-1) * sizeof(double));
		
	get_A1_A2(a, m, minindex, a1, a2);
	
#ifdef DEBUG
	printf("A1 and A2 is:\n");
	print_mat(a1, m);
	print_mat(a2, m-1);
	puts("----------------------------------------------------------");
#endif      
	
	double perman_1 = perman_hybrid_d1(a1, m, rdegs, cdegs);
	double perman_2 = perman_hybrid_d1(a2, m-1, rdegs, cdegs);
	
	free(a1);
	free(a2);
	return perman_1 + perman_2;
      }
    } else {
      return perman(a, m);
    } 
  } else {
    return perman(a, m);
  }
}




