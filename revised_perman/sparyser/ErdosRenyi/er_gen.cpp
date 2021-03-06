#include <iostream>
#include <stdio.h>
#include <random>
#include <cstring>

int main(int argc, char** argv) {
  
  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(0.0, 1.0);

  double ps[11] = {0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70};
  double ns[6] = {30, 32, 34, 36, 38, 40};
  int x = 10; 

  char f[1000];
  
  for(int pindex = 0; pindex < 11; pindex++) {
    double p = ps[pindex]; 
    for(int nindex = 0; nindex < 6; nindex++) {
      int n = ns[nindex];
      int *mat = new int[n * n];

      for(int i = 0; i < x; i++) {
	int nnz = 0;
	sprintf(f, "erdos_%d.%.2f.%d.mtx", n, p, i);
	FILE* fp = fopen(f, "w");
	
	memset(mat, 0, sizeof(int) * n * n);
	for(int r = 0; r < n; r++) {
	  for(int c = 0; c < n; c++) {
	    if(dis(gen) < p) {
	      mat[r * n + c] = 1;
	      nnz++;
	    }
	  }
	}
	
	fprintf(fp, "%d %d %d\n", n, n, nnz);
	for(int r = 0; r < n; r++) {
	  for(int c = 0; c < n; c++) {
	    if(mat[r * n + c] == 1) {
	      fprintf(fp, "%d %d\n", r+1, c+1); 
	    }
	  }
	}
	
	fclose(fp);
      }
      delete [] mat;
    }
  }

  return 1;
}
