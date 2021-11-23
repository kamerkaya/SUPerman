#include <iostream> 
#include "kutils.h"
#include "algos.h"
#include "algos_w.h"
#include <omp.h>

using namespace std;

string matrix = "";
int pcall_counter = 0;

int recursive_counter = 0;

double getPerman(int* mat, int nov, int algo, int sorting) {
  cout << "Perman: [veni] running with " << nov << " rows:\n";  fflush(0);
  if(nov > 64) {
    cout << "matrix is too large " << endl;
    return 0;
  }

  // sort the matrix rows with respect to degrees ****************
  if(sorting == 1) {
    sortWColDeg(mat, nov, true);
    firstSeenRow(mat, nov);
  } else if(sorting == 2) {
    sortRCM(mat, nov);
    firstSeenRow(mat, nov);
  } else if(sorting == 3) {
    sortMinNew(mat, nov);
    firstSeenRow(mat, nov);
  } else if(sorting == 4) {
    bfsOrder(mat, nov);
    firstSeenRow(mat, nov);
  }
    /*
  if(sorting != 0) {
    int* mat_new = new int[nov * nov];
    for(int i = 0; i < nov; i++) {
      for(int j = 0; j < nov; j++) {
	mat_new[i * nov + j] = mat[i * nov + (nov - j - 1)];
      }
    }
    delete [] mat;
    mat = mat_new;
  }*/

  //matrix change
  if(algo == 4) {
    int* mat_new = new int[(nov + 1) * (nov + 1)];
    for(int i = 0; i < nov; i++) {
      int rs = 0;
      for(int j = 0; j < nov; j++) {
	int val = mat[(i * nov) + j];
   	mat_new[(i * (nov + 1)) + j] = val;
	rs += val;
      }
      if(i < nov - 3) {
	mat_new[(i * (nov + 1)) + nov] = 0;
      } else {
	mat_new[(i * (nov + 1)) + nov] = rs;
      }
    }
    for(int j = 0; j < nov; j++) {
      mat_new[(nov * (nov + 1)) + j] = 0;
    }
    mat_new[(nov * (nov + 1)) + nov] = 1;
    
    delete [] mat;
    mat = mat_new;
    nov = nov + 1;
  }
  //--------------------------------------------------------------

  printMatrix(mat, nov);

  int *xadj, *adj, *val;
  matrix2graph(mat, nov, xadj, adj, val);

  //unsigned long long perman;
  double perman;
  double start = omp_get_wtime();
  if(algo == 0) {
    perman = brute_w(xadj, adj, val, 2 * nov);
  } else if(algo == 1) {
    perman = perman64(mat, nov);                                                                  
  } else if(algo == 2) {
    //perman = sparse_perman64_w(xadj, adj, val, nov); 
    perman = sparser_perman64_w(xadj, adj, val, nov); 
  } else if(algo == 3 || algo == 4) {
    perman = sparser_skip_perman64_w(xadj, adj, val, mat, nov);
    //cout << "F perman: " << perman << std::endl;
  } else if(algo == 10) {
    perman = parallel_skip_perman64_w(xadj, adj, val, mat, nov);
  } else if(algo == 11) {
    perman = parallel_skip_perman64_w_balanced(xadj, adj, val, mat, nov);
  } 

  double wtime = omp_get_wtime() - start;
  cout << " [vidi] " << algo << " " << sorting << " " << matrix << "_" << pcall_counter << " " << nov << " " << perman << " " << wtime << endl; 
  pcall_counter++;

  //cout << "deleting ----- " << endl;
  delete [] xadj;
  delete [] adj;
  delete [] val;
  //cout << "deleted ----- " << endl;
  return perman;
}


//Well, that will be the driver code, simple as that
double getPermanRecursive(int* mat, int nov, int algo, int sorting) {
  //unsigned long long perman = 0;
  //  cout << "Recursive count: " << recursive_counter++ << endl;
  double perman = 0;
  int minDeg = getMinNnz(mat, nov);
  //cout << "##MINDEG: " << minDeg << endl;
  if(minDeg < 5 && nov > 30) {
    if(minDeg == 1) {
      d1compress(mat, nov);
      //cout << "d1: matrix is reduced to " << nov << " rows" << endl;
      return getPermanRecursive(mat, nov, algo, sorting);
    } else if (minDeg == 2) {
      d2compress(mat, nov);
      //cout << "d2: matrix is reduced to " << nov << " rows" << endl;
      return getPermanRecursive(mat, nov, algo, sorting);
    } else if (minDeg == 3 || minDeg == 4) {
      int* mat2 = nullptr;
      int nov2;
      d34compress(mat, nov, mat2, nov2, minDeg);
      //cout << "d34: matrix is reduced to matrices with " << nov << " and " << nov2 << " rows" << endl;
      perman = getPermanRecursive(mat, nov, algo, sorting) + getPermanRecursive(mat2, nov2, algo, sorting);
      //      cout << "deleting2 " << " " << mat2 << endl;
      if(mat2 != nullptr) {
	delete [] mat2;
	mat2 = nullptr;
      }
      // cout << "deleted2 " << endl;
    } 
  } else {
    perman = getPerman(mat, nov, algo, sorting);
  }
  return perman;
}

int main(int argc, char** argv) {
  int nov, nnz;
  int* mat;

  int algo = 0;
  int compressing = 0;
  int dulmenning = 0;
  int sorting = 0;
  int singling = 1;

  if(argc == 1) {
    cout << "usage: sparyser filename [algo] [compressed] [dulmage mendehlson] [sort] [singling]" << endl;
    exit(1);
  }
  
  if(argc > 1) {  
    string filename(argv[1]);
    matrix = filename;
    //readTxtFile(filename, mat, 1, nov, nnz);
    readINTFile(filename, mat, 1, nov, nnz);
    printMatrix(mat, nov);
  }

  if(argc > 2) {
    algo = atoi(argv[2]);
  }
  cout << "Algo is " << algo << endl;

  if(argc > 3) {
    compressing = atoi(argv[3]);
  }
  cout << "Compress matrix " << compressing << endl;

  if(argc > 4) {
    dulmenning = atoi(argv[4]);
  }
  cout << "Apply Dulmage-Mendehlson " << dulmenning << endl;

  if(argc > 5) {
    sorting = atoi(argv[5]);
  }
  cout << "Sorting " << sorting << endl;

  if(argc > 6) {
    singling = atoi(argv[6]);
  }
  cout << "Single run " << singling << endl;
  cout << seperator << endl;

  // apply dulmage mendehlson
  if(dulmenning) {
    cout << "Applying Dulmage-Mendelson "<< endl;
    dulmen(mat, nov);
    cout << seperator << endl;
  }

  
  // compress the matrix and remove singleton rows ***************
  if(compressing) {
    cout << "Compressing phase " << endl;
    bool comp = true;
    while(comp && nov > 1) {
      comp = d1compress(mat, nov);
      if(comp) {
	cout << "Matrix is compressed by d1 compress: nov is " << nov << endl;
      } else {
	comp = d2compress(mat, nov);
	if(comp) {
	  cout << "Matrix is compressed by d2 compress: nov is " << nov << endl;
	}
      }
       
      if(comp) {
	if(checkEmpty(mat, nov)) {
	  cout << "Matrix is rank deficient" << endl;
	  cout << "Perman is 0" << endl;
	  exit(1);
	} 
      }
    }
    cout << "Compressing is done" << endl;
    cout << seperator << endl;
  }
  

  //unsigned long long perman = 0;
  double perman = 0;

  double start = omp_get_wtime();
  if(singling) {
    perman = getPerman(mat, nov, algo, sorting);
  } else {
    cout << "Calling, getPermanRecursive.. " << endl;
    perman = getPermanRecursive(mat, nov, algo, sorting);
  }
  double wtime = omp_get_wtime() - start;
  cout << "Overall perman is: " << perman << " in " << wtime << endl;     
  return 0;
}

