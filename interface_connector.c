#include "algo.h"
#include "matlab_calculate_return.h"


template <class T>
void print_mat(T mat, int nov){

  std::cout << "Your matrix: " << std::endl;
  for(int i = 0; i < nov*nov; i++){
    std::cout << mat[i] << " ";
    if(i != 0 && i%nov == 0)
      std::cout << endl;
  }
  
}
template <class T>
double decide_and_call(T* mat, int nov, int nnz, int nt, int x, int y, int z, int algo,
		       int* cptrs, int* rows, int* rptrs, int*cols, T* rvals, T* cvals){

  int perman = 0;
  std::cout << "Will call: " << algo << std::endl;
  
  if(algo == 0){
    perman = rasmussen_sparse(mat, rptrs, cols, nov, x, nt);
  }
  else if(algo == 1){
    perman = rasmussen(mat, nov, x, nt);
  }
  else if(algo == 2){
    perman = approximation_perman64_sparse(cptrs, rows, rptrs, cols, nov, x, y, z, nt);
    //Note that scale intervals is 'y'
  }
  else if(algo == 3){
    perman = approximation_perman64(mat, nov, x, y, z, nt);
  }
  else if(algo == 4){
    perman = parallel_perman64_sparse(mat, cptrs, rows, cvals, nov, nt);
  }
  else if(algo == 5){
    perman = parallel_perman64(mat, nov, nt);
  }
  else if(algo == 6){
    perman = parallel_skip_perman64_w(rptrs, cols, rvals, cptrs, rows, cvals, nov, nt);
  }
  else if(algo == 7){
    perman = parallel_skip_perman64_w_balanced(rptrs, cols, rvals, cptrs, rows, cvals, nov, nt);
  }
  else if(algo == 8){
    perman = perman64(mat, nov);
  }
  else{
    std::cout << "Algo unavailable, exiting..";
    exit(1);
  }

  return perman;
}

extern "C" void connect(){
  std::cout << "SUPerman Connected.." << std::endl;
}

extern "C" double read_calculate_return(char* filename, int algorithm, int nt, int x, int y, int z)
{

  //std::cout << "File: " << filename.c_str() << std::endl;
  std::cout << algorithm << " " << nt << " " << x << " " << y << " " << z << std::endl;
  
  //Reading the matrix
  int nov, nnz;
  string type;   
  ifstream inFile(filename);                                                      
  string line;
  bool generic = 0;
  double perman = 0;

  int preprocessing = 0;
  std::cout << "Preprocessing: " << preprocessing << std::endl;
  if(algorithm == 0 || algorithm == 2 || algorithm == 4){
    preprocessing = 1;
  }
  else if(algorithm == 6 || algorithm == 7){
    preprocessing = 2;
  }
  else{
    //std::cout << "SUPerpython can't achieve that right now, exiting.." << std::endl;
    preprocessing = 0;
  }
      
  
  
  getline(inFile, line);
  istringstream iss(line);
  iss >> nov >> nnz >> type;
  
  if (type == "int") {
    int* mat = new int[nov*nov];
    ReadMatrix(mat, inFile, nov, generic);
    //print_mat(mat, nov);
    int *cvals, *rvals;
    int *cptrs, *rows, *rptrs, *cols;
    if (preprocessing == 1) {
      matrix2compressed_sortOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    } else if (preprocessing == 2) {
      matrix2compressed_skipOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    } else {
      matrix2compressed(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    }
    perman = decide_and_call(mat, nov, nnz, nt, x, y, z, algorithm, cptrs, rows, rptrs, cols, rvals, cvals);
  }

  else if (type == "double") {                                                                     
    double* mat = new double[nov*nov];
    ReadMatrix(mat, inFile, nov, generic);
    //print_mat(mat, nov);
    double *cvals, *rvals;
    int *cptrs, *rows, *rptrs, *cols;
    if (preprocessing == 1) {
      matrix2compressed_sortOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    } else if (preprocessing == 2) {
      matrix2compressed_skipOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    } else {
      matrix2compressed(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    }
    perman = decide_and_call(mat, nov, nnz, nt, x, y, z, algorithm, cptrs, rows, rptrs, cols, rvals, cvals);
  }

  else if (type == "double") {                                                                     
    double* mat = new double[nov*nov];
    ReadMatrix(mat, inFile, nov, generic);
    //print_mat(mat, nov);
    double *cvals, *rvals;
    int *cptrs, *rows, *rptrs, *cols;
    if (preprocessing == 1) {
      matrix2compressed_sortOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    } else if (preprocessing == 2) {
      matrix2compressed_skipOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    } else {
      matrix2compressed(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    }
    perman = decide_and_call(mat, nov, nnz, nt, x, y, z, algorithm, cptrs, rows, rptrs, cols, rvals, cvals);
  }

  else{
    std::cout << "Corrupted file, exiting.. " << std::endl;
    exit(1);
  }

  return perman;
}

extern "C" double matlab_calculate_return(int* mat, int algorithm, int nt, int x, int y, int z,
					  int nov, int nnz)
{

  //std::cout << algorithm << " " << nt << " " << x << " " << y << " " << z << std::endl;
  std::cout << "Your matrix from Matlab: " << std::endl;
  print_mat(mat, nov);
  
  //Reading the matrix
  bool generic = 0;
  double perman = 0;

  int preprocessing = 0;
  std::cout << "Preprocessing: " << preprocessing << std::endl;
  if(algorithm == 0 || algorithm == 2 || algorithm == 4){
    preprocessing = 1;
  }
  else if(algorithm == 6 || algorithm == 7){
    preprocessing = 2;
  }
  else{
    //std::cout << "SUPerpython can't achieve that right now, exiting.." << std::endl;
    preprocessing = 0;
  }
   
  int *cvals, *rvals;
  int *cptrs, *rows, *rptrs, *cols;
  if (preprocessing == 1) {
    matrix2compressed_sortOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
  } else if (preprocessing == 2) {
    matrix2compressed_skipOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
  } else {
    matrix2compressed(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
  }
  perman = decide_and_call(mat, nov, nnz, nt, x, y, z, algorithm, cptrs, rows, rptrs, cols, rvals, cvals);


  return perman;
}


int main(){

  shell();
  std::cout << "Just an empty shell of a man" << std::endl;
  
  return 0;
  
}
