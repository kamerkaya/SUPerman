#ifndef FLAGS_H
#define FLAGS_H

#include<string>
#include<iostream>

template<class C>
struct ScaleCompanion{

  C* r_v;
  C* c_v;


  ScaleCompanion(int nov)
  {
    r_v = new C[nov];
    c_v = new C[nov];
  }

  ~ScaleCompanion()
  {
    delete r_v;
    delete c_v;
  }
  
};

struct Result{

  double permanent;
  double time;

  Result(){
    permanent = 0;
    time = 0;
  }

  Result(double p, double t):
    permanent(p), time(t) {}

  Result operator +(Result const &r2){
    return Result(permanent + r2.permanent, time + r2.time);
  }
  
};

//Parameters struct 
struct flags {
  
  bool cpu;
  bool gpu;
  bool gpu_stated;
  
  bool sparse;
  bool dense;

  bool exact;
  bool approximation;

  bool grid_graph;
  bool calculation_half_precision;
  bool calculation_quad_precision;
  bool storage_half_precision;
  bool storage_quad_precision;
  bool binary_graph;
  
  int gridm;
  int gridn;
  
  int perman_algo;
  int threads;

  int scale_intervals;
  int scale_times;
  
  const char* filename;
  const char* type;
  std::string algo_name;
  int preprocessing;

  int gpu_num;
  int number_of_times;

  int grid_dim;
  int block_dim;

  int device_id;
  int rep;

  int grid_multip;

  bool compression;

  double scaling_threshold;
  
  flags(){
    
    cpu = 0;
    gpu = 1; //Assumed gpu
    gpu_stated = 0; //This is to enable hybrid execution and prevent running multiple algos

    sparse = 0;
    dense = 1; //Assumed dense

    exact = 1; //Assumed exact
    approximation = 0; 
    
    grid_graph = 0; //Assumed it is not a grid graph
    storage_half_precision = 0; //Assumed double data type
    storage_quad_precision = 0; //Assumed double data type
    calculation_half_precision = 0; //Assumed double data type
    calculation_quad_precision = 0; //Assumed double data type
    binary_graph = 0;
    
    gridm = -1; //If stay -1, means there is a problem getting the actual value
    gridn = -1; //If stay -1, means there is a problem getting the actual value

    perman_algo = 0; //If no algorithm is stated, run the first algo

    threads = 1; //Assumed sequential execution if not stated

    scale_intervals = 4; //If stay -1, means there is a problem getting the actual value
    scale_times = 5; //If stay -1, means there is a problem getting the actual value

    //filename = "";
    type = ""; //If stay empty string, means there is a problem getting the actual value
    //These are not initialized until data is seen
    
    preprocessing = 0; //If stay 0, means no preprocessing. Else 1->sortorder 2->skiporder

    gpu_num = 1; //Assumed we will use 1 GPU
    number_of_times = 100000; //Assumed 100K

    grid_dim = 2048; // Assumed 2048 / 256, will override if 
    block_dim = 256; // desired otherwise

    device_id = 0; //Assumed one and only GPU
    rep = 1; //Assume one repetition

    grid_multip = 1; //Assume maximum possible shared memory
    compression = 0; //Assume will not compress unless stated otherwise
    scaling_threshold = -1.0; //It is doubly stochastic if called by mistake
  }
  
};


//These structs should be relocated to "flags.h"
template <class T>
struct DenseMatrix{
  T* mat;
  int nov;
  int nnz;

  ~DenseMatrix(){
    delete mat;
  }
};

template <class T>
DenseMatrix<T>* copy_dense(DenseMatrix<T>* to_copy){

  DenseMatrix<T>* ret = new DenseMatrix<T>;
  
  int nov = to_copy->nov;
  int nnz = to_copy->nnz;

  ret->nov = nov;
  ret->nnz = nnz;


  ret->mat = new T[nov * nov];
  for(int i = 0; i < nov * nov; i++){
    ret->mat[i] = to_copy->mat[i];
  }

  return ret;
}

template <class T>
struct SparseMatrix{
  int* cptrs;
  int* rptrs;
  int* rows;
  int* cols;
  T* cvals;
  T* rvals;
  int nov;
  int nnz;

  ~SparseMatrix(){
    delete[] cptrs;
    delete[] rptrs;
    delete[] rows;
    delete[] cols;
    delete[] cvals;
    delete[] rvals;
  }
  
};

template<class T>
SparseMatrix<T>* copy_sparse(SparseMatrix<T>* to_copy){

  SparseMatrix<T>* ret = new SparseMatrix<T>;

  int nov = to_copy->nov;
  int nnz = to_copy->nnz;

  ret->nov = nov;
  ret->nnz = nnz;

  ret->rvals = new T[nnz];
  ret->cvals = new T[nnz];
  ret->cptrs = new int[nov + 1];
  ret->rptrs = new int[nov + 1];
  ret->rows = new int[nnz];
  ret->cols = new int[nnz];
  
  for(int i = 0; i < nov + 1; i++){
    ret->cptrs[i] = to_copy->cptrs[i];
    ret->rptrs[i] = to_copy->rptrs[i];
  }
  
  for(int i = 0; i < nnz; i++){
    ret->rvals[i] = to_copy->rvals[i];
    ret->cvals[i] = to_copy->cvals[i];
    
    ret->rows[i] = to_copy->rows[i];
    ret->cols[i] = to_copy->cols[i];
  }

  return ret;
}


#endif
