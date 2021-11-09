#ifndef FLAGS_H
#define FLAGS_H

//Parameters struct 
struct flags {
  
  bool cpu;
  bool gpu;
  
  bool sparse;
  bool dense;

  bool exact;
  bool approximation;

  bool grid_graph;
  bool half_precision;
  bool binary_graph;
  
  int gridm;
  int gridn;
  
  int perman_algo;
  int threads;

  int scale_intervals;
  int scale_times;
  
  const char* filename;
  const char* type;
  int preprocessing;

  int gpu_num;
  int number_of_times;

  int grid_dim;
  int block_dim;

  int device_id;
  int rep;

  int grid_multip;
  
  flags(){
    
    cpu = 0;
    gpu = 1; //Assumed gpu

    sparse = 0;
    dense = 1; //Assumed dense

    exact = 1; //Assumed exact
    approximation = 0; 
    
    grid_graph = 0; //Assumed it is not a grid graph
    half_precision = 0; //Assumed double data type
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


#endif
