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
  
  flags(){
    
    cpu = 0;
    gpu = 1; //Assumed gpu

    sparse = 0;
    dense = 1; //Assumed dense

    exact = 1; //Assumed exact
    approximation = 0; 
    
    grid_graph = 0; //Assumed it is not a grid graph
    gridm = -1; //If stay -1, means there is a problem getting the actual value
    gridn = -1; //If stay -1, means there is a problem getting the actual value

    perman_algo = -1; //If stay -1, means there is a problem getting the actual value
    threads = 1; //Assumed sequential execution if not stated

    scale_intervals = 4; //If stay -1, means there is a problem getting the actual value
    scale_times = 5; //If stay -1, means there is a problem getting the actual value

    //filename = "";
    //type = ""; //If stay empty string, means there is a problem getting the actual value
    //These are not initialized until data is seen
    
    preprocessing = -1; //If stay -1, means no preprocessing :)

    gpu_num = 1; //Assumed we will use 1 GPU
    number_of_times = 100000; //Assumed 100K

    grid_dim = 2048; // Assumed 2048 / 256, will override if 
    block_dim = 256; // desired otherwise
  }
  
};

#endif
