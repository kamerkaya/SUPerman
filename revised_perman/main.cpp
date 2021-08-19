#include <iostream>
#include <string>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <omp.h>
#include <stdio.h>
#include <getopt.h>
#include "util.h" //evaluate_data_return_parameters() --> To be implemented
#include "gpu_wrappers.h" //All GPU wrappers will be stored there to simplify things

//Excluding GPU algos for a minimal start
//#include "gpu_exact_dense.cu" // Can't include now due to cuda f.
//#include "gpu_exact_sparse.cu"
//#include "gpu_approximation_dense.cu"
//#include "gpu_approximation_sparse.cu"
//Excluding GPU algos for a minimal start
//
#include "cpu_algos.hpp"
//
#include <math.h>
using namespace std;

void print_flags(flags flags){

  std::cout << "*~~~~~~~~~~~~FLAGS~~~~~~~~~~~~*" << std::endl;
  std::cout << "- cpu: " << flags.cpu << std::endl;
  std::cout << "- gpu: " << flags.gpu << std::endl;
  std::cout << "- sparse: " << flags.sparse << std::endl;
  std::cout << "- dense: " << flags.dense << std::endl;
  std::cout << "- exact: " << flags.exact << std::endl;
  std::cout << "- approximation: " << flags.approximation << std::endl;
  std::cout << "- grid_graph: " << flags.grid_graph << std::endl;
  std::cout << "- gridm: " << flags.gridm << std::endl;
  std::cout << "- gridn: " << flags.gridn << std::endl;
  std::cout << "- perman_algo: " << flags.perman_algo << std::endl;
  std::cout << "- threads: " << flags.threads << std::endl;
  std::cout << "- scale_intervals: " << flags.scale_intervals << std::endl;
  std::cout << "- scale_times: " << flags.scale_times << std::endl;
  std::string fname = &flags.filename[0];
  std::cout << "- fname: " << fname << std::endl;
  std::cout << "- type: " << flags.type << std::endl;
  std::cout << "- preprocessing: " << flags.preprocessing << std::endl;
  std::cout << "- gpu_num: " << flags.gpu_num << std::endl;
  std::cout << "- number_of_times: " << flags.number_of_times << std::endl;
  std::cout << "- grid_dim: " << flags.grid_dim << std::endl;
  std::cout << "- block_dim: " << flags.block_dim << std::endl;
  std::cout << "*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*" << std::endl;

}


template <class T>
void RunAlgo(T *mat, int *cptrs, int *rows, T *cvals, int *rptrs, int *cols, T *rvals, int nov, int nnz, flags flags) 
{
  
  print_flags(flags);

  int grid_dim = 2048;
  int block_dim = 256;
  
  if(flags.type == "double"){ //This is possible error because 
    //flags.type is a char*
    block_dim = 128;
  }

  //This is just for a more readable code
  bool cpu = flags.cpu;
  bool gpu = flags.gpu;

  bool dense = flags.dense;
  bool sparse = flags.sparse;

  bool exact = flags.exact;
  bool approximation = flags.approximation;
  
  int perman_algo = flags.perman_algo;
  int gpu_num = flags.gpu_num;
  int threads = flags.threads;

  int number_of_times = flags.number_of_times;
  int scale_intervals = flags.scale_intervals;
  int scale_times = flags.scale_times;
  //This is just for a more readable code
  
  double start, end, perman;
  
  if(cpu && dense && exact){    
    
    if(perman_algo == 0){
      start = omp_get_wtime();
      perman = parallel_perman64(mat, nov, flags);
      end = omp_get_wtime();
      cout << "Result: parallel_perman64 " << perman << " in " << (end - start) << endl;
    }
    
    else{
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }
    
  }
  
  if(cpu && sparse && exact){
    
    if (perman_algo == 1) {
      start = omp_get_wtime();
      perman = parallel_perman64_sparse(mat, cptrs, rows, cvals, nov, flags);
      end = omp_get_wtime();
      cout << "Result: parallel_perman64_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 2) {
      start = omp_get_wtime();
      perman = parallel_skip_perman64_w(rptrs, cols, rvals, cptrs, rows, cvals, nov, flags);
      end = omp_get_wtime();
      cout << "Result: parallel_skip_perman64_w " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 3) {
      start = omp_get_wtime();
      perman = parallel_skip_perman64_w_balanced(rptrs, cols, rvals, cptrs, rows, cvals, nov, flags);
      end = omp_get_wtime();
      cout << "Result: parallel_skip_perman64_w_balanced " << perman << " in " << (end - start) << endl;
    }
    
    else{
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }
    
  }
  
  if(cpu && dense && approximation){
    
    if (perman_algo == 1) { // rasmussen
      start = omp_get_wtime();
      perman = rasmussen(mat, nov, flags);
      end = omp_get_wtime();
      printf("Result: rasmussen %2lf in %lf\n", perman, end-start);
      cout << "Result: rasmussen " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 2) { // approximation_with_scaling
      start = omp_get_wtime();
      perman = approximation_perman64(mat, nov, flags);
      end = omp_get_wtime();
      printf("Result: approximation_perman64 %2lf in %lf\n", perman, end-start);
      cout << "Result: approximation_perman64 " << perman << " in " << (end - start) << endl;
    }
    
    else{
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }
    
  }
  
  
  if(cpu && sparse && approximation){
    
    if (perman_algo == 1) { // rasmussen
      start = omp_get_wtime();
      perman = rasmussen_sparse(mat, rptrs, cols, nov, flags);
      end = omp_get_wtime();
      printf("Result: rasmussen_sparse %2lf in %lf\n", perman, end-start);
      cout << "Result: rasmussen_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 2) { // approximation_with_scaling
      start = omp_get_wtime();
      perman = approximation_perman64_sparse(cptrs, rows, rptrs, cols, nov, flags);
      end = omp_get_wtime();
      printf("Result: approximation_perman64_sparse %2lf in %lf\n", perman, end-start);
      cout << "Result: approximation_perman64_sparse " << perman << " in " << (end - start) << endl;
    }
    
    else{
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }

  }
  
  
  if(gpu && dense && exact){
    
    if (perman_algo == 0) {
      start = omp_get_wtime();
      perman = gpu_perman64_xglobal(mat, nov, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xlocal " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 1) {
      start = omp_get_wtime();
      perman = gpu_perman64_xlocal(mat, nov, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xlocal " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 2) {
      start = omp_get_wtime();
      perman = gpu_perman64_xshared(mat, nov, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 3) {
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing(mat, nov, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 4) {
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared(mat, nov, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing_mshared " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 5) {
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared_multigpu(mat, nov, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing_mshared_multigpu " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 6) {
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks(mat, nov, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 66) {
      flags.gpu_num = 4;
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared_multigpu_manual_distribution(mat, nov, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_xshared_coalescing_mshared_multigpu_manual_distribution %2lf in %lf\n", perman, end-start);
      
    }else{
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }
  }
  
  /*
  if(gpu && dense && approximation){
      
    if (perman_algo == 1) { // rasmussen
      start = omp_get_wtime();
      perman = gpu_perman64_rasmussen(mat, nov, number_of_times);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_rasmussen %2lf in %lf\n", perman, end-start);
      cout << "Result: gpu_perman64_rasmussen " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 2) { // approximation_with_scaling
      start = omp_get_wtime();
      perman = gpu_perman64_approximation(mat, nov, number_of_times, scale_intervals, scale_times);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_approximation %2lf in %lf\n", perman, end-start);
      cout << "Result: gpu_perman64_approximation " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 3) { // rasmussen
      start = omp_get_wtime();
      perman = gpu_perman64_rasmussen_multigpucpu_chunks(mat, nov, number_of_times, gpu_num, cpu, threads);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_rasmussen_multigpucpu_chunks %2lf in %lf\n", perman, end-start);
      cout << "Result: gpu_perman64_rasmussen_multigpucpu_chunks " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 4) { // approximation_with_scaling
      start = omp_get_wtime();
      perman = gpu_perman64_approximation_multigpucpu_chunks(mat, nov, number_of_times, gpu_num, cpu, scale_intervals, scale_times, threads);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_approximation_multigpucpu_chunks %2lf in %lf\n", perman, end-start);
      cout << "Result: gpu_perman64_approximation_multigpucpu_chunks " << perman << " in " << (end - start) << endl;
    } else {
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    } 
  }
  
  */

  /*
  if(gpu && sparse && approximation){
    
    if (perman_algo == 1) { // rasmussen
	start = omp_get_wtime();
          perman = gpu_perman64_rasmussen_sparse(rptrs, cols, nov, nnz, number_of_times, false);
          end = omp_get_wtime();
          printf("Result: gpu_perman64_rasmussen_sparse %2lf in %lf\n", perman, end-start);
          cout << "Result: gpu_perman64_rasmussen_sparse " << perman << " in " << (end - start) << endl;
        } else if (perman_algo == 2) { // approximation_with_scaling
          start = omp_get_wtime();
          perman = gpu_perman64_approximation_sparse(cptrs, rows, rptrs, cols, nov, nnz, number_of_times, scale_intervals, scale_times, false);
          end = omp_get_wtime();
          printf("Result: gpu_perman64_approximation_sparse %2lf in %lf\n", perman, end-start);
          cout << "Result: gpu_perman64_approximation_sparse " << perman << " in " << (end - start) << endl;
        } else if (perman_algo == 3) { // rasmussen
          start = omp_get_wtime();
          perman = gpu_perman64_rasmussen_multigpucpu_chunks_sparse(cptrs, rows, rptrs, cols, nov, nnz, number_of_times, gpu_num, cpu, threads, false);
          end = omp_get_wtime();
          printf("Result: gpu_perman64_rasmussen_multigpucpu_chunks_sparse %2lf in %lf\n", perman, end-start);
          cout << "Result: gpu_perman64_rasmussen_multigpucpu_chunks_sparse " << perman << " in " << (end - start) << endl;
        } else if (perman_algo == 4) { // approximation_with_scaling
          start = omp_get_wtime();
          perman = gpu_perman64_approximation_multigpucpu_chunks_sparse(cptrs, rows, rptrs, cols, nov, nnz, number_of_times, gpu_num, cpu, scale_intervals, scale_times, threads, false);
          end = omp_get_wtime();
          printf("Result: gpu_perman64_approximation_multigpucpu_chunks_sparse %2lf in %lf\n", perman, end-start);
          cout << "Result: gpu_perman64_approximation_multigpucpu_chunks_sparse " << perman << " in " << (end - start) << endl;
    } 
    else {
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
    } 
  }
  */
}//RunAlgo2()
  

/*
void RunPermanForGridGraphs(int m, int n, int perman_algo, bool gpu, bool cpu, int gpu_num, int threads, int number_of_times, int scale_intervals, int scale_times) {
  int *mat, *cptrs, *rows, *rptrs, *cols;
  int nov = m * n / 2;
  int nnz = gridGraph2compressed(m, n, mat, cptrs, rows, rptrs, cols);
  if (nnz == -1) {
    delete[] mat;
    delete[] cptrs;
    delete[] rows;
    delete[] rptrs;
    delete[] cols;
    return;
  }
  double start, end, perman;
  if (gpu) {
    if (perman_algo == 1) { // rasmussen
      start = omp_get_wtime();
      perman = gpu_perman64_rasmussen_sparse(rptrs, cols, nov, nnz, number_of_times, true);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_rasmussen_sparse %2lf in %lf\n", perman, end-start);
      cout << "Try: gpu_perman64_rasmussen_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 2) { // approximation_with_scaling
      start = omp_get_wtime();
      perman = gpu_perman64_approximation_sparse(cptrs, rows, rptrs, cols, nov, nnz, number_of_times, scale_intervals, scale_times, true);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_approximation_sparse %2lf in %lf\n", perman, end-start);
      cout << "Try: gpu_perman64_approximation_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 3) { // rasmussen
      start = omp_get_wtime();
      perman = gpu_perman64_rasmussen_multigpucpu_chunks_sparse(cptrs, rows, rptrs, cols, nov, nnz, number_of_times, gpu_num, cpu, threads, true);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_rasmussen_multigpucpu_chunks %2lf in %lf\n", perman, end-start);
      cout << "Try: gpu_perman64_rasmussen_multigpucpu_chunks " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 4) { // approximation_with_scaling
      start = omp_get_wtime();
      perman = gpu_perman64_approximation_multigpucpu_chunks_sparse(cptrs, rows, rptrs, cols, nov, nnz, number_of_times, gpu_num, cpu, scale_intervals, scale_times, threads, true);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_approximation_multigpucpu_chunks_sparse %2lf in %lf\n", perman, end-start);
      cout << "Try: gpu_perman64_approximation_multigpucpu_chunks_sparse " << perman << " in " << (end - start) << endl;
    } else {
      cout << "Unknown Algorithm ID" << endl;
    }
  } else if (cpu) {
    if (perman_algo == 1) { // rasmussen
      start = omp_get_wtime();
      perman = rasmussen_sparse(mat, rptrs, cols, nov, number_of_times, threads);
      end = omp_get_wtime();
      printf("Result: rasmussen_sparse %2lf in %lf\n", perman, end-start);
      cout << "Try: rasmussen_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 2) { // approximation_with_scaling
      start = omp_get_wtime();
      perman = approximation_perman64_sparse(cptrs, rows, rptrs, cols, nov, number_of_times, scale_intervals, scale_times, threads);
      end = omp_get_wtime();
      printf("Result: approximation_perman64_sparse %2lf in %lf\n", perman, end-start);
      cout << "Try: approximation_perman64_sparse " << perman << " in " << (end - start) << endl;
    } else {
      cout << "Unknown Algorithm ID" << endl;
    } 
  }
  delete[] mat;
  delete[] cptrs;
  delete[] rows;
  delete[] rptrs;
  delete[] cols;
}
*/

int main (int argc, char **argv)
{ 
  bool generic = true;
  bool dense = true;
  bool approximation = false;
  bool gpu = false;
  bool cpu = false;
  int gpu_num = 2;
  int threads = 16;
  //string filename;
  int perman_algo = 1;
  int preprocessing = 0;

  int number_of_times = 100000;
  int scale_intervals = 4;
  int scale_times = 5;

  bool grid_graph = false;
  int gridm = 36;
  int gridn = 36;

  // We need to handle this part with a constructor

  flags flags;
  
  /* A string listing valid short options letters.  */
  const char* const short_options = "bsr:t:f:gd:cap:x:y:z:im:n:";
  /* An array describing valid long options.  */
  const struct option long_options[] = {
    { "binary",     0, NULL, 'b' },
    { "sparse",     0, NULL, 's' },
    { "preprocessing",   1, NULL, 'r' },
    { "threads",  1, NULL, 't' },
    { "file",  1, NULL, 'f' },
    { "gpu",  0, NULL, 'g' },
    { "device",  1, NULL, 'd' },
    { "cpu",  0, NULL, 'c' },
    { "approximation",  0, NULL, 'a' },
    { "perman",  1, NULL, 'p' },
    { "numOfTimes",  1, NULL, 'x' },
    { "scaleIntervals",  1, NULL, 'y' },
    { "scaleTimes",  1, NULL, 'z' },
    { "grid",  0, NULL, 'i' },
    { "gridm",  1, NULL, 'm' },
    { "gridn",  1, NULL, 'n' },
    { NULL,       0, NULL, 0   }   /* Required at end of array.  */
  };
  
  
  std::string holder;
  int next_option;
  do {
    next_option = getopt_long(argc, argv, short_options, long_options, NULL);
    switch (next_option)
      {
      case 'b':
        generic = false;
        break;
      case 's':
	flags.dense = 0;
	flags.sparse = 1;
        break;
      case 'r':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -t requires an argument.\n");
          return 1;
        }
	flags.preprocessing = atoi(optarg);
	break;
      case 't':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -t requires an argument.\n");
          return 1;
        }
        flags.threads = atoi(optarg);
        break;
      case 'f':
	if (optarg[0] == '-'){
	  fprintf (stderr, "Option -f requires an argument.\n");
          return 1;
        }
	holder = optarg;
        flags.filename = holder.c_str();
	break;
      case 'a':
        flags.approximation = true;
        break;
      case 'g':
	flags.gpu = 1;
	flags.cpu = 0;
        break;
      case 'd':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -d requires an argument.\n");
          return 1;
        }
        flags.gpu_num = atoi(optarg);
        break;
      case 'c':
        flags.cpu = 1;
	flags.gpu = 0;
        break;
      case 'p':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -p requires an argument.\n");
          return 1;
        }
        flags.perman_algo = atoi(optarg);
        break;
      case 'x':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -x requires an argument.\n");
          return 1;
        }
        flags.number_of_times = atoi(optarg);
        break;
      case 'y':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -y requires an argument.\n");
          return 1;
        }
        flags.scale_intervals = atoi(optarg);
        break;
      case 'z':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -z requires an argument.\n");
          return 1;
        }
        flags.scale_times = atoi(optarg);
        break;
      case 'i':
        flags.grid_graph = true;
        break;
      case 'm':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -m requires an argument.\n");
          return 1;
        }
        flags.gridm = atoi(optarg);
        break;
      case 'n':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -n requires an argument.\n");
          return 1;
        }
        flags.gridn = atoi(optarg);
        break;
      case '?':
        return 1;
      case -1:    /* Done with options.  */
        break;
      default:
        abort ();
      }
    
  } while (next_option != -1);
  
  
  if (!grid_graph && flags.filename == "") {
    fprintf (stderr, "Option -f is a required argument.\n");
    return 1;
  }
  
  for (int index = optind; index < argc; index++)
    {
      printf ("Non-option argument %s\n", argv[index]);
    }
  
  if (!flags.cpu && !flags.gpu) {
    gpu = true;
  }
  
  if (flags.grid_graph) {
    std::cout << "Grid graphs are out of support for a limited time, exiting.. " << std::endl;
    exit(1);
    //RunPermanForGridGraphs(gridm, gridn, perman_algo, gpu, cpu, gpu_num, threads, number_of_times, scale_intervals, scale_times);
    return 0;
  }
  
  int nov, nnz;
  string type;


  std::string fname = &flags.filename[0];
  //This is to have filename in the struct, but ifstream don't like 
  //char*, so.
  //Type also goes same.
  //The reason they are being char* is they are also included in .cu
  //files
  ifstream inFile(fname);
  string line;
  getline(inFile, line);
  std::cout << "line: " << line << std::endl;
  istringstream iss(line);
  iss >> nov >> nnz >> type;
  flags.type = type.c_str(); 

  if (type == "int") {
    int* mat = new int[nov*nov];
    ReadMatrix(mat, inFile, nov, generic);
    //printMatrix()?
    
    int *cvals, *rvals;
    int *cptrs, *rows, *rptrs, *cols;
    if (flags.preprocessing == 1) {
      matrix2compressed_sortOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    } else if (flags.preprocessing == 2) {
      matrix2compressed_skipOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    } else {
      matrix2compressed(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    }
    
    RunAlgo(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz, flags);
    
    delete[] mat;
    delete[] cptrs;
    delete[] rows;
    delete[] cvals;
    delete[] rptrs;
    delete[] cols;
    delete[] rvals;
    
  } else if (flags.type == "float") {
    float* mat = new float[nov*nov];
    ReadMatrix(mat, inFile, nov, generic);
    //for (int i = 0; i < nov; i++) {
    //for(int j = 0; j < nov; j++) {
    //if (mat[i*nov+j] == 0) {
    //cout << "0.0 ";
    //} else {
    //cout << to_string(mat[i*nov+j]).substr(0,3) << " ";
    //}
    //}
    //cout << endl;
    //}

    float *cvals, *rvals;
    int *cptrs, *rows, *rptrs, *cols;
    if (preprocessing == 1) {
      matrix2compressed_sortOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    } else if (preprocessing == 2) {
      matrix2compressed_skipOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    } else {
      matrix2compressed(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    }
    RunAlgo(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz, flags);

    delete[] mat;
    delete[] cptrs;
    delete[] rows;
    delete[] cvals;
    delete[] rptrs;
    delete[] cols;
    delete[] rvals;

  } else if (flags.type == "double") {
    double* mat = new double[nov*nov];
    ReadMatrix(mat, inFile, nov, generic);
    //for (int i = 0; i < nov; i++) {
    //for(int j = 0; j < nov; j++) {
    //if (mat[i*nov+j] == 0) {
    //cout << "0.0 ";
    //} else {
    //cout << to_string(mat[i*nov+j]).substr(0,3) << " ";
    //}
    //}
    //cout << endl;
    //}

    double *cvals, *rvals;
    int *cptrs, *rows, *rptrs, *cols;
    if (preprocessing == 1) {
      matrix2compressed_sortOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    } else if (preprocessing == 2) {
      matrix2compressed_skipOrder(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    } else {
      matrix2compressed(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz);
    }

    RunAlgo(mat, cptrs, rows, cvals, rptrs, cols, rvals, nov, nnz, flags);

    delete[] mat;
    delete[] cptrs;
    delete[] rows;
    delete[] cvals;
    delete[] rptrs;
    delete[] cols;
    delete[] rvals;

  }

  return 0;
}
