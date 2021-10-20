#include <iostream>
#include <string>
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <omp.h>
#include <stdio.h>
#include <getopt.h>
#include "util.h" //evaluate_data_return_parameters() --> To be implemented

#ifndef ONLYCPU
#include "gpu_wrappers.h" //All GPU wrappers will be stored there to simplify things
#else
#include "flags.h"
#endif

//Functions from following files are called from this file
//#include "gpu_exact_sparse.cu"
//#include "gpu_approximation_dense.cu"
//#include "gpu_approximation_sparse.cu"
//Excluding GPU algos for a minimal start
//
#include "cpu_algos.hpp"
//
#include <math.h>
//
#include "read_matrix.hpp"
//
#ifndef ONLYCPU
#include "mmio.c" //This is erroneous but works for now
#else
#include "mmio.h"
#endif

#define DEBUG

using namespace std;


void print_flags(flags flags){

  std::cout << "*~~~~~~~~~~~~FLAGS~~~~~~~~~~~~*" << std::endl;
  std::cout << "- cpu: " << flags.cpu << std::endl;
  std::cout << "- gpu: " << flags.gpu << std::endl;
  std::cout << "- sparse: " << flags.sparse << std::endl;
  std::cout << "- dense: " << flags.dense << std::endl;
  std::cout << "- exact: " << flags.exact << std::endl;
  std::cout << "- approximation: " << flags.approximation << std::endl;
  std::cout << "- half-precision: " << flags.half_precision << std::endl;
  std::cout << "- binary graph: " << flags.binary_graph << std::endl;
  std::cout << "- grid_graph: " << flags.grid_graph << std::endl;
  std::cout << "- gridm: " << flags.gridm << std::endl;
  std::cout << "- gridn: " << flags.gridn << std::endl;
  std::cout << "- perman_algo: " << flags.perman_algo << std::endl;
  std::cout << "- threads: " << flags.threads << std::endl;
  std::cout << "- scale_intervals: " << flags.scale_intervals << std::endl;
  std::cout << "- scale_times: " << flags.scale_times << std::endl;
  printf("- fname: %s \n", flags.filename);
  std::cout << "- type: " << flags.type << std::endl;
  std::cout << "- preprocessing: " << flags.preprocessing << std::endl;
  std::cout << "- gpu_num: " << flags.gpu_num << std::endl;
  std::cout << "- number_of_times: " << flags.number_of_times << std::endl;
  std::cout << "- grid_dim: " << flags.grid_dim << std::endl;
  std::cout << "- block_dim: " << flags.block_dim << std::endl;
  std::cout << "*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*" << std::endl;
  //
}


template <class T>
void RunAlgo(DenseMatrix<T>* densemat, SparseMatrix<T>* sparsemat, flags flags) 
{

  print_flags(flags);

  int grid_dim = 2048;
  int block_dim = 256;
  
  if(std::string(flags.type) == std::string("double")){
    std::cout << "==SC== Resetted block_dim to 128 due to double entries.. " << std::endl;
    block_dim = 128;
  }

  //Pack flags
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
  //Pack flags
  
  double start, end, perman;
  
  if(cpu && dense && exact){    
    
    if(perman_algo == 1){
#ifdef DEBUG
      cout << "Calling, parallel_perman64()" << endl;
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      start = omp_get_wtime();
      perman = parallel_perman64(densemat, flags);
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
#ifdef DEBUG
      cout << "Calling, parallel_perman64_sparse()" << endl;
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      start = omp_get_wtime();
      perman = parallel_perman64_sparse(densemat, sparsemat, flags);
      end = omp_get_wtime();
      cout << "Result: parallel_perman64_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 2) {
#ifdef DEBUG
      cout << "Calling, parallel_skip_perman64_w()" << endl;
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      start = omp_get_wtime();
      perman = parallel_skip_perman64_w(sparsemat, flags);
      end = omp_get_wtime();
      cout << "Result: parallel_skip_perman64_w " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 3) {
#ifdef DEBUG
      cout << "Calling, parallel_skip_perman64_w_balanced()" << endl;
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = parallel_skip_perman64_w_balanced(sparsemat, flags);
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
#ifdef DEBUG
      printf("Calling, rasmussen() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      start = omp_get_wtime();
      perman = rasmussen(densemat, flags);
      end = omp_get_wtime();
      printf("Result: rasmussen %2lf in %lf\n", perman, end-start);
      //cout << "Result: rasmussen " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 2) { // approximation_with_scaling
#ifdef DEBUG
      printf("Calling, approximation_perman64() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      start = omp_get_wtime();
      perman = approximation_perman64(densemat, flags);
      end = omp_get_wtime();
      printf("Result: approximation_perman64 %2lf in %lf\n", perman, end-start);
      //cout << "Result: approximation_perman64 " << perman << " in " << (end - start) << endl;
    }
    
    else{
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }
    
  }
  
  
  if(cpu && sparse && approximation){
    
    if (perman_algo == 1) { // rasmussen
#ifdef DEBUG
      printf("Calling, rasmussen_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = rasmussen_sparse(densemat, sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: rasmussen_sparse %2lf in %lf\n", perman, end-start);
      //cout << "Result: rasmussen_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 2) { // approximation_with_scaling
#ifdef DEBUG
      printf("Calling, approximation_perman64_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = approximation_perman64_sparse(sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: approximation_perman64_sparse %2lf in %lf\n", perman, end-start);
      //cout << "Result: approximation_perman64_sparse " << perman << " in " << (end - start) << endl;
    }
    
    else{
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }

  }


#ifndef ONLYCPU
  if(gpu && dense && exact){
    
    if (perman_algo == 21) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xglobal() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_xglobal(densemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xglobal " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 1) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xlocal() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_xlocal(densemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xlocal " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 2) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_xshared(densemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 3) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing(densemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 4) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared(densemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing_mshared " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 5) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpu() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared_multigpu(densemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing_mshared_multigpu " << perman << " in " << (end - start) << endl;
    } 
    else if (perman_algo == 6) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpu_manual_distribution() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.gpu_num = 4;
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared_multigpu_manual_distribution(densemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_xshared_coalescing_mshared_multigpu_manual_distribution %2lf in %lf\n", perman, end-start);
    }
    
    else if (perman_algo == 6) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks(densemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks " << perman << " in " << (end - start) << endl;
    }
    
    
    else{
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }
  }
  
  if(gpu && sparse && exact){
    
    if (perman_algo == 1) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xlocal_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_xlocal_sparse(densemat, sparsemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xlocal_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 2) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_sparse(densemat, sparsemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 3) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_sparse(densemat, sparsemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 4) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared_sparse(densemat, sparsemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing_mshared_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 5) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpu_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared_multigpu_sparse(densemat, sparsemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing_mshared_multigpu_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 7) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse(densemat, sparsemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse " << perman << " in " << (end - start) << endl;
    }
    else if (perman_algo == 14){
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_skipper() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared_skipper(densemat, sparsemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing_mshared_skipper " << perman << " in " << end - start << endl;
    }
    else if (perman_algo == 17){
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpu_chunks_skipper() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_skipper(densemat, sparsemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_skipper " << perman << " in " << end - start << endl;
    }
    else if (perman_algo == 6) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpu_sparse_manual_distribution \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      flags.gpu_num = 4; //This is a manual setting specialized for GPUs we have, so recommend not to use it. 
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared_multigpu_sparse_manual_distribution(densemat, sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_xshared_coalescing_mshared_multigpu_manual_distribution %2lf in %lf\n", perman, end-start);
      
    }else{
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }
  }
  
  
  if(gpu && dense && approximation){
      
    if (perman_algo == 1) { // rasmussen
#ifdef DEBUG
      printf("Calling, gpu_perman64_rasmussen() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      start = omp_get_wtime();
      perman = gpu_perman64_rasmussen(densemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_rasmussen %2lf in %lf\n", perman, end-start);
      //cout << "Result: gpu_perman64_rasmussen " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 2) { // approximation_with_scaling
#ifdef DEBUG
      printf("Calling, gpu_perman64_approximation() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      start = omp_get_wtime();
      perman = gpu_perman64_approximation(densemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_approximation %2lf in %lf\n", perman, end-start);
      //cout << "Result: gpu_perman64_approximation " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 3) { // rasmussen
#ifdef DEBUG
      printf("Calling, gpu_perman64_rasmussen_multigpucpu_chunks() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_rasmussen_multigpucpu_chunks(densemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_rasmussen_multigpucpu_chunks %2lf in %lf\n", perman, end-start);
      //cout << "Result: gpu_perman64_rasmussen_multigpucpu_chunks " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 4) { // approximation_with_scaling
#ifdef DEBUG
      printf("gpu_perman64_approximation_multigpucpu_chunks() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_approximation_multigpucpu_chunks(densemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_approximation_multigpucpu_chunks %2lf in %lf\n", perman, end-start);
      //cout << "Result: gpu_perman64_approximation_multigpucpu_chunks " << perman << " in " << (end - start) << endl;
    } else {
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    } 
  }
  
  
  if(gpu && sparse && approximation){
    
    if (perman_algo == 1) { // rasmussen
#ifdef DEBUG
      printf("gpu_perman64_rasmussen_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      start = omp_get_wtime();
      perman = gpu_perman64_rasmussen_sparse(sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_rasmussen_sparse %2lf in %lf\n", perman, end-start);
      //cout << "Result: gpu_perman64_rasmussen_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 2) { // approximation_with_scaling
#ifdef DEBUG
      printf("gpu_perman64_approximation_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_approximation_sparse(sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_approximation_sparse %2lf in %lf\n", perman, end-start);
      //cout << "Result: gpu_perman64_approximation_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 3) { // rasmussen
#ifdef DEBUG
      printf("gpu_perman64_rasmussen_multigpucpu_chunks_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_rasmussen_multigpucpu_chunks_sparse(sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_rasmussen_multigpucpu_chunks_sparse %2lf in %lf\n", perman, end-start);
      //cout << "Result: gpu_perman64_rasmussen_multigpucpu_chunks_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 4) { // approximation_with_scaling
#ifdef DEBUG
      printf("gpu_perman64_approximation_multigpucpu_chunks_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_approximation_multigpucpu_chunks_sparse(sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_approximation_multigpucpu_chunks_sparse %2lf in %lf\n", perman, end-start);
      //cout << "Result: gpu_perman64_approximation_multigpucpu_chunks_sparse " << perman << " in " << (end - start) << endl;
    } 
    else {
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
    } 
  }

  if(gpu && cpu && dense && exact){
    if (perman_algo == 6) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks(densemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks " << perman << " in " << (end - start) << endl;
    }
  }

  if (gpu && cpu && sparse && exact) {
    if(perman_algo == 7){
      
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse(densemat, sparsemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse " << perman << " in " << (end - start) << endl;
    }
  }
  else if(perman_algo == 17){
    #ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpu_chunks_skipper() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_skipper(densemat, sparsemat, flags);
      end = omp_get_wtime();
      cout << "Result: gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_skipper " << perman << " in " << end - start << endl;
  }

  if(gpu && cpu && dense && approximation){
    if (perman_algo == 3) { // rasmussen
#ifdef DEBUG
      printf("Calling, gpu_perman64_rasmussen_multigpucpu_chunks() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_rasmussen_multigpucpu_chunks(densemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_rasmussen_multigpucpu_chunks %2lf in %lf\n", perman, end-start);
      //cout << "Result: gpu_perman64_rasmussen_multigpucpu_chunks " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 4) { // approximation_with_scaling
#ifdef DEBUG
      printf("gpu_perman64_approximation_multigpucpu_chunks() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_approximation_multigpucpu_chunks(densemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_approximation_multigpucpu_chunks %2lf in %lf\n", perman, end-start);
      //cout << "Result: gpu_perman64_approximation_multigpucpu_chunks " << perman << " in " << (end - start) << endl;
    }
  }
  
  if(gpu && cpu && sparse && approximation){
    if (perman_algo == 3) { // rasmussen
#ifdef DEBUG
      printf("gpu_perman64_rasmussen_multigpucpu_chunks_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_rasmussen_multigpucpu_chunks_sparse(sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_rasmussen_multigpucpu_chunks_sparse %2lf in %lf\n", perman, end-start);
      //cout << "Result: gpu_perman64_rasmussen_multigpucpu_chunks_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 4) { // approximation_with_scaling
#ifdef DEBUG
      printf("gpu_perman64_approximation_multigpucpu_chunks_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif      
      start = omp_get_wtime();
      perman = gpu_perman64_approximation_multigpucpu_chunks_sparse(sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_approximation_multigpucpu_chunks_sparse %2lf in %lf\n", perman, end-start);
      //cout << "Result: gpu_perman64_approximation_multigpucpu_chunks_sparse " << perman << " in " << (end - start) << endl;
    } 
  }
#endif
}


//RunAlgo2()

//template <class T>
//void RunAlgo(DenseMatrix<T>* densemat, SparseMatrix<T>* sparsemat, flags flags) 

//void RunPermanForGridGraphs(int m, int n, int perman_algo, bool gpu, bool cpu, int gpu_num, int threads, int number_of_times, int scale_intervals, int scale_times) {
void RunPermanForGridGraphs(flags flags) {

  
  print_flags(flags);
  
  //Pack Parameters//
  int m = flags.gridm;
  int n = flags.gridn;
  int perman_algo = flags.perman_algo;
  bool gpu = flags.gpu;
  bool cpu = flags.cpu;
  int gpu_num = flags.gpu_num;
  int threads = flags.threads;
  int number_of_times = flags.number_of_times;
  int scale_intervals = flags.scale_intervals;
  int scale_times = flags.scale_times;
  //Pack parameters

  //Object oriented grids//
  flags.type = "int";
  SparseMatrix<int>* dummy;
  SparseMatrix<int>* sparsemat;
  DenseMatrix<int>* densemat;
  sparsemat = new SparseMatrix<int>();
  dummy = new SparseMatrix<int>();
  densemat = new DenseMatrix<int>(); 
  sparsemat->nov = m;
  dummy->nov = m;
  densemat->nov = m; 
  //Object oriented grids//
  
  int *mat, *cptrs, *rows, *rptrs, *cols;
  int nov = m * n / 2;
  int nnz = gridGraph2compressed(m, n, densemat->mat, dummy->cptrs, dummy->rows, dummy->rptrs, dummy->cols);

  
  if (nnz == -1) {
    delete[] mat;
    delete[] cptrs;
    delete[] rows;
    delete[] rptrs;
    delete[] cols;
    return;
  }

  sparsemat->rvals = new int[nnz];
  sparsemat->cvals = new int[nnz];
  sparsemat->cptrs = new int[m + 1];
  sparsemat->rptrs = new int[m + 1];
  sparsemat->rows = new int[nnz];
  sparsemat->cols = new int[nnz];

  sparsemat->nnz = nnz;
  densemat->nnz = nnz;
  print_densematrix(densemat);
  matrix2compressed_o(densemat, sparsemat);
  cout << "Compressed densemat " << endl;  
  print_sparsematrix(sparsemat);

  delete dummy;
    
  double start, end, perman;
#ifndef ONLYCPU
  if (gpu) {
    if (perman_algo == 1) { // rasmussen
      start = omp_get_wtime();
      perman = gpu_perman64_rasmussen_sparse(sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_rasmussen_sparse %2lf in %lf\n", perman, end-start);
      cout << "Try: gpu_perman64_rasmussen_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 2) { // approximation_with_scaling
      start = omp_get_wtime();
      perman = gpu_perman64_approximation_sparse(sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_approximation_sparse %2lf in %lf\n", perman, end-start);
      cout << "Try: gpu_perman64_approximation_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 3) { // rasmussen
      start = omp_get_wtime();
      perman = gpu_perman64_rasmussen_multigpucpu_chunks_sparse(sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_rasmussen_multigpucpu_chunks %2lf in %lf\n", perman, end-start);
      cout << "Try: gpu_perman64_rasmussen_multigpucpu_chunks " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 4) { // approximation_with_scaling
      start = omp_get_wtime();
      perman = gpu_perman64_approximation_multigpucpu_chunks_sparse(sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: gpu_perman64_approximation_multigpucpu_chunks_sparse %2lf in %lf\n", perman, end-start);
      cout << "Try: gpu_perman64_approximation_multigpucpu_chunks_sparse " << perman << " in " << (end - start) << endl;
    } else {
      cout << "Unknown Algorithm ID" << endl;
    }
  } else if (cpu) {
    if (perman_algo == 1) { // rasmussen
      start = omp_get_wtime();
      perman = rasmussen_sparse(densemat, sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: rasmussen_sparse %2lf in %lf\n", perman, end-start);
      cout << "Try: rasmussen_sparse " << perman << " in " << (end - start) << endl;
    } else if (perman_algo == 2) { // approximation_with_scaling
      start = omp_get_wtime();
      perman = approximation_perman64_sparse(sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: approximation_perman64_sparse %2lf in %lf\n", perman, end-start);
      cout << "Try: approximation_perman64_sparse " << perman << " in " << (end - start) << endl;
    } else {
      cout << "Unknown Algorithm ID" << endl;
    } 
  }
#else
  if (perman_algo == 1) { // rasmussen
    start = omp_get_wtime();
    perman = rasmussen_sparse(densemat, sparsemat, flags);
    end = omp_get_wtime();
    printf("Result: rasmussen_sparse %2lf in %lf\n", perman, end-start);
    cout << "Try: rasmussen_sparse " << perman << " in " << (end - start) << endl;
  } else if (perman_algo == 2) { // approximation_with_scaling
    start = omp_get_wtime();
    perman = approximation_perman64_sparse(sparsemat, flags);
    end = omp_get_wtime();
    printf("Result: approximation_perman64_sparse %2lf in %lf\n", perman, end-start);
    cout << "Try: approximation_perman64_sparse " << perman << " in " << (end - start) << endl;
  } else {
    cout << "Unknown Algorithm ID" << endl;
  } 
#endif
  
  delete[] mat;
  delete[] cptrs;
  delete[] rows;
  delete[] rptrs;
  delete[] cols;
  
}


int main (int argc, char **argv)
{ 
  bool generic = true;
  bool dense = true;
  bool approximation = false;
  bool half_precision = false;
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
  const char* const short_options = "bsr:t:f:gd:cap:x:y:z:im:n:h";
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
    { "halfprec" , 0, NULL, 'h'},
    { NULL,       0, NULL, 0   }   /* Required at end of array.  */
  };

    
  std::string holder;
  int next_option;
  do {
    next_option = getopt_long(argc, argv, short_options, long_options, NULL);
    switch (next_option)
      {
      case 'b':
        flags.binary_graph = true;
        break;
      case 's':
	flags.dense = 0;
	flags.sparse = 1;
        break;
      case 'r':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -r requires an argument.\n");
          return 1;
        }
	flags.preprocessing = atoi(optarg); //1->sortorder | 2->skiporder
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
	//flags.gpu = 0; //Prevents hybrid execution
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
      case 'h':
	flags.half_precision = 1;
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
  
  if(flags.grid_graph) {
    //std::cout << "Grid graphs are out of support for a limited time, exiting.. " << std::endl;
    //exit(1);
    RunPermanForGridGraphs(flags);
    return 0;
  }
  
  int nov, nnz;
  string type;

  //std::string fname = &flags.filename[0];
  //This is to have filename in the struct, but ifstream don't like 
  //char*, so.
  //Type also goes same.
  //The reason they are being char* is they are also included in .cu
  //files
  
  FILE* f;
  int ret_code;
  MM_typecode matcode;
  int M, N, nz;
  int i;
  int *I, *J;
  
  if((f = fopen(flags.filename, "r")) == NULL){
    printf("Error opening the file, exiting.. \n");
    exit(1);
  }
  
  if(mm_read_banner(f, &matcode) != 0){
    printf("Could not process Matrix Market Banner, exiting.. \n");
    exit(1);
  }
  
  if(mm_is_matrix(matcode) != 1){
    printf("SUPerman only supports matrices, exiting.. \n");
    exit(1);
  }

  if(mm_is_coordinate(matcode) != 1){
    printf("SUPerman only supports mtx format at the moment, exiting.. \n");
    exit(1);
  }
  
  if((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) != 0){
    printf("Matrix size cannot be read, exiting.. \n");
  }

#ifdef DEBUG
  std::cout << "M: " << M << " N: " << N << " nz: " << nz << std::endl;
#endif
  nov = M;
  nnz = nz;

  if(M != N){
    printf("SUPerman only works with nxn matrices, exiting.. ");
    exit(1);
  }
  
  if(mm_is_complex(matcode) == 1){
    printf("SUPerman does not support complex type, exiting.. ");
    exit(1);
    //Instead of exit(1), there should be an escape function
    //which frees the allocated memory
  }
  
  bool is_pattern = false;
  if(mm_is_pattern(matcode) == 1)
    is_pattern = true;
  
  if(flags.binary_graph)
    is_pattern = true;

  bool is_symmetric = false;
  if(mm_is_symmetric(matcode) == 1 || mm_is_skew(matcode))
    is_symmetric = true;

  if(is_symmetric)
  nz *= 2;

 

  if(mm_is_real(matcode) == 1 && !flags.half_precision && !is_pattern){
#ifdef DEBUG
    std::cout << "Read Case: 0" << std::endl;
#endif
    flags.type = "double";
    SparseMatrix<double>* sparsemat;
    DenseMatrix<double>* densemat;
    sparsemat = new SparseMatrix<double>();
    densemat = new DenseMatrix<double>(); 
    densemat->mat = new double[M*N];
    sparsemat->rvals = new double[nz];
    sparsemat->cvals = new double[nz];
    sparsemat->cptrs = new int[nov + 1];
    sparsemat->rptrs = new int[nov + 1];
    sparsemat->rows = new int[nz];
    sparsemat->cols = new int[nz];
    sparsemat->nov = M; 
    densemat->nov = M; 
    sparsemat->nnz = nz;
    densemat->nnz = nz;

    if(!is_symmetric)
      readDenseMatrix(densemat, flags.filename, is_pattern);
    else 
      readSymmetricDenseMatrix(densemat, flags.filename, is_pattern);
    
#ifdef DEBUG
    std::cout << "Read.. OK! -- Compressing.. " << std::endl;
#endif
    if(flags.preprocessing == 0)
      matrix2compressed_o(densemat, sparsemat); 
    if(flags.preprocessing == 1)
      matrix2compressed_sortOrder_o(densemat, sparsemat); 
    if(flags.preprocessing == 2)
      matrix2compressed_skipOrder_o(densemat, sparsemat);
    
#ifdef DEBUG
    std::cout << "Compression..OK!" << std::endl;

    print_sparsematrix(sparsemat);
    print_densematrix(densemat);
#endif

    RunAlgo(densemat, sparsemat, flags);
  }
  
  else if(mm_is_real(matcode) == 1 && flags.half_precision && !is_pattern){
#ifdef DEBUG
    std::cout << "Read Case: 1" << std::endl;
#endif
    flags.type = "float";
    SparseMatrix<float>* sparsemat;
    DenseMatrix<float>* densemat;
    sparsemat = new SparseMatrix<float>();
    densemat = new DenseMatrix<float>(); 
    densemat->mat = new float[M*N];
    sparsemat->rvals = new float[nz];
    sparsemat->cvals = new float[nz];
    sparsemat->cptrs = new int[nov + 1];
    sparsemat->rptrs = new int[nov + 1];
    sparsemat->rows = new int[nz];
    sparsemat->cols = new int[nz];
    sparsemat->nov = M; 
    densemat->nov = M; 
    sparsemat->nnz = nz;
    densemat->nnz = nz;
    
    if(!is_symmetric)
      readDenseMatrix(densemat, flags.filename, is_pattern);
    else 
      readSymmetricDenseMatrix(densemat, flags.filename, is_pattern);
    
#ifdef DEBUG
    std::cout << "Read.. OK! -- Compressing.. " << std::endl;
#endif
    if(flags.preprocessing == 0)
      matrix2compressed_o(densemat, sparsemat); 
    if(flags.preprocessing == 1)
      matrix2compressed_sortOrder_o(densemat, sparsemat); 
    if(flags.preprocessing == 2)
      matrix2compressed_skipOrder_o(densemat, sparsemat);
#ifdef DEBUG
    std::cout << "Compression..OK!" << std::endl;

    print_sparsematrix(sparsemat);
    print_densematrix(densemat);
#endif
    
    RunAlgo(densemat, sparsemat, flags);    
  }
  
  else if(mm_is_integer(matcode) == 1 || is_pattern){
#ifdef DEBUG
    std::cout << "Read Case: 2" << std::endl;
#endif
    flags.type = "int";
    SparseMatrix<int>* sparsemat;
    DenseMatrix<int>* densemat;
    sparsemat = new SparseMatrix<int>();
    densemat = new DenseMatrix<int>();
    densemat->mat = new int[M*M];
    sparsemat->rvals = new int[nz];
    sparsemat->cvals = new int[nz];
    sparsemat->cptrs = new int[nov + 1];
    sparsemat->rptrs = new int[nov + 1];
    sparsemat->rows = new int[nz];
    sparsemat->cols = new int[nz];
    sparsemat->nov = M; 
    densemat->nov = M; 
    sparsemat->nnz = nz;
    densemat->nnz = nz;    
    
    if(!is_symmetric)
      readDenseMatrix(densemat, flags.filename, is_pattern);
    else 
      readSymmetricDenseMatrix(densemat, flags.filename, is_pattern);


#ifdef DEBUG
    std::cout << "Read.. OK! -- Compressing.. " << std::endl;
#endif
    if(flags.preprocessing == 0)
      matrix2compressed_o(densemat, sparsemat); 
    if(flags.preprocessing == 1)
      matrix2compressed_sortOrder_o(densemat, sparsemat); 
    if(flags.preprocessing == 2)
      matrix2compressed_skipOrder_o(densemat, sparsemat);
#ifdef DEBUG
    std::cout << "Compression..OK!" << std::endl;

    print_sparsematrix(sparsemat);
    print_densematrix(densemat);
#endif

    RunAlgo(densemat, sparsemat, flags);
  }
  
  else{
    std::cout << "Matrix or flags have overlapping features.. " <<std::endl;
    print_flags(flags);
    exit(1);
  }
  
  return 0;
}
