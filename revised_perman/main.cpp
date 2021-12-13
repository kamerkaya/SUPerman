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

#include <cfenv>

//#define HEAVYDEBUG

using namespace std;

int recursive_count = 0;

template<class S>
bool max20(DenseMatrix<S>* densemat){

  int nov = densemat->nov;
  

  for(int i = 0; i < nov*nov; i++){
    if(densemat->mat[i] > (S)20)
      return true;
    }

  return false;
}

template<class S>
Result scale_and_calculate(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags &flags, bool compressing);

void print_flags(flags flags){
  //
  std::cout << "*~~~~~~~~~~~~FLAGS~~~~~~~~~~~~*" << std::endl;
  std::cout << "- cpu: " << flags.cpu << std::endl;
  std::cout << "- gpu: " << flags.gpu << std::endl;
  std::cout << "- sparse: " << flags.sparse << std::endl;
  std::cout << "- dense: " << flags.dense << std::endl;
  std::cout << "- exact: " << flags.exact << std::endl;
  std::cout << "- approximation: " << flags.approximation << std::endl;
  std::cout << "- calculation half-precision: " << flags.calculation_half_precision << std::endl;
  std::cout << "- calculation quad-precision: " << flags.calculation_quad_precision << std::endl;
  std::cout << "- storage half-precision: " << flags.storage_half_precision << std::endl;
  std::cout << "- storage quad-precision: " << flags.storage_quad_precision << std::endl;
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
  std::cout << "- no rep.: " << flags.rep << std::endl;
  std::cout << "- preprocessing: " << flags.preprocessing << std::endl;
  std::cout << "- gpu_num: " << flags.gpu_num << std::endl;
  std::cout << "- number_of_times: " << flags.number_of_times << std::endl;
  std::cout << "- grid_dim: " << flags.grid_dim << std::endl;
  std::cout << "- block_dim: " << flags.block_dim << std::endl;
  std::cout << "- device_id: " << flags.device_id << std::endl;
  std::cout << "- grid_multip: " << flags.grid_multip << std::endl;
  std::cout << "- compression: " << flags.compression << std::endl;
  std::cout << "- scaling_threshold: " << flags.scaling_threshold << std::endl;
  std::cout << "*~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*" << std::endl;
  //
}


template <class S>
Result RunAlgo(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags &flags, bool supress) 
{

  int grid_dim = 2048; //For any case, if it's failed to determined by CUDA
  int block_dim = 256;
  
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
  int no_repetition = flags.rep;
  //Pack flags
  
  double start, end, perman;
  
  Result result;
  
  if(cpu && dense && exact && !gpu){    
    
    if(perman_algo == 1){
#ifdef DEBUG
      cout << "Calling, parallel_perman64()" << endl;
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "parallel_perman";
      if(flags.calculation_half_precision)
	result = parallel_perman64<float, S>(densemat, flags);
      else if(flags.calculation_quad_precision)
	result = parallel_perman64<__float128, S>(densemat, flags);
      else
	result = parallel_perman64<double, S>(densemat, flags);	
    }
    else{
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }
  }
  
  if(cpu && sparse && exact && !gpu){
    
    if (perman_algo == 1) {
#ifdef DEBUG
      cout << "Calling, parallel_perman64_sparse()" << endl;
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "parallel_perman_sparse";
      if(flags.calculation_half_precision)
	result = parallel_perman64_sparse<float, S>(densemat, sparsemat, flags);
      else if(flags.calculation_quad_precision)
	result = parallel_perman64_sparse<__float128, S>(densemat, sparsemat, flags);
      else
	result = parallel_perman64_sparse<double, S>(densemat, sparsemat, flags); 
    }
    else if (perman_algo == 2) {
#ifdef DEBUG
      cout << "Calling, parallel_skip_perman64_w()" << endl;
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "parallel_skip_perman";

      if(flags.calculation_half_precision)
	result = parallel_skip_perman64_w<float, S>(sparsemat, flags);
      else if(flags.calculation_quad_precision)
	result = parallel_skip_perman64_w<__float128, S>(sparsemat, flags);
      else
	result = parallel_skip_perman64_w<double, S>(sparsemat, flags);
    }
    else if (perman_algo == 3) {
#ifdef DEBUG
      cout << "Calling, parallel_skip_perman64_w_balanced()" << endl;
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "parallel_skip_perman_balanced";
      if(flags.calculation_half_precision)
	result = parallel_skip_perman64_w_balanced<float, S>(sparsemat, flags);
      else if(flags.calculation_quad_precision)
	result = parallel_skip_perman64_w_balanced<__float128, S>(sparsemat, flags);
      else
	result = parallel_skip_perman64_w_balanced<double, S>(sparsemat, flags);
    }
    else
      {
	std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
	exit(1);
      }
  }
  
  if(cpu && dense && approximation && !gpu){
    
    if (perman_algo == 1) { // rasmussen
#ifdef DEBUG
      printf("Calling, rasmussen() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "rasmussen";
      if(flags.calculation_half_precision)
	result = rasmussen<float, S>(densemat, flags);
      else if(flags.calculation_quad_precision)
	result = rasmussen<__float128, S>(densemat, flags);
      else
	result = rasmussen<double, S>(densemat, flags);
    }
    
    else if (perman_algo == 2) { // approximation_with_scaling
#ifdef DEBUG
      printf("Calling, approximation_perman64() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "approximation";
      if(flags.calculation_half_precision)
	result = approximation_perman64<float, S>(densemat, flags);
      else if(flags.calculation_quad_precision)
	result = approximation_perman64<__float128, S>(densemat, flags);
      else
	result = approximation_perman64<double, S>(densemat, flags); 
    }
    else{
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }  
  }
  
  if(cpu && sparse && approximation && !gpu){
    
    if (perman_algo == 1) { // rasmussen
#ifdef DEBUG
      printf("Calling, rasmussen_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "rasmussen_sparse";
      if(flags.calculation_half_precision)
	result = rasmussen_sparse<float, S>(densemat, sparsemat, flags);
      else if(flags.calculation_quad_precision)
	result = rasmussen_sparse<__float128, S>(densemat, sparsemat, flags);
      else
	result = rasmussen_sparse<double, S>(densemat, sparsemat, flags);
    }
    else if (perman_algo == 2) { // approximation_with_scaling
#ifdef DEBUG
      printf("Calling, approximation_perman64_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "approximation_perman64";
      if(flags.calculation_half_precision)
	result = approximation_perman64_sparse<float, S>(sparsemat, flags);
      else if(flags.calculation_quad_precision)
	result = approximation_perman64_sparse<__float128, S>(sparsemat, flags);
      else
	result = approximation_perman64_sparse<double, S>(sparsemat, flags);
    }
    else{
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }  
  }
  
  
#ifndef ONLYCPU
  if(gpu && dense && exact && !cpu){
    
    if (perman_algo == 21) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xglobal() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xglobal";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xglobal<float, S>(densemat, flags);
      else
	result = gpu_perman64_xglobal<double, S>(densemat, flags);
    }
    else if (perman_algo == 1) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xlocal() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xlocal";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xlocal<float, S>(densemat, flags);
      else
	result = gpu_perman64_xlocal<double, S>(densemat, flags);
      
    }
    else if (perman_algo == 2) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared<float, S>(densemat, flags);
      else
	result = gpu_perman64_xshared<double, S>(densemat, flags);
      
    }
    else if (perman_algo == 3) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing<float, S>(densemat, flags);
      else
	result = gpu_perman64_xshared_coalescing<double, S>(densemat, flags);
    }
    else if (perman_algo == 4) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_mshared<float, S>(densemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_mshared<double, S>(densemat, flags);
      
    }
    else if (perman_algo == 5) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpu() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman64_xshared_mshared_multigpu";
      //Will have it's change
      perman = gpu_perman64_xshared_coalescing_mshared_multigpu(densemat, flags);      
    } 
    else if (perman_algo == 6) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpu_manual_distribution() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_multigpu_manual_distribution";
      flags.gpu_num = 4; //This will change accordingly
      
      perman = gpu_perman64_xshared_coalescing_mshared_multigpu_manual_distribution(densemat, flags);	
    }
    else if (perman_algo == 7) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_multigpucpu_chunks";
      perman = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks(densemat, flags);
    }
    
    else{
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }
  }
  
  if(gpu && sparse && exact && !cpu){
    
    if (perman_algo == 1) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xlocal_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xlocal_sparse";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xlocal_sparse<float, S>(densemat, sparsemat, flags);
      else
	result = gpu_perman64_xlocal_sparse<double, S>(densemat, sparsemat, flags);
    }
    else if (perman_algo == 2) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_sparse";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_sparse<float, S>(densemat, sparsemat, flags);
      else
	result = gpu_perman64_xshared_sparse<double, S>(densemat, sparsemat, flags);
      
    }
    else if (perman_algo == 3) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      
      flags.algo_name = "gpu_perman_xshared_coalescing_sparse";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_sparse<float, S>(densemat, sparsemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_sparse<double, S>(densemat, sparsemat, flags);
      
      
    }
    else if (perman_algo == 4) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_sparse";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_mshared_sparse<float, S>(densemat, sparsemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_mshared_sparse<double, S>(densemat, sparsemat, flags);
      
    }
    else if (perman_algo == 5) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpu_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_multigpu_sparse";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_mshared_multigpu_sparse<float, S>(densemat, sparsemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_mshared_multigpu_sparse<double, S>(densemat, sparsemat, flags);
    }
    else if (perman_algo == 7) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_multigpucpu_chunks_sparse";
      perman = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse(densemat, sparsemat, flags);
    }
    else if (perman_algo == 14){
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_skipper() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_skipper";
      if(flags.calculation_half_precision)
	result = gpu_perman64_xshared_coalescing_mshared_skipper<float, S>(densemat, sparsemat, flags);
      else
	result = gpu_perman64_xshared_coalescing_mshared_skipper<double, S>(densemat, sparsemat, flags);
    }
    else if (perman_algo == 17){
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpu_chunks_skipper() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_multigpucpu_chunks_skipper";
      perman = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_skipper(densemat, sparsemat, flags);
    }
    else if (perman_algo == 6) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpu_sparse_manual_distribution \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_multigpu_sparse_manual_distribution";
      flags.gpu_num = 4; //This is a manual setting specialized for GPUs we have, so recommend not to use it.
	perman = gpu_perman64_xshared_coalescing_mshared_multigpu_sparse_manual_distribution(densemat, sparsemat, flags);

    
    }
    else{
    std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    }
  }
  
  
  if(gpu && dense && approximation && !cpu){
    
    if (perman_algo == 1) { // rasmussen
#ifdef DEBUG
      printf("Calling, gpu_perman64_rasmussen() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif

      flags.algo_name = "gpu_rasmussen";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_rasmussen<float, S>(densemat, flags);
      else
	result = gpu_perman64_rasmussen<double, S>(densemat, flags);
      
    }
    else if (perman_algo == 2) { // approximation_with_scaling
#ifdef DEBUG
      printf("Calling, gpu_perman64_approximation() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      
      flags.algo_name = "gpu_approximation";
      if(flags.calculation_half_precision)
	result = gpu_perman64_approximation<float, S>(densemat, flags);
      else
	result = gpu_perman64_approximation<double, S>(densemat, flags);
      
      
    }
    else if (perman_algo == 3) { // rasmussen
#ifdef DEBUG
      printf("Calling, gpu_perman64_rasmussen_multigpucpu_chunks() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif

      //This will change accordingly
      flags.algo_name = "gpu_rasmussen_multigpucpu_chunks";
      perman = gpu_perman64_rasmussen_multigpucpu_chunks(densemat, flags);
      
    
    }
    else if (perman_algo == 4) { // approximation_with_scaling
#ifdef DEBUG
      printf("gpu_perman64_approximation_multigpucpu_chunks() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif

      flags.algo_name = "gpu_approximation_multigpucpu_chunks";
      perman = gpu_perman64_approximation_multigpucpu_chunks(densemat, flags);
      
    }
    else {
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
      exit(1);
    } 
  }
  
  
  if(gpu && sparse && approximation && !cpu){
    
    if (perman_algo == 1) { // rasmussen
#ifdef DEBUG
      printf("gpu_perman64_rasmussen_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif

      flags.algo_name = "gpu_rasmussen_sparse";
      if(flags.calculation_half_precision)
	result = gpu_perman64_rasmussen_sparse<float, S>(sparsemat, flags);
      else
	result = gpu_perman64_rasmussen_sparse<double, S>(sparsemat, flags);
      
      
    }
    else if (perman_algo == 2) { // approximation_with_scaling
#ifdef DEBUG
      printf("gpu_perman64_approximation_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif
      

      flags.algo_name = "gpu_approximation_sparse";
      
      if(flags.calculation_half_precision)
	result = gpu_perman64_approximation_sparse<float, S>(sparsemat, flags);
      else
	result = gpu_perman64_approximation_sparse<double, S>(sparsemat, flags);
      
      
    }
    else if (perman_algo == 3) { // rasmussen
#ifdef DEBUG
      printf("gpu_perman64_rasmussen_multigpucpu_chunks_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif

      flags.algo_name = "gpu_rasmussen_multigpucpu_chunks_sparse";
      //This will change accordingly
      perman = gpu_perman64_rasmussen_multigpucpu_chunks_sparse(sparsemat, flags);
	
    } else if (perman_algo == 4) { // approximation_with_scaling
#ifdef DEBUG
      printf("gpu_perman64_approximation_multigpucpu_chunks_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif

      flags.algo_name = "gpu_approximation_myultigpucpu_chunks_sparse";
      //This will change accordingly
      perman = gpu_perman64_approximation_multigpucpu_chunks_sparse(sparsemat, flags);
	
    } 
    else {
      std::cout << "No algorithm with specified setting, exiting.. " << std::endl;
    } 
  }
  
  if(gpu && cpu && dense && exact){
    if (perman_algo == 7) {
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif

      flags.algo_name = "gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks";
      //This will change accordingly
      perman = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks(densemat, flags);
	
      
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

      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_multigpucpu_chunks_sparse";
      //This will change accordingly
      perman = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_sparse(densemat, sparsemat, flags);

    }
  }
  else if(perman_algo == 17){
#ifdef DEBUG
      printf("Calling, gpu_perman64_xshared_coalescing_mshared_multigpu_chunks_skipper() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif

      flags.algo_name = "gpu_perman_xshared_coalescing_mshared_multigpucpu_chunks_skipper";
      //This will change accordingly
      perman = gpu_perman64_xshared_coalescing_mshared_multigpucpu_chunks_skipper(densemat, sparsemat, flags);
  
  }
  
  if(gpu && cpu && dense && approximation){
    if (perman_algo == 3) { // rasmussen
#ifdef DEBUG
      printf("Calling, gpu_perman64_rasmussen_multigpucpu_chunks() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif

      flags.algo_name = "rasmussen_multigpucpu_chunks";
      perman = gpu_perman64_rasmussen_multigpucpu_chunks(densemat, flags);

    }
    else if (perman_algo == 4) { // approximation_with_scaling
#ifdef DEBUG
      printf("gpu_perman64_approximation_multigpucpu_chunks() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif

      flags.algo_name = "approximation_mulltigpucpu_chunks";
      perman = gpu_perman64_approximation_multigpucpu_chunks(densemat, flags);
	
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

      flags.algo_name = "gpu_perman6_rasmussen_multigpucpu_chunks_sparse";
      perman = gpu_perman64_rasmussen_multigpucpu_chunks_sparse(sparsemat, flags);

    
    }
    else if (perman_algo == 4) { // approximation_with_scaling
#ifdef DEBUG
      printf("gpu_perman64_approximation_multigpucpu_chunks_sparse() \n");
#endif
#ifdef STRUCTURAL
      exit(1);
#endif

      flags.algo_name = "gpu_perman_approximation_multigpucpu_chunks_sparse";
      perman = gpu_perman64_approximation_multigpucpu_chunks_sparse(sparsemat, flags);

      
    }
  }
#endif
  //return perman;
  return result;
}


//RunAlgo2()

//template <class T>
//void RunAlgo(DenseMatrix<T>* densemat, SparseMatrix<T>* sparsemat, flags flags) 

//void RunPermanForGridGraphs(int m, int n, int perman_algo, bool gpu, bool cpu, int gpu_num, int threads, int number_of_times, int scale_intervals, int scale_times) {
void RunPermanForGridGraphs(flags flags) {
  
  print_flags(flags);

  Result result;
  
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
  int no_repetition = flags.rep;
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
      for(int i = 0; i < no_repetition; i++){
	//start = omp_get_wtime();
	if(flags.calculation_half_precision)
	  result = gpu_perman64_rasmussen_sparse<float, int>(sparsemat, flags);
	else
	  result = gpu_perman64_rasmussen_sparse<double, int>(sparsemat, flags);
	printf("Result | gpu_perman64_rasmussen_sparse | %.10e in %f \n", result.permanent, result.time);
      }
    } else if (perman_algo == 2) { // approximation_with_scaling
      for(int i = 0; i < no_repetition; i++){
	start = omp_get_wtime();
	if(flags.calculation_half_precision)
	  result = gpu_perman64_approximation_sparse<float, int>(sparsemat, flags);
	else
	  result = gpu_perman64_approximation_sparse<double, int>(sparsemat, flags);
	end = omp_get_wtime();
	//printf("Result: gpu_perman64_approximation_sparse %2lf in %lf\n", perman, end-start);
	//cout << "Try: gpu_perman64_approximation_sparse " << perman << " in " << (end - start) << endl;
	printf("Result | gpu_perman64_approximation_sparse | %.16e in %f \n", result.permanent, result.time);
      }
    } else if (perman_algo == 3) { // rasmussen
      for(int i = 0; i < no_repetition; i++){
	start = omp_get_wtime();
	perman = gpu_perman64_rasmussen_multigpucpu_chunks_sparse(sparsemat, flags);
	end = omp_get_wtime();
	printf("Result: gpu_perman64_rasmussen_multigpucpu_chunks %2lf in %lf\n", perman, end-start);
	cout << "Try: gpu_perman64_rasmussen_multigpucpu_chunks " << perman << " in " << (end - start) << endl;
      }
    } else if (perman_algo == 4) { // approximation_with_scaling
      for(int i = 0; i < no_repetition; i++){
	start = omp_get_wtime();
	perman = gpu_perman64_approximation_multigpucpu_chunks_sparse(sparsemat, flags);
	end = omp_get_wtime();
	printf("Result: gpu_perman64_approximation_multigpucpu_chunks_sparse %2lf in %lf\n", perman, end-start);
	//cout << "Try: gpu_perman64_approximation_multigpucpu_chunks_sparse " << perman << " in " << (end - start) << endl;
      }
    } else {
      cout << "Unknown Algorithm ID" << endl;
    }
  } else if (cpu) {
    if (perman_algo == 1) { // rasmussen
      for(int i = 0; i < no_repetition; i++){
	start = omp_get_wtime();
	if(flags.calculation_half_precision)
	  result = rasmussen_sparse<float, int>(densemat, sparsemat, flags);
	else
	  result = rasmussen_sparse<double, int>(densemat, sparsemat, flags);
	end = omp_get_wtime();
	printf("Result: rasmussen_sparse %.16e in %lf\n", perman, end-start);
	//cout << "Try: rasmussen_sparse " << perman << " in " << (end - start) << endl;
      }
    } else if (perman_algo == 2) { // approximation_with_scaling
      for(int i = 0; i < no_repetition; i++){
	start = omp_get_wtime();
	if(flags.calculation_half_precision)
	  result = approximation_perman64_sparse<float, int>(sparsemat, flags);
	else
	  result = approximation_perman64_sparse<double, int>(sparsemat, flags);
	end = omp_get_wtime();
	printf("Result: approximation_perman64_sparse %.16e in %lf\n", perman, end-start);
	//cout << "Try: approximation_perman64_sparse " << perman << " in " << (end - start) << endl;
      }
    } else {
      cout << "Unknown Algorithm ID" << endl;
    } 
  }
#else
  if (perman_algo == 1) { // rasmussen
    for(int i = 0; i < no_repetition; i++){
      start = omp_get_wtime();
      if(flags.calculation_half_precision)
	result = rasmussen_sparse<float, int>(densemat, sparsemat, flags);
      else
	result = rasmussen_sparse<double, int>(densemat, sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: rasmussen_sparse %.16e in %lf\n", perman, end-start);
      //cout << "Try: rasmussen_sparse " << perman << " in " << (end - start) << endl;
    }
  } else if (perman_algo == 2) { // approximation_with_scaling
    for(int i = 0; i < no_repetition; i++){
      start = omp_get_wtime();
      if(flags.calculation_half_precision)
	result = approximation_perman64_sparse<float, int>(sparsemat, flags);
      else
	result = approximation_perman64_sparse<double, int>(sparsemat, flags);
      end = omp_get_wtime();
      printf("Result: approximation_perman64_sparse %.16e in %lf\n", perman, end-start);
      //cout << "Try: approximation_perman64_sparse " << perman << " in " << (end - start) << endl;
    }
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

//Looks like we need them:
//crate_densematrix_from_mat2()
//crate_sparsematrix_from_mat2()

template<class S>
int getNnz(S* mat2, int nov2){
  int nnz2 = 0;

  for(int i = 0; i < nov2*nov2; i++){
    if(mat2[i] > (S)0)
      nnz2++;
  }
  
  return nnz2;
}

template<class S>
DenseMatrix<S>* create_densematrix_from_mat2(S* mat2, int nov2){
  
  DenseMatrix<S>* densemat2 = new DenseMatrix<S>();
  densemat2->mat = mat2;
  densemat2->nov = nov2;
  densemat2->nnz = getNnz(mat2, nov2);
  
  return densemat2;
}

template<class S>
SparseMatrix<S>* create_sparsematrix_from_densemat2(DenseMatrix<S>* densemat2, flags flags){

  int nnz = densemat2->nnz;
  int nov = densemat2->nov;
  
  SparseMatrix<S>* sparsemat2 = new SparseMatrix<S>();
  sparsemat2->rvals = new S[nnz];
  sparsemat2->cvals = new S[nnz];
  sparsemat2->cptrs = new int[nov + 1];
  sparsemat2->rptrs = new int[nov + 1];
  sparsemat2->rows = new int[nnz];
  sparsemat2->cols = new int[nnz];
  sparsemat2->nov = nov;
  sparsemat2->nnz = nnz;

  if(flags.preprocessing == 0)
    matrix2compressed_o(densemat2, sparsemat2);
  if(flags.preprocessing == 1)
    matrix2compressed_sortOrder_o(densemat2, sparsemat2);
  if(flags.preprocessing == 2)
    matrix2compressed_skipOrder_o(densemat2, sparsemat2);

  return sparsemat2;
}

template<class S>
Result compress_and_calculate_recursive(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags &flags){

  //cout << "In, compress_and_calculate_recursive-> " << recursive_count++ << endl;
  
  //cout << "densemat->nov: " << densemat->nov << endl;
  //cout << "densemat->nnz: " << densemat->nnz << endl;
  
  bool silent = 0;
  
  //double final_perman = 0;
  Result result;
  int minDeg = getMinNnz(densemat->mat, densemat->nov);
  cout << "##MINDEG: " << minDeg << endl;
  if(minDeg < 5 && densemat->nov > 30){

    if(minDeg == 1){
      d1compress(densemat->mat, densemat->nov);
      if(!silent)
	cout << "d1: matrix is reduced to: " << densemat->nov << " rows" << endl;
      densemat->nnz = getNnz(densemat->mat, densemat->nov);
      delete sparsemat;
      sparsemat = create_sparsematrix_from_densemat2(densemat, flags);
      return compress_and_calculate_recursive(densemat, sparsemat, flags);
    }
    
    else if(minDeg == 2){
      d2compress(densemat->mat, densemat->nov);
      if(!silent)
	cout << "d2: matrix is reduced to: " << densemat->nov << " rows" << endl;
      densemat->nnz = getNnz(densemat->mat, densemat->nov);
      delete sparsemat;
      sparsemat = create_sparsematrix_from_densemat2(densemat, flags);
      return compress_and_calculate_recursive(densemat, sparsemat, flags);
    }
    
    else if(minDeg == 3 || minDeg == 4){
      S* mat2 = nullptr;
      int nov2;
      d34compress(densemat->mat, densemat->nov, mat2, nov2, minDeg);
      DenseMatrix<S>* densemat2 = create_densematrix_from_mat2(mat2, nov2);
      SparseMatrix<S>* sparsemat2 = create_sparsematrix_from_densemat2(densemat2, flags);
      
      if(!silent)
	cout << "d34: matrix is reduced to matrices with: " << densemat->nov << " and " << densemat2->nov <<" rows" << endl;
      
      result = compress_and_calculate_recursive(densemat, sparsemat, flags) +
	compress_and_calculate_recursive(densemat2, sparsemat2, flags);

      if(mat2 != nullptr){
	delete[] mat2;
	mat2 = nullptr;
      }
    }
  }
  else{

    //if(flags.scaling_threshold != -1.0 && !max20(densemat))
    if(flags.scaling_threshold != -1.0)
      result = scale_and_calculate(densemat, sparsemat, flags, true);
    else
      result = RunAlgo(densemat, sparsemat, flags, true);

  }
  return result;
}

template<class S>
Result compress_singleton_and_then_recurse(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags &flags){

  bool comp = true;
  while(comp && densemat->nov > 1){
    comp = d1compress(densemat->mat, densemat->nov);
    if(comp){
      cout << "Removing singleton -- d1: matrix is reduced to: " << densemat->nov << " rows" << endl;
      densemat->nnz = getNnz(densemat->mat, densemat->nov);
      //delete sparsemat; //Possible memory leak, will deal with later
      sparsemat = create_sparsematrix_from_densemat2(densemat, flags);
    }
    else{
      comp = d2compress(densemat->mat, densemat->nov);
      if(comp){
	cout << "Removing singleton -- d2: matrix is reduced to: " << densemat->nov << " rows" << endl;
	densemat->nnz = getNnz(densemat->mat, densemat->nov);
	//delete sparsemat; //Possible memory leak, will deal with later
	sparsemat = create_sparsematrix_from_densemat2(densemat, flags);
      }
    }

    if(comp){
      if(checkEmpty(densemat->mat, densemat->nov)){
	cout << "Matrix is rank deficient!" << endl;
	cout << "Perman is 0" << endl;
	exit(1);
      }
    }
  }
  
  cout << "Singleton compressing is done" << endl;
  return compress_and_calculate_recursive(densemat, sparsemat, flags);
  
}

template<class S>
Result scale_and_calculate(DenseMatrix<S>* densemat, SparseMatrix<S>* sparsemat, flags &flags, bool compressing){

  //Do not delete sparsemat inside of scale_and_calculate because it is already deleted in main
  
    
  //Pack parameters//
  int nov = densemat->nov;
  //Pack parameters//

  Result result;
  
  if(!flags.storage_half_precision && flags.type == "int"){
    
    flags.type = "double";
    
    DenseMatrix<double>* densemat2 = swap_types<S, double>(densemat);
    
#ifdef DEBUG
    std::cout << "Swapped mat type: double" << std::endl;
#endif
    
    delete densemat;
    //delete sparsemat;
    
    SparseMatrix<double>* sparsemat2 = create_sparsematrix_from_densemat2(densemat2, flags);
    ScaleCompanion<double>* sc = scalesk(sparsemat2, flags);
    scaleMatrix(densemat2, sc);
    delete sparsemat2;
    
    SparseMatrix<double>* sparsemat3 = create_sparsematrix_from_densemat2(densemat2, flags);
    
#ifdef HEAVYDEBUG
    std::cout << "Scaled matrix: " << std::endl;
    print_float_densematrix(densemat2);
    print_sparsematrix(sparsemat3);
#endif
    
    if(flags.compression && !compressing)
      result = compress_singleton_and_then_recurse(densemat2, sparsemat2, flags);
    else
      result = RunAlgo(densemat2, sparsemat3, flags, false);
    
#ifdef DEBUG
    printf("Mid Result || %s | %s | %.16e in %f \n", flags.algo_name.c_str(), flags.filename, result.permanent, result.time);
#endif
    
    for(int i = 0; i < nov; i++){
      result.permanent /= sc->c_v[i];
    }
    
    for(int i = 0; i < nov; i++){
      result.permanent /= sc->r_v[i];
    }
    
  }
  
  
  else if(flags.storage_half_precision && flags.type == "int"){
    
    flags.type = "float";
  
    DenseMatrix<float>* densemat2 = swap_types<S, float>(densemat);

#ifdef DEBUG
    std::cout << "Swapped mat type: float" << std::endl;
#endif

    
    delete densemat;
    delete sparsemat;
    
    SparseMatrix<float>* sparsemat2 = create_sparsematrix_from_densemat2(densemat2, flags);
    ScaleCompanion<float>* sc = scalesk(sparsemat2, flags);
    scaleMatrix(densemat2, sc);
    delete sparsemat2;
    
    SparseMatrix<float>* sparsemat3 = create_sparsematrix_from_densemat2(densemat2, flags);

#ifdef HEAVYDEBUG
    std::cout << "Scaled matrix: " << std::endl;
    print_float_densematrix(densemat2);
    print_sparsematrix(sparsemat3);
#endif
    
    
    if(flags.compression && !compressing)
      result = compress_singleton_and_then_recurse(densemat2, sparsemat2, flags);
    else
      result = RunAlgo(densemat2, sparsemat3, flags, false);

#ifdef DEBUG
    printf("Mid Result || %s | %s | %.16e in %f \n", flags.algo_name.c_str(), flags.filename, result.permanent, result.time);
#endif
    
    for(int i = 0; i < nov; i++){
      result.permanent /= sc->c_v[i];
    }
    
    for(int i = 0; i < nov; i++){
      result.permanent /= sc->r_v[i];
    }
    
  }
  
  else if(!flags.storage_half_precision && flags.type == "double"){
    ScaleCompanion<S>* sc = scalesk(sparsemat, flags);
    scaleMatrix(densemat, sc);
    //delete sparsemat;
    
    SparseMatrix<S>* sparsemat2 = create_sparsematrix_from_densemat2(densemat, flags);

#ifdef HEAVYDEBUG
    print_float_densematrix(densemat);
    print_sparsematrix(sparsemat2);
#endif

    if(flags.compression && !compressing)
      result = compress_singleton_and_then_recurse(densemat, sparsemat2, flags);
    else
      result = RunAlgo(densemat, sparsemat2, flags, false);


#ifdef DEBUG
    printf("Mid Result || %s | %s | %.16e in %f \n", flags.algo_name.c_str(), flags.filename, result.permanent, result.time);
#endif
      
    for(int i = 0; i < nov; i++){
      result.permanent /= sc->c_v[i];
    }
    
    for(int i = 0; i < nov; i++){
      result.permanent /= sc->r_v[i];
    }
  }
  
  
  else if(flags.storage_half_precision && flags.type == "float"){
    ScaleCompanion<S>* sc = scalesk(sparsemat, flags);
    scaleMatrix(densemat, sc);
    //delete sparsemat;
    
    SparseMatrix<S>* sparsemat2 = create_sparsematrix_from_densemat2(densemat, flags);
        
    if(flags.compression && !compressing)
      result = compress_singleton_and_then_recurse(densemat, sparsemat2, flags);
    else
      result = RunAlgo(densemat, sparsemat2, flags, false);
    
#ifdef DEBUG
    printf("Mid Result || %s | %s | %.16e in %f \n", flags.algo_name.c_str(), flags.filename, result.permanent, result.time);
#endif
    
    for(int i = 0; i < nov; i++){
      result.permanent /= sc->c_v[i];
    }
    
    for(int i = 0; i < nov; i++){
      result.permanent /= sc->r_v[i];
    }  
  }

  else{
    std::cout << "Why do you want to scale? Exiting.. " << std::endl;
    exit(1);
  }

  return result;
}




int main (int argc, char **argv)
{ 
  bool generic = true;
  bool dense = true;
  bool approximation = false;
  bool calculation_half_precision = false;
  bool storage_half_precision = false;
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

  bool compression = false;
  
  // We need to handle this part with a constructor

  flags flags;
  /* A string listing valid short options letters.  */
  const char* const short_options = "bsr:t:f:gd:cap:x:y:z:im:n:hwqk:e:ol:vu:";
  /* An array describing valid long options.  */
  const struct option long_options[] = {
    { "binary",     0, NULL, 'b' },
    { "sparse",     0, NULL, 's' },
    { "preprocessing",   1, NULL, 'r' },
    { "threads",  1, NULL, 't' },
    { "file",  1, NULL, 'f' },
    { "gpu",  0, NULL, 'g' },
    { "nodevice",  1, NULL, 'd' },
    { "deviceid" , 0, NULL, 'l'},
    { "cpu",  0, NULL, 'c' },
    { "approximation",  0, NULL, 'a' },
    { "perman",  1, NULL, 'p' },
    { "numOfTimes",  1, NULL, 'x' },
    { "scaleIntervals",  1, NULL, 'y' },
    { "scaleTimes",  1, NULL, 'z' },
    { "grid",  0, NULL, 'i' },
    { "gridm",  1, NULL, 'm' },
    { "gridn",  1, NULL, 'n' },
    { "calchalfprec" , 0, NULL, 'h'},
    { "calcquadprec", 0, NULL, 'q'},
    { "storhalfprec" , 0, NULL, 'w'},
    { "storquadprec" , 0, NULL, 'v'},
    { "norep", 1, NULL, 'k'},
    { "gridmultip", 1, NULL, 'e'},
    { "compression", 0, NULL, 'o'},
    { "scalingthresh", 0, NULL, 'u'},
    
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
	flags.exact = false;
        break;
      case 'g':
	flags.gpu = 1;
	flags.gpu_stated = 1;
        break;
      case 'c':
        flags.cpu = 1;
	if(!flags.gpu_stated)
	  flags.gpu = 0; //Prevents multiple execution in case of sole -c
        break;
      case 'd':
        if (optarg[0] == '-'){
          fprintf (stderr, "Option -d requires an argument.\n");
          return 1;
        }
        flags.gpu_num = atoi(optarg);
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
	flags.calculation_half_precision = 1;
	break;
      case 'q':
	flags.calculation_quad_precision = 1;
	break;
      case 'w':
	flags.storage_half_precision = 1;
	break;
      case 'v':
	flags.storage_quad_precision = 1;
	break;
      case 'l':
	if(optarg[0] == '-'){
	  fprintf(stderr, "Option -q requires an argument. \n");
	  return 1;
	}
	flags.device_id = atoi(optarg);
	break;
      case 'k':
	if(optarg[0] == '-'){
	  fprintf(stderr, "Option -k requires an argument \n");
	}
	flags.rep = atoi(optarg);
	break;
      case 'e':
	if(optarg[0] == '-'){
	  fprintf(stderr, "Option -e requires an argument \n");
	}
	flags.grid_multip = atoi(optarg);
	break;
      case 'o':
	flags.compression = 1;
	break;
      case 'u':
	flags.scaling_threshold = (double)atoi(optarg);
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

  if(flags.gpu && (flags.storage_quad_precision || flags.calculation_quad_precision)){
    std::cout << "Quad precision is only available with cpu.. exiting.. " << std::endl;
    exit(1);
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

  bool is_binary = false;
  if(flags.binary_graph)
    is_binary = true;

  bool is_symmetric = false;
  if(mm_is_symmetric(matcode) == 1 || mm_is_skew(matcode))
    is_symmetric = true;

  if(is_symmetric)
    nz *= 2;

#ifdef DEBUG
  std::cout << "Matrix is pattern: " << is_pattern << std::endl;
#endif

  double final_perman;

#ifdef DEBUG
  cout << "matcode - stor_half - !is_pattern - !is_binary: " << mm_is_real(matcode) << " " <<!flags.storage_half_precision << " " <<!is_pattern << " " <<!is_binary << endl;
#endif
  
  if(mm_is_real(matcode) == 1 && !flags.storage_half_precision && !is_pattern && !is_binary && !flags.storage_quad_precision){ 
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
      readDenseMatrix(densemat, flags.filename, is_pattern, is_binary);
    else 
      readSymmetricDenseMatrix(densemat, flags.filename, is_pattern, is_binary);

        
#ifdef DEBUG
    std::cout << "Read.. OK! -- Sparse Representation Compressing.. " << std::endl;
#endif
    if(flags.preprocessing == 0)
      matrix2compressed_o(densemat, sparsemat); 
    if(flags.preprocessing == 1)
      matrix2compressed_sortOrder_o(densemat, sparsemat); 
    if(flags.preprocessing == 2)
      matrix2compressed_skipOrder_o(densemat, sparsemat);
    
#ifdef DEBUG
    std::cout << "Sparse Compression..OK!" << std::endl;

    print_sparsematrix(sparsemat);
    print_densematrix(densemat);
#endif
    
    print_flags(flags);

    bool scaling_chosen = 0;

    if(flags.scaling_threshold != -1.0)
      scaling_chosen = 1;
    
#ifdef DEBUG
    std::cout << "Scaling chosen: " << scaling_chosen << " threshold: " << flags.scaling_threshold << std::endl;
#endif
    
    for(int i = 0; i < flags.rep; i++){
      //std::cout << "i: " << i << std::endl;
      //print_float_densematrix(densemat);
      
      Result result;

      if(scaling_chosen){
	DenseMatrix<double>* copy_densemat = copy_dense(densemat);
	SparseMatrix<double>* copy_sparsemat = copy_sparse(sparsemat);
	result = scale_and_calculate(copy_densemat, copy_sparsemat, flags, false);
	delete copy_densemat;
	delete copy_sparsemat;
	flags.type = "double"; //In case if scale_and_calculate change it
      }

      else{
	if(flags.compression)
	  result = compress_singleton_and_then_recurse(densemat, sparsemat, flags);
	else
	  result = RunAlgo(densemat, sparsemat, flags, false);
      }
      printf("Result || %s | %s | %.16e in %f \n", flags.algo_name.c_str(), flags.filename, result.permanent, result.time);
    }
  }

 
#ifdef ONLYCPU
  
  else if(mm_is_real(matcode) == 1 && !flags.storage_half_precision && !is_pattern && !is_binary && flags.storage_quad_precision){
#ifdef DEBUG
    std::cout << "Read Case: 1" << std::endl;
#endif
    flags.type = "__float128";
    SparseMatrix<__float128>* sparsemat;
    DenseMatrix<__float128>* densemat;
    sparsemat = new SparseMatrix<__float128>();
    densemat = new DenseMatrix<__float128>(); 
    densemat->mat = new __float128[M*N];
    sparsemat->rvals = new __float128[nz];
    sparsemat->cvals = new __float128[nz];
    sparsemat->cptrs = new int[nov + 1];
    sparsemat->rptrs = new int[nov + 1];
    sparsemat->rows = new int[nz];
    sparsemat->cols = new int[nz];
    sparsemat->nov = M; 
    densemat->nov = M; 
    sparsemat->nnz = nz;
    densemat->nnz = nz;

    if(!is_symmetric)
      readDenseMatrix(densemat, flags.filename, is_pattern, is_binary);
    else 
      readSymmetricDenseMatrix(densemat, flags.filename, is_pattern, is_binary);
    
#ifdef DEBUG
    std::cout << "Read.. OK! -- Sparse Representation Compressing.. " << std::endl;
#endif
    if(flags.preprocessing == 0)
      matrix2compressed_o(densemat, sparsemat); 
    if(flags.preprocessing == 1)
      matrix2compressed_sortOrder_o(densemat, sparsemat); 
    if(flags.preprocessing == 2)
      matrix2compressed_skipOrder_o(densemat, sparsemat);
    
#ifdef DEBUG
    std::cout << "Sparse Compression..OK!" << std::endl;

    print_sparsematrix(sparsemat);
    print_densematrix(densemat);
#endif

    Result result;

    print_flags(flags);

    bool scaling_chosen = 0;
    if(flags.scaling_threshold != -1.0)
      scaling_chosen = 1;
    
#ifdef DEBUG
    std::cout << "Scaling chosen: " << scaling_chosen << " threshold: " << flags.scaling_threshold << std::endl;
#endif

    //No scaling for __float128
    //This is due to some standard library functions does not support __float128
    //and it already does not require scaling for accurate result
    
    for(int i = 0; i < flags.rep; i++){
      
      if(flags.compression)
	result = compress_singleton_and_then_recurse(densemat, sparsemat, flags);
      else
	result = RunAlgo(densemat, sparsemat, flags, false);
      
      printf("Result || %s | %s | %.16e in %f \n", flags.algo_name.c_str(), flags.filename, result.permanent, result.time);
    }
  }
  
#endif
  
  
  else if(mm_is_real(matcode) == 1 && flags.storage_half_precision && !is_pattern && !is_binary){
#ifdef DEBUG
    std::cout << "Read Case: 2" << std::endl;
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
      readDenseMatrix(densemat, flags.filename, is_pattern, is_binary);
    else 
      readSymmetricDenseMatrix(densemat, flags.filename, is_pattern, is_binary);
    
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

    print_flags(flags);

    bool scaling_chosen = 0;
    if(flags.scaling_threshold != -1.0)
      scaling_chosen = 1;

#ifdef DEBUG
    std::cout << "Scaling chosen: " << scaling_chosen << " threshold: " << flags.scaling_threshold << std::endl;
#endif
    
    for(int i = 0; i < flags.rep; i++){
      Result result;

      if(scaling_chosen){
	//result = scale_and_calculate(densemat, sparsemat, flags);
	DenseMatrix<float>* copy_densemat = copy_dense(densemat);
	SparseMatrix<float>* copy_sparsemat = copy_sparse(sparsemat);
	result = scale_and_calculate(copy_densemat, copy_sparsemat, flags, false);
	delete copy_densemat;
	delete copy_sparsemat;
	flags.type = "float";//In case if scale_and_calculate change it
      }

      else{
      
	if(flags.compression)
	  result = compress_singleton_and_then_recurse(densemat, sparsemat, flags);
	else
	  result = RunAlgo(densemat, sparsemat, flags, false);
      }
      printf("Result || %s | %s | %.16e in %f \n", flags.algo_name.c_str(), flags.filename, result.permanent, result.time);
    }
  }
  
  else if(mm_is_integer(matcode) == 1 || is_pattern || is_binary){
#ifdef DEBUG
    std::cout << "Read Case: 3" << std::endl;
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
      readDenseMatrix(densemat, flags.filename, is_pattern, is_binary);
    else 
      readSymmetricDenseMatrix(densemat, flags.filename, is_pattern, is_binary);


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

    print_flags(flags);
    
    bool scaling_chosen = 0;
        
    if(flags.scaling_threshold != -1.0)
      scaling_chosen = 1;
    
#ifdef DEBUG
    std::cout << "Scaling chosen: " << scaling_chosen << " threshold: " << flags.scaling_threshold << std::endl;
#endif
    for(int i = 0; i < flags.rep; i++){
      Result result;
        
      if(scaling_chosen){
	//result = scale_and_calculate(densemat, sparsemat, flags);
	DenseMatrix<int>* copy_densemat = copy_dense(densemat);
	SparseMatrix<int>* copy_sparsemat = copy_sparse(sparsemat);
	result = scale_and_calculate(copy_densemat, copy_sparsemat, flags, false);
	delete copy_densemat;
	delete copy_sparsemat;
	flags.type = "int";
      }
      
      else{
	if(flags.compression)
	  result = compress_singleton_and_then_recurse(densemat, sparsemat, flags);
	else
	  result = RunAlgo(densemat, sparsemat, flags, false);
      }
      printf("Result || %s | %s | %.16e in %f \n", flags.algo_name.c_str(), flags.filename, result.permanent, result.time);
    }
  }
  
  else{
    std::cout << "Matrix or flags have overlapping features.. " <<std::endl;
    print_flags(flags);
    exit(1);
  }
  
  //std::feclearexcept(FE_OVERFLOW);
  //std::cout << "Is any overflow happened: " << (bool)std::fetestexcept(FE_OVERFLOW) << std::endl;
  return 0;
}
