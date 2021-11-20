#ifndef READ_MATRIX_H
#define READ_MATRIX_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <bits/stdc++.h> 
//#define HARDDEBUG 

template <class T>
void readDenseMatrix(DenseMatrix<T>* mat, const char* filename, bool is_pattern, bool is_binary){

#ifdef DEBUG
  std::cout << "In function: readDenseMatrix()" << std::endl;
  std::cout << "- is_pattern: " << is_pattern << std::endl;
  std::cout << "- is_binary: " << is_binary << std::endl;
#endif
  
  std::ifstream file(filename);
  
  //Ignore comment headers
  while (file.peek() == '%') file.ignore(2048, '\n');
  
  int no_row, no_col, no_lines;
  file >> no_row >> no_col >> no_lines;
  
#ifdef DEBUG
  std::cout << "No rows: " << no_row << " No cols: " << no_col << " No Lines: "<< no_lines << std::endl;
#endif
  
  mat->mat = new T[no_row*no_col];
  
  for(int i = 0; i < (no_row*no_col); i++){  
    mat->mat[i] = (T)0;
  }
  
  double cast;
  int x, y;
  
  for(int i = 0; i < no_lines; i++){
    
    if(is_pattern){
      file >> x >> y;
      cast = (int)1; //Just for safety
    }
    else if(!is_pattern && is_binary){
      file >> x >> y >> cast;
      cast = (int)1; //Just for safety
    }
    else{
      file >> x >> y >> cast;
    }
    
    x -= 1; //Convert from 1-based to 0-based
    y -= 1;

#ifdef HARDDEBUG
    std::cout << "x: " << x << " y: " << y << " x*no_row+y: " << x*no_row+y << std::endl;
#endif
    
    if(is_pattern || is_binary)
      mat->mat[x*no_row+y] = (int)1;
    else
      mat->mat[x*no_row+y] = (T)cast;
  }

#ifdef HARDDEBUG
  std::cout << "I've read something like that: " << std::endl;
  for(int i = 0; i < no_row; i++){
    for(int j = 0; j < no_row; j++){
      std::cout << mat->mat[i*no_row+j] << " ";
    }
    std::cout << std::endl;
  }
#endif
  
  file.close();
  
}

template <class T>
void readSymmetricDenseMatrix(DenseMatrix<T>* mat, const char* filename, bool is_pattern, bool is_binary){

#ifdef DEBUG
  std::cout << "In function: readSymmetricDenseMatrix()" << std::endl;
  std::cout << "- is_pattern: " << is_pattern << std::endl;
  std::cout << "- is_binary: " << is_binary << std::endl;
#endif
  
  std::ifstream file(filename);
  
  //Ignore comment headers
  while (file.peek() == '%') file.ignore(2048, '\n');
  
  int no_row, no_col, no_lines;
  file >> no_row >> no_col >> no_lines;
  
#ifdef DEBUG
  std::cout << "No rows: " << no_row << " No cols: " << no_col << " No Lines: "<< no_lines << std::endl;
#endif
  
  mat->mat = new T[no_row*no_col];
  
  for(int i = 0; i < (no_row*no_col); i++){  
    mat->mat[i] = (T)0;
  }
  
  double cast;
  int x, y;
  for(int i = 0; i < no_lines; i++){
    if((is_pattern && !is_binary) || (is_pattern && is_binary)){
      file >> x >> y;
    }
    else if(!is_pattern && is_binary){
      file >> x >> y >> cast;
      cast = int(1); //Just for safety
    }
    else
      file >> x >> y >> cast;

    x -= 1; //Convert from 1-based to 0-based
    y -= 1;

#ifdef HARDDEBUG
    std::cout << "x: " << x << " y: " << y << " x*no_row+y: " << x*no_row+y << std::endl;
#endif
        
    if(is_pattern || is_binary)
      mat->mat[x*no_row+y] = (int)1;
    else
      mat->mat[x*no_row+y] = (T)cast;

    if(x != y){
      if(is_pattern || is_binary)
      mat->mat[y*no_row+x] = (int)1;
    else
      mat->mat[y*no_row+x] = (T)cast;
    }
  }

#ifdef HARDDEBUG
  std::cout << "I've read something symmetric like that: " << std::endl;
  for(int i = 0; i < no_row; i++){
    for(int j = 0; j < no_col; j++){
      std::cout << mat->mat[i*no_row+j] << " ";
    }
    std::cout << std::endl;
  }
#endif
  
#ifdef DEBUG
  std::cout << "Returning from readSymmetricDenseMatrix()" << std::endl;
#endif

  file.close();
}

#endif
