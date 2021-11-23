#include <bitset>
#include <omp.h>

#define REPORT

unsigned long long brute(int *xadj, int *adj, int nov) {
  unsigned long long perman = 0;

  int* matched = new int[nov];
  for(int i = 0; i < nov; i++) matched[i] = 0;
  
  int h_nov = nov/2;
  int* ptrs = new int[h_nov];
  for(int i = 0; i < h_nov; i++) {ptrs[i] = xadj[i];}

  matched[0] = adj[0];
  matched[adj[0]] = 1;
  ptrs[0] = 1;

  int curr = 1;
  while(curr >= 0) {
#ifdef DEBUG
    cout << "curr: " << curr << endl;
    //cout << "\tmatches: "; for(int i = 0; i< nov; i++) {cout << matched[i] << " ";} cout << endl;
#endif

    //clear the existing matching on current
    if(matched[curr] != 0) {
      matched[matched[curr]] = 0;
      matched[curr] = 0;
    }

    //check if we can increase the matching by matching curr
    int ptr = ptrs[curr];
#ifdef DEBUG
    cout << "\tptr init: " << ptr << endl;
#endif
    int partner;
    for(; ptr < xadj[curr + 1]; ptr++) {
      if(matched[adj[ptr]] == 0) {
	partner = adj[ptr];
	ptrs[curr] = ptr + 1;
	break;
      }
    }
#ifdef DEBUG
    cout << "\tptr: " << ptr << endl;
    cout << "\tnext: " << xadj[curr + 1] << endl;
#endif
    if(ptr < xadj[curr + 1]) { //we can extend matching
#ifdef DEBUG
      cout << "\t" << curr << " " << partner << endl;
#endif
      if(curr == h_nov - 1) {
	perman++;
	if(perman % 10000000 == 0) {
	  cout << perman << endl;
	}
	ptrs[curr] = xadj[curr];
	curr--;
      } else {
	matched[curr] = partner;
	matched[partner] = 1;
	curr++;
      }
    } else {
      ptrs[curr] = xadj[curr];
      curr--;
    }
  }
  return perman;
}


void print_x(double* x, int nov){

  for(int i = 0; i < nov; i++){
    cout << x[i] << " ";
  }
  cout << endl;
  cout << "--------" << endl;
}

unsigned long int perman64(int* mat, int nov) {
  double x[64];   
  double rs; //row sum
  double s;  //+1 or -1 
  double prod; //product of the elements in vector 'x'
  double p = 1; //product of the elements in vector 'x'
  double *xptr; 
  int j, k;
  unsigned long long int i, tn11 = (1ULL << (nov-1)) - 1ULL;
  unsigned long long int gray;
  
  //create the x vector and initiate the permanent
  for (j = 0; j < nov; j++) {
    rs = .0f;
    for (k = 0; k < nov; k++)
      rs += mat[(j * nov) + k];  // sum of row j
    x[j] = mat[(j * nov) + (nov-1)] - rs/2;  // see Nijenhuis and Wilf - x vector entry
    p *= x[j];   // product of the elements in vector 'x'
  }

  cout << "perman64 x: " << endl;
  print_x(x, nov);
  cout << "p init: " << p << endl;
  
  //create the transpose of the matrix
  int* mat_t = new int[nov * nov];
  for (int i = 0; i < nov; i++) {
    for (int j = 0; j < nov; j++) {
      mat_t[(i * nov) + j] = mat[(j * nov) + i];
    }
  }

  gray = 0;
  unsigned long long ctr = 0LL;
  unsigned long long one = 1;

  unsigned long long counter = 0;

  double t_start = omp_get_wtime();
  for (i = 1; i <= tn11; i++) {
#ifdef REPORT   
    if(i % 100000000 == 0) cout << "\t" << i/100000000 << " of " << tn11/100000000 << " in " << omp_get_wtime() - t_start << " - percent " << (counter + .0f) / i << endl;
#endif

    //compute the gray code
    k = __builtin_ctzll(++ctr);
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    //decide if subtract of not - if the kth bit of gray is one then 1, otherwise -1
    s = ((one << k) & gray) ? 1 : -1;
    
    counter++;
    prod = 1.0;
    xptr = (double*)x;
    for (j = 0; j < nov; j++) {
      *xptr += s * mat_t[(k * nov) + j]; // see Nijenhuis and Wilf - update x vector entries
      prod *= *xptr++;  //product of the elements in vector 'x'
    }

    p += ((i&1ULL)? -1.0:1.0) * prod; 
  }

  delete [] mat_t;

  return((4*(nov&1)-2) * p);
}

unsigned long int sparse_perman64(int* xadj, int* adj, int nov) {
  double x[64];   
  double rs; //row sum
  double s;  //+1 or -1
  double prod; //product of the elements in vector 'x'
  double p = 1; //product of the elements in vector 'x'
  double *xptr; 
  int j, k;
  int ptr;
  unsigned long long int i, tn11 = (1ULL << (nov-1)) - 1ULL;
  unsigned long long int gray;
  
  for(j = 0; j < nov; j++) x[j] = (xadj[j+1] - xadj[j]) / (-2.0f);
  for(ptr = xadj[2 * nov - 1]; ptr < xadj[2 * nov]; ptr++) x[adj[ptr]]++;

  int nzeros = 0;
  prod = 1;
  for(j = 0; j < nov; j++) {
    if(x[j] == 0) {
      nzeros++;
    } else {
      prod *= x[j]; 
    }
  }

  if(nzeros == 0) {
    p = prod;
  } else {
    p = 0;
  }

  gray = 0;
  unsigned long long ctr = 0LL;
  unsigned long long one = 1;

  for (i = 1; i <= tn11; i++) {
    if(i % 100000000 == 0)  cout << i << " of " << tn11 << endl;
    
    k = __builtin_ctzll(++ctr);
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    s = ((one << k) & gray) ? 1 : -1;

    if(s == 1) {
      for (ptr = xadj[nov + k]; ptr < xadj[nov + k + 1]; ptr++) {
	int index = adj[ptr];
	double value = x[index];
	
	if(value == 0) {
	  nzeros--;
	  x[index] = 1;
	} else if(value == -1) {
	  nzeros++;
	  x[index] = 0;
	  prod = -prod;
	} else {
	  x[index] = x[index] + 1;
	  prod /= value;
	  prod *= (value + 1);
	}
      }
    } else {
      for (ptr = xadj[nov + k]; ptr < xadj[nov + k + 1]; ptr++) {
        int index = adj[ptr];
        double value = x[index];
	
	if(value == 0) {
	  nzeros--;
	  x[index] = -1;
	  prod = -prod;
	} else if(value == 1) {
	  nzeros++;
	  x[index] = 0;
	} else {
          x[index] = x[index] - 1;
	  prod /= value;
          prod *= (value - 1);
        }
      }
    }
    
    if(nzeros == 0) {
      p += ((i&1ULL)? -1.0:1.0) * prod; 
    }
  }
  
  return((4*(nov&1)-2) * p);
}

unsigned long int sparser_perman64(int* xadj, int* adj, int nov) {
  double x[64];   
  double rs; //row sum
  double s;  //+1 or -1
  double prod; //product of the elements in vector 'x'
  double p = 0; 
  double *xptr; 
  int j, k;
  int ptr;
  unsigned long long int i, tn11 = (1ULL << (nov-1)) - 1ULL;
  unsigned long long int gray;

  for(j = 0; j < nov; j++) x[j] = (xadj[j+1] - xadj[j]) / (-2.0f);
  for(ptr = xadj[2 * nov-1]; ptr < xadj[2 * nov]; ptr++) x[adj[ptr]]++;

  int nzeros = 0;
  prod = 1;
  for(j = 0; j < nov; j++) {
    prod *= x[j]; 
    if(x[j] == 0) {
      nzeros++;
    } 
  }
  p += prod;

  gray = 0;
  unsigned long long ctr = 0LL;
  unsigned long long one = 1;

  for(j = 0; j < nov; j++) {
    cout << x[j] << " ";
  } 
  cout << endl;


  for (i = 1; i <= tn11; i++) {
    if(i % 100000000 == 0) cout << i << " of " << tn11 << endl;

    k = __builtin_ctzll(++ctr);
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    s = ((one << k) & gray) ? 1 : -1;

    for (ptr = xadj[nov + k]; ptr < xadj[nov + k + 1]; ptr++) {
      int index = adj[ptr];
      if(x[index] == 0) nzeros--;
      x[index] += s;
      if(x[index] == 0) nzeros++;
    }
      
    if(nzeros == 0) {
      prod = 1; for(int j = 0; j < nov; j++) prod *= x[j]; 
      p += ((i&1ULL)? -1.0:1.0) * prod; 
    }
  }
  
  return((4*(nov&1)-2) * p);
}



#define WORKING
unsigned long int sparse_perman64_skip(int* xadj, int* adj, int nov) {
  double x[64];   
  
  int old_nzeros;
  int counters[64];
  
  double rs; //row sum
  double s;  //+1 or -1
  double prod; //product of the elements in vector 'x'
  double old_prod;
  double p = 1; //product of the elements in vector 'x'
  double *xptr;//, *aptr; 
  int j, k;
  int ptr;
  unsigned long long int i, tn11 = (1ULL << (nov-1)) - 1ULL, old_i;
  unsigned long long int gray, prevgray = 0, old_gray, two_to_k;
  bool flag = false;

  for(j = 0; j < nov; j++) x[j] = (xadj[j+1] - xadj[j]) / (-2.0f);
  for(ptr = xadj[2*nov-1]; ptr < xadj[2*nov]; ptr++) x[adj[ptr]]++;

  int nzeros = 0;
  prod = 1;
  for(j = 0; j < nov; j++) {
    if(x[j] == 0) {
      nzeros++;
    } else {
      prod *= x[j]; 
    }
  }

  long long int count = 0;

  if(nzeros == 0) {
    p = prod;
    old_prod = prod;
    old_i = 0;
    old_gray = 0;
    flag = false;
  } else {
    flag = true;
    old_nzeros = nzeros;
    cout << " " << i << " First time zero: - ";
    for(int jj = 0; jj < nov; jj++) {
      counters[jj] = 0;
      if(x[jj] == 0) {
	cout << jj << " ";
      }
    }
    cout << endl;

    for(int jj = 0; jj < nov; jj++) {
      if(x[jj] == 0) {
	cout << "\t" << jj << ": ";
	for(int ptr = xadj[jj]; ptr < xadj[jj + 1]; ptr++) {
	  cout << adj[ptr] - nov << " ";
	  counters[adj[ptr] - nov]++;
	}
	cout << endl;
      }
    }

    p = 0;
  }
#ifdef FAST
  gray = 0;
  unsigned long long ctr = 0LL;
  unsigned long long one = 1;
#endif
  for (i = 1; i <= tn11; i++) {

    if(i % 100000000 == 0) {
      cout << i << " of " << tn11 << endl;
    }

#ifdef FAST
    k = __builtin_ctzll(++ctr);
    gray ^= (one << k); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    s = ((one << k) & gray) ? 1 : -1;
#else
    gray = i ^ (i >> 1); // Gray-code order: 1,3,2,6,7,5,4,12,13,15,...
    two_to_k = 1;    // two_to_k = 2 raised to the k power (2^k)
    k = 0;
    while (two_to_k < (gray^prevgray)) {
      two_to_k <<= 1;  // two_to_k is a bitmask to find location of 1
      k++;
    }
    s = (two_to_k & gray) ? 1 : -1;
    prevgray = gray;        
#endif

#ifdef WORKING
    cout << "k : " << k << endl;
    cout << "Bef: ";
    for (int ii = 0; ii < nov; ii++) {
      if (x[ii] == 0) {
	cout << ii << " ";
      }
    }
    cout << endl;

    cout << "Touching: ";
    for (ptr = xadj[nov + k]; ptr < xadj[nov + k + 1]; ptr++) {
      cout << adj[ptr] << " ";
    }
    cout << endl;
#endif
    if(s == 1) {
      for (ptr = xadj[nov + k]; ptr < xadj[nov + k + 1]; ptr++) {
	int index = adj[ptr];
	double value = x[index];
	
	if(value == 0 || value == -1) {
	  if(value == 0) {
	    nzeros--;
	    x[index] = 1;
	  } else {
	    nzeros++;
	    x[index] = 0;
	    prod = -prod;
	  }
	} else {
	  x[index] = x[index] + 1;
	  prod /= value;
	  prod *= (value + 1);
	}
      }
    } else {
      for (ptr = xadj[nov + k]; ptr < xadj[nov + k + 1]; ptr++) {
        int index = adj[ptr];
        double value = x[index];
	
        if(value == 0 || value == 1) {
          if(value == 0) {
            nzeros--;
            x[index] = -1;
            prod = -prod;
          } else {
            nzeros++;
            x[index] = 0;
          }
        } else {
          x[index] = x[index] - 1;
	  prod /= value;
          prod *= (value - 1);
        }
      }
    }
#ifdef WORKING
    cout << "Aft: ";
    for (int ii = 0; ii < nov; ii++) {
      if (x[ii] == 0) {
	cout << ii << " ";
      }
    }
    cout << endl;
#endif
    if(nzeros == 0) {
      std::bitset<64> y(gray);
      cout << " gray: " << y << "\ti: " << i << "\tprod: " << prod << " -  ";
      for(int jj = 0; jj < nov; jj++) {
	cout << x[jj] << " ";
      }
      cout << endl;


      p += ((i&1ULL)? -1.0:1.0) * prod; 
      old_prod = prod;
      old_i = i;
      old_gray = gray;
      flag = false;
    } else {
      count++;

      if(flag == false) {
	old_nzeros = nzeros;
	flag = true;
	cout << " " << i << " - " << k << " First time zero: - ";
	for(int jj = 0; jj < nov; jj++) {
	  counters[jj] = 0;
	  if(x[jj] == 0) {
	    cout << jj << " ";
	  }
	}
	cout << endl;

	for(int jj = 0; jj < nov; jj++) {
	  if(x[jj] == 0) {
	    cout << "\t" << jj << ": ";
	    for(int ptr = xadj[jj]; ptr < xadj[jj + 1]; ptr++) {
	      cout << adj[ptr] - nov << " ";
	      counters[adj[ptr] - nov]++;
	    }
	    cout << endl;
	  }
	}
      }
    }   
    
#ifdef WORKING
    cout << "p is: " << p << " contribute: " << ((i&1ULL)? -1.0:1.0) * prod << endl << endl;
#endif
    //if(i % 1000000 == 0) {
    // cout << i << ", " << nzeros << ", " << (count + 0.0f) / i << ": prod =  " << prod << endl;
    //}
  }
  
  return((4*(nov&1)-2) * p);
}


