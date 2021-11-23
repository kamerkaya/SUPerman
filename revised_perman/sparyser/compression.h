
void compress() {
#ifdef COMP
  vtype novd2 = hnov;
  
#define SMALL_MAT_DEBUG
#ifdef SMALL_MAT_DEBUG
    cout << "Original " << endl;
    for(vtype i = 0; i < hnov; i++) {
      for(vtype j = 0; j < hnov; j++) {
        cout << mat[i * novd2 + j] << " ";
      }
      cout << endl;
    }
    cout << "--------------------------------------" << endl;
#endif
  //compress the matrix and convert to the sparse form
  bool cont = true;
  vtype deg, n1, n2;

  unsigned long long v1, v2;

  while(hnov > 2 && cont) {
    cont = false;
    
    for(vtype i = 0; i < hnov; i++) {
      deg = 0;
      for(vtype j = 0; j < hnov; j++) {
	if(mat[(i * novd2) + j] > 0) {
	  if(deg == 0) {n1 = j; v1 = mat[(i * novd2) + j];}
	  if(deg == 1) {n2 = j; v2 = mat[(i * novd2) + j];}
	  deg++;
	}
      }
      if(deg == 0) {cout << "perman is 0" << endl; exit(1);}
      else if(deg == 1 || deg == 2) {
	//cout << "-D: " << deg << ": " << i << " " << n1 << " " << n2 << endl;

	cont = true;

	if(deg == 2) {
	  for(vtype j = 0; j < hnov; j++) {
	    mat[(novd2 * j) + n2] = v1 * mat[(novd2 * j) + n2] + v2 * mat[(novd2 * j) + n1];
	  }
	}

	for(vtype j = i + 1; j < hnov; j++) {
	  for(vtype k = 0; k < hnov; k++) {
	    mat[(j-1) * novd2 + k] = mat[j * novd2 + k];
	  }
	}

	for(vtype j = n1 + 1; j < hnov; j++) {
	  for(vtype k = 0; k < hnov; k++) {
	    mat[k * novd2 + (j - 1)] = mat[(k * novd2) + j];
	  }
	}
	hnov--;
	//cout << "+Compressed to " << hnov << " matrix " << endl;
	break;
      }
    }

    for(vtype i = 0; i < hnov; i++) {
      deg = 0;
      for(vtype j = 0; j < hnov; j++) {
        if(mat[(j * novd2) + i] > 0) {
          if(deg == 0) {n1 = j; v1 = mat[(j * novd2) + i];}
          if(deg == 1) {n2 = j; v2 = mat[(j * novd2) + i];}
          deg++;
        }
      }
      if(deg == 0) {cout << "perman is 0" << endl; exit(1);}
      else if(deg == 1 || deg == 2) {
	cont = true;
	//cout << "+D: " << deg << ": " << i << " " << n1 << " " << n2 << endl;
	if(deg == 2) {
          for(vtype j = 0; j < hnov; j++) {
            mat[(novd2 * n2) + j] = v1 * mat[(novd2 * n2) + j] + v2 * mat[(novd2 * n1) + j];
          }
        }
	
        for(vtype j = n1 + 1; j < hnov; j++) {
	  for(vtype k = 0; k < hnov; k++) {
            mat[((j-1) * novd2) + k] = mat[(j * novd2) + k];
          }
	}
	
	for(vtype k = 0; k < hnov - 1; k++) {
	  for(vtype j = i + 1; j < hnov; j++) {
            mat[(k * novd2) + (j-1)] = mat[(k * novd2) + j];
          }
        }
	hnov--;
	break;
      }
    }
  }
  cout << "Compressed to " << hnov << " matrix " << endl;

#ifdef SMALL_MAT_DEBUG
  cout << "Final" << endl;
  for(vtype ii = 0; ii < hnov; ii++) {
    for(vtype j = 0; j < hnov; j++) {
      cout << mat[ii*novd2 + j] << " ";
    }
    cout << endl;
  }
  cout << "--------------------------------------" << endl;
#endif

  if(hnov < nov / 2) {
    compressed = true;
    //squeeze matrix
    for(vtype i = 0; i < hnov; i++) {
      for(vtype j = 0; j < hnov; j++) {
	mat[i * hnov + j] = mat[i * novd2 + j];
      }
    }
     
    //squeeze graph
    xadj[0] = 0;
    etype ptr = 0;
    for(vtype i = 0; i < hnov; i++) {
      vtype deg = 0;
      for(vtype j = 0; j < hnov; j++) {
	unsigned long long cval =  mat[i * hnov + j];
	if(cval != 0) {
	  deg++;
	  adj[ptr] = hnov + j;
	  val[ptr++] = cval;
	}
      }
      xadj[i+1] = xadj[i] + deg;
    }

    for(vtype i = 0; i < hnov; i++) {
      vtype deg = 0;
      for(vtype j = 0; j < hnov; j++) {
	unsigned long long cval =  mat[j * hnov + i];
        if(cval != 0) {
          deg++;
          adj[ptr] = j;
	  val[ptr++] = cval;
	}
      }
      xadj[hnov + i + 1] = xadj[hnov + i] + deg;
    }
    nov = 2 * hnov;
  }
#endif
}
