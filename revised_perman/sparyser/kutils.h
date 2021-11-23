#include <iostream>
#include <vector>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string.h>
#include <cassert>
#include "matching.h"
#include "rcm.hpp"

using namespace std;

string seperator = "************************************************************************";

bool pairCompare_inc(const pair<int, int> &a,const pair<int, int> &b) {
    return a.first < b.first;
}

bool pairCompare_dec(const pair<int, int> &a,const pair<int, int> &b) {
    return a.first > b.first;
}

void printGraph(int* xadj, int* adj, int* val, int nov) {
  cout << seperator << endl;
  for(int i = 0; i < nov; i++) {
    cout << i << "( " << xadj[i] << ", " << xadj[i+1] << ") : ";

    int eptr = xadj[i+1];
    for(int e = xadj[i]; e < eptr; e++) {
      cout << "(" << adj[e] << ", " << val[e] << "), ";
    }
    cout << endl;
  }
}

int getRowNnz(int i, int* mat, int nov) {
  int nnz = 0;
  for(int j = 0; j < nov; j++) {
    if(mat[(i * nov) + j] > 0) {
      nnz++;
    }
  }
  return nnz;
}

int getColNnz(int i, int* mat, int nov) {
  int nnz = 0;
  for(int j = 0; j < nov; j++) {
    if(mat[(j * nov) + i] > 0) {
      nnz++;
    }
  }
  return nnz;
}

void printMatrix(int* mat, int nov) {
  cout << seperator << endl;
  //  cout << "Rdegs: "; for(int i = 0; i < nov; i++) cout << getRowNnz(i, mat, nov) << " "; cout << endl;
  //  cout << "Cdegs: "; for(int i = 0; i < nov; i++) cout << getColNnz(i, mat, nov) << " "; cout << endl;


  //  cout << seperator << endl;

  for(int i = 0; i < nov; i++) {
    for(int j = 0; j < nov; j++) {
      cout << mat[i * nov + j] << " ";
    }
    cout << endl;
  }
}

int getMinNnz(int* mat, int nov) {
  int minDeg = nov;
  for(int i = 0; i < nov; i++) {
    int deg = getRowNnz(i, mat, nov);
    if(deg < minDeg) {
      minDeg = deg;
    }
    
    deg = getColNnz(i, mat, nov);
    if(deg < minDeg) {
      minDeg = deg;
    }
  }
  return minDeg;
}

void readTxtFile(string filename, int*& mat, int offset, int& nov, int& nnz) {
  std::ifstream infile(filename.c_str());
  if(infile.good() == false) { cout << "File cannot be opened" << endl; exit(1);}

  std::string line;

  bool symmetric = false;

  //skip the comments and get the info line                                                                                                                                
  while (std::getline(infile, line)) {
    if(line[0] == '%' && line[1] == '%') {
      if(line.find("symmetric") != string::npos) {
        symmetric = true;
      }
    }
    if(line[0] != '%' && line[0] != ' ') { break; }
  }
  //cout << line << endl;

  //get the information for number of vertices and edges                                                                                                                   
  int nov_;
  std::istringstream iss(line);
  if (!(iss >> nov >> nov_ >> nnz)) {
    cout << "Exiting due to weird initial info on the graph " << endl;
    exit(1);
  }
  if(nov != nov_) {
    cout << "Matrix is not square!!!" << endl;
    exit(1);
  }
  mat = new int[nov * nov];
  memset(mat, 0, sizeof(int) * (nov * nov));

  int i, j;
  int* row_degs = new int[nov]; memset(row_degs, 0, sizeof(int) * nov);
  int* col_degs = new int[nov]; memset(col_degs, 0, sizeof(int) * nov);
  nnz = 0;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    if (!(iss >> i >> j)) { continue; } // erroneous line                                                                                                                  
    i -= offset;
    j -= offset;
    mat[i * nov + j] = 1;
    row_degs[i]++;
    col_degs[j]++;
    nnz++;
    if(symmetric && (i != j)) {
      mat[j * nov + i] = 1;
      row_degs[j]++;
      col_degs[i]++; nnz++;
    }
  }

  int zero_deg = 0;
  int one_deg = 0;
  int two_deg = 0;
  for(int i = 0; i < nov; i++) {
    if(row_degs[i] == 0) {
      zero_deg++;
      cout << "Row " << i << " has no nonzeros" << endl;
    } else if(row_degs[i] == 1) {
      one_deg++;
    } else if(row_degs[i] == 2) {
      two_deg++;
    }

    if(col_degs[i] == 0) {
      zero_deg++;
      cout << "Col " << i << " has no nonzeros" << endl;
    } else if(col_degs[i] == 1) {
      one_deg++;
    } else if(col_degs[i] == 2) {
      two_deg++;
    }
  }
  delete [] row_degs;
  delete [] col_degs;
  
  if(zero_deg > 0) {
    cout << "Exiting due to non-empty rows/columns " << endl;
    exit(1);
  } else {
    cout << "Number of rows/cols is " << nov << endl;
    cout << "Number of nnz is    " << nnz << endl;
    cout << "#d1 vertices: " << one_deg << endl;
    cout << "#d2 vertices: " << two_deg << endl;
    cout << "min degree: " << getMinNnz(mat,nov) << endl;
  }
  cout << seperator << endl;
}

void readINTFile(string filename, int*& mat, int offset, int& nov, int& nnz) {
  std::ifstream infile(filename.c_str());
  if(infile.good() == false) { cout << "File cannot be opened" << endl; exit(1);}

  std::string line;

  bool symmetric = false;

  //skip the comments and get the info line                                                                                                                                
  while (std::getline(infile, line)) {
    if(line[0] == '%' && line[1] == '%') {
      if(line.find("symmetric") != string::npos) {
        symmetric = true;
      }
    }
    if(line[0] != '%' && line[0] != ' ') { break; }
  }
  //cout << line << endl;

  //get the information for number of vertices and edges                                                                                                                   
  int nov_;
  std::istringstream iss(line);
  if (!(iss >> nov >> nov_ >> nnz)) {
    cout << "Exiting due to weird initial info on the graph " << endl;
    exit(1);
  }
  if(nov != nov_) {
    cout << "Matrix is not square!!!" << endl;
    exit(1);
  }
  mat = new int[nov * nov];
  memset(mat, 0, sizeof(int) * (nov * nov));

  int i, j, val;
  int* row_degs = new int[nov]; memset(row_degs, 0, sizeof(int) * nov);
  int* col_degs = new int[nov]; memset(col_degs, 0, sizeof(int) * nov);
  nnz = 0;
  while (std::getline(infile, line)) {
    std::istringstream iss(line);
    if (!(iss >> i >> j >> val)) { continue; } // erroneous line                                                                                                                  
    i -= offset;
    j -= offset;
    mat[i * nov + j] = val;
    row_degs[i]++;
    col_degs[j]++;
    nnz++;
    if(symmetric && (i != j)) {
      mat[j * nov + i] = val;
      row_degs[j]++;
      col_degs[i]++; nnz++;
    }
  }

  int zero_deg = 0;
  int one_deg = 0;
  int two_deg = 0;
  for(int i = 0; i < nov; i++) {
    if(row_degs[i] == 0) {
      zero_deg++;
      cout << "Row " << i << " has no nonzeros" << endl;
    } else if(row_degs[i] == 1) {
      one_deg++;
    } else if(row_degs[i] == 2) {
      two_deg++;
    }

    if(col_degs[i] == 0) {
      zero_deg++;
      cout << "Col " << i << " has no nonzeros" << endl;
    } else if(col_degs[i] == 1) {
      one_deg++;
    } else if(col_degs[i] == 2) {
      two_deg++;
    }
  }
  delete [] row_degs;
  delete [] col_degs;
  
  if(zero_deg > 0) {
    cout << "Exiting due to non-empty rows/columns " << endl;
    exit(1);
  } else {
    cout << "Number of rows/cols is " << nov << endl;
    cout << "Number of nnz is    " << nnz << endl;
    cout << "#d1 vertices: " << one_deg << endl;
    cout << "#d2 vertices: " << two_deg << endl;
    cout << "min degree: " << getMinNnz(mat,nov) << endl;
  }
  cout << seperator << endl;
}

void matrix2graph(int* mat, int nov,
		  int*& xadj, int*& adj, int*& val) {
  int nnz = 0;
  for(int i = 0; i < nov * nov; i++) {
    assert(mat[i] >= 0);
    if(mat[i] > 0) {
      nnz++;
    }
  }

  xadj = new int[(2 * nov) + 1];
  adj = new int[2 * nnz];
  val = new int[2 * nnz];
  
  nnz = 0;
  for(int i = 0; i < nov; i++) {
    xadj[i] = nnz;
    for(int j = 0; j < nov; j++) {
      assert(mat[(i * nov) + j] >= 0);
      if(mat[(i * nov) + j] > 0) {
	adj[nnz] = nov + j;
	val[nnz] = mat[(i * nov) + j];
	nnz++;
      }
    }  
  }

  for(int i = 0; i < nov; i++) {
    xadj[i + nov] = nnz;
    for(int j = 0; j < nov; j++) {
      assert(mat[(j * nov) + i] >= 0);
      if(mat[(j * nov) + i] > 0) {
        adj[nnz] = j;
        val[nnz] = mat[(j * nov) + i];
        nnz++;
      }
    }
  }
  xadj[2 * nov] = nnz;
}

void sortWRowDeg(int* mat, int nov, bool inc) {
  vector<pair<int, int> > r_degs;

  for(int i = 0; i < nov; i++) {
    int rdeg = 1;
    for(int j = 0; j < nov; j++) {
      if(mat[(i * nov) + j] != 0) {
        rdeg++;
      }
    }
    r_degs.push_back(make_pair(rdeg, i));
  }
  if(inc) {
    sort(r_degs.begin(), r_degs.end(), pairCompare_inc);
  } else {
    sort(r_degs.begin(), r_degs.end(), pairCompare_dec);
  }

  int* n_mat = new int[nov * nov];
  memset(n_mat, 0, sizeof(int) * nov * nov);
  for(int i = 0; i < nov; i++) {
    for(int j = 0; j < nov; j++) {
      n_mat[(i * nov) + j] = mat[(r_degs[i].second * nov) + j];
    }
  }
  memcpy(mat, n_mat, nov * nov * sizeof(int));

  delete [] n_mat;
}

void sortWColDeg(int* mat, int nov, bool inc) {
  vector<pair<int, int> > c_degs;

  for(int j = 0; j < nov; j++) {
    int cdeg = 0;
    for(int i = 0; i < nov; i++) {
      if(mat[(i * nov) + j] != 0) {
	cdeg++;
      }
    }
    c_degs.push_back(make_pair(cdeg, j)); 
  }
  
  if(inc) {
    sort(c_degs.begin(), c_degs.end(), pairCompare_inc);
  } else {
    sort(c_degs.begin(), c_degs.end(), pairCompare_dec);
  }

  int* n_mat = new int[nov * nov];
  memset(n_mat, 0, sizeof(int) * nov * nov);
  for(int i = 0; i < nov; i++) {
    for(int j = 0; j < nov; j++) {
      n_mat[(i * nov) + j] = mat[(i * nov) + c_degs[j].second];
    }
  }
  memcpy(mat, n_mat, nov * nov * sizeof(int));

  delete [] n_mat;
}

void firstSeenRow(int* mat, int nov) {
  int* rperm = new int[nov]; 
  int* mark = new int[nov]; 
  memset(mark, 0, sizeof(int) * nov);
  int current = 0;
  for(int j = 0; j < nov; j++) {
    for(int i = 0; i < nov; i++) {
      int val = mat[i * nov + j];
      if(val > 0 && mark[i] == 0) {
	rperm[current++] = i;
	mark[i] = 1;
      }
    }
  }

  int* n_mat = new int[nov * nov];
  memset(n_mat, 0, sizeof(int) * nov * nov);
  for(int i = 0; i < nov; i++) {
    for(int j = 0; j < nov; j++) {
      n_mat[(i * nov) + j] = mat[(rperm[i] * nov) + j];
    }
  }
  memcpy(mat, n_mat, nov * nov * sizeof(int));
  delete [] n_mat;
  delete [] rperm;
  delete [] mark;
}

void firstSeenCol(int* mat, int nov) {
  int* cperm = new int[nov];
  int* mark = new int[nov];
  memset(mark, 0, sizeof(int) * nov);
  int current = 0;
  for(int i = 0; i < nov; i++) {
    for(int j = 0; j < nov; j++) {
      int val = mat[i * nov + j];
      if(val > 0 && mark[j] == 0) {
        cperm[current++] = j;
	mark[j] = 1;
      }
    }
  }

  int* n_mat = new int[nov * nov];
  memset(n_mat, 0, sizeof(int) * nov * nov);
  for(int i = 0; i < nov; i++) {
    for(int j = 0; j < nov; j++) {
      n_mat[(i * nov) + j] = mat[(i * nov) + cperm[j]];
    }
  }
  memcpy(mat, n_mat, nov * nov * sizeof(int));
  delete [] n_mat;
  delete [] cperm;
  delete [] mark;
}

void sortRCM(int* mat, int nov) {
  int* n_mat = new int[nov * nov];
  memcpy(n_mat, mat, nov * nov * sizeof(int));

  for(int i = 0; i < nov; i++) {
    for(int j = 0; j < nov; j++) {
      if(n_mat[i*nov + j] != 0) {
	n_mat[(i * nov) + j] = 1;//n_mat[(j * nov) + i] = 1;
      }
    }
  }

  int* adj_row = new int[nov + 1];  
  int nnz = 1;
  for(int i = 0; i < nov; i++) {
    adj_row[i] = nnz;
    for(int j = 0; j < nov; j++) {
      if(n_mat[(i * nov) + j] != 0) {
	nnz++;
      }
    }
  }
  adj_row[nov] = nnz;
  
  int* adj = new int[nnz-1];
  nnz = 0;
  for(int i = 0; i < nov; i++) {
    for(int j = 0; j < nov; j++) {
      if(n_mat[(i * nov) + j] != 0) {
	adj[nnz++] = j + 1;
      }
    }
  }
  
  int* perm = new int[nov];
  genrcm (nov, nnz, adj_row, adj, perm);

  memset(n_mat, 0, sizeof(int) * nov * nov);
  for(int i = 0; i < nov; i++) {
    for(int j = 0; j < nov; j++) {
      n_mat[(i * nov) + j] = mat[((perm[i] - 1) * nov) + perm[j] - 1];
    }
  }
  memcpy(mat, n_mat, nov * nov * sizeof(int));

  delete [] perm;  
  delete [] adj;
  delete [] adj_row;
  delete [] n_mat;
}

void bfsOrder(int* mat, int nov) {

  vector<pair<int, int> > c_degs;

  for(int j = 0; j < nov; j++) {
    int cdeg = 0;
    for(int i = 0; i < nov; i++) {
      if(mat[(i * nov) + j] != 0) {
        cdeg++;
      }
    }
    c_degs.push_back(make_pair(cdeg, j));
  }
  sort(c_degs.begin(), c_degs.end(), pairCompare_inc);
  int* corder = new int[nov];
  
  int* visited = new int[nov];
  memset(visited, 0, sizeof(int) * nov);
  int* que = new int[nov];

  int qp = 0;
  int qep = 0;
  
  int cr = 0;
  int* rque = new int[nov];
  int* rvisited = new int[nov];
  memset(rvisited, 0, sizeof(int) * nov);

  for(int j = 0; j < nov; j++) {
    //  for(int j = nov-1; j >- 0; j--) {
    int col = c_degs[j].second;
    if(visited[col] == 0) {
      que[qep++] = col;
      visited[col] = 1;
      //      cout <<col << " " << qp << " " << qep << " " << nov << endl;
      
      while(qp < qep) {
	int ccol = que[qp++];
	int startq = qep;
       	for(int i = 0; i < nov; i++) {
	  if(mat[i* nov + ccol] != 0) {
	    if(rvisited[i] == 0) {
	      rvisited[i] = 1;
	      rque[cr++] = i;
	      //      cout << "\t" << i << " " << cr << endl;
	    }
	    for(int j2 = 0; j2 < nov; j2++) {
	      if(mat[i * nov + j2] != 0) {
		if(visited[j2] == 0) {
		  que[qep++] = j2;
		  //  cout << j2 << " " << qp << " " << qep << " " << nov << endl;
	  
		  visited[j2] = 1;
		}
	      }
	    }
	  }
	}
	int endq = qep;

	//	cout << "cord: ";
	memset(corder, 0, sizeof(int) * nov);
	for(int si = startq; si < endq; si++) {
	  int ncnt = 0;
	  int tcol = que[si];
	  for(int i = 0; i < nov; i++) {
	    if(mat[i * nov + tcol] != 0) {
	      if(rvisited[i] == 0) {
		ncnt++;
	      }
	    }
	  }
	  corder[tcol] = ncnt;
	  //cout << ncnt << " ";
	}
	//cout << endl;

	for(int si = startq; si < endq - 1; si++) {
	  for(int sj = startq; sj < endq - 1; sj++) {
	    if(corder[que[sj]] > corder[que[sj + 1]]) {
	      int tmp = que[sj];
	      que[sj] = que[sj+1];
	      que[sj+1] = tmp;
	    }
	  }
	} 


	/*cout << "cord: ";
        for(int si = startq; si < endq; si++) {
	  cout << corder[que[si]] << " ";
	}
	cout << endl;*/
      }
    }
  }

  int* n_mat = new int[nov * nov];
  memset(n_mat, 0, sizeof(int) * nov * nov);
  for(int i = 0; i < nov; i++) {
    for(int j = 0; j < nov; j++) {
      n_mat[(i * nov) + j] = mat[(rque[i] * nov) + que[j]];
    }
  }
  memcpy(mat, n_mat, nov * nov * sizeof(int));

  delete [] n_mat;
  delete [] que;
  delete [] rque;
  delete [] visited;
  delete [] rvisited;
}

void sortMinNew(int* mat, int nov) {
  int* cdegs = new int[nov];
  memset(cdegs, 0, sizeof(int) * nov);
  for(int i = 0; i < nov; i++) {
    for(int j = 0; j < nov; j++) {
      if(mat[(i * nov) + j] != 0) {
	cdegs[j]++;
      }
    }
  }
  
  int* rperm = new int[nov]; int rc = 0;
  int* rseen = new int[nov]; memset(rseen, 0, sizeof(int) * nov);
  int* cperm = new int[nov]; int cc = 0;
  int* cseen = new int[nov]; memset(cseen, 0, sizeof(int) * nov);

  for(int j = 0; j < nov; j++) {
    int mindegcol = -1;
    for(int j2 = 0; j2 < nov; j2++) {
      if(cseen[j2] == 0) {
	if(mindegcol == -1 || cdegs[mindegcol] > cdegs[j2]) {
	  mindegcol = j2;
	}
      }
    }

    cseen[mindegcol] = 1;
    cperm[cc++] = mindegcol;
    
    for(int i = 0; i < nov; i++) {
      if(mat[(i * nov) + mindegcol] != 0) {
	if(rseen[i] == 0) {
	  rseen[i] = 1;
	  rperm[rc++] = i;
	  for(int j3 = 0; j3 < nov; j3++) {
	    if(mat[(i * nov) + j3] != 0) {
	      cdegs[j3]--;
	    }
	  }
	}
      }
    }
  }
  
int* n_mat = new int[nov * nov];
  memset(n_mat, 0, sizeof(int) * nov * nov);
  for(int i = 0; i < nov; i++) {
    for(int j = 0; j < nov; j++) {
      n_mat[(i * nov) + j] = mat[(rperm[i] * nov) + cperm[j]];
    }
  }
  memcpy(mat, n_mat, nov * nov * sizeof(int));

  delete [] rseen;
  delete [] cseen;
  delete [] rperm;
  delete [] cperm;
}

bool checkEmpty(int* mat, int nov) {
  for(int i = 0; i < nov; i++) {
    if(getRowNnz(i, mat, nov) == 0) {
      return true;
    }
    if(getColNnz(i, mat, nov) == 0) {
      return true;
    }
  }
  return false;
}

bool d1compress(int* mat, int& nov) {
  int d1row = -1, d1col = -1;
  for(int i = 0; i < nov; i++) {
    if(getRowNnz(i, mat, nov) == 1) {
      d1row = i;
    } 
    if(getColNnz(i, mat, nov) == 1) {
      d1col = i;
    }
  }

  if(d1row == -1 && d1col == -1) {
    return false;
  }
  
  int val;
  if(d1row != -1) {
    for(int j = 0; j < nov; j++) {
      if(mat[(d1row * nov) + j] > 0) {
	val = mat[(d1row * nov) + j];
	d1col = j;
	break;
      }
    }
  } else if(d1col != -1) {
    for(int j = 0; j < nov; j++) {
      if(mat[(j * nov) + d1col] > 0) {
	val = mat[(j * nov) + d1col];
        d1row = j;
        break;
      }
    }
  }

  //remove d1row and d1col now
  int* n_mat = new int[nov * nov];
  for(int i = 0; i < nov; i++) {
    for(int j = 0; j < nov; j++) {
      if(i != d1row && j != d1col) {
	int val = mat[(i * nov) + j];
	
	int rloc = i; if(i > d1row) rloc--;
	int cloc = j; if(j > d1col) cloc--;

	n_mat[rloc * (nov - 1) + cloc] = val;
      }
    }
  }
  nov = nov - 1;
  memcpy(mat, n_mat, sizeof(int) * nov * nov);
 
  for(int j = 0; j < nov; j++) {
    mat[j] *= val;
  }
 
  delete [] n_mat;
  return true;
}

//double threshold = 20;
bool d2compress(int* mat, int& nov) {
  int d2row = -1, d2col = -1;

  for(int i = 0; i < nov; i++) {
    if(getRowNnz(i, mat, nov) == 2) d2row = i;
    if(getColNnz(i, mat, nov) == 2) d2col = i;
    if(d2row != -1 || d2col != -1) break;
  }

  if(d2row == -1 && d2col == -1) return false;

  int nbr1 = -1, nbr2 = -1;
  if(d2row != -1) {
    for(int j = 0; j < nov; j++) {
      if(mat[(d2row * nov) + j] > 0) {
	if(nbr1 == -1) { nbr1 = j; } 
	else { nbr2 = j; break; }
      }
    }
    //logic bu olmali
    /*if(fabs(mat[d2row * nov + nbr1]) > threshold || fabs(mat[d2row * nov + nbr2]) > threshold) {
      return false;
      }*/
  } else if(d2col != -1) {
    for(int j = 0; j < nov; j++) {
      if(mat[(j * nov) + d2col] > 0) {
	if(nbr1 == -1) { nbr1 = j; } 
	else { nbr2 = j; break; }
      }
    }
  }
  
  int* n_mat = new int[nov * nov];
  if(d2row != -1) {
    //deleting d2row and col nbr2
    for(int i = 0; i < nov; i++) {
      for(int j = 0; j < nov; j++) {
	if(i != d2row && j != nbr2) {
	  int rloc = i; if(i > d2row) rloc--;
	  int cloc = j; if(j > nbr2) cloc--;

	  int val = mat[(i * nov) + j];
	  if(j == nbr1) {
	    val = (mat[(i * nov) + nbr1] * mat[(d2row * nov) + nbr2]) +
	          (mat[(i * nov) + nbr2] * mat[(d2row * nov) + nbr1]);

	  }
	  n_mat[(rloc * (nov - 1)) + cloc] = val;
	}
      }
    }
  } else if(d2col != -1) {
    //deleting d2col and row nbr2 
    for(int i = 0; i < nov; i++) {
      for(int j = 0; j < nov; j++) {
	if(i != nbr2 && j != d2col) {
          int rloc = i; if(i > nbr2) rloc--;
          int cloc = j; if(j > d2col) cloc--;

          int val = mat[(i * nov) + j];
          if(i == nbr1) {
            val = (mat[(nbr1 * nov) + j] * mat[(nbr2 * nov) + d2col]) +
	          (mat[(nbr2 * nov) + j] * mat[(nbr1 * nov) + d2col]);
          }
          n_mat[(rloc * (nov - 1)) + cloc] = val;
        }
      }
    }
  }
  nov = nov - 1;
  memcpy(mat, n_mat, sizeof(int) * nov * nov);

  delete [] n_mat;
  return true;
}

bool d34compress(int* mat, int& nov, int*& mat2, int& nov2, int minDeg) {
  int drow = -1, dcol = -1;

  for(int i = 0; i < nov; i++) {
    if(getRowNnz(i, mat, nov) == minDeg) drow = i;
    if(getColNnz(i, mat, nov) == minDeg) dcol = i;
    if(drow != -1 || dcol != -1) break;
  }
  if(drow == -1 && dcol == -1) return false;

  int* t_mat = new int[nov * nov];
  if(drow == -1) {
    for(int i = 0; i < nov; i++) {
      for(int j = 0; j < nov; j++) {
	t_mat[j * nov + i] = mat[i * nov + j];
      }
    }
    drow = dcol;
  } else {
    memcpy(t_mat, mat, sizeof(int) * nov * nov);
  }

  int nbrs[4] = {-1, -1, -1, -1};
  int index = 0;
  int zeroloc = -1;
  for(int j = 0; j < nov; j++) {
    if(t_mat[drow * nov + j] != 0) {
      nbrs[index++] = j;
    } else {
      zeroloc = j;
    }
  }

  if(nbrs[3] == -1) {
    nbrs[3] = zeroloc;
  }

  mat2 = new int[nov * nov];
  //  cout << "generated " << mat2 << endl;
  memset(mat, 0, sizeof(int) * nov * nov);
  memset(mat2, 0, sizeof(int) * nov * nov);

  for(int i = 0; i < nov; i++) {
    if(i != drow) {
      int iloc = i; if(i > drow) iloc--;

      for(int j = 0; j < nov; j++) {
	if(j != nbrs[1]) {
	  int jloc = j; if(j > nbrs[1]) jloc--;
	  if(j != nbrs[0]) {
	    mat[iloc * (nov-1) + jloc] = t_mat[i * nov + j];
	  } else {
	    mat[iloc * (nov-1) + jloc] = (t_mat[drow * nov + nbrs[0]] * t_mat[i * nov + nbrs[1]]) + 
	                             (t_mat[drow * nov + nbrs[1]] * t_mat[i * nov + nbrs[0]]);
	  }
	}

	if(j != nbrs[3]) {
          int jloc = j; if(j > nbrs[3]) jloc--;
          if(j != nbrs[2]) {
            mat2[iloc * (nov-1) + jloc] = t_mat[i * nov + j];
          } else {
	    mat2[iloc * (nov-1) + jloc] = (t_mat[drow * nov + nbrs[2]] * t_mat[i * nov + nbrs[3]]) +
	                                  (t_mat[drow * nov + nbrs[3]] * t_mat[i * nov + nbrs[2]]);
          }
	}
      }
    }
  }

  nov = nov - 1;
  nov2 = nov;
  delete [] t_mat;
  return true;
}

void reach(int* xadj, int* adj, int nov, bool* visited, int* que, int source) {
  for(int i = 0; i < nov; i++) {
    visited[i] = false;
  }

  que[0] = source;
  visited[source] = true;
  int qp = 0, qe = 1;
  
  while(qp < qe) {
    int curr = que[qp++];
    
    for(int ptr = xadj[curr]; ptr < xadj[curr + 1]; ptr++) {
      int nbr = adj[ptr];
      if(!visited[nbr]) {
	visited[nbr] = true;
	que[qe++] = nbr;
      }
    }
  }
}

void dulmen(int* mat, int nov) {
  int *xadj, *adj, *val;
  matrix2graph(mat, nov, xadj, adj, val);

  //matching array - we need a valid matching
  int* rmatch = new int[nov];
  int* cmatch = new int[nov];
  for(int i = 0; i < nov; i++) {rmatch[i] = cmatch[i] = -1;}

  //mapping the larger vertex ids to [0...nov-1]
  int* nadj = new int[xadj[nov]];
  for(int ptr = 0; ptr < xadj[nov]; ptr++) {
    nadj[ptr] = adj[ptr] - nov;
  }
  match(xadj, nadj, rmatch, cmatch, nov, nov);
  
  /*  for(int i = 0; i < nov; i++) {
    cout << i << " " << rmatch[i] << " " << cmatch[rmatch[i]] << endl;
    }*/

  //check if the matching is full or not
  int mcount = 0;
  for(int i = 0; i < nov; i++) {
    if(rmatch[i] >= 0) {
      mcount++;
      if(cmatch[rmatch[i]] != i) {
	cout << "Weird matching " << endl;
	exit(1);
      }
    }
  }

  cout << "Match count is " << mcount << endl;
  if(mcount != nov) {
    cout << "Not matching" << endl;
    cout << "Perman is 0" << endl;
    exit(1);
  }

  // a new graph type wrt to the obtained matching
  int* gxadj = new int[nov + 1];
  int* gadj = new int[xadj[nov]];
  
  gxadj[0] = 0;
  int ptr = 0;
  for(int i = 0; i < nov; i++) {
    int matched = rmatch[i];
    for(int ptr2 = xadj[i]; ptr2 < xadj[i+1]; ptr2++) {
      int nbor = adj[ptr2];
      if(nbor != matched + nov) {
	gadj[ptr++] = cmatch[nbor - nov];
      }
    }
    gxadj[i+1] = ptr;
  }
  
  // find the components in DM decomposition
  int* component = new int[nov];
  bool* visit1 = new bool[nov];
  bool* visit2 = new bool[nov];
  int* que = new int[nov];
  for(int i = 0; i < nov; i++) {
    component[i] = -1;
  }

  int cid = 0;
  for(int i = 0; i < nov; i++) {
    if(component[i] == -1) {
      component[i] = cid;
      reach(gxadj, gadj, nov, visit1, que, i);
      for(int j = 0; j < nov; j++) {
	if(i != j && component[j] == -1 && visit1[j]) {
	  reach(gxadj, gadj, nov, visit2, que, j);
	  if(visit2[i]) {
	    component[j] = cid;
	  }
	}
      }
      cid++;
    }
  }
  cout << "Components: ";
  for(int i = 0; i < nov; i++) {
    cout << component[i] << " ";
  }
  cout << endl;
  
  int erased = 0;
  for(int i = 0; i < nov; i++) {
    for(int j = 0; j < nov; j++) {
      if(mat[(i * nov) + j] > 0) {
	if(component[i] != component[j]) {
	  mat[(i * nov) + j] = 0;
	  erased++;
	}
      }
    }
  }
  cout << "no erased edges: " << erased << endl;

  delete [] component;
  delete [] visit1;
  delete [] visit2;
  delete [] que;
  delete [] gxadj;
  delete [] gadj;
  delete [] nadj;
  delete [] rmatch;
  delete [] cmatch;
}
