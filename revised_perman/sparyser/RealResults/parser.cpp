#include <iostream>
#include <string>
#include <cstring>
#include <vector>
#include <map>
#include <cmath>
#include <stdio.h>
#include <iomanip>      // std::setprecision

using namespace std;

const int x = 15;
struct data_p {
public:
  int index;
  char* matrix;
  double* permans;
  double* times;

  data_p() {
    matrix = new char[1000];
    permans = new double[x];
    times = new double[x];
  }
};

int main(int argc, char* argv[]) {
  
  char* filename = argv[1];
  FILE* fp = fopen(filename, "r");
  
  char file[1000], temp[1000], matrix[1000];
  int algo, sorting, id, nov; 
  double perman, time;
  
  std::map<std::string, int> m;
  std::map<std::string, int> mc;
  
  int res_count = 0;
  vector<data_p*> results;

  //algo: 1,2,3,10
  //sorting; 0,1,3

  cout << "reading " << endl;
  while (fscanf(fp, "%s %s %d %d %s %d %lf %lf\n", file, temp, &algo, &sorting, matrix, &nov, &perman, &time)  != EOF) {
    cout << file << " " << temp << " " << algo << " " << sorting << " " << matrix << " " << nov << " " << perman << " " << time << endl;    
    //if(algo == 11) continue;
    //if(algo == 10) continue;
    //if(algo == 1 && sorting != 0) continue;
    //if(algo == 2 && sorting != 1) continue;
    //if(algo == 3 && sorting != 3) continue;

    if (mc.find(file) != mc.end()) {
      mc[file] = mc[file] + 1;
    } else {
      mc[file] = 0;
    }
    if(mc[file] > 1000) continue;
    
    int index = -1;
    data_p* p;
    if (m.find(matrix) != m.end()) {
      int index = m[matrix];
      p = results[index];
    } else {
      p = new data_p();
      memcpy(p->matrix, matrix, 1000 * sizeof(char));
      for(int i = 0; i < x; i++) {
	p->permans[i] = -1;
	p->times[i] = -1;
      }
      p->index = res_count;      
      results.push_back(p);

      index = res_count;
      m[matrix] = index;
      res_count++;
    }
    algo--;
    if(algo == 10) algo = 4;
    if(algo == 9) algo = 3;
    if(sorting == 3) sorting = 2;

    int index2 = (algo * 3) + sorting;
    cout << index2 << endl;
    p->permans[index2] = perman;
    p->times[index2] = time;
  }
  cout << res_count << endl;
  //print permans
  for(int i = 0; i < res_count; i++) {
    cout << i << " " << results[i]->matrix << ": ";
    for(int j = 0; j < x; j++) {
      cout << results[i]->permans[j] << " ";
    }
    cout << endl;
  }

  //print times
  for(int i = 0; i < res_count; i++) {
    cout << i << " " << results[i]->matrix << ": ";
    for(int j = 0; j < x; j++) {
      cout << results[i]->times[j] << " ";
    }
    cout << endl;
  }

  int counts[15]; memset(counts, 0, sizeof(int) * 15);
  int best_counts[15]; memset(best_counts, 0, sizeof(int) * 15);
  double rel_perf[15]; memset(rel_perf, 0, sizeof(int) * 15);
  for(int j = 0; j < 15; j++) {best_counts[j] = counts[j] = rel_perf[j] = 0;}
  for(int i = 0; i < res_count; i++) {    
    double min_time = 1000000;
    for(int j = 0; j < 15; j++) {		
      if(results[i]->permans[j] != -1) {
	if(results[i]->times[j] < min_time) {
	  min_time = results[i]->times[j];
	}
      }
    }
    
    for(int j = 0; j < 15; j++) {
      if(results[i]->times[j] != -1) {
	counts[j]++;
	if((results[i]->times[j]/min_time) <= 1.05) {
	  best_counts[j]++;
	}
      }
    }
  }
  
  cout << "counts ----- " << res_count << endl;
  for(int a = 0; a <= 4; a++) {
    for(int s = 0; s < 3; s++) {
      int index2 = (a * 3) + s;
      cout << counts[index2] << "\t";
    }
    cout << endl;
  }
  cout << "-----------------------" << endl;

  cout << "best_counts ----- " << res_count << endl;
  for(int a = 0; a <= 4; a++) {
    for(int s = 0; s < 3; s++) {
      int index2 = (a * 3) + s;
      cout << best_counts[index2] << "\t";
    }
    cout << endl;
  }
  cout << "-----------------------" << endl;

  int algos[15] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  int nalgo = 15;
  for(int i = 0; i < nalgo; i++) {
    int a1 = algos[i];
    for(int j = 0; j < nalgo; j++) {
      int a2 = algos[j];
      
      double metric = 0;
      for(int k = 0; k < res_count; k++) {
	double t1 = results[k]->times[a1];
	double t2 = results[k]->times[a2];

	if(t1 != -1) {
	  if(t1/t2 <= 1.05) {
	    metric++;
	  } 
	}
      }
      cout << metric << "\t";
    }
    cout << endl;
  }
  cout << "-------------" << endl;
  cout.precision(3);
  for(int i = 0; i < nalgo; i++) {
    for(int j = 0; j < nalgo; j++) {
      int a1 = algos[i];
      int a2 = algos[j];
      
      double metric = 0;
      int cnt = 0;
      for(int k = 0; k < res_count; k++) {
	double t1 = results[k]->times[a1];
	double t2 = results[k]->times[a2];
	if((t1 != -1) && (t2 != -1)) {
	  cnt++;
	  metric += t2 / t1;
	  //	  if(a1 == 0 && a2 == 8) cout << t1 << " " << " " << t2 << " " << t2/t1 << endl;
	}
      }
      if(a1 < a2) {
	cout << "--\t";
      }	else {
	cout <<  metric/cnt << "\t";
      }
    }
    cout << endl;
  }
  cout << "-------------" << endl;
    
  return 0;
}
