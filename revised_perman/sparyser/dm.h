#define etype int
#define vtype int


void reach(etype* xadj, vtype* adj, vtype nov, bool* visited, vtype* que, vtype source) {
  for(vtype i = 0; i < nov; i++) {
    visited[i] = false;
  }

  que[0] = source;
  visited[source] = true;
  vtype qp = 0, qe = 1;
  
  while(qp < qe) {
    vtype curr = que[qp++];
    
    for(etype ptr = xadj[curr]; ptr < xadj[curr + 1]; ptr++) {
      vtype nbr = adj[ptr];
      if(!visited[nbr]) {
	visited[nbr] = true;
	que[qe++] = nbr;
      }
    }
  }
}


void dulmage_mendehlson() {
  int* rmatch = new int[hnov];
  int* cmatch = new int[hnov];
  vtype* nadj = new vtype[xadj[hnov]];
  for(etype ptr = 0; ptr < xadj[hnov]; ptr++) {nadj[ptr] = adj[ptr] - hnov;}
  for(vtype i = 0; i < hnov; i++) {rmatch[i] = cmatch[i] = -1;}
  match(xadj, nadj, rmatch, cmatch, hnov, hnov);

  
  vtype mcount = 0;
  for(vtype i = 0; i < hnov; i++) {
    if(rmatch[i] >= 0) {
      mcount++;
      if(cmatch[rmatch[i]] != i) {
	cout << "Weird matching " << endl;
	exit(1);
      }
    }
  }
  cout << "Match count is " << mcount << endl;
  if(mcount != hnov) {
    cout << "Perman is 0" << endl;
    exit(1);
  }

  vtype* gxadj = new vtype[hnov+1];
  etype* gadj = new etype[xadj[hnov]];
  
  gxadj[0] = 0;
  etype ptr = 0;
  for(vtype i = 0; i < hnov; i++) {
    vtype matched = rmatch[i];
    for(etype ptr2 = xadj[i]; ptr2 < xadj[i+1]; ptr2++) {
      vtype nbor = adj[ptr2];
      if(bor != matched + hnov) {
	gadj[ptr++] = cmatch[nbor - hnov];
      }
    }
    gxadj[i+1] = ptr;
  }
  
  // printGraph(gxadj, gadj, val, hnov);

  int* component = new int[hnov];
  bool* visit1 = new bool[hnov];
  bool* visit2 = new bool[hnov];
  int* que = new int[hnov];
  for(vtype i = 0; i < hnov; i++) {
    component[i] = -1;
  }

  int cid = 0;
  for(vtype i = 0; i < hnov; i++) {
    if(component[i] == -1) {
      component[i] = cid;
      
      reach(gxadj, gadj, hnov, visit1, que, i);
      
      for(vtype j = 0; j < hnov; j++) {
	if(i != j && component[j] == -1 && visit1[j]) {
	  reach(gxadj, gadj, hnov, visit2, que, j);
	  
	  if(visit2[i]) {
	    component[j] = cid;
	  }
	}
      }
      cid++;
    }
  }
  
  cout << "comps: ";
  for(vtype i = 0; i < hnov; i++) {
    cout << component[i] << " ";
  }
  cout << endl;

  vtype erased = 0;
  ptr = 0;
  etype* xadj_t = new etype[nov+1];
  for(vtype i = 0; i <= nov; i++) {
    xadj_t[i] = xadj[i];
  }

  for(vtype i = 0; i < hnov; i++) {
    for(etype ptr2 = xadj_t[i]; ptr2 < xadj_t[i+1]; ptr2++) {
      if(component[i] == component[adj[ptr2] - hnov]) {
	adj[ptr++] = adj[ptr2];
      } else {
	mat[(i * hnov) + adj[ptr2] - hnov] = 0;
	erased++;
      }
    }
    xadj[i+1] = ptr;
  }

  
  for(vtype i = hnov; i < nov; i++) {
    for(etype ptr2 = xadj_t[i]; ptr2 < xadj_t[i+1]; ptr2++) {
      if(mat[adj[ptr2] * hnov + (i - hnov)] == 1) {
        adj[ptr++] = adj[ptr2];
      }
    }
    xadj[i+1] = ptr;
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
  delete [] xadj_t;
}
