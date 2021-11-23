#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>

void match(int* col_ptrs, int* col_ids, int* match, int* row_match, int n, int m) {
  int* visited = (int*)malloc(sizeof(int) * m);
  int* stack = (int*)malloc(sizeof(int) * n);
  int* colptrs = (int*)malloc(sizeof(int) * n);
  int* lookahead = (int*)malloc(sizeof(int) * n);
  int* unmatched = (int*)malloc(sizeof(int) * n);

  int i, j, row, col, stack_col, temp, ptr, eptr, stack_last,
    stop = 0, pcount = 1, stack_end_ptr, nunmatched = 0, nextunmatched = 0,
    current_col, inc = 1;

  memset(visited, 0, sizeof(int) * m);
  memcpy(lookahead, col_ptrs, sizeof(int) * n);

  for(i = 0; i < n; i++) {
    if(match[i] == -1 && col_ptrs[i] != col_ptrs[i+1]) {
      unmatched[nunmatched++] = i;
    }
  }

  while(!stop) {
    stop = 1; stack_end_ptr = n;
    if(inc) {
      for(i = 0; i < nunmatched; i++) {
	current_col = unmatched[i];
	stack[0] = current_col; stack_last = 0; colptrs[current_col] = col_ptrs[current_col];

	while(stack_last > -1) {
	  stack_col = stack[stack_last];

	  eptr = col_ptrs[stack_col + 1];
	  for(ptr = lookahead[stack_col]; ptr < eptr && row_match[col_ids[ptr]] != -1; ptr++){}
	  lookahead[stack_col] = ptr + 1;

	  if(ptr >= eptr) {
	    for(ptr = colptrs[stack_col]; ptr < eptr; ptr++) {
	      temp = visited[col_ids[ptr]];
	      if(temp != pcount && temp != -1) {
		break;
	      }
	    }
	    colptrs[stack_col] = ptr + 1;

	    if(ptr == eptr) {
	      if(stop) {stack[--stack_end_ptr] = stack_col;}
	      --stack_last;
	      continue;
	    }

	    row = col_ids[ptr]; visited[row] = pcount;
	    col = row_match[row]; stack[++stack_last] = col; colptrs[col] = col_ptrs[col];
	  } else {
	    row = col_ids[ptr]; visited[row] = pcount;
	    while(row != -1){
	      col = stack[stack_last--];
	      temp = match[col];
	      match[col] = row; row_match[row] = col;
	      row = temp;
	    }
	    stop = 0;
	    break;
	  }
	}

	if(match[current_col] == -1) {
	  if(stop) {
	    for(j = stack_end_ptr + 1; j < n; j++) {
	      visited[match[stack[j]]] = -1;
	    }
	    stack_end_ptr = n;
	  } else {
	    unmatched[nextunmatched++] = current_col;
	  }
	}
      }
    } else {
      for(i = 0; i < nunmatched; i++) {
	current_col = unmatched[i];
	stack[0] = current_col; stack_last = 0; colptrs[current_col] = col_ptrs[current_col + 1] - 1;

	while(stack_last > -1) {
	  stack_col = stack[stack_last];

	  eptr = col_ptrs[stack_col + 1];
	  for(ptr = lookahead[stack_col]; ptr < eptr && row_match[col_ids[ptr]] != -1; ptr++){}
	  lookahead[stack_col] = ptr + 1;

	  if(ptr >= eptr) {
	    eptr = col_ptrs[stack_col] - 1;
	    for(ptr = colptrs[stack_col]; ptr > eptr; ptr--) {
	      temp = visited[col_ids[ptr]];
	      if(temp != pcount && temp != -1) {
		break;
	      }
	    }
	    colptrs[stack_col] = ptr - 1;

	    if(ptr == eptr) {
	      if(stop) {stack[--stack_end_ptr] = stack_col;}
	      --stack_last;
	      continue;
	    }

	    row = col_ids[ptr]; visited[row] = pcount;
	    col = row_match[row]; stack[++stack_last] = col;
	    colptrs[col] = col_ptrs[col + 1] - 1;

	  } else {
	    row = col_ids[ptr]; visited[row] = pcount;
	    while(row != -1){
	      col = stack[stack_last--];
	      temp = match[col];
	      match[col] = row; row_match[row] = col;
	      row = temp;
	    }
	    stop = 0;
	    break;
	  }
	}

	if(match[current_col] == -1) {
	  if(stop) {
	    for(j = stack_end_ptr + 1; j < n; j++) {
	      visited[match[stack[j]]] = -1;
	    }
	    stack_end_ptr = n;
	  } else {
	    unmatched[nextunmatched++] = current_col;
	  }
	}
      }
    }
    pcount++; nunmatched = nextunmatched; nextunmatched = 0; inc = !inc;
  }

  free(unmatched);
  free(lookahead);
  free(colptrs);
  free(stack);
  free(visited);
}
