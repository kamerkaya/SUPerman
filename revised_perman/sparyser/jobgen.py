import os
import math
import os.path
import sys
import datetime
import time

if len(sys.argv) <> 3:
    print "python <jobgen.py> <path_to_matrix_folder> <exec:gather>"   
    sys.exit()
    
probs = ["0.20", "0.30", "0.40", "0.50"]
algos = ["11"] #, ["1", "2", "3", "10", "11"]
ns = ["32", "34", "36"]
sort = ["3"]
noexec = 3

path = os.path.abspath(sys.argv[1])
option = sys.argv[2]
currentPath = os.path.dirname(os.path.abspath(sys.argv[0])) + "/"
                              
for nindex in range(len(ns)):
    n = ns[nindex];
    for pindex in range(len(probs)):
        p = probs[pindex];
        for index in range(len(algos)):
            algo = algos[index]
            for sindex in range(len(sort)):
                srt = sort[sindex]
                for x in range(noexec):
                    mat_file = path + "/erdos_" + n + "." + p + "." + str(x) + ".mtx";
                
                    single_out_file = currentPath + "Results/" + "erdos_n" + n + "p" + p + "a" + algo + "s" + srt + "x" + str(x) + ".single.out";
                    single_exec_string = "./sparyser " + mat_file + " " + algo + " " + "0 0 " + srt + " 1 > " + single_out_file; 
                    print single_exec_string
                    if (algo == "10") or (algo == "11"):
                        os.environ["OMP_NUM_THREADS"] = "16";
                        os.system("timeout 3600s " + single_exec_string)
                    else:
                        os.system("timeout 3600s " + single_exec_string + " &")
