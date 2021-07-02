from ctypes import *
import argparse

stdc = cdll.LoadLibrary("libc.so.6") # or similar to load c library
stdcpp = cdll.LoadLibrary("libstdc++.so.6") # or similar to load c++ library
libConnect = cdll.LoadLibrary('./libConnect.so')
libConnect.connect()
##Connection initiated##

parser = argparse.ArgumentParser(description="SUPerman: Fast Permanent Calculating Tool")
parser.add_argument("-f", "--filename", type=str)
parser.add_argument("-a", "--algorithm", type=int)
parser.add_argument("-t", "--nothreads", type=int)
parser.add_argument("-x", "--repeat", type=int)
parser.add_argument("-y", "--sinterval", type=int)
parser.add_argument("-z", "--stime", type=int)
args = parser.parse_args()
##Parse arguments


libConnect.read_calculate_return.argtypes = [c_char_p, c_int, c_int, c_int, c_int, c_int]
#->fname, algorithm, no_threads
libConnect.read_calculate_return.restype = c_double


#fname = c_char_p('double/33_0.10_0'.encode('utf-8'))
fname = args.filename.encode('utf-8')

perman = libConnect.read_calculate_return(fname, args.algorithm, args.nothreads, args.repeat, args.sinterval, args.stime)


print('Perman: ', perman)
