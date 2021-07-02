unloadlibrary libConnect
loadlibrary("libConnect.so", "matlab_calculate_return.h")

m = libfunctions("libConnect", '-full')
m

%mat = [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1]
mat = [1,0,0,5,6,0,1,0,0,4,8,0,1,0,0,2,0,0,1,0,0,1,1,1,1]


perman = calllib('libConnect', 'matlab_calculate_return', mat, 2,2,2,2,2,5,13)
perman