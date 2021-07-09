%unloadlibrary libConnect
loadlibrary("libConnect.so", "matlab_calculate_return.h")

m = libfunctions("libConnect", '-full')
m

mat = [1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1]
mat2 = [1.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0,1.0,0.0,1.0,0.0,1.0]


perman = calllib('libConnect', 'matlab_calculate_return_int', mat, 2,2,2,2,2,5,13)
perman

perman2 = calllib('libConnect', 'matlab_calculate_return_double', mat2, 2,2,2,2,2,5,13)
perman2

perman3 = calllib('libConnect', 'read_calculate_return', 'int/32_0.35_1', 6, 8, 10000, 5, 5)
perman3 