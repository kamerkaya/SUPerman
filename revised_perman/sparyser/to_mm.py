reader = open("mat.txt", "r")
lines = reader.readlines()
#print(lines)

lines2 = []
for item in lines:
    lines2.append(item.strip("\n"))

#print(lines2)

writer = open("mat2.txt", "w")
writer.write("%%MatrixMarket matrix coordinate integer general\n")
writer.write("%-------------------------------------------------------------------------------\n")

nnzs = 0
int_nnzs = []

for i in range(len(lines2)):
    str_nnzs = lines2[i].split(' ')
    print(str_nnzs)
    
    for j in range(len(str_nnzs)):
        nnz = int(str_nnzs[j])
        if(nnz != 0):
            nnzs += 1
            int_nnzs.append([nnz, [i+1, j+1]])

            
writer.write(str(len(lines2)) + ' ' + str(len(lines2)) + ' ' + str(nnzs) + '\n')
    
for item in int_nnzs:
    writer.write(str(item [1][0]) + ' ' + str(item[1][1]) + ' ' + str(item[0]) + '\n')
