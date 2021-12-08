import sys




def gen_gpu_commands(gpu_base_str, gpu_algos, gpu_types, gpu_grid_size, scaling):

    commands = []
    
    for algo in gpu_algos:
        for types in gpu_types:
            for gs in gpu_grid_size:
                for scale in scaling:

                    if(algo == "-p4"):
                        new_algo = "-p4 -r1"
                    elif(algo == "-p14"):
                        new_algo = "-p14 -r2"
                    else:
                        print("wut", algo)
                        exit(1)

                    command = gpu_base_str + " " + new_algo

                    if(types != ""):
                        command += " " + types

                    if(gs != ""):
                        command += " " + gs

                    if(scale != ""):
                        command += " " + scale

                    commands.append(command)

    return commands


def gen_cpu_commands(cpu_base_str, cpu_algos, cpu_types, scaling):

    commands = []

    for algo in cpu_algos:
        for types in cpu_types:
            for scale in scaling:

                #print(scaling, types, scale=="-u2", types=="-v")
                if(scale == "-u2" and types == "-v"):
                    #print("Here types: ", types)
                    continue

                if(scale == "-u2" and types == "-q -v"):
                    #print("Here types: ", types)
                    continue
                

                if(algo == "-p1"):
                    new_algo = "-p1 -r1"
                elif(algo == "-p3"):
                    new_algo = "-p3 -r2"
                else:
                    print("wut", algo)
                    exit(1)

                command = cpu_base_str + " " + new_algo

                if(types != ""):
                    command += " " + types

                if(scale != ""):
                    command += " " + scale

                commands.append(command)

    return commands
    



if __name__ == "__main__":

    gpu_base_str = "./gpu_perman -f mat2r.txt -s -k5"
    cpu_base_str = "./cpu_perman -f mat2r.txt -s -c -t88 -k5"

    gpu_algos = ["-p4", "-p14"]
    gpu_types = ["", "-h", "-h -w", "-w"]
    gpu_grid_size = ["", "-e2", "-e4", "-e8"]
    #scaling = ["", "-u1"]

    
    #gpu_commands = gen_gpu_commands(gpu_base_str, gpu_algos, gpu_types, gpu_grid_size, scaling)
    
    #for item in gpu_commands:
        #print(item)

    #writer = open('gpu_commmands.txt', 'w+')
    #for item in gpu_commands:
        #writer.write(item + '\n')


    cpu_algos = ["-p1", "-p3"]
    cpu_types = ["", "-h", "-h -w", "-w", "-q", "-v", "-q -v"]
    scaling = ["", "-u2"]

    cpu_commands = gen_cpu_commands(cpu_base_str, cpu_algos, cpu_types, scaling)

    for item in cpu_commands:
        print(item)

    writer = open('cpu_commands.txt', 'w+')
    for item in cpu_commands:
        writer.write(item + '\n')
