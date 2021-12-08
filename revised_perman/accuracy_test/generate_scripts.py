import os
import sys

def scrape_command(command):

    scraped = command[2:]
    scraped = scraped.replace(" ", "_")

    return scraped


if __name__ == "__main__":

    commands_file = sys.argv[1]
    reader = open(commands_file, 'r')
    lines = reader.readlines()
    reader.close()
    #print(lines)

    scrl = []
    counter = 0;

    for line in lines:
        to_write = ''
        command = line.strip('\n')
        to_write += '####' + command + '####' + '\n'
        out_scraped = scrape_command(command)
        out_name = 'out_' + str(counter) + '_' + out_scraped
        to_write += out_name + '=$(' + command + ' 1>' + ' ' + out_name + '.stdtxt' + ' 2> ' + ' ' + out_name + '.errtxt' + ')'+ ' \n'
        to_write += 'wait \n'
        scrl.append(to_write)

        counter += 1


    writer = open(commands_file.strip('.txt') + '_scripts.sh', 'w+')
    for item in scrl:
        writer.write(item)

    writer.close()
        
