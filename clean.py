import random
import sys


input_file = './data/cat-cluster.txt'
output_file = './data/cat-clean.txt'

cluster1 = [5, 13, 2]
reserve1 = 0.3

cluster2 = [25, 46, 89, 15, 41, 14, 67, 100, 171, 38, 128, 10, 17]
reserve2 = 0.5


def read_file():
    f = open(input_file)
    con = []
    for line in f:
        if line.startswith('Conversation'):
            con = []
            tmp = line
        elif line == '\n':
            if con == []:
                continue
            if con[0][0] == '1':
                con = con[1:]
            yield con
        else:
            con.append(line)
    f.close()


if __name__ == '__main__':
    state1 = {}
    for c in cluster1:
        state1[c] = [0, 0]
    state2 = {}
    for c in cluster2:
        state2[c] = [0, 0]

    data = read_file()
    fout = open(output_file, 'w')
    for conv in data:
        for i in range(0, len(conv) - 6, 2):
            c = int(conv[i + 5].split()[1])

            if c in cluster1:
                if random.random() > reserve1:
                    state1[c][1] += 1
                    continue
                else:
                    state1[c][0] += 1

            if c in cluster2:
                if random.random() > reserve2:
                    state2[c][1] += 1
                    continue
                else:
                    state2[c][0] += 1

            for j in range(i, i + 6):
                s = conv[j].split()
                t = conv[j].find(s[2])
                fout.write(s[0] + ' ' + conv[j][t:])
            fout.write('\n')
    fout.close()

    f = open('./data/clean-report.txt', 'w')
    sys.stdout = f
    print('Cluster', '  Total', 'Reserve', ' Reduce', '  Ratio')
    for c, a in state1.items():
        print('%7d %7d %7d %7d    %.2f' % (c, sum(a), a[0], a[1], a[1] / sum(a)))
    for c, a in state2.items():
        print('%7d %7d %7d %7d    %.2f' % (c, sum(a), a[0], a[1], a[1] / sum(a)))
