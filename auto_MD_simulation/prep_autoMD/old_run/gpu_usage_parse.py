#!/usr/bin/python3
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('usage_file')
parser.add_argument('timestamp')
args = parser.parse_args()

if __name__ == '__main__':
    with open(args.usage_file, 'r') as f:
        gpu_usage = [x.strip() for x in f.readlines()]

    # print(gpu_usage)
    nodes = dict()
    current_node = None
    for line in gpu_usage:
        cols = line.split()
        if line.startswith('node'):
            node = cols[0][:-1]
            current_node = node
            if node in ('node75', 'node76', 'node77', 'node78'):
                nodes[node] = ['PROWAVE', 'PROWAVE']
                continue
            elif node not in nodes:
                nodes[node] = [None]
            else:
                nodes[node].append(None)
        else:
            if len(cols) == 1:
                continue

            if cols[1] not in ('pmemd.cuda', 'pmemd.cuda.MPI', 'rism3d-x', '/opt/nbcc/bin/rism3d-x'):
                continue

            if not nodes[current_node][0]:
                nodes[current_node][0] = cols[0]
            else:
                nodes[current_node][1] = cols[0]

    print('Checked at %s' % args.timestamp)
    print('<table>')
    print('<tr>')
    print('<td>node</td><td>Gpu1</td><td>Gpu2</td>')
    print('</tr>')
    empty = '<td><span style="color: red">EMPTY</span></td>'
    no_gpu = '<td><span style="color: grey">No GPU</span></td>'
    for node, users in sorted(nodes.items(), key=lambda x: int(x[0][4:])):
        print('<tr>')
        print('<td>%s</td>' % node, end=' ')
        if len(users) > 1:
            print('<td>%s</td>' % users[0] if users[0] else empty, '<td>%s</td>' % users[1] if users[1] else empty)
        else:
            print('<td>%s</td>' % users[0] if users[0] else empty, no_gpu)
        print('</tr>')
    print('</table>')
