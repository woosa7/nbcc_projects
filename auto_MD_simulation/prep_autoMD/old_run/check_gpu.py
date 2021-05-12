#!/usr/bin/python3
import os
import errno
import signal
import argparse
import subprocess
from io import StringIO
from xml.etree.ElementTree import parse

parser = argparse.ArgumentParser()
parser.add_argument('node')
args = parser.parse_args()


# class TimeoutError(Exception):
#     pass
def handle_timeout(signum, frame):
    raise TimeoutError(os.strerror(errno.ETIME))


def main(**kwargs):
    if kwargs['node'] != 'all':
        nodes = [kwargs['node']]
    else:
        nodes = ['node%d' % i for i in range(11, 146)]

    for node in nodes:
        if node in ('node103', 'node104'):
            continue

        if node in ('node75', 'node76', 'node77', 'node78'):
            print('%s:' % node)
            print('PROWAVE')
            continue

        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(10)
        try:
            with subprocess.Popen(['ssh', node, 'nvidia-smi', '-q', '-x'], stdout=subprocess.PIPE) as p0:
                output, errors = p0.communicate()

                ret = []
                tree = parse(StringIO(output.decode('utf-8')))
                gpus = tree.findall('gpu')

                for gpu in gpus:
                    product_name = gpu.findtext('product_name')
                    temp = gpu.find('temperature').findtext('gpu_temp')
                    processes = gpu.findall('processes')

                    processes_text = []
                    for i, proc in enumerate(processes):
                        process_info = proc.find('process_info')
                        if process_info:
                            processes_text.append('%d: %s' % (i, process_info.findtext('process_name')))

                    processes_text = ', '.join(processes_text)
                    ret.append('%s: %s %s %s ' % (node, product_name, temp, processes_text))

                print('\n'.join(ret))

            cmd = ['ssh', node, 'ps', 'aux', '|', 'grep', '-E', '\'pmemd.cuda|rism3d-x\'', '|', 'grep', '-v', 'grep']
            with subprocess.Popen(cmd, stdout=subprocess.PIPE) as p1:
                output, errors = p1.communicate()

                procs = output.decode().splitlines()
                for proc in procs:
                    if not proc or not len(proc.split()):
                        continue

                    user = proc.split()[0]
                    proc_name = proc.split()[10]
                    print(user, proc_name)

                # print('len(output.splitlines())-2)
        except:
            import traceback
            traceback.print_exc()
        finally:
            signal.alarm(0)


if __name__ == '__main__':
    main(node=args.node)
