from copy import deepcopy
from queue import Queue


def dis_rot(rot):
    return min(abs(90 - rot), abs(270 - rot))


near_q = [(0.2, 0.2, 30), (0.18, 0.18, 30)]
queue = Queue()
for item in near_q:
    queue.put(item)
while not queue.empty():
    print(queue.get())
