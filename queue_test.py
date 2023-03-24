import time
import threading
from queue import Queue
import numpy as np


def get_nir():
    print("get nir")
    color_out = 1
    nir_out = 2
    return color_out, nir_out


def get_color():
    print("get color")
    color_out = 3
    return color_out


# Worker 類別，負責處理資料
class Input(threading.Thread):
    def __init__(self, color_queue, nir_queue, choose_input='color'):
        threading.Thread.__init__(self)
        self.rgb_queue = color_queue
        self.nir_queue = nir_queue
        self.choose_input = choose_input

    def run(self):
        print("run")
        if self.choose_input == 'color':
            color_data = get_color()
            print("put")
            self.rgb_queue.put(color_data)
            print("over")
        else:
            color_data, nir_data = get_nir()
            self.rgb_queue.put(color_data)
            self.nir_queue.put(nir_data)

    def get_data(self):
        if self.choose_input == 'color':
            print("getting color data ")
            print("queue size " + str(self.rgb_queue.qsize()))
            rgb_out = self.rgb_queue.get()
            print("color data "+str(rgb_out))
            return rgb_out
        else:
            rgb_out = self.rgb_queue.get()
            nir_out = self.nir_queue.get()
            print("nir data "+str(nir_out))
            return rgb_out, nir_out


class Porcess(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self, rgb_frame, nir_frame=None):
        if nir_frame is not None:
            print("--------------color process-------------yy")
            print("rgb frame number " + str(rgb_frame))
        else:
            print("--------------nir process-------------yy")
            print("rgb frame number " + str(rgb_frame))
            print("nir frame number " + str(nir_frame))


# 建立佇列
rgb_queue = Queue()
nir_queue = Queue()

choose_input = 'color'

# 建立兩個 Worker
work_input = Input(rgb_queue, nir_queue, choose_input)

for i in range(10):
    print("index : "+str(i))
    work_input.start()

    work_input.join()
    print("get data")
    if choose_input == 'color' and work_input.rgb_queue.qsize() >= 1:
        print("get color data")
        color_data = work_input.get_data()
    elif choose_input == 'nir' and work_input.nir_queue.qsize() >= 1:
        print("get nir data")
        color_data, nir_data = work_input.get_data()
    else:
        print("queue did not have value")

    print("process data")
    if work_input.nir_queue.qsize() >= 1:
        print("--------------nir process-------------yy")
        print("rgb frame number " + str(color_data))
        print("nir frame number " + str(nir_data))
    elif work_input.rgb_queue.qsize() >= 1:
        print("--------------color process-------------yy")
        print("rgb frame number " + str(nir_data))

    print("end")



print("Done.")
