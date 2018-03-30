import threading
import queue
import datetime
from multiprocessing import Process, Lock, Queue, current_process, Pool
import sys
from _config import ConfigMap, Logging
import os


logging = Logging("trajectory")
system = ConfigMap("System")
cores = int(system['cores'])
date_time_field = 3
base_folder = "/large/"
file_to_split = os.path.join(base_folder, "yellow_tripdata_2015-12.csv")


class FileSplitter:

    def process_line(self, l):
        dt_tm_str = l.split(',')[1]
        dt_tm = datetime.datetime.strptime(dt_tm_str, '%Y-%m-%d %H:%M:%S')
        file_name = os.path.join(base_folder, "_" + str(dt_tm.hour) + "_" + str(dt_tm.weekday()))
        f = open(file_name, "a")
        f.write(l)
        f.close()

    def get_next_line(self):
        linectr = 0
        with open(file_to_split,'r') as f:
            for line in f:
                linectr += 1
                if linectr > 1 :
                    yield line

    def process_file(self):
        f = self.get_next_line()

        t = Pool(processes=cores)

        for i in f:
            t.map(self.process_line, (i,))
        t.join()
        t.close()


    # def do_work(self, in_queue, out_queue):
    #     while True:
    #         item = in_queue.get()
    #         # process
    #         result = item
    #         out_queue.put(result)
    #         in_queue.task_done()
    #
    #
    # def split_file(self):
    #     work = Queue.Queue()
    #     results = Queue.Queue()
    #     total = 20
    #
    #     # start for workers
    #     for i in xrange(4):
    #         t = threading.Thread(target=do_work, args=(work, results))
    #         t.daemon = True
    #         t.start()
    #
    #     # produce data
    #     for i in xrange(total):
    #         work.put(i)
    #
    #     work.join()
    #
    #     # get the results
    #     for i in xrange(total):
    #         print results.get()
    #
    #     sys.exit()

if __name__ == "__main__":
    filesplitter = FileSplitter()
    filesplitter.process_file()





