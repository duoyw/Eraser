import os
import time


class TimeStatistic:
    _start_time = {}
    _end_time = {}
    _count = {}
    _total_time = {}

    @classmethod
    def start(cls, name):
        cls._start_time[name] = time.time()

    @classmethod
    def end(cls, name):
        if name in cls._start_time:
            pass_time = time.time() - cls._start_time[name]
            if name in cls._count:
                cls._count[name] = cls._count[name] + 1
                cls._total_time[name] = cls._total_time[name] + pass_time
            else:
                cls._count[name] = 1
                cls._total_time[name] = pass_time
        else:
            raise RuntimeError("end but no start")

    @classmethod
    def print(cls):
        names = list(cls._total_time.keys())
        print("############################")
        for name in names:
            total = cls._total_time[name]
            count = cls._count[name]
            aver = total / count
            print("{}: total time is {}, aver time is {}, count  is {}".format(name, total, aver, count))
        print("############################")
