#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/12/25 17:22
"""
import os
import sys
import time
import threading
import collections

from threading import Thread
import multiprocessing
from multiprocessing import Process

class MyTestSay(Thread):
    def __init__(self, name,sleep=3):
        super().__init__()
        self.name = name
        self.sleep = sleep

    def run(self):
        for i in range(1000):
            print(self.name,' ',i)
        print("name:{} hello".format(self.name))

if __name__ == '__main__':
    mts1 = MyTestSay("one")
    mts2 = MyTestSay("two")
    mts1.start()
    mts2.start()
    print("main")
    print("mts1:\t", mts1.is_alive())
    print("mts2:\t", mts2.is_alive())