#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/9/30 9:48
"""
import os
import sys
__metaclass__ = type

class A(object):
    def __init__(self):
        self.lists = [1,2,3]
        self.maps = {"A":11,"B":22}
        self.__asd = 2
        self._aa = 3
    def _getLists(self):
        return self.lists
    def __setLists(self,ll):
        self.lists = ll
    #property 实现了属性的访问、修改、删除等，表现我
    #0个参数既不能访问也不能修改，1个参数为只读
    l = property(_getLists,__setLists)
    def __getitem__(self, item):
        """
        返回集合中项目数量 a[]方法调用 a[2]
        :param item:
        :return:
        """
        if isinstance(item,int):
            return self.lists[item]
        if isinstance(item,str):
            return self.maps.get(item)
    def __setitem__(self, key, value):
        """
        设置序列的值 a["C"] = 33
        :param key:
        :param value:
        :return:
        """
        if isinstance(key,str):
            self.maps[key] = value
        if isinstance(key,int):
            if key > len(self.lists):
                raise IndexError("key 超出list界线")
            else:
                self.lists.insert(key,value)
    def __delitem__(self, key):
        """
        删除序列 del(a["C"])
        :param key:
        :return:
        """
        self.maps.pop(key)

    def __getattr__(self, item):
        if item is "lens":
            print("get lens")
        else:
            print("get other")
    def __setattr__(self, key, value):
        if key is "lens":
            print("set lens\t",value)
        else:
            self.__dict__[key] = value

class Fibs:
    """
    fib = Fibs()
    for i in range(10):
        print(next(fib))
    print(list(fib))
    """
    def __init__(self):
        self.a = 0
        self.b = 1
    #迭代器 next方法
    #可从迭代器得到序列
    def __iter__(self):
        return self
    def __next__(self):
        self.a ,self.b = self.b,self.a+self.b
        #A值超过50就停止
        if self.a > 50:
            raise StopIteration
        return self.a

"""
生成器--是一种用普通的函数语法定义的迭代器（任何包含yield语句的函数都称为生成器）
"""
def flatten(nested):
    for sublist in nested:
        for ele in sublist:
            yield ele
def recurrence_flatten(nested):
    """
    递归生成器
    :param nested:
    :return:
    """
    try:
        for sublist in nested:
            for ele in flatten(sublist):
                yield ele
    except TypeError:
        yield nested

def test_generator():
    nested = ['foo',['bar',['baz']]]
    print(list(recurrence_flatten(nested)))
test_generator()
