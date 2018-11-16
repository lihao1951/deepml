#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/10/10 15:19
"""
import os
import sys
import tkinter
import pprint

#检查当前模块的可用查询目录
#自己定义的模块最好放入 site-packages目录下，这样在使用时，就不用添加sys.path.append()
#还有一种方法是添加至Pythonpath 环境变量中，该路径会因为OS不同而不同
pprint.pprint(sys.path)
print(pprint.__file__)
#__all__定义的是共有的方法名称
tk = tkinter.Tk()
tk.mainloop()
