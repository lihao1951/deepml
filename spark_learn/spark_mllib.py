#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/11/22 15:40
"""
import os
import sys
import numpy as np
# 向量
from pyspark.mllib.linalg import Vectors
from pyspark.context import SparkContext
from pyspark.sql import SQLContext

sc =SparkContext(appName="testApp")
sqlContext = SQLContext(sc)
#增加这个命令 会消除  'PipelinedRDD' object has no attribute 'toDF 错误
"""
toDF method is a monkey patch executed inside SparkSession (SQLContext constructor in 1.x) 
constructor so to be able to use it you have to create a SQLContext (or SparkSession) first:
"""
people = sqlContext.read.json("people.json")
people.show()
people.printSchema()
people.select(people["name"],people["age"]+10).show()
allRows = people.collect()
for row in allRows:
    print(row.age)
sc.stop()
# numpy 可以直接表示未稠密向量
# 稀疏向量
# from pyspark.mllib.linalg import SparseVector
# 稀疏向量代表的是 Vectors.sparse(3->向量长度, [0, 2]->索引, [1.0, 3.0]->值)
# 标记点 代表的是一个标注好的训练数据 回归/分类
# from pyspark.mllib.regression import LabeledPoint
# ld = LabeledPoint(1.0,Vectors.sparse(3,[0,2],[1.0,3.0]))
# print('ld labels ',ld.label)
# print('ld features ',ld.features)
# 稀疏数据 MLlib支持读取一个以LIBSVM格式存储训练例子。LIBSVM是LIBSVM和LIBLINEAR默认的格式。
# from pyspark.mllib.util import MLUtils
#examples = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_libsvm_data.txt")

# 本地矩阵 存储在单机上
# from pyspark.mllib.linalg import Matrix,Matrices
# Create a dense matrix ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0))
# m1 = Matrices.dense(3,2,[1,3,5,2,4,6])#稠密矩阵是按照列存储的
# print('m1 values',m1.values)
# print('m1 rows',m1.numRows)
# print('m1 cols',m1.numCols)

# Create a sparse matrix ((9.0, 0.0), (0.0, 8.0), (0.0, 6.0))
# m2 = Matrices.sparse(3, 2, [0, 1, 3], [0, 2, 1], [9, 6, 8])#稀疏数据 行、列、行指标、列指标、值

# 分布式矩阵（Long 类型行索引，Long类型列索引，Double类型值）分布在一个或多个RDD上
# 存储巨大的分布式矩阵需要选择正确格式 否则会出错
# 另外，转换分布式矩阵的格式 需要全局Shuffle 目前有三种分布式矩阵RowMatrix IndexRowMatrix CoordinateMatrix
# from pyspark.mllib.linalg.distributed import IndexedRow,RowMatrix,IndexedRowMatrix,CoordinateMatrix,MatrixEntry
# mat1 = RowMatrix(sc.parallelize([[1,2,3],[2,3,4],[2,3,5]]))
# rows = sc.parallelize([IndexedRow(0, [1, 2, 3]),IndexedRow(1, [4, 5, 6])])
# mat2 = IndexedRowMatrix(rows)
# entries = sc.parallelize([MatrixEntry(0, 0, 1.2),MatrixEntry(6, 4, 2.1)])
# mat3 = CoordinateMatrix(entries)
"""
统计分析模块
"""
from pyspark.mllib.stat import Statistics
# 汇总分析

# rdd = sc.parallelize([Vectors.dense([2, 0, 0, -2]), \
#                   Vectors.dense([4, 5, 0,  3]),    \
#                   Vectors.dense([6, 7, 0,  8])])
# summary = Statistics.colStats(rdd)
# print(summary.mean())
# print(summary.variance())
# print(summary.numNonzeros())

# 相关性分析
# n1 = sc.parallelize(np.array([1,1,2,2],dtype=np.float32))
# n2 = sc.parallelize(np.array([1,1,2,2],dtype=np.float32))
# corr = Statistics.corr(n1,n2,method="spearman")
# print(corr)
# n2 = sc.parallelize(np.array([1,1,2,2],dtype=np.float32))
# n2.randomSplit()可以进行分割
# splits = n2.randomSplit([0.5,0.5],seed=1)
# s0 = splits[0]
# s1 = splits[1]
# s0.foreach(print)
# sc.stop()