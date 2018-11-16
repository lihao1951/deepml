#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
CV 学习 1
Author LiHao
Time 2018/10/25 17:18
"""
from __future__ import print_function
import os
import sys
import cv2

"""
cv2不是代表了opencv是2版本，而是代表的是新的面对对象编写的版本
cv2.cv / cv 代表的是旧版本 过程化编写的

使用opencv读取图片时， 默认的通道顺序是BGR而非RGB，
在RGB为主流的当下， 这种默认给我们带来了一点不便。
那么， opencv 为什么要使用BGR而非RGB呢？ 
目前看到的一种解释说是因为历史原因：早期BGR也比较流行，
opencv一开始选择了BGR，到后来即使RGB成为主流，但也不好改了。 
"""
import numpy as np
def bw_image():
    img = np.zeros((20,20),dtype=np.uint8) #图像RGB一般为0-255 HSV为0-180
    print(img.shape)
    img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)#灰色转RGB
    print(img.shape)

def transfor_img(img):
    # imread 第二个参数 为读取方式 参照cv::ImreadModes
    # IMREAD_ANYCOLOR
    # IMREAD_ANYDEPTH
    # IMREAD_COLOR
    # IMREAD_GRAYSCALE
    # IMREAD_LOAD_GDAL
    # IMREAD_UNCHANGED
    image = cv2.imread(img,cv2.IMREAD_LOAD_GDAL)
    cv2.imwrite('transfer2.jpg',image)
"""
可以用numpy访问图像数据
一般通过索引来更新数据最好
"""
def test_numpy():
    img=cv2.imread("../images/lena.bmp")
    print(img.item(0,2,0))
    print(img[:,:,0])
    img.itemset((100,100,0),255)
    print(img.item(100, 100, 0))

def change_roi(img,start,end):
    roi = img[start:end,start:end]
    img[start+50:end+50,start+50:end+50] = roi

def show(wname,image,sec=5):
    img=cv2.imread(image)
    print(img.size)#宽度*高度*通道数
    print(img.shape)#显示(宽度,高度,通道数)
    print(img.dtype)#一般为uint8
    change_roi(img,100,200)
    cv2.imshow(wname,img)
    cv2.waitKey(sec*1000)
    cv2.destroyWindow(wname)
"""
视频的读写
cv2.VideoCapture(0) 获取摄像头
cv2.VideoCapture(video) 获取某一个
"""
def video_read(video):
    videoCapture = cv2.VideoCapture(video)

def video_camera_write():
    """
    读取摄像头的头像 写入到一个文件中
    VideoWriter_fourcc('I','4','2','0') 未压缩的YUV颜色编码 420色度子采样 会产生较大文件 后缀 avi
    VideoWriter_fourcc('P','I','M','1') MPEG-1编码 后缀 avi
    VideoWriter_fourcc('X','V','I','D') MPEG-4编码 大小为平均值 后缀avi
    VideoWriter_fourcc('T','H','E','O') Ogg Vorbis 后缀为ogv
    VideoWriter_fourcc('F','L','V','1') Flash视频 后缀.flv
    :return:
    """
    cameraCapture = cv2.VideoCapture(0)
    fps=30 #指定帧数
    size =(int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    videoWriter = cv2.VideoWriter('../videos/myFirst.avi',cv2.VideoWriter_fourcc('X','V','I','D'),fps,size)
    success,frame = cameraCapture.read()
    print(cameraCapture.isOpened())
    numFrameRemaining = 10*fps-1
    while success and numFrameRemaining>0:
        videoWriter.write(frame)
        success,frame = cameraCapture.read()
        numFrameRemaining -= 1
    cameraCapture.release()

clicked = False
def onMouse(event,x,y,flags,param):
    global clicked
    if event == cv2.EVENT_LBUTTONUP:
        clicked = True
def show_camera_ontime():
    """
    实时显示摄像头数据
    通过onMouse方法来捕获鼠标动作
    waitKey来捕捉键盘动作
    :return:
    """
    cameraCaputer = cv2.VideoCapture(0)
    cv2.namedWindow("MyWindow")
    cv2.setMouseCallback("MyWindow",onMouse)
    print('Show camera feed. Click window or press any key to stop')
    success,frame=cameraCaputer.read()
    while success and cv2.waitKey(1)==-1 and not clicked:
        cv2.imshow("MyWindow",frame)
        success,frame = cameraCaputer.read()
    cv2.destroyWindow("MyWindow")
    cameraCaputer.release()


if __name__ == '__main__':
    #bw_image()
    #show('test','../images/lena.bmp',5)
    #test_numpy()
    #video_camera_write()
    show_camera_ontime()