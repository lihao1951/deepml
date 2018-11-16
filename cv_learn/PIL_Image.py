#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/9/1 21:55
"""

import os
import sys
import PIL
import matplotlib.pyplot as plt
from PIL import Image
pil_im = Image.open("images/lena.bmp")
#pil_im.save("images/lena_l.jpg")
out=pil_im.rotate(20)
plt.imshow(out)
plt.show()