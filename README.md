# 9种轻量级人脸检测算法的比拼
在下载完程序后，需要下载模型文件，才能正常运行程序。由于模型文件太多，无法直接上传，可以去百度云盘下载
百度云盘下载链接: https://pan.baidu.com/s/1AYqDypnpsmHwqDScoCwzKw 提取码: uncm

下载完成后一共有10个文件夹，把它们放到本仓库代码文件夹里。程序依赖pytorch和opencv，如果你的python环境里没有这两个库，那在Terminal输入pip安装。
pytorch的安装命令是 pip install torch和pip install torchvision
而opencv的安装命令是 pip install opencv-python
配置好运行环境之后，就可以运行程序了。
运行 Run_all_compare_time.py，就可以看到9种人脸检测算法的结果和运行耗时统计直方图。效果可以看我的csdn博客文章(地址是 https://blog.csdn.net/nihate/article/details/108798831 )里的图

每一种人脸检测算法，我都用一个类包装起来的，在这个类有构造函数__init__和成员函数detect
centerface，dbface，retinaface，mtcnn，yoloface 这五个人脸检测网络在输出检测框的同时还输出人脸里的5个关键点，
用这5个关键点可以做人脸对齐的。它们的类构造函数里有个初始化参数align是用来决定是否做人脸对齐的开关，在人脸识别系统里，人脸对齐这一步不是必选项的。

如果你想构建一个人脸识别系统，那可以先运行get_face_feature.py，它是获取人脸特征向量的，最后会生成一个已知身份的人脸特征向量的pkl文件。
在这个.py文件里，使用的人脸检测器是yoloface，如果你想换用其它的人脸检测器，那就修改主函数开头的from...import...即可。
假如你使用的人脸检测器的输出没有关键点，而你想在人脸检测后做人脸对齐，那么检测人脸关键点的功能可以使用
mtcnn_pfld_landmark.py里的类pfld_landmark，它就是PFLD人脸检测，需要注意的是，它的输入是在人脸检测之后剪切出的人脸roi区域。
接下来是提取人脸特征向量，用的是arcface。

在得到人脸特征向量的pkl文件后，运行detect_face_align_rec.py，就是做人脸检测→人脸对齐→人脸识别，需要注意的是
在上一步提取已知身份的人脸特征向量的pkl文件时，get_face_feature.py使用的人脸检测器和现在detect_face_align_rec.py里使用的人脸检测器一致。

看近几年顶会的paper，在人脸检测→人脸对齐→人脸识别这三大模块里，做的最多的是人脸检测这个模块，几乎每年都有新的网络提出，我关心和感兴趣的也是人脸检测这个模块。
而在人脸识别提取特征向量这个模块，创新性的工作都是在最后的全连接层后面的loss做改动创新，怎样让同类人脸更加聚拢，让不同类人脸的间距更大。
在提取人脸特征向量这一步，目前也有很多种网络，比如ArcFace、SphereFace、CosFace等等的，如果想继续扩充现在的程序，读者可以把这些模块添加进来。
