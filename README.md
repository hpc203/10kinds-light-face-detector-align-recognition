在下载完程序后，需要下载模型文件，才能正常运行程序。由于模型文件太多，无法直接上传，可以去百度云盘下载
百度云盘下载链接: https://pan.baidu.com/s/1AYqDypnpsmHwqDScoCwzKw 提取码: uncm

下载完成后一共有10个文件夹，把它们放到本仓库代码文件夹里。程序依赖pytorch和opencv，如果你的python环境里没有这两个库，那在Terminal输入pip安装。
pytorch的安装命令是 pip install torch和pip install torchvision
opencv的安装命令是 pip install opencv-python
配置好运行环境之后，就可以运行程序了。
运行 Run_all_compare_time.py，就可以看到9种人脸检测算法的结果和运行耗时统计直方图


每一种人脸检测算法，我都用一个类包装起来的，在这个类有构造函数__init__和成员函数detect
centerface，dbface，retinaface，mtcnn，yoloface 这五个人脸检测网络在输出检测框的同时还输出人脸里的5个关键点，
用这5个关键点可以做人脸对齐的。它们的类构造函数里有个初始化参数align是用来决定是否做人脸对齐的开关，在人脸识别系统里，人脸对齐这一步不是必选项的

如果你想构建一个人脸识别系统，那可以先运行get_face_feature.py，它是获取人脸特征向量的，
在这个.py文件里，我选用的人脸检测器是yoloface,如果你想换用其它的人脸检测器，那就修改主函数开头的from...import...即可
假如你使用的人脸检测器的输出没有关键点，而你想在人脸检测后做人脸对齐，那么检测人脸关键点的功能可以使用
mtcnn_pfld_landmark.py里的类pfld_landmark，它就是PFLD人脸检测，需要注意的是，它的输入是在人脸检测之后剪切出的人脸roi区域
