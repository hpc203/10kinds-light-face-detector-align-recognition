在下载完程序后，需要下载模型文件，才能正常运行程序。由于模型文件太多，无法直接上传，可以去百度云盘下载
百度云盘下载链接: https://pan.baidu.com/s/1AYqDypnpsmHwqDScoCwzKw 提取码: uncm

下载完成后一共有10个文件夹，把它们放到本仓库代码文件夹里。程序依赖pytorch和opencv，如果你的python环境里没有这两个库，那在Terminal输入pip安装。
pytorch的安装命令是 pip install torch和pip install torchvision
opencv的安装命令是 pip install opencv-python
配置好运行环境之后，就可以运行程序了。
运行 Run_all_compare_time.py，就可以看到9种人脸检测算法的结果和运行耗时统计直方图
每一种人脸检测算法，我都用一个类包装起来的，在这个类有构造函数__init__和成员函数detect
centerface，dbface，retinaface，mtcnn，yoloface 这五个人脸检测网络在输出检测框的同时还输出人脸里的5个关键点，用这5个关键点可以做人脸对齐的
