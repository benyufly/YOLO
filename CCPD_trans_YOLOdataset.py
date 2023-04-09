import cv2
import os

# 参考 https://blog.csdn.net/qq_36516958/article/details/114274778
# https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data#2-create-labels

path = r'E:\LPRnet\yolov3_spp\my_yolo_dataset\train\images'
outputPath = r'E:\LPRnet\yolov3_spp\my_yolo_dataset\train\labels\\'

for filename in os.listdir(path):
    list1 = filename.split("-", 3)  # 第一次分割，以减号'-'做分割
    subname = list1[2]
    lt, rb = subname.split("_", 1)  # 第二次分割，以下划线'_'做分割
    lx, ly = lt.split("&", 1)
    rx, ry = rb.split("&", 1)
    width = int(rx) - int(lx)
    height = int(ry) - int(ly)  # bounding box的宽和高
    cx = float(lx) + width / 2
    cy = float(ly) + height / 2  # bounding box中心点

    img = cv2.imread(path + '\\' +filename)
    width = width / img.shape[1]
    height = height / img.shape[0]
    cx = cx / img.shape[1]
    cy = cy / img.shape[0]

    txtname = filename.split(".", 1)
    txtfile = outputPath + txtname[0] + ".txt"  # 生成的txt文件名和jpg文件名对应
    # 绿牌是第0类，蓝牌是第1类
    with open(txtfile, "w") as f:  # w表示写入txt文件，如果txt文件不存在将会创建
        f.write(str(0) + " " + str(cx) + " " + str(cy) + " " + str(width) + " " + str(height))

    # print(filename)
    # print(txtfile)
    # print(img.shape)
    print(cx)
    print(cy)
    print(width)
    print(height)
