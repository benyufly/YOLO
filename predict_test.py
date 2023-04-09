import collections
import os
import json
import time

import torch
import cv2
import numpy as np
import argparse
from torch.autograd import Variable


from LPR_model.LPRNet import build_lprnet
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from test_utils.load_data import CHARS, CHARS_DICT, LPRDataLoader
from build_utils import img_utils, torch_utils, utils
from models import Darknet
from draw_box_utils import draw_box

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    parser.add_argument('--img_size', default=[94, 24], help='the image size')
    parser.add_argument('--test_img_dirs', default=r"./my_yolo_dataset/val/images/", help='the test images path')
    parser.add_argument('--cfg_path', default="cfg/my_yolov3.cfg", help='cfg path')
    parser.add_argument('--yolo_weights', default="weights/yolov3spp-29.pt", help='yolo weight path')
    parser.add_argument('--json_path', default="./data/my_data.json", help='json path')
    parser.add_argument('--yolo_img_size', default= 512 , help='yolo net input image size(# 必须是32的整数倍 [416, 512, 608])')
    parser.add_argument('--dropout_rate', default=0, help='dropout rate.')
    parser.add_argument('--lpr_max_len', default=8, help='license plate number max length.')
    parser.add_argument('--test_batch_size', default=1, help='testing batch size.')
    parser.add_argument('--phase_train', default=False, type=bool, help='train or test phase flag.')
    parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
    parser.add_argument('--cuda', default=False, type=bool, help='Use cuda to train model')
    parser.add_argument('--show', default=True, type=bool, help='show test image and its predict result or not.')
    parser.add_argument('--pretrained_model', default='./weights/myLPRweight.pth', help='pretrained base model')

    args = parser.parse_args()

    return args

def show(img, label, box):
    lb = ""
    for i in label:
        lb += CHARS[i]

    img = LprImgAddText(img, lb,box)
    # cv2.imshow("test", img)  linux无法imshow
    cv2.imwrite(r"./result/{}.jpg".format(lb), img)
    #img.save(r"./LRP_result",lb)
    print("predict: ", lb)
    cv2.waitKey()
    return lb

def LprImgAddText(img, text, box, color=(0, 255, 0), textSize=12):
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)
    draw = ImageDraw.Draw(img)
    im_width, im_height = img.size
    box = tuple(box[0].tolist())  # numpy -> list -> tuple
    xmin, ymin, xmax, ymax = box
    (left, right, top, bottom) = (xmin * 1, xmax * 1,
                                  ymin * 1, ymax * 1)

    try:
        font = ImageFont.truetype("data/NotoSansCJK-Regular.ttc", textSize, encoding="utf-8")
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    LPR_display_str_map = collections.defaultdict(list)
    LPR_display_str_map[box].append(str(text))
    text_width, text_height = font.getsize(text)
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * text_height

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    margin = np.ceil(0.05 * text_height)
    draw.rectangle([(left + 100, text_bottom - text_height - 2 * margin),
                    (left + text_width + 100, text_bottom)], fill=color)
    draw.text((left + margin + 100, text_bottom - text_height - margin),
                text,
                fill='black',
                font=font)
    text_bottom -= text_height - 2 * margin
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

# LPR预处理
def LPR_transform(img):
    img = np.asarray(img)  # 转成array,变成24*94*3
    img = img.astype('float32')
    img -= 127.5
    img *= 0.0078125
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img)  # 转成tensor类型
    img = img.unsqueeze(0)  # 添加batch维度 1*24*94*3
    return img

def main():
    args = get_parser()
    total = 0
    tp = 0
    for img_name in os.listdir(args.test_img_dirs):  #(r"./my_yolo_dataset/val/images/"):

        target_label = ""
        _, _, box, points, plate, brightness, blurriness = img_name.split('-')
        list_plate = plate.split('_')  # 读取车牌
        target_label += provinces[int(list_plate[0])]
        target_label += alphabets[int(list_plate[1])]
        target_label += ads[int(list_plate[2])] + ads[int(list_plate[3])] + ads[int(list_plate[4])] + ads[
            int(list_plate[5])] + ads[int(list_plate[6])] + ads[int(list_plate[7])]  # 读取车牌信息

        img_path = os.path.join(args.test_img_dirs,img_name)
        img_size = args.yolo_img_size
        cfg = args.cfg_path  # 改成生成的.cfg文件
        weights = args.yolo_weights  # 改成自己训练好的yolo权重文件
        json_path = args.json_path  # json标签文件
        assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
        assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
        assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
        assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

        json_file = open(json_path, 'r')
        class_dict = json.load(json_file)
        json_file.close()
        category_index = {v: k for k, v in class_dict.items()}

        input_size = (img_size, img_size)

        device = torch.device("cuda:0" if args.cuda else "cpu")

        model = Darknet(cfg, img_size)

        model.load_state_dict(torch.load(weights, map_location=device)["model"])
        model.to(device)
        model.eval()  #  加载yolo网络
        with torch.no_grad():  # 禁止反向传播，test不需要反向传播
            # init
            img = torch.zeros((1, 3, img_size, img_size), device=device)
            model(img)

            img_o = cv2.imread(img_path)  # BGR
            assert img_o is not None, "Image Not Found " + img_path

            img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
            # Convert
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416

            img = np.ascontiguousarray(img)

            img = torch.from_numpy(img).to(device).float()  # 转成tensor类型
            img /= 255.0  # scale (0, 255) to (0, 1)
            img = img.unsqueeze(0)  # add batch dimension

            t1 = torch_utils.time_synchronized()
            pred = model(img)[0]  # only get inference result
            t2 = torch_utils.time_synchronized()
            print(t2 - t1)

            pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
            t3 = time.time()
            print(t3 - t2)

            if pred is None:
                print("No target detected.")
                exit(0)

            # process detections
            pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
            print(pred.shape)

            bboxes = pred[:, :4].detach().cpu().numpy()
            scores = pred[:, 4].detach().cpu().numpy()
            classes = pred[:, 5].detach().cpu().numpy().astype(int) + 1

            img_o = draw_box(img_o[:, :, ::-1], bboxes, classes, scores, category_index)  # [:, :, ::-1]将该图片按照第三维倒序取出，img_o从rgb格式变成了numpy.array
            img_LPR_draw = np.asarray(img_o)

            ymin = int(bboxes[0][1])
            ymax = int(bboxes[0][3])
            xmin = int(bboxes[0][0])
            xmax = int(bboxes[0][2])

            img_crop = img_o.crop((xmin, ymin, xmax, ymax))  # 裁剪出车牌位置
            img_crop = img_crop.resize((94, 24), Image.LANCZOS)  # resize到LPR网络的输入大小,返回PIL Image OBJECT

            plt.imshow(img_o)  # img_o为数组,没有channel维
            plt.show()

            #########################################################

            lprnet = build_lprnet(lpr_max_len=args.lpr_max_len, phase=args.phase_train, class_num=len(CHARS),
                                  dropout_rate=args.dropout_rate)
            LPR_device = torch.device("cuda:0" if args.cuda else "cpu")
            lprnet.to(LPR_device)  # 加载LPR网络
            print("Successful to build lprnetwork!")

            # load pretrained model 加载训练好的权重
            if args.pretrained_model:
                lprnet.load_state_dict(torch.load(args.pretrained_model))
                print("load LPR model successful!")
            else:
                print("[Error] Can't found LPR model, please check!")
                return False

            img_crop = LPR_transform(img_crop)  # LPR预处理
            try:
                if args.cuda:
                    img_crop = Variable(img_crop.cuda())
                else:
                    img_crop = Variable(img_crop)
                # forward
                prebs = lprnet(img_crop)
                prebs = prebs.cpu().detach().numpy()
                preb = prebs[0, :, :] # preb为68*18的矩阵，车牌有68个可选字符
                preb_label = list()
                for j in range(preb.shape[1]):
                    preb_label.append(np.argmax(preb[:, j], axis=0))  # argmax返回最大值的索引值
                no_repeat_blank_label = list()  # 存最后8个字符label的list，18个字符要去掉部分重复的和‘-’
                pre_c = preb_label[0]  # preb_label为1*18
                if pre_c != len(CHARS) - 1:
                    no_repeat_blank_label.append(pre_c)
                for c in preb_label:  # dropout repeate label and blank label
                    if (pre_c == c) or (c == len(CHARS) - 1):
                        if c == len(CHARS) - 1:
                            pre_c = c
                        continue
                    no_repeat_blank_label.append(c)
                    pre_c = c
                if args.show:
                    predict_lb = show(img_LPR_draw, no_repeat_blank_label,bboxes)
            finally:
                total += 1
                if target_label == predict_lb:
                    tp += 1
                cv2.destroyAllWindows()
    print("[Info] Test Accuracy: {},total:{},tp:{}".format(tp / total,total,tp))


if __name__ == "__main__":
    main()
