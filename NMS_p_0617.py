"""Just a quick script to show functionality of opencv."""
import numpy as np
import cv2
from xml.dom.minidom import parse
import math
import argparse

# ==============================================
# Load up the classifier using frontal face data
# ==============================================
# frontalface_location = './opencvdata/haarcacade_frontalface_default.xml'
# face_cascade = cv2.CascadeClassifier(frontalface_location)
#eye_cascade = cv2.CascadeClassifier('./opencvdata/haarcascade_eye.xml')
# =====================================
# Assign the source and the output file
# =====================================
# Use VideoCapture(0) if you want to use webcam
# cap = cv2.VideoCapture('./videos/ttd-lgbt.mp4')
# out = cv2.VideoWriter('lgbt2.mp4', cv2.cv.CV_FOURCC('X', '2', '6', '4'), 30, (1280, 720))
# ===========================================
# Main loop
# - Iterate through each frame of video
# - Detect face
# - And draw a rectangle around detected face
# ===========================================

import os
def file_name(file_dir):
  nc=["aeroplane","bicycle","bird","boat","bottle","bus","car","cat","chair","cow","diningtable","dog","horse","motorbike","person","pottedplant","sheep","sofa","train","tvmonitor"]
  for root, dirs, files in os.walk(file_dir):
    # print(root) #当前目录路径
    # print(dirs) #当前路径下所有子目录
    # print(files) #当前路径下所有非目录子文件
    for file in files:
        # print(file[:-4])
        readXML(nc,file[:-4])

""
# 输入xml文件名称,所有种类
# 返回字典类型种类  类型:(类型:(xmax xmin ymax ymin))
""
def readXML(nc, filename):
        domTree = parse("../Annotations/"+filename+".xml")
        img=cv2.imread(filename+".jpg")
        # 文档根元素
        rootNode = domTree.documentElement
        # print(rootNode.nodeName)
        elements = rootNode.getElementsByTagName('object')
        for element in elements:
            for node in element.childNodes:
                # 通过nodeName判断是否是文本
                if node.nodeName=='name':
                    # text = node.data.replace('\n', '')
                    if node.childNodes[0].nodeValue in nc:
                        # position_node=element.childNodes[8]
                        xmin=int(element.getElementsByTagName('xmin')[0].childNodes[0].nodeValue)
                        xmax=int(element.getElementsByTagName('xmax')[0].childNodes[0].nodeValue)
                        ymin = int(element.getElementsByTagName('ymin')[0].childNodes[0].nodeValue)
                        ymax = int(element.getElementsByTagName('ymax')[0].childNodes[0].nodeValue)
                        # print(xmin,type(xmax),ymin,ymax)
                        cv2.rectangle(img, (xmin,ymin), (xmax,ymax),(255, 255, 255),2)
                        cv2.imwrite("../save/"+filename+".jpg",img)
                        # print(element.getElementsByTagName('xmin')[0].childNodes[0].nodeValue


                    # print(node.nodeName,node.childNodes[0].nodeValue,node.nodeType)
                    # if node.nodeName in nc:
                    #     # 用data属性获取文本内容
                    #     text = node.data.replace('\n', '')
                    #     # 这里的文本需要特殊处理一下，会有多余的'\n'
                    #     print(text)

            # print(element)

        print("   ")
        # class_name = rootNode.getElementsByTagName("name")
        # print(class_name[0].txt)

#

# for nameclass in nc:
    #
	# # 所有顾客
	# customers = rootNode.getElementsByTagName("customer")
	# print("****所有顾客信息****")
	# for customer in customers:
	# 	if customer.hasAttribute("ID"):
	# 		print("ID:", customer.getAttribute("ID"))
	# 		# name 元素
	# 		name = customer.getElementsByTagName("name")[0]
	# 		print(name.nodeName, ":", name.childNodes[0].data)
	# 		# phone 元素
	# 		phone = customer.getElementsByTagName("phone")[0]
	# 		print(phone.nodeName, ":", phone.childNodes[0].data)
	# 		# comments 元素
	# 		comments = customer.getElementsByTagName("comments")[0]
	# 		print(comments.nodeName, ":", comments.childNodes[0].data)
# input point and line to return the distance between
def get_point_line_distance(point, line):
      point_x = float(point[0])
      point_y = float(point[1])
      line_s_x = float(line[0][0])
      line_s_y = float(line[0][1])
      line_e_x = float(line[1][0])
      line_e_y =float( line[1][1])
      #若直线与y轴平行，则距离为点的x坐标与直线上任意一点的x坐标差值的绝对值
      if line_e_x - line_s_x == 0:
            return math.fabs(point_x - line_s_x)
      #若直线与x轴平行，则距离为点的y坐标与直线上任意一点的y坐标差值的绝对值
      if line_e_y - line_s_y == 0:
            return math.fabs(point_y - line_s_y)
      #斜率
      k = (line_e_y - line_s_y) / (line_e_x - line_s_x)
      #截距
      b = line_s_y - k * line_s_x
      #带入公式得到距离dis
      dis = math.fabs(k * point_x - point_y + b) / math.pow(k * k + 1, 0.5)
      return dis

def readTXT(filename,args):
    file = open(filename)
    while 1:
        labelname_per_image=[]
        labels_per_image=[]
        line = file.readline()
        if not line:
            break

        labels=line.split(" ")
        # check whether the image file exist
        if not os.path.exists("."+labels[0]):
            print("image not exit: pass")
            pass
        img=cv2.imread("."+labels[0])
        # print(img)
        # for label in labels:
        for i in range(len(labels)-1):
            parameters=labels[i+1].split(",")
            # label_num[int(parameters[4])]+=1
            if len(parameters)>1:
                # print(parameters, len(parameters))
                labelname_per_image.append(int(parameters[4]))
                labels_per_image.append(parameters)
        # print(args.mainc)
        # print(labelname_per_image)
        if (args.mainc in labelname_per_image and args.checkbox==1) or (args.checkbox==2 and (args.mainc in labelname_per_image or args.secondmain in labelname_per_image) ) :
            listNMS=CustonNMS(args,labels_per_image)
            # print(listNMS)
            if len(listNMS)>0:
                for NMS_box in listNMS:
                    # print((type(NMS_box[0]), NMS_box[1]), (NMS_box[2], NMS_box[3]))
                    # print()
                    if NMS_box[4]=="green":
                        cv2.rectangle(img, (int(NMS_box[0]), int(NMS_box[1])), (int(NMS_box[2]), int(NMS_box[3])), (0, 255, 255), 2)
                    else:
                        cv2.rectangle(img, (int(NMS_box[0]), int(NMS_box[1])), (int(NMS_box[2]), int(NMS_box[3])), (255, 255, 0), 2)

                    # print(img)
                    cv2.imwrite("./save/"+labels[0], img)

        pass  # do something
    file.close()
def CustonNMS(args,labels_per_image):
    listNMS=[]
    careboundings=[]
    mainboundings=[]
    # print(type(args.otherc[0]))
    for label in labels_per_image:
        if (label[4]) in args.otherc:
            careboundings.append(label)
        elif int(label[4]) ==args.mainc:
            mainboundings.append(label)
    # print(mainboundings)
    for mainbounding in mainboundings:
        box1=(float(mainbounding[0]),float(mainbounding[1]),float(mainbounding[2]),float(mainbounding[3]))
        line1,line2,line3,line4=[[mainbounding[0],mainbounding[1]],[mainbounding[2],mainbounding[1]]],[[mainbounding[0],mainbounding[1]],[mainbounding[0],mainbounding[3]]],[[mainbounding[2],mainbounding[1]],[mainbounding[2],mainbounding[3]]],[[mainbounding[0],mainbounding[3]],[mainbounding[2],mainbounding[3]]]
        bigbox=[box1]
        for carebounding in careboundings:
            box2=(float(carebounding[0]),float(carebounding[1]),float(carebounding[2]),float(carebounding[3]))
            #judge overlape area
            if mat_inter(box1,box2):
                bigbox.append(box2)
            center_care=[(float(carebounding[0])+float(carebounding[2]))/2,(float(carebounding[1])+float(carebounding[3]))/2]
            d1,d2,d3,d4=get_point_line_distance(center_care,line1),get_point_line_distance(center_care,line2),get_point_line_distance(center_care,line3),get_point_line_distance(center_care,line4)
            min_d=min(d1,d2,d3,d4)
            if min_d< args.distance_threhod:
                bigbox.append(box2)
        if len(bigbox)>1:
            # print(np.array(bigbox))
            bigbox=np.array(bigbox)
            newbox=[min(bigbox[:,0]),min(bigbox[:,1]),max(bigbox[:,2]),max(bigbox[:,3]),args.color_1]
            listNMS.append(newbox)
    #once the checkboxes==2
    if args.checkbox==2:
        careboundings = []
        mainboundings = []
        for label in labels_per_image:
            if (label[4]) in args.secondotherc:
                careboundings.append(label)
            elif int(label[4]) == args.secondmainc:
                mainboundings.append(label)
        # print(mainboundings)
        for mainbounding in mainboundings:
            box1 = (float(mainbounding[0]), float(mainbounding[1]), float(mainbounding[2]), float(mainbounding[3]))
            line1, line2, line3, line4 = [[mainbounding[0], mainbounding[1]], [mainbounding[2], mainbounding[1]]], [
                [mainbounding[0], mainbounding[1]], [mainbounding[0], mainbounding[3]]], [
                                             [mainbounding[2], mainbounding[1]], [mainbounding[2], mainbounding[3]]], [
                                             [mainbounding[0], mainbounding[3]], [mainbounding[2], mainbounding[3]]]
            bigbox = [box1]
            for carebounding in careboundings:
                box2 = (float(carebounding[0]), float(carebounding[1]), float(carebounding[2]), float(carebounding[3]))
                # judge overlape area
                if mat_inter(box1, box2):
                    bigbox.append(box2)
                center_care = [(float(carebounding[0]) + float(carebounding[2])) / 2,
                               (float(carebounding[1]) + float(carebounding[3])) / 2]
                d1, d2, d3, d4 = get_point_line_distance(center_care, line1), get_point_line_distance(center_care,
                                                                                                      line2), get_point_line_distance(
                    center_care, line3), get_point_line_distance(center_care, line4)
                min_d = min(d1, d2, d3, d4)
                if min_d < args.distance_threhod:
                    bigbox.append(box2)
            if len(bigbox) > 1:
                # print(np.array(bigbox))
                bigbox = np.array(bigbox)
                newbox = [min(bigbox[:, 0]), min(bigbox[:, 1]), max(bigbox[:, 2]), max(bigbox[:, 3]),args.color_2]
                listNMS.append(newbox)
    return listNMS



def mat_inter(box1, box2):
    # judge whether the two scalrs have overlap area
    # box=(xA,yA,xB,yB)
    x01, y01, x02, y02 = box1
    x11, y11, x12, y12 = box2

    lx = abs((x01 + x02) / 2 - (x11 + x12) / 2)
    ly = abs((y01 + y02) / 2 - (y11 + y12) / 2)
    sax = abs(x01 - x02)
    sbx = abs(x11 - x12)
    say = abs(y01 - y02)
    sby = abs(y11 - y12)
    if lx <= (sax + sbx) / 2 and ly <= (say + sby) / 2:
        return True
    else:
        return False



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.description = 'please enter two parameters a and b ...'
    parser.add_argument("-checkbox", "--checkbox", help="this means main class", dest="checkbox", type=int, default="1")

    parser.add_argument("-main", "--mainc", help="this means main class", dest="mainc", type=int, default="0")
    parser.add_argument("-other", "--otherc", help="this is other classes", dest="otherc", type=list, default=[3,1,2])
    parser.add_argument("-distance_threhod", "--distance_mini", dest="distance_threhod", help="distance threhold", type=float, default="0.05")
    parser.add_argument("-boudingcolor_1", "--boudingcolor_1", dest="color_1", help="colorhelp", type=str, default="red")

    parser.add_argument("-secondmain", "--secondmainc", help="this means main class", dest="secondmainc", type=int, default="0")
    parser.add_argument("-secondother", "--secondotherc", help="this is other classes", dest="secondotherc", type=list, default=[3,1,2])
    # parser.add_argument("-seconddistance_threhod", "--seconddistance_mini", dest="seconddistance_threhod", help="seconddistance threhold", type=float, default="0.05")
    parser.add_argument("-boudingcolor_2", "--boudingcolor_2", dest="color_2", help="color2help", type=str, default="green")



    parser.add_argument("-groundfile", "--file_name", dest="filename", help="filename", type=str, default="sogou_train.txt")
    args = parser.parse_args()
    print("parameters","mian_class:",args.mainc,"other_classes:",args.otherc,"distance_threhod",args.distance_threhod)
    ground_truth_file=args.filename
    # file_name(ground_truth_file,)
    readTXT(ground_truth_file,args)
# counter = 0
# while True:
#     counter += 1
#     ret, frame = cap.read()
#     # Change the frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Apply classifier to find face
#     faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#     # Draw a rectangle
#     for (x, y, w, h) in faces:
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         roi_gray = gray[y:y+h, x:x+w]
#         roi_color = frame[y:y+h, x:x+w]
#     # Draw the frame with the rectangle back
#     out.write(frame)
#     # cv2.imshow('frame', frame)
#     # if counter % 3 == 0:
#     #     cv2.imshow('frame', frame)
#     # if cv2.waitKey(1) & 0xFF == ord('q'):
#     if ret != True:
#         break
# # Clean up
# cap.release()
# out.release()
# cv2.destroyAllWindows()
