# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 11:32:40 2019

@author: FanXudong
"""

from os.path import join
import argparse
import numpy as np
import sys
import os
import random 
import torch

 
def IOU(x,centroids):
    '''
    :param x: 某一个ground truth的w,h
    :param centroids:  anchor的w,h的集合[(w,h),(),...]，共k个
    :return: 单个ground truth box与所有k个anchor box的IoU值集合
    '''
    IoUs = []
    #w, h = x  # ground truth的w,h
    w = x
    for centroid in centroids:
        #c_w,c_h = centroid   #anchor的w,h
        c_w = centroid
        # if c_w>=w and c_h>=h:   #anchor包围ground truth
        if c_w>=w:
            # iou = w*h/(c_w*c_h)
            iou = w/c_w
        # elif c_w>=w and c_h<=h:    #anchor宽矮
        elif c_w>=w:
            # iou = w*c_h/(w*h + (c_w-w)*c_h)
            iou = w/(w + (c_w-w))
        elif c_w<=w:    #anchor瘦长
            iou = c_w/(w + c_w)
        else: #ground truth包围anchor     means both w,h are bigger than c_w and c_h respectively
            iou = (c_w)/(w)
        IoUs.append(iou) # will become (k,) shape
    return np.array(IoUs)
 
def avg_IOU(X,centroids):
    '''
    :param X: ground truth的w,h的集合[(w,h),(),...]
    :param centroids: anchor的w,h的集合[(w,h),(),...]，共k个
    '''
    #n,d = X.shape
    n = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        sum+= max(IOU(X[i],centroids))  #返回一个ground truth与所有anchor的IoU中的最大值
    return sum/n    #对所有ground truth求平均
 
def write_anchors_to_file(centroids,X,anchor_file,input_shape):
    '''
    :param centroids: anchor的w,h的集合[(w,h),(),...]，共k个
    :param X: ground truth的w,h的集合[(w,h),(),...]
    :param anchor_file: anchor和平均IoU的输出路径
    '''
    f = open(anchor_file,'w')
    
    anchors = centroids.copy()
    print(anchors.shape)
 
    for i in range(anchors.shape[0]):
        #求出yolov3相对于原图的实际大小
        #anchors[i][0]*=input_shape
        anchors[i]*=input_shape
        #anchors[i][1]*=input_shape

    #widths = anchors[:,0]
    widths = anchors[:]
    sorted_indices = np.argsort(widths)
 
    print('Anchors = ', anchors[sorted_indices])
        
    for i in sorted_indices[:-1]:
        #f.write('%0.2f,%0.2f, '%(anchors[i,0],anchors[i,1]))
        #f.write('%0.2f, '%(anchors[i,0]))
        f.write('%0.2f, '%(anchors[i]))
 
    #there should not be comma after last anchor, that's why
    #f.write('%0.2f,%0.2f\n'%(anchors[sorted_indices[-1:],0],anchors[sorted_indices[-1:],1]))
    #f.write('%0.2f\n'%(anchors[sorted_indices[-1:],0]))
    f.write('%0.2f\n'%(anchors[sorted_indices[-1:]]))
    
    f.write('avg IoU: %f\n'%(avg_IOU(X,centroids)))
    print()
 
def k_means(X,centroids,eps,anchor_file,input_shape):
    
    N = X.shape[0] #ground truth的个数
    iterations = 200
    print("centroids.shape",centroids)
    #k,dim = centroids.shape  #anchor的个数k以及w,h两维，dim默认等于2
    k = centroids.shape[0]
    prev_assignments = np.ones(N)*(-1)    #对每个ground truth分配初始标签
    iter = 0
    old_D = np.zeros((N,k))  #初始化每个ground truth对每个anchor的IoU
 
    while iter < iterations:
        D = []
        iter+=1           
        for i in range(N):
            d = 1 - IOU(X[i],centroids)
            D.append(d)
        D = np.array(D) # D.shape = (N,k)  得到每个ground truth对每个anchor的IoU
        
        print("iter {}: dists = {}".format(iter,np.sum(np.abs(old_D-D))))  #计算每次迭代和前一次IoU的变化值
            
        #assign samples to centroids 
        assignments = np.argmin(D,axis=1)  #将每个ground truth分配给距离d最小的anchor序号
        
        if (assignments == prev_assignments).all() :  #如果前一次分配的结果和这次的结果相同，就输出anchor以及平均IoU
            print("Centroids = ",centroids)
            write_anchors_to_file(centroids,X,anchor_file,input_shape)
            return
 
        #calculate new centroids
        #centroid_sums=np.zeros((k,dim),np.float)   #初始化以便对每个簇的w,h求和
        centroid_sums=np.zeros((k), float)
        for i in range(N):
            centroid_sums[assignments[i]]+=X[i]         #将每个簇中的ground truth的w和h分别累加
        for j in range(k):            #对簇中的w,h求平均
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j)+1)
        
        prev_assignments = assignments.copy()     
        old_D = D.copy()  
 
def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default = r'.\yolov3.txt',
                        help='path to filelist\n' ) #这个文件是由运行scripts文件夹中的    
                             #voc_label.py文件得到的，scripts文件夹中会生成几个TXT文件。
                             #python voc_label.py
                             #目前yolo打标签可以使用labelimg中的yolo格式
    parser.add_argument('--output_dir', default = r'.\save', type = str,
                        help='Output anchor directory\n' )
    parser.add_argument('--num_clusters', default = 7, type = int, 
                        help='number of clusters\n' )
    '''
    需要注意的是yolov2输出的值比较小是相对特征图来说的，
    yolov3输出值较大是相对原图来说的，
    所以yolov2和yolov3的输出是有区别的
    '''
    parser.add_argument('--input_shape', default=8192, type=int,
                        help='input images shape，multiples of 32. etc. 416*416\n')
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
 
    for dataset in ["ballroom", "hains", "carnatic"]:
        root_path = os.getcwd() + "/" + args.dir + "/" + dataset + "/label"
        file_names = os.listdir(root_path)

        for file_name in file_names:
            if file_name == ".DS_Store":
                continue

            f = open(os.path.join(root_path, file_name))
          
            lines = [line.rstrip('\n') for line in f.readlines()]
            
            annotation_dims = []
         
        #     for line in lines:
        # #        line = line.replace('JPEGImages','labels')
        #         #line = line.replace('JPEGImages','YOLO')
        #         line = line.replace('.wav','.beats')
        #         #line = line.replace('.png','.txt')
        # #        print(line)
        #         f2 = open(line)
        #         for line in f2.readlines():
        #             line = line.rstrip('\n')
        # #            print(line)
        #             temp = line.strip().split(" ")
        # #            print(temp)
                    
        #             w = float(temp[3])
        #             h = float(temp[4])
        # #            w,h = line.split(' ')[3:]            
        #             #print(w,h)
        #             annotation_dims.append((float(w),float(h)))

            beat_times, downbeat_times = [], []
            for line in lines:
                if dataset == "carnatic":
                    beat_time, beat_type = line.split(",")
                else:
                    line = line.replace('\t', ' ')
                    beat_time, beat_type = line.split(" ")

                if beat_type == "1":
                    downbeat_times.append(float(beat_time))
                else:
                    beat_times.append(float(beat_time))

            beat_intervals, downbeat_intervals = [], []

            for beat_index, current_beat_location in enumerate(beat_times[:-1]):
                next_beat_location = beat_times[beat_index + 1]

                beat_length = (next_beat_location - current_beat_location) * 22050 / 256
                annotation_dims.append(beat_length)

            for downbeat_index, current_downbeat_location in enumerate(downbeat_times[:-1]):
                next_downbeat_location = downbeat_times[downbeat_index + 1]

                downbeat_length = (next_downbeat_location - current_downbeat_location) * 22050 / 256
                annotation_dims.append(downbeat_length)

    annotation_dims = np.array(annotation_dims) #保存所有ground truth框的(w,h)

    eps = 0.005
 
    anchor_file = join( args.output_dir,'anchors_%d.txt'%(args.num_clusters))
    indices = [ random.randrange(annotation_dims.shape[0]) for i in range(args.num_clusters)]
    centroids = annotation_dims[indices]
    k_means(annotation_dims,centroids,eps,anchor_file,args.input_shape)
    print('centroids.shape', centroids.shape)
 
if __name__=="__main__":
    main(sys.argv)
