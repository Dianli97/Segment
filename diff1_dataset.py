import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader,WeightedRandomSampler
import os
import pandas as pd
import random as rd
import numpy as np
import glob
# import norm
# import FE_model as fe
# from diff_test import *

# class MyData(Dataset):
#     def __init__(self, root_dir, dstype, subject):  #dstype:数据集类型，train或者test
        # dt_dir = os.path.join(root_dir, dstype)
        # self.glo_dir = os.path.join(dt_dir, 'glove_tar')
        # self.hyb_dir = os.path.join(dt_dir, 'hybrid_src')
#         self.glo_file_name = os.listdir(self.glo_dir)
#         self.hyb_file_name = os.listdir(self.hyb_dir)
#         # all_data = torch.cat(self.data, dim=2)  # dim=2表示沿着长度维度拼接
#         # all_labels = torch.cat(self.labels, dim=2) 
#         # self.glo_file_paths = []
#         # self.hyb_file_paths = []
#         self.subject = str('_'+subject+'_')
        

#         for i in range(len(self.glo_file_name)): 
#             self.glo_file_paths.append(os.path.join(self.glo_dir, self.glo_file_name[i])) # 建立一个列表，里面的元素为每一个glo数据的path
#             name = str(self.glo_file_name[i])
#             filename = name.split(self.subject)[1]
#             tar_hyb_name = 'hybrid'+self.subject+filename
#             self.hyb_file_paths.append(os.path.join(self.hyb_dir, tar_hyb_name))

    
#     def __getitem__(self, idx): # 我们要让getitem做必要的范围内最少的事情
#         glo_data = pd.read_csv(self.glo_file_paths[idx], header=None, skiprows=10, usecols=[0,1,2,3,4], names=['ch_0','ch_1','ch_2','ch_3','ch_4'])
#         hyb_data = pd.read_csv(self.hyb_file_paths[idx], header=None, skiprows=11, usecols=[0,1,2,3,4,5], names=['ch_0','ch_1','ch_2','ch_3','ch_4','ch_5'])
#         # print(hyb_data.head())
#         # print(self.glo_file_paths[idx])
#         # print(self.hyb_file_paths[idx])
#         glo_data = torch.tensor(glo_data.values, dtype=torch.float32)
#         hyb_data = torch.tensor(hyb_data.values, dtype=torch.float32)
#         glo_data = torch.transpose(glo_data, 0, 1) #
#         hyb_data = torch.transpose(hyb_data, 0, 1) # 

#         sample_size = 200
#         step = 200
#         glo_samples = []
#         hyb_samples = []

#         for i in range(0, glo_data.shape[1] - sample_size + 1, step):
#             glo_slice = glo_data[ : ,i:i + sample_size] 

#             # ========================================================
#             # 这里使一个新的尝试，把目标数据转换成20长度的数据的首末差值
#             glo_slice_diff = glo_slice[:, -1] - glo_slice[:, 0] # 每个切片值都变成了初末位置的差值。
#             targets = np.zeros_like(glo_slice_diff)
#             targets[glo_slice_diff > 550] = 2
#             targets[glo_slice_diff < -550] = 0
#             # print(glo_slice.shape) # (5,1)
#             # print(targets)
#             # =========================================================
            
#             glo_samples.append(targets)
#         # for i in range(65):
#         #     print('sample_num={}\n'.format(i),glo_samples[i])
#         for i in range(0, hyb_data.shape[1] - sample_size + 1, step):
#             hyb_slice = hyb_data[ : ,i:i + sample_size] 
#             hyb_samples.append(hyb_slice)

#         return glo_samples, hyb_samples
        
    
#     def __len__(self): # 关于len和getitem的联动，目前为止一种能理解的解释是，len会返回数据数量n,从而影响getitem里的idx，使idx为一个[0,n-1]的数字，从而实现用索引的方式提取某一个数据。
#         return(len(self.glo_file_name)) #

# 200长度切片
# class MyDataset(Dataset):
#     def __init__(self, data_folder_path, dstype, subject, step=20):  # <== 修改部分: 增加了一个步长参数 step，默认为20
#         dt_dir = os.path.join(data_folder_path, dstype)
#         self.glo_dir = os.path.join(dt_dir, 'glove_tar')
#         self.hyb_dir = os.path.join(dt_dir, 'hybrid_src')
#         self.data_files = sorted(glob.glob(self.hyb_dir + "/hybrid*.csv")) 
#         self.label_files = sorted(glob.glob(self.glo_dir + "/glove*.csv"))
#         self.subject = str('_'+subject+'_')

#         self.all_data = []
#         self.all_labels = []
#         sample_size = 200
#         step = 20
#         for data_file, label_file in zip(self.data_files, self.label_files):
#             data = pd.read_csv(data_file, header=None, skiprows=11, usecols=[0,1,2,3,4,5], names=['ch_0','ch_1','ch_2','ch_3','ch_4','ch_5'])
#             labels = pd.read_csv(label_file, header=None, skiprows=10, usecols=[0,1,2,3,4], names=['ch_0','ch_1','ch_2','ch_3','ch_4'])
#             # 将数据和标签转化为张量
#             data = torch.tensor(data.values)
#             labels = torch.tensor(labels.values)

#             # 对每个文件进行切割和过滤
#             data_list = []
#             labels_list = []
#             for i in range(0, len(data) - 199, 10):  # 更改了步长为20，使数据有重叠部分
#                 segment_data = data[i:i+200]
#                 segment_labels = labels[i:i+200]
#                 labels_diff = segment_labels[199] - segment_labels[0]
#                 if ((labels_diff < -550) | (labels_diff > 550)).any():  # 根据阈值过滤标签，并获取相应的索引
#                     data_list.append(segment_data)
#                     labels_list.append(segment_labels)
#             if data_list:
#                 self.all_data.extend(data_list)
#             if labels_list:
#                 self.all_labels.extend(labels_list)
#             # 在所有标签已加载后，将其转换为一个大的张量
#             # self.all_labels_tensor = torch.stack([torch.mode(labels).values for labels in self.all_labels])

#     def __getitem__(self, index):
#         x = self.all_data[index]
#         y = self.process_labels(self.all_labels[index])
#         return x, y

#     def process_labels(self, labels):
#         diff = labels[199] - labels[0]
#         # print(diff)
#         result = torch.where(diff < -550, torch.tensor(0), torch.where((diff >= -550) & (diff <= 550), torch.tensor(2), torch.tensor(1)))
#         # print(result)
#         return result

#     def __len__(self):
#         return len(self.all_data)

# 100长度
class MyDataset(Dataset):
    def __init__(self, data_folder_path, dstype, subject, step=20):  # <== 修改部分: 增加了一个步长参数 step，默认为20
        dt_dir = os.path.join(data_folder_path, dstype)
        self.glo_dir = os.path.join(dt_dir, 'glove_tar')
        self.hyb_dir = os.path.join(dt_dir, 'hybrid_src')
        self.data_files = sorted(glob.glob(self.hyb_dir + "/hybrid*.csv")) 
        self.label_files = sorted(glob.glob(self.glo_dir + "/glove*.csv"))
        self.subject = str('_'+subject+'_')

        self.all_data = []
        self.all_labels = []
        for data_file, label_file in zip(self.data_files, self.label_files):
            data = pd.read_csv(data_file, header=None, skiprows=11, usecols=[0,1,2,3,4,5], names=['ch_0','ch_1','ch_2','ch_3','ch_4','ch_5'])
            labels = pd.read_csv(label_file, header=None, skiprows=10, usecols=[0,1,2,3,4], names=['ch_0','ch_1','ch_2','ch_3','ch_4'])
            # 将数据和标签转化为张量
            data = torch.tensor(data.values)
            labels = torch.tensor(labels.values)
            # print(labels.shape)
            # 对每个文件进行切割和过滤
            data_list = []
            labels_list = []
            for i in range(0, len(data) - 99, 10):  # 更改了步长为20，使数据有重叠部分
                segment_data = data[i:i+100]
                segment_labels = labels[i:i+100]
                # print(segment_labels.shape) # torch.Size([100, 5])
                labels_diff = segment_labels[99] - segment_labels[0]
                # print(labels_diff.shape)
                if ((labels_diff < -250) | (labels_diff > 250)).any():  # 根据阈值过滤标签，并获取相应的索引
                    data_list.append(segment_data)
                    labels_list.append(segment_labels)
            if data_list:
                self.all_data.extend(data_list)
            if labels_list:
                self.all_labels.extend(labels_list) 
        
            # 在所有标签已加载后，将其转换为一个大的张量
        # self.all_labels_tensor = torch.stack(self.all_labels)
        # for i in range(self.all_labels_tensor.shape[0]):
            
        # print(self.all_labels_tensor.shape)
        
    def __getitem__(self, index):
        x = self.all_data[index]
        y = self.process_labels(self.all_labels[index])
        return x, y

    def process_labels(self, labels):
        diff = labels[99] - labels[0]
        # print(diff)
        result = torch.where(diff < -250, torch.tensor(0), torch.where((diff >= -250) & (diff <= 250), torch.tensor(2), torch.tensor(1)))
        # print(result)
        return result

    def __len__(self):
        return len(self.all_data)


data_folder_path = r'C:\csv_data_pre'
train_data = MyDataset(data_folder_path, dstype='train', subject='a')
test_data = MyDataset(data_folder_path, dstype='test', subject='a')
train_dataloader = DataLoader(train_data, batch_size=256, drop_last=True)
test_dataloader = DataLoader(test_data, batch_size=256, drop_last=True)

# train_counts = train_data.all_labels_tensor.bincount(minlength=3)  # minlength参数确保输出tensor的长度至少为3，即使某些类别不存在
# test_counts = test_data.all_labels_tensor.bincount(minlength=3)

# print("Train dataset counts: ", train_counts)
# print("Test dataset counts: ", test_counts)
# 初始化一个字典来存储每个通道的分类数量
# channel_counts = {f'ch_{i}': {j: 0 for j in range(3)} for i in range(5)}

# for data in train_data:
#     hyb, glo = data  # glo是分类标签
#     # 对于每个通道，统计每个类别的数量，并更新到字典中
#     for i in range(5):  # 对于每个通道
#         channel = f'ch_{i}'
#         label = glo[i].item()  # 提取此通道的标签
#         channel_counts[channel][label] += 1  # 更新字典

# # 打印结果
# for channel, counts in channel_counts.items():
#     print(f'{channel}: {counts}')
###################################
# ch_0: {0: 12298, 1: 10472, 2: 21887}
# ch_1: {0: 9584, 1: 8735, 2: 26338}
# ch_2: {0: 12247, 1: 11889, 2: 20521}
# ch_3: {0: 12838, 1: 12424, 2: 19395}
# ch_4: {0: 12244, 1: 12229, 2: 20184}
###################################
# for data in train_data:
#     hyb, glo = data 
#     print(f'hyb,{hyb.shape}')
#     # print(hyb)
#     print(f'glo,{glo.shape}')
    # print(glo)
    
