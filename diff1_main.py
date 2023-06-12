import torch
from torch import nn
import torchmetrics
from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
import numpy as np
import random as rd
# import norm
import diff_model_altfusion as fe
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import math
from sklearn.metrics import accuracy_score, f1_score
import time

from diff1_dataset import test_dataloader, train_dataloader

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
# =================================================================== 
# 这个副本目的是，把原本的三分类任务中，阈值区间以内的那些直接不参与分类任务，从而变成一个二分类任务
# ===================================================================

emg_model = fe.EMGModel(num_channels=5)
fmg_model = fe.FMGModel(num_channels=5)
fusion_model = fe.FusionModel(num_channels=5)
# emg_model = torch.load('G:\我的云端硬盘\Python\EMG_clmodel_50.pth')
# fmg_model = torch.load('G:\我的云端硬盘\Python\FMG_clmodel_50.pth')
# fusion_model = torch.load('G:\我的云端硬盘\Python\Fusion_clmodel_50.pth')

emg_model.to(device)
fmg_model.to(device)
fusion_model.to(device)
# multi_task_model.to(device)

# 选择损失函数
weights0 = [3.63,4.26,2.04] # 这里给三个分类分别分配一个权重，因为012中2的样本太多了所以权重给小一点
weights1 = [4.66,5.11,1.70]
weights2 = [3.65,3.76,2.18]
weights3 = [3.48,3.59,2.30]
weights4 = [3.65,3.65,2.21]
weights0_tensor = torch.tensor(weights0, dtype=torch.float32)
weights1_tensor = torch.tensor(weights1, dtype=torch.float32)
weights2_tensor = torch.tensor(weights2, dtype=torch.float32)
weights3_tensor = torch.tensor(weights3, dtype=torch.float32)
weights4_tensor = torch.tensor(weights4, dtype=torch.float32)
loss_fn0 = nn.CrossEntropyLoss(weight=weights0_tensor) 
loss_fn1 = nn.CrossEntropyLoss(weight=weights1_tensor) 
loss_fn2 = nn.CrossEntropyLoss(weight=weights2_tensor) 
loss_fn3 = nn.CrossEntropyLoss(weight=weights3_tensor) 
loss_fn4 = nn.CrossEntropyLoss(weight=weights4_tensor) 

if torch.cuda.is_available():
    loss_fn0 = loss_fn0.cuda()
    loss_fn1 = loss_fn1.cuda()
    loss_fn2 = loss_fn2.cuda()
    loss_fn3 = loss_fn3.cuda()
    loss_fn4 = loss_fn4.cuda()
    
loss_fns = [loss_fn0, loss_fn1, loss_fn2, loss_fn3, loss_fn4]
# loss_fn = nn.CrossEntropyLoss()

# loss_fn = torch.nn.L1Loss()



# 优化器

optimizer_emg = torch.optim.Adam(emg_model.parameters(), lr=0.01)
optimizer_fmg = torch.optim.Adam(fmg_model.parameters(), lr=0.01)
optimizer_fusion = torch.optim.Adam(fusion_model.parameters(), lr=0.01)
# optimizer = torch.optim.Adam(mymodel.parameters(), lr=learning_rate,weight_decay=0.0001)
# 设置训练网络的一些参数
#记录训练的次数
total_train_step = 0
#记录测试的次数
total_test_step = 0
#设置训练轮数
epoch = 100

test_dataloader = test_dataloader
train_dataloader = train_dataloader

def to_one_hot(data, num_classes):
    # 获取输入数据的形状
    shape = data.shape

    # 创建一个全零张量，形状为 (batch_size, channels, num_classes)
    one_hot_tensor = torch.zeros(shape[0], shape[1], num_classes, dtype=torch.float32)

    # 将数据转换为 int64 类型
    data = data.to(torch.int64)

    # 使用 scatter_ 方法将输入数据转换为 one-hot 编码形式
    one_hot_tensor.scatter_(2, data.unsqueeze(2), 1)

    return one_hot_tensor

# 创建f1 score
emg_f1_score = torchmetrics.F1Score(num_classes=3, average='macro', task='multiclass').cuda()
fmg_f1_score = torchmetrics.F1Score(num_classes=3, average='macro', task='multiclass').cuda()
fusion_f1_score = torchmetrics.F1Score(num_classes=3, average='macro', task='multiclass').cuda()
# ==


# def compute_accuracy(preds, labels):
#     _, predicted = torch.max(preds, -1)  # 在最后一维上找到最大值
#     total = labels.numel()  # 计算所有元素的数量
#     correct = (predicted == labels.view(-1)).sum().item()  # 将标签转换为1D张量，然后计算正确预测的数量
#     return correct / total  # 返回准确率
# idx = 0

# def calculate_accuracy(true_labels, predictions):
#     num_classes = predictions.max().item() + 1
#     accuracies = []

#     for class_id in range(num_classes):
#         correct = ((predictions == class_id) & (true_labels == class_id)).float().sum()
#         total = (true_labels == class_id).float().sum()

#         if total > 0:
#             accuracies.append((correct / total).item())
#         else:
#             accuracies.append(0.0)
    
#     return accuracies
def calculate_accuracy(y_true, y_pred):
    if isinstance(y_pred, list):
        y_pred = torch.Tensor(y_pred)  # 将列表转化为PyTorch张量
        y_true = torch.Tensor(y_true)
    y_true = y_true.cpu().numpy() if y_true.is_cuda else y_true.numpy()
    y_pred = y_pred.cpu().numpy() if y_pred.is_cuda else y_pred.numpy()
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy

def calculate_f1(y_true, y_pred):
    if isinstance(y_pred, list):
        y_pred = torch.Tensor(y_pred)  # 将列表转化为PyTorch张量
        y_true = torch.Tensor(y_true)
    y_true = y_true.cpu().numpy() if y_true.is_cuda else y_true.numpy()
    y_pred = y_pred.cpu().numpy() if y_pred.is_cuda else y_pred.numpy()
    f1 = f1_score(y_true, y_pred, average='macro')
    return f1

for i in range(epoch):
    print('---------第{}轮训练开始---------'.format(i+1))
    # 训练步骤开始
    fmg_model.train()
    fmg_model.train()
    fusion_model.train()
    batch_count = 0
    # total_train_step = 0
    # multi_task_model.train()
    # avg_Foutset_train=[]
    for data in train_dataloader:
        hyb, glo = data # 这里是hyb (200,6) glo (5)
        batch_count += 1
        # emg_running_loss = 0.0
        # fmg_running_loss = 0.0
        # fu_running_loss = 0.0
        # 初始化变量
        emg_running_loss = [0.0]*5  # 假设你有5个通道
        fmg_running_loss = [0.0]*5
        fu_running_loss = [0.0]*5
        
        # running_loss = 0.0
        # print(hyb.shape) # torch.Size([64, 100, 6])
        # print(glo.shape) # torch.Size([64, 5])

        num_classes = 3
        one_hot_glo = to_one_hot(glo, num_classes)
        one_hot_glo = one_hot_glo.to(device)
        # print(one_hot_glo[0])
        
        # print(one_hot_glo)
        # print(glo_labels[0].shape)
        if torch.cuda.is_available():

            glo = glo.float().cuda()
            hyb = hyb.float().cuda()
        hyb = hyb.permute(0,2,1)
        # print(hyb.shape) # torch.Size([64, 6, 200])
        hyb_emg = hyb[:, :3, :] # 前三列是emg信号
        hyb_fmg = hyb[:, 3:, :] # 后三列是fmg信号
        # 定义损失组
        losses_emg = []
        losses_fmg = []
        losses_fu = []
        
        # 训练 EMGModel
        optimizer_emg.zero_grad()
        emg_outputs = emg_model(hyb_emg)
        emg_outputs = torch.stack(emg_outputs).squeeze(dim=2).permute(1,0,2)
        
        # print(emg_outputs.shape) # torch.Size([64, 5, 3])
        for ch_output, ch_label, loss_fn in zip(emg_outputs, one_hot_glo, loss_fns):
            ch_loss = loss_fn(ch_output, ch_label)
            losses_emg.append(ch_loss)
            ch_loss.backward(retain_graph=True)  # 对每个通道的损失进行反向传播
        optimizer_emg.step()  # 更新参数
        
        # 训练 FMGModel
        optimizer_fmg.zero_grad()
        fmg_outputs = fmg_model(hyb_fmg)
        fmg_outputs = torch.stack(fmg_outputs).squeeze(dim=2).permute(1,0,2)
        for ch_output, ch_label, loss_fn in zip(fmg_outputs, one_hot_glo, loss_fns):
            ch_loss = loss_fn(ch_output, ch_label)
            losses_fmg.append(ch_loss)
            ch_loss.backward(retain_graph=True)  # 对每个通道的损失进行反向传播
        optimizer_fmg.step()  # 更新参数
        
        # 使用训练好的 EMGModel 和 FMGModel 获取输出
        # with torch.no_grad():
        #     emg_outputs = emg_model(hyb_emg)
        #     fmg_outputs = fmg_model(hyb_fmg)

        # 训练 FusionModel
        optimizer_fusion.zero_grad()
        # fusion_outputs = fusion_model(emg_outputs, fmg_outputs)
        fusion_outputs = fusion_model(hyb_emg,hyb_fmg)
        fusion_outputs = torch.stack(fusion_outputs).squeeze(dim=2).permute(1,0,2)
        for ch_output, ch_label, loss_fn in zip(fusion_outputs, one_hot_glo, loss_fns):
            ch_loss = loss_fn(ch_output, ch_label)
            losses_fu.append(ch_loss)
            ch_loss.backward(retain_graph=True)  # 对每个通道的损失进行反向传播
        optimizer_fusion.step()  # 更新参数

        # 计算损失，更新损失变量
        for i in range(5):  # 假设你有5个通道
            emg_running_loss[i] += losses_emg[i].item()
            fmg_running_loss[i] += losses_fmg[i].item()
            fu_running_loss[i] += losses_fu[i].item()

        total_train_step += 1
        if total_train_step % 100 == 0:  # 让每100个切片作为一个整体输出一次
            print(f'------------训练次数:{total_train_step}------------')  # loss.item()作用是只显示其对应的数字  
            for i in range(5):  # 假设你有5个通道
                print(f'---------Channel {i}---------')
                print(f'------fmg_Loss:{fmg_running_loss[i] / batch_count}------')
                print(f'------emg_Loss:{emg_running_loss[i] / batch_count}------')
                print(f'------fu_Loss:{fu_running_loss[i] / batch_count}------')
            
                
                
    
    # 测试步骤开始
    num_samples_to_show = 1  # 设置要显示的样本数量
    sample_counter = 0
    
    fmg_model.eval()
    fmg_model.eval()  
    fusion_model.eval()
    # multi_task_model.eval()
    total_accuracy_emg = 0
    total_accuracy_fmg = 0
    total_accuracy_fusion = 0
    num_samples = 0
    emg_correct = [0,0,0,0,0]
    fmg_correct = [0,0,0,0,0]
    fu_correct = [0,0,0,0,0]
    total = 0
    emgte_f1_scores = [0.0 for _ in range(5)]
    fmgte_f1_scores = [0.0 for _ in range(5)]
    fute_f1_scores = [0.0 for _ in range(5)]
    emg_accuracy = [0.0 for _ in range(5)]
    fmg_accuracy = [0.0 for _ in range(5)]
    fu_accuracy = [0.0 for _ in range(5)]
    test_num = 0
    all_emg_preds = [[None]*65 for _ in range(5)] # 嵌套列表，5行65列
    all_fmg_preds = [[None]*65 for _ in range(5)]
    all_fu_preds = [[None]*65 for _ in range(5)]
    all_labels = [[None]*65 for _ in range(5)]
    slice_emg_preds = [[None]*65 for _ in range(5)] # 嵌套列表，5行65列
    slice_fmg_preds = [[None]*65 for _ in range(5)]
    slice_fu_preds = [[None]*65 for _ in range(5)]
    slice_labels = [[None]*65 for _ in range(5)]
    
    all_emg_preds = []
    all_fmg_preds = []
    all_fusion_preds = []
    all_glo_t = []
    
    emg_preds_all = []
    fmg_preds_all = []
    fusion_preds_all = []
    glo_all = []

    with torch.no_grad(): # 测试要在无梯度时进行，这时候模型里只剩下各种权重值
        for data in test_dataloader: # 这里提取出来的每个glo_list都是包含batchsize的
            hyb, glo = data # 这里glo_list(646,174,5,100) 拼接起点应该为(0,0,5,100)，下一步则是(1,0,5,100)

            
            if torch.cuda.is_available():

                glo = glo.float().cuda()
                hyb = hyb.float().cuda()
            
            hyb = hyb.permute(0,2,1)
            # print(hyb.shape) # torch.Size([64, 6, 200])
            hyb_emg = hyb[:, :3, :]
            hyb_fmg = hyb[:, 3:, :]

            # num_samples += len(glo)
            emg_outputs = emg_model(hyb_emg)
            emg_outputs_p = torch.stack(emg_outputs).squeeze(dim=2) # torch.Size([5, 64, 3])
            fmg_outputs = fmg_model(hyb_fmg)
            fmg_outputs_p = torch.stack(fmg_outputs).squeeze(dim=2) # torch.Size([5, 64, 3])
            fusion_outputs = fusion_model(hyb_emg,hyb_fmg)
            fusion_outputs_p = torch.stack(fusion_outputs).squeeze(dim=2) # torch.Size([5, 64, 3])
            # print(emg_outputs_p.shape) # torch.Size([5, 64, 3])
            # print(glo.shape) # torch.Size([64, 5])
            glo_t = torch.tensor(glo).transpose(0,1) # # torch.Size([5, 64])
            # 获取预测类别并添加到列表中
            all_emg_preds.append(torch.argmax(emg_outputs_p, dim=2))
            all_fmg_preds.append(torch.argmax(fmg_outputs_p, dim=2))
            all_fusion_preds.append(torch.argmax(fusion_outputs_p, dim=2))

            # 添加真实标签到列表中
            all_glo_t.append(glo_t)
            
        # 连接所有批次的数据
        all_emg_preds = torch.cat(all_emg_preds, dim=1)
        all_fmg_preds = torch.cat(all_fmg_preds, dim=1)
        all_fusion_preds = torch.cat(all_fusion_preds, dim=1)
        all_glo_t = torch.cat(all_glo_t, dim=1)
        # print(all_emg_preds.shape) # torch.Size([5, 20736])
        # print(all_glo_t.shape) # torch.Size([5, 20736])

        # 计算总体准确率
        for ch in range(all_glo_t.shape[0]):
            emg_accuracy = calculate_accuracy(all_glo_t[ch], all_emg_preds[ch])
            fmg_accuracy = calculate_accuracy(all_glo_t[ch], all_fmg_preds[ch])
            fusion_accuracy = calculate_accuracy(all_glo_t[ch], all_fusion_preds[ch])
            emg_f1 = calculate_f1(all_glo_t[ch], all_emg_preds[ch])
            fmg_f1 = calculate_f1(all_glo_t[ch], all_fmg_preds[ch])
            fu_f1 = calculate_f1(all_glo_t[ch], all_fusion_preds[ch])
            print(f"For channel {ch}")
            print(f"EMG accuracy: {emg_accuracy *100:.4f}%, EMG F1score:{emg_f1:.4f}")
            print(f"FMG accuracy: {fmg_accuracy *100:.4f}%, FMG F1score:{fmg_f1:.4f}")
            print(f"Fusion accuracy: {fusion_accuracy *100:.4f}%, Fusion F1score:{fu_f1:.4f}")

        # print("EMG accuracy for class 0: ", emg_accuracy[0])
        # print("EMG accuracy for class 1: ", emg_accuracy[1])
        # # print("EMG accuracy for class 2: ", emg_accuracy[2])
        # print("FMG accuracy for class 0: ", fmg_accuracy[0])
        # print("FMG accuracy for class 1: ", fmg_accuracy[1])
        # # print("FMG accuracy for class 2: ", fmg_accuracy[2])
        # print("Fusion accuracy for class 0: ", fusion_accuracy[0])
        # print("Fusion accuracy for class 1: ", fusion_accuracy[1])
        random_indices = np.random.choice(all_glo_t.shape[1], 3, replace=False)

        for idx in random_indices:
            print(f"For sample {idx}:")
            print(f"Actual label: {all_glo_t[:, idx]}")
            print(f"EMG model prediction: {all_emg_preds[:, idx]}")
            print(f"FMG model prediction: {all_fmg_preds[:, idx]}")
            print(f"Fusion model prediction: {all_fusion_preds[:, idx]}")
        # print("Fusion accuracy for class 2: ", fusion_accuracy[2])
        #     # test_num += 1
        #     for ch_idx in range(glo.shape[1]):
        #         slice_emg_preds[ch_idx][i] = emg_outputs_p[ch_idx]
        #         slice_fmg_preds[ch_idx][i] = fmg_outputs_p[ch_idx]
        #         slice_fu_preds[ch_idx][i] = fusion_outputs_p[ch_idx]
        #         slice_labels[ch_idx][i] = glo[:,ch_idx]

        #     # 某个batch完成了全部的65个切片之后
        #     batch_emg_preds = torch.stack([torch.stack(row_tensors) for row_tensors in slice_emg_preds])
        #     batch_fmg_preds = torch.stack([torch.stack(row_tensors) for row_tensors in slice_fmg_preds])
        #     batch_fu_preds = torch.stack([torch.stack(row_tensors) for row_tensors in slice_fu_preds])
        #     batch_labels = torch.stack([torch.stack(row_tensors) for row_tensors in slice_labels])
        #     # print(batch_emg_preds.shape) # torch.Size([5, 65, 64, 13])
        #     # print(batch_labels.shape) # torch.Size([5, 65, 64])
            
        #     # 按照通道区分，在batch维度上拼接
        #     if test_num == 0: # 如果是第0次测试，则初始化第0个
        #         for ch_idx in range(batch_labels.shape[0]):
        #             all_emg_preds[ch_idx] = batch_emg_preds[ch_idx]
        #             # print(all_emg_preds[ch_idx].shape)
        #             all_fmg_preds[ch_idx] = batch_fmg_preds[ch_idx]
        #             all_fu_preds[ch_idx] = batch_fu_preds[ch_idx]
        #             all_labels[ch_idx] = batch_labels[ch_idx]
        #     if test_num != 0:
        #         for ch_idx in range(batch_labels.shape[0]):
        #             all_emg_preds[ch_idx] = torch.cat([all_emg_preds[ch_idx], batch_emg_preds[ch_idx]], dim=1)
        #             all_fmg_preds[ch_idx] = torch.cat([all_fmg_preds[ch_idx], batch_fmg_preds[ch_idx]], dim=1)
        #             all_fu_preds[ch_idx] = torch.cat([all_fu_preds[ch_idx], batch_fu_preds[ch_idx]], dim=1)
        #             all_labels[ch_idx] = torch.cat([all_labels[ch_idx], batch_labels[ch_idx]], dim=1)
                    
        #     test_num += 1 
        # all_emg_preds = torch.stack(all_emg_preds)
        # all_fmg_preds = torch.stack(all_fmg_preds)
        # all_fu_preds = torch.stack(all_fu_preds)
        # all_labels = torch.stack(all_labels)

        # all_emg_preds_reshaped = all_emg_preds.view(5, -1, 3)
        # all_fmg_preds_reshaped = all_fmg_preds.view(5, -1, 3)
        # all_fu_preds_reshaped = all_fu_preds.view(5, -1, 3)
        # all_labels_reshaped = all_labels.view(5, -1)

        

        # for ch in range(5):
        #     # Compute F1 score
        #     average_emg_f1 = emg_f1_score(all_emg_preds_reshaped[ch], all_labels_reshaped[ch]).item()
        #     average_fmg_f1 = fmg_f1_score(all_fmg_preds_reshaped[ch], all_labels_reshaped[ch]).item()
        #     average_fu_f1 = fusion_f1_score(all_fu_preds_reshaped[ch], all_labels_reshaped[ch]).item()
        #     # Compute accuracy
        #     _, emg_predicted = torch.max(all_emg_preds_reshaped[ch], -1)
        #     _, fmg_predicted = torch.max(all_fmg_preds_reshaped[ch], -1)
        #     _, fu_predicted = torch.max(all_fu_preds_reshaped[ch], -1)
        #     # print(len(emg_predicted))
        #     # print(all_labels[ch_idx].shape)
        #     total_emg_correct = (emg_predicted== all_labels_reshaped[ch]).sum().item()
        #     total_fmg_correct = (fmg_predicted == all_labels_reshaped[ch]).sum().item()
        #     total_fu_correct = (fu_predicted == all_labels_reshaped[ch]).sum().item()
        #     average_emg_accuracy = total_emg_correct / all_emg_preds.numel()
        #     average_fmg_accuracy = total_fmg_correct / all_fmg_preds.numel()
        #     average_fu_accuracy = total_fu_correct / all_fu_preds.numel()
        #     print(f'ch_{ch}, emg_f1={average_emg_f1:.3f}, fmg_f1={average_fmg_f1:.3f}, fu_f1={average_fu_f1:.3f}')
        #     print(f'ch_{ch}, emg_ac={average_emg_accuracy * 100:.3f}%, fmg_ac={average_fmg_accuracy * 100:.3f}%, fu_ac={average_fu_accuracy * 100:.3f}%')
    
    # timenow = int(time.time())
    # torch.save(fmg_model,f'G:\\我的云端硬盘\\Python\\model\\{timenow}_EMG_clmodel_{epoch}.pth') # 保存每一轮的模型  
    # torch.save(fmg_model,f'G:\\我的云端硬盘\\Python\\model\\{timenow}_FMG_clmodel_{epoch}.pth')
    # torch.save(fusion_model,f'G:\\我的云端硬盘\\Python\\model\\{timenow}_FU_clmodel_{epoch}.pth')
    torch.save(fmg_model,f'EMG_3cmodel_{epoch}.pth') # 保存每一轮的模型  
    torch.save(fmg_model,f'FMG_3cmodel_{epoch}.pth')
    torch.save(fusion_model,f'FU_3cmodel_{epoch}.pth')
    # torch.save(multi_task_model,'multi_task_model_{}.pth'.format(epoch))
    # path = 'model_{}.pth'.format(epoch)
    # torch.save(model, path)  
            
# 目前没有解决的问题：
# 2023/4/19
# 消除每个切片初始结束位置的奇异值（特别离谱的值，比如过大或过小）---几乎完成
# 2023/4/20 05:48
# 明天（划掉）今天模型试一下TCN因果卷积 ---试了，效果不明显，晚点试下并行卷积分支的网络层
# 2023/4/21 
# 今天有事去都内。晚点回来再说
# 2023/4/22 17:10
# 先找到一个接近的成功例子看一看，看一下对方是怎么实现的。如果使用的是深度学习，试图搞清楚对方是如何用某个网络实现这个目的的
# 2023/4/24 19:31
# 这数据处理绝对还tm有问题，为啥还是能出现这么极端的数值而且反向增长。
# 而且在训练轮数约为67作用的时候数值变得乱七八糟。接下来好好检查一下那里可能出了问题。
# 2023/4/25 23:56
# 莱斯特神助攻！！之前那种切片方式存在致命问题！我试图用一小段数据去预测一个积累值，这原理上就不可能
# 回头试一下增加样本长度，增长率为20，然后让每次新生成的这20长度的为训练样本，目标值为数据手套这20长度的始末差值
# 2023/4/28 17:24
# 预测数值波动太小，是归一化用的之前四万多的数据的原因。回头试一下重新基于这组新的数据归一化.
# 有一个奇思妙想，如果令所有手套实际的正向波动为1，正负10%以内的波动为0，反向波动为-1，然后改模型为分类？
# 接上一条，四万多的数值的话，变大是手指张开，变小是手指收回
# 初步设定1，如果电压变换值在[-250,+250]之间，则令其=0，若在以上则为1，以下则为-1（后续可以考虑再加一个-2，表示用力收回，可以设置一个单独的分类标准）
# 一个可能的问题，如果动的太慢（比如200样本长度的|末-初|<250），则可能导致完全识别不到
# 手指被动变化位置（比如放到一个鼠标上，手指发生弯曲），基准位置不会发生改变----后话
# 2023/4/29 
# 上述方法存在的几个问题
# 1.在这种处理方法中，大部分数据都被处理成了0，导致0分类的数据量相对过大。需要考虑怎么解决。（使用权重损失函数或者重新采样，去掉一部分0数据，复制一部分1和-1数据）
# 2.glo数据被这样处理后丢失了很多信息。（chatgpt提供的解决方案是为每个通道训练独立的模型或者使用多任务学习）
# 2023/5/2 17点59分
# 对于torch中的dim问题，假设存在这么一个数据尺寸是(64,5,3),对其dim=1指的是将这个=5的维度视为一个整体，然后对这个整体处理。
# 所以_, index = torch.max(tensor,dim=1)后index尺寸变成了(65,3),即每5个元素中最大值的位置索引
# 19点53分
# 未解决的问题：
# 准确率低的离谱。到底是这些数据的预测结果就这么拉还是我代码哪里有问题。好好检查。
# 2023/5/3 15点57分
# 现在准确率倒是高了，但问题是分类任务中绝大多数都是1，而0和2的数据过少，所以准确率高了也是虚高。必须要把分类数据分布平均化一下才行
# 2023/5/5 
# 测试了分配权重，效果一般。准确度还是很高，但是实测之后发现在乱分类。再尝试一下重采样
# 2023/5/8 19点39分
# 重采样效果也一般。明天再试一下多加几个分类，多分割几个类别
# 2023/5/10 15点29分 
# 昨天把分类增加到了13种。重采样的时候有一个问题，样本数小于batchsize导致根本没有进入训练过程
# 目前为止的fusion_model是使用emg和fmg模型的（结果）作为输入数据。
# 接下来试一下在最开始就把fmg和emg分别提取特征然后共同参与分类
# 还是试一下一开始就把fmg和emg混在一起提取信号吧
# 23点28分
# 效果仍然很烂，
# EMG Model Accuracy: 69.7580%,f1_score:0.0714
# FMG Model Accuracy: 25.6891%,f1_score:0.0364
# Fusion Model Accuracy: 40.7724%,f1_score:0.0444
# 非常感人 明天想想办法.


# 对labels的每一个通道，使模型仅基于阈值区间的外的分类结果（>550与<-550这两类）进行训练，并基于以下三种情况对结果进行最终分类
# 1.当>550的类别的可能性大于80%，则最终分类为>550所属的类别
# 2.当<-550的类别的可能性大于80%，则最终分类为<-550所属的类别
# 3.如果上述两种的可能性都小于80%，则最终分类为阈值区间的内的类别

# 2023/5/26 04点59分
# 我想要的效果是，在测试阶段，五个通道分别输出一个自己这个通道的分类结果，然后只看自己这个分类结果和实际是否对的上
# 2023/5/30 16点58分
# 关于训练部分，我希望的是，对于五个通道，只对实际类别不为第2类的那几个通道进行学习。从而集中注意于0和1类。
# 尽可能拉高0和1这两个分类的准确度，令第2类one_hot之后一直保持在0，因此，当在0，1这两个类的可能性都较低时
# 第2类的可能性自然会较高