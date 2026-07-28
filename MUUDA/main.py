import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import mmd
import numpy as np
from sklearn import metrics
import time
import utils
from torch.utils.data import TensorDataset, DataLoader

from loss import *
from select import select_high_confidence_samples_by_voting
import argparse
from pytorch_metric_learning.losses import SupConLoss

parser = argparse.ArgumentParser(description='Selection')

parser.add_argument('--dataset', type=str, default='Indiana', choices=['Pavia', 'Houston', 'S-H', 'Indiana'],
                    help='dataset')
args = parser.parse_args()

##################################
# dataset
if args.dataset == 'Pavia':
    from datasets.Pavia.config_UP2PC import *
    data_s, label_s = utils.load_data_pavia(data_path_s, label_path_s)
    data_t, label_t = utils.load_data_pavia(data_path_t, label_path_t)

elif args.dataset == 'Houston':
    from datasets.Houston.config_Houston import *
    data_s, label_s = utils.load_data_houston(data_path_s, label_path_s)
    data_t, label_t = utils.load_data_houston(data_path_t, label_path_t)

elif args.dataset == 'S-H':
    from datasets.Shanghai_Hangzhou.config_SH2HZ import *
    data_s, data_t, label_s, label_t = utils.cubeData(file_path)

elif args.dataset == 'Indiana':
    from datasets.Indiana.config_Indiana import *
    data_s, data_t, label_s, label_t = utils.cubeData(file_path)

# Loss Function
crossEntropy = nn.CrossEntropyLoss().cuda()
loss_fn = SupConLoss(temperature=0.1).cuda()
loss_crit = consistency_loss


acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
f1 = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G, best_RandPerm, best_Row, best_Column, best_nTrain = None, None, None, None, None

utils.set_seed(seeds[0])

trainX, trainY = utils.get_sample_data(data_s, label_s, HalfWidth, num_per_class)
testID, testX, testY, G, RandPerm, Row, Column = utils.get_all_data(data_t, label_t, HalfWidth)

train_dataset = TensorDataset(torch.tensor(trainX), torch.tensor(trainY, dtype=torch.long))
test_dataset = TensorDataset(torch.tensor(testX), torch.tensor(testY, dtype=torch.long))

train_loader_s = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
train_loader_t = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

len_source_loader = len(train_loader_s)
len_target_loader = len(train_loader_t)

# model
from models.net import DSAN

feature_encoder = DSAN(n_band=nBand, patch_size=patch_size, num_class=CLASS_NUM).cuda()
# 学习率
LEARNING_RATE = lr
# 优化器
optimizer = torch.optim.SGD(feature_encoder.parameters(), lr=LEARNING_RATE, momentum=momentum, weight_decay=l2_decay)

print("Training...")

last_accuracy = 0.0
best_episdoe = 0
train_loss = []
test_acc = []
running_D_loss, running_F_loss = 0.0, 0.0
running_label_loss = 0
running_domain_loss = 0
total_hit, total_num = 0.0, 0.0
size = 0.0
test_acc_list = []
pseudo_data = None
pseudo_labels = None
pseudo_loader = None

train_start = time.time()

for epoch in range(1, epochs + 1):

    feature_encoder.train()

    iter_source = iter(train_loader_s)
    iter_target = iter(train_loader_t)

    num_iter = len_source_loader

    for i in range(1, num_iter):
        source_data, source_label = next(iter_source)
        target_data, target_label = next(iter_target)

        source_features, source_band_weights, source_outputs = feature_encoder(source_data.cuda())
        target_features, target_band_weights, target_outputs = feature_encoder(target_data.cuda())
        
        # Loss Cls
        cls_loss_s = crossEntropy(source_outputs, source_label.cuda())
        # Loss Lmmd
        lmmd_loss = mmd.lmmd(source_features, target_features, source_label,
                             torch.nn.functional.softmax(target_outputs, dim=1), BATCH_SIZE=BATCH_SIZE,
                             CLASS_NUM=CLASS_NUM)
        # Loss Contrastive
        loss_contrastive_s = loss_fn(source_features, source_label.cuda())
        category_loss_s = loss_crit(source_band_weights, source_data, source_label.cuda())

        lambd = 2 / (1 + math.exp(-10 * (epoch) / epochs)) - 1

        loss = cls_loss_s + alpha * lambd * lmmd_loss + category_loss_s + beta * loss_contrastive_s
        # loss = cls_loss_s + alpha * lambd * lmmd_loss + beta * category_loss_s

        # Update parameters - 使用统一优化器
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= train_num and pseudo_loader is not None:
            iter_pse = iter(pseudo_loader)
            pseudo_data, pseudo_label = next(iter_pse)
            pseudo_features, pseudo_band_weights, pseudo_outputs = feature_encoder(pseudo_data.cuda())

            cls_loss_t = crossEntropy(pseudo_outputs, pseudo_label.cuda())
            # 计算类别一致性损失
            category_loss_t = loss_crit(pseudo_band_weights, pseudo_data, pseudo_label.cuda())
            loss_contrastive_t = loss_fn(pseudo_features, pseudo_label.cuda())

            loss_t = cls_loss_t + category_loss_t + beta * loss_contrastive_t
            # Update parameters
            optimizer.zero_grad()
            loss_t.backward()
            optimizer.step()

        pred = source_outputs.data.max(1)[1]
        total_hit += pred.eq(source_label.data.cuda()).sum()
        size += source_label.data.size()[0]
        test_accuracy = 100. * float(total_hit) / size

    print(
        'epoch {:>3d}:   cls_loss_s: {:6.4f}, lmmd loss:{:6f}, loss_contra_s: {:6.4f},loss_contrastive_s: {:6.4f}, acc {:6.4f}, total loss: {:6.4f}'
        .format(epoch, cls_loss_s.item(), lmmd_loss.item(),
                category_loss_s.item(), loss_contrastive_s.item(),
                total_hit / size, loss.item()))

    # ==================== 伪标签更新 ====================
    if epoch >= train_num and epoch % CONFIDENT_INTERVAL == 0 and epoch != epochs:
        print("Updating pseudo labels...")

        all_test_features = []
        all_test_outputs = []
        all_test_datas = []
        all_test_labels = []

        feature_encoder.eval()
        with torch.no_grad():

            # 分批处理测试数据，避免内存溢出
            for test_datas, test_labels in test_loader:
                # 将数据移到GPU并计算特征
                test_datas_gpu = test_datas.cuda()
                test_features, test_band_weights, test_outputs = feature_encoder(test_datas_gpu)

                # 将结果移回CPU保存
                all_test_features.append(test_features.cpu())
                all_test_outputs.append(test_outputs.cpu())
                all_test_datas.append(test_datas)
                all_test_labels.append(test_labels)

                # 立即释放GPU内存
                del test_datas_gpu, test_features, test_outputs
                torch.cuda.empty_cache()

            # 合并所有批次的数据
            all_test_features = torch.cat(all_test_features, dim=0)
            all_test_outputs = torch.cat(all_test_outputs, dim=0)
            all_test_datas = torch.cat(all_test_datas, dim=0)
            all_test_labels = torch.cat(all_test_labels, dim=0)

            # 转换为概率（在CPU上进行）
            test_probs = F.softmax(all_test_outputs, dim=1)

            selected_idx, pseudo_labels = select_high_confidence_samples_by_voting(
                all_test_features.numpy(),
                test_probs.numpy(),
                num_components=20,
                num_clusters=multiple * CLASS_NUM,
                seed=seeds[0],
                balance_sample_total=num_per_class * CLASS_NUM,
            )

            if len(selected_idx) > 0:
                # 保存伪标签数据
                pseudo_data = all_test_datas[selected_idx]
                print(f"Selected {len(pseudo_data)} target domain pseudo-labeled samples")

                batch_true_labels = all_test_labels.numpy()[selected_idx]

                # 计算准确率
                correct = (pseudo_labels == batch_true_labels).sum()
                accuracy = 100. * correct / len(pseudo_labels)
                print(f"Selected {len(pseudo_data)} samples | Pseudo-label Accuracy: {accuracy:.2f}%")

                pseudo_labels_tensor = torch.tensor(pseudo_labels, dtype=torch.long)

                # 创建伪标签DataLoader
                pseudo_dataset = TensorDataset(pseudo_data, pseudo_labels_tensor)
                pseudo_loader = DataLoader(pseudo_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

            else:
                print("No pseudo-labeled samples selected")
                pseudo_loader = None

            del all_test_features, all_test_outputs, all_test_datas

        feature_encoder.train()

    train_end = time.time()

    if epoch % 20 == 0:
        print("Testing ...")
        feature_encoder.eval()
        total_rewards = 0
        counter = 0
        accuracies = []
        predict = np.array([], dtype=np.int64)
        labels = np.array([], dtype=np.int64)
        with torch.no_grad():
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                test_features, test_band_weights, test_outputs = feature_encoder(Variable(test_datas).cuda())

                pred = test_outputs.data.max(1)[1]

                test_labels = test_labels.numpy()
                rewards = [1 if pred[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, pred.cpu().numpy())
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter  #
                accuracies.append(accuracy)

        test_accuracy = 100. * total_rewards / len(test_loader.dataset)
        acc[0] = 100. * total_rewards / len(test_loader.dataset)
        OA = acc
        C = metrics.confusion_matrix(labels, predict)
        A[0, :] = np.diag(C) / np.sum(C, 1, dtype=float)
        f1[0] = metrics.f1_score(labels, predict, average='macro')
        k[0] = metrics.cohen_kappa_score(labels, predict)
        print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format(total_rewards, len(test_loader.dataset),
                                                       100. * total_rewards / len(test_loader.dataset)))
        test_end = time.time()

        # Training mode
        if test_accuracy > last_accuracy:
            # save networks
            # torch.save(feature_encoder.state_dict(),str("../checkpoints/DFSL_feature_encoder_" + "houston_cl_lmmd_dis_attention" +str(iDataSet) +".pkl"))
            print("save networks for epoch:", epoch + 1)
            last_accuracy = test_accuracy
            best_episdoe = epoch
            best_predict_all = predict
            best_G, best_RandPerm, best_Row, best_Column = G, RandPerm, Row, Column
            print('best epoch:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))

AA = np.mean(A, 1)
AAMean = np.mean(AA,0)
AAStd = np.std(AA)
AMean = np.mean(A, 0)
AStd = np.std(A, 0)
OAMean = np.mean(acc)
OAStd = np.std(acc)
kMean = np.mean(k)
kStd = np.std(k)
print("train time per DataSet(s): " + "{:.5f}".format(train_end - train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end - train_end))
print("OA: " + "{:.2f}".format(OAMean) + " +- " + "{:.2f}".format(OAStd))
print("AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print("Kappa: " + "{:.4f}".format(100 * kMean) + " +- " + "{:.4f}".format(100 * kStd))
# print("F1-score: " + "{:.4f}".format(100 * f1Mean))  # 输出F1-score
print("Accuracy for each class: ")
for i in range(CLASS_NUM):
    print("Class " + str(i + 1) + ": " + "{:.2f}".format(100 * AMean[i]))

################classification map################################

for i in range(len(best_predict_all)):  # predict ndarray <class 'tuple'>: (9729,)
    best_G[best_Row[best_RandPerm[i]]][best_Column[best_RandPerm[i]]] = best_predict_all[i] + 1

hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 0:
            hsi_pic[i, j, :] = [0, 0, 0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [0, 0, 1]
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0, 1, 0]
        if best_G[i][j] == 3:
            hsi_pic[i, j, :] = [0, 1, 1]
        if best_G[i][j] == 4:
            hsi_pic[i, j, :] = [1, 0, 0]
        if best_G[i][j] == 5:
            hsi_pic[i, j, :] = [1, 0, 1]
        if best_G[i][j] == 6:
            hsi_pic[i, j, :] = [1, 1, 0]
        if best_G[i][j] == 7:
            hsi_pic[i, j, :] = [0.5, 0.5, 1]
import matplotlib.pyplot as plt

# 保存图像
plt.imsave(f'results/{args.dataset}_classification_map.png', hsi_pic)