import numpy as np
import argparse
import time, os
# import random
from src.early_stopping import *
from src import process_twitter as process_data
import copy
import pickle as pickle
from random import sample
import torchvision
import torch
from trans_padding import Conv1d as conv1d

from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
print(torch.version.cuda)
if(torch.cuda.is_available()):
    print("CUDA 存在")
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
#多模线性池化
from src.pytorch_compact_bilinear_pooling import  CountSketch, CompactBilinearPooling
import sys
# from logger import Logger
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
#sys.stdout = Logger("/tmp/pycharm_project_169/src/result/multi_fusion/multi_twitter02.txt")  # 保存到D盘

from sklearn import metrics
from transformers import AutoConfig, TFAutoModel, AutoTokenizer, BertModel, BertTokenizer
#可视化
#from tensorboardX import SummaryWriter
#writer = SummaryWriter('runs/multi_fusion_twitter02')
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#预训练模型
#config = AutoConfig.from_pretrained('/root/autodl-tmp/pre_trainmodel/ber-base-uncased/config.json')
#MODEL = '/root/autodl-tmp/pre_trainmodel/ber-base-uncased'
MODEL = 'bert-base-uncased'
N_LABELS = 1

class Rumor_Data(Dataset):
    def __init__(self, dataset):
        self.text = torch.from_numpy(np.array(dataset['post_text']))
        self.image = list(dataset['image'])
        # self.social_context = torch.from_numpy(np.array(dataset['social_feature']))
        self.mask = torch.from_numpy(np.array(dataset['mask']))
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.event_label = torch.from_numpy(np.array(dataset['event_label']))
        print(
            '数量统计',
            'text: %d, image: %d, label: %d, event_label: %d'
            % (len(self.text), len(self.image), len(self.label), len(self.event_label))
        )
        print('TEXT: %d, Image: %d, label: %d, Event: %d'
              % (len(self.text), len(self.image), len(self.label), len(self.event_label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx], self.mask[idx]), self.label[idx], self.event_label[idx]


class ReverseLayerF(Function):

    @staticmethod
    def forward(self, x):
        self.lambd = args.lambd
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x):
    return ReverseLayerF()(x)


# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(self, args):
        super(CNN_Fusion, self).__init__()
        self.args = args
        self.event_num = args.event_num
        vocab_size = args.vocab_size
        emb_dim = args.embed_dim
        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19
        #text
        # bert
        self.size_pool = 3
        self.drop_rate = 0.2
        self.final_hid = 32
        #bert后面5层
        self.sequence_out = 3840
        #隐藏层设置true，可以取出后面的信息
        bert_model = BertModel.from_pretrained(MODEL,output_hidden_states=True)
        self.bert_hidden_size = args.bert_hidden_dim
        for param in bert_model.parameters():
            param.requires_grad = False
        self.bertModel = bert_model
        self.dropout = nn.Dropout(args.dropout)
        '''卷积'''
        # 4卷积
        self.convs4_2 = nn.Sequential(
            nn.Conv1d(self.sequence_out, 768, 2),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.size_pool)
        )
        self.convs4_3 = nn.Sequential(
            nn.Conv1d(self.sequence_out, 768, 3),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.size_pool)
        )
        self.convs4_4 = nn.Sequential(
            nn.Conv1d(self.sequence_out, 768, 4),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.size_pool)
        )
        self.convs4_5 = nn.Sequential(
            nn.Conv1d(self.sequence_out, 768, 5),
            nn.BatchNorm1d(768),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.size_pool)
        )
        #两卷积
        self.l2_pool = 768
        self.convs2_1 = nn.Sequential(
            conv1d(self.l2_pool ,768 , 3),
            nn.BatchNorm1d(self.l2_pool),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.size_pool)
        )
        self.convs2_2 = nn.Sequential(
            conv1d(self.l2_pool,768,3),
            nn.BatchNorm1d(self.l2_pool),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.size_pool)
        )
        # text_append_改正，因为不知道准确的长度，所以进行查找后计算
        self.text_flatten = 9216
        self.text_append_layer = nn.Sequential(
            nn.Linear(self.text_flatten, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(512, self.final_hid),
            nn.BatchNorm1d(32),
            nn.LeakyReLU()
        )

        # IMAGE
        # hidden_size = args.hidden_dim
        resnet_1 = torchvision.models.resnet50(pretrained=True)  # 1000
        resnet_1.fc = nn.Linear(2048, 2048)  # 重新定义最后一层
        for param in resnet_1.parameters():
            param.requires_grad = False
        param.requires_grad = False

        resnet_3 = torchvision.models.resnet50(pretrained=True)
        for param in resnet_3.parameters():
            param.requires_grad = False
        self.resnet_1 = resnet_1  # 2048
        # 视觉处理的取到倒数的含有区域的一层
        resnet_3 = torch.nn.Sequential(*list(resnet_3.children())[:-2])  # 提取最后一层了
        self.resnet_3 = resnet_3  # 2048*7*7
        #image_append
        self.image_append_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(1024, self.final_hid),
            nn.BatchNorm1d(self.final_hid),
            nn.LeakyReLU()
        )
        # region_image,49块区域进行
        self.region = 49
        self.region_image = nn.Sequential(
            nn.Linear(2048, self.final_hid),
            nn.BatchNorm1d(self.region),
            nn.ReLU()
        )
        # attetion att_img
        self.img_dim = 32
        self.att_hid = 32
        self.head = 1
        self.img_key_layer = nn.Linear(self.img_dim, int(self.att_hid / self.head))
        self.ima_value_layer = nn.Linear(self.img_dim, int(self.att_hid / self.head))
        self.text_query = nn.Linear(self.final_hid, int(self.att_hid / self.head))
        # self.score_softmax = nn.Softmax(dim=1)
        # 注意力均值化
        # 注意力均值化
        self.att_average = nn.Sequential(
            nn.Linear(32, self.final_hid),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.re_Dro = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )

        # 层激活
        self.layer_norm = nn.LayerNorm(32)

        # attention :attention text
        self.img_query = nn.Linear(self.final_hid, int(self.att_hid / self.head))
        self.text_key_layer = nn.Linear(self.bert_hidden_size, int(self.att_hid / self.head))
        self.text_value_layer = nn.Linear(self.bert_hidden_size, int(self.att_hid / self.head))
        #   soft用上一层的 att_averrage  #均值后用上面的    #层激活
        # self_attention
        self.self_img_query = nn.Linear(32, int(self.att_hid / self.head))
        self.self_img_key = nn.Linear(32, int(self.att_hid / self.head))
        self.self_img_value = nn.Linear(32, int(self.att_hid / self.head))
        # soft同上
        # self_注意力均值化
        self.self_att_average = nn.Sequential(
            nn.Linear(32, self.final_hid),
            nn.BatchNorm1d(49),
            nn.ReLU()
        )
        # 均值后数句
        self.self_re_Dro = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(49),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )
        # 层标准化
        self.self_layer_norm = nn.LayerNorm([49, 32])
        # flatten_self
        self.self_faltten = nn.Sequential(
            nn.Linear(49 * 32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        # 文本自注意力机制-text_region
        # self_attention 32 21 768
        self.self_text_query = nn.Linear(768, int(self.att_hid / self.head))
        self.self_text_key = nn.Linear(768, int(self.att_hid / self.head))
        self.self_text_value = nn.Linear(768, int(self.att_hid / self.head))
        # soft同上
        # self_注意力均值化
        self.self_text_att_average = nn.Sequential(
            nn.Linear(32, self.final_hid),
            nn.BatchNorm1d(12),
            nn.ReLU()
        )
        # 均值后数句
        self.self_text_re_Dro = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(12),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )
        # 层标准化
        self.self_text_layer_norm = nn.LayerNorm([12, 32])
        # flatten_self
        self.self_text_faltten = nn.Sequential(
            nn.Linear(12 * 32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        # image
        self.multi = nn.Linear(768, 32)
        self.merge = nn.Linear(1024,32)
        self.mcb = CompactBilinearPooling(12, 49, 32)
        # 融合激活
        self.merge_feature = nn.Sequential(
            nn.Linear(128, self.final_hid),
            nn.Dropout(self.drop_rate),
        )
        # Class Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.hidden_size, 2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

    def init_hidden(self, batch_size):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
                to_var(torch.zeros(1, batch_size, self.lstm_size)))

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        # x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)

        return x

    def forward(self, text, image, mask):
        # print(text.shape)#torch.Size([32, 87])
        # print(image.shape)#torch.Size([32, 3, 224, 224])
        out = self.bertModel(text)  # berts输出
        '''bert隐藏层四层'''
        all_hidden_state = out[2]  # bert模型的原因，因为，out【2】是所有层的输出
        sequence_output_1 = all_hidden_state[-1]
        sequence_output_2 = all_hidden_state[-2]
        sequence_output_3 = all_hidden_state[-3]
        sequence_output_4 = all_hidden_state[-4]
        sequence_output_5 = all_hidden_state[-5]
        '''将bert的后面四层堆叠'''
        sequence_output = torch.cat(
            (sequence_output_1, sequence_output_2, sequence_output_3, sequence_output_4, sequence_output_5), 2)
        # print(sequence_output.shape)#torch.Size([32, 87, 3840])
        # 由于torch卷积的原因，需要将max_len和词向量维度进行转变使用permute函数
        sequence_output = sequence_output.permute(0, 2, 1)
        '''四卷积'''
        convs = []
        l_pool_2 = self.convs4_2(sequence_output)  # torch.Size([32, 768, 28])
        # print(l_pool_2.shape)
        convs.append(l_pool_2)
        l_pool_3 = self.convs4_3(sequence_output)  # torch.Size([32, 768, 28])
        # print(l_pool_3.shape)
        convs.append(l_pool_3)
        l_pool_4 = self.convs4_4(sequence_output)  # torch.Size([32, 768, 28])
        # print(l_pool_4.shape)
        convs.append(l_pool_4)
        l_pool_5 = self.convs4_5(sequence_output)  # torch.Size([32, 768, 27])
        # print(l_pool_5.shape)
        convs.append(l_pool_5)
        '''拼接4卷积-convs'''
        l2_pool = torch.cat(convs, dim=2)  # torch.Size([32, 768, 111])
        # print(l2_pool.shape)
        '''卷积完成，将词向量的维度和max——len进行调换位置-TM：【batch，区域数，向量维度】'''
        # print(l2_pool.shape)
        Tm = l2_pool.permute(0, 2, 1)
        l2_pool = Tm  # torch.Size([32, 111, 768]),后面通过这个矩阵跟图像进行attention操作
        '''展平,（batch 不展平）此时后面的两个维度交不交换没有必要性，因为后面要进行flatten操作'''
        '''双流1：'''
        # print(l2_pool.shape)
        multi_text = self.multi(l2_pool)  # 32,111,32
        multi_text = multi_text.permute(0, 2, 1)  # 32 ,32 ,111
        '''双流2'''        '''两个全连接--32维度-便于合并'''
        text = torch.flatten(l2_pool, start_dim=1, end_dim=-1)  # end_dim=-1
        merge = []
        text_append = self.text_append_layer(text)
        '''text_append[batch_size,32],两个用途，用于attetntion和直接拼接'''
        merge.append(text_append)
        '''IMAGE'''
        '''image_append——倒数一层resnet'''
        image_1 = self.resnet_1(image)  # 【2048】
        # print(image_1.shape)  # torch.Size([32, 2048])
        image_append = self.image_append_layer(image_1)
        # print(image_append.size())  # torch.Size([32, 32])
        merge.append(image_append)  # [32,32]
        '''区域image——倒数3层'''
        image_3 = self.resnet_3(image)  #
        # print(image_3.shape)  # torch.Size([32, 2048, 7, 7])
        # 调换位置，torch和tensorflow的架构
        image_3 = image_3.permute(0, 2, 3, 1)  # torch.Size([32, 7, 7, 2048])
        '''reshape【batch，region，维度】'''
        image_3 = torch.reshape(image_3, ((image_3.shape[0], -1, image_3.shape[-1])))
        # print(image_3.shape)#torch.Size([32, 49, 2048]),49个区域
        image = self.region_image(image_3)
        '''Im区域块的image用于attention'''
        Im = image  # print(Im.shape)#torch.Size([32, 49, 32])
        #多模线性池化块
        # print("-" * 50, "multi-model", "-" * 50)
        '''
        multi_image = Im.permute(0, 2,1)  # b, w, h, c) should work. Just make sure that “c” is last. That’s because the FFT works on the last dimension.
        # print(multi_image.shape)#torch.Size([32, 32, 49])
        x = multi_text  # torch.Size([32, 32, 12])
        # print(x.shape)
        y = multi_image  # torch.Size([32, 32, 49])
        # print(y.shape)
        # 线性池化
        Merge_feture = self.mcb(x, y)
        # print(Merge_feture.shape)#torch.Size([32, 32, 32])
        Merge_feture = torch.flatten(Merge_feture, start_dim=1, end_dim=2)
        Merge_feture = self.merge(Merge_feture)
        merge.append(Merge_feture)
        '''
        #'''注意力机制'''
        head = 1
        att_layers = 1
        att_hid = 32

        '''自注意力机制-Im-Im'''
        # print("-" * 50, "self_attention_img", "-" * 50)
        in_AttSelf_key = Im  # torch.Size([32, 49, 32])
        in_AttSelf_query = Im  # torch.Size([32, 49, 32])
        for layer in range(att_layers):
            self_att_img = []
            for _ in range(head):
                self_img_query = self.self_img_query(in_AttSelf_query)
                self_img_key = self.self_img_key(in_AttSelf_key)
                self_img_value = self.self_img_value(in_AttSelf_key)
                # torch.Size([32, 49, 32])  torch.Size([32, 49, 32])
                '''计算sccore Q*K-公式第一步'''
                score = torch.tensordot(self_img_query, self_img_key, dims=([2], [2]))  # torch.Size([32, 32, 21])
                # print(score.shape)  # torch.Size([32, 49,32, 49])
                '''改变合并方法'''
                score = torch.stack([score[i, :, i, :] for i in range(len(score))])
                # print(score.shape)  # torch.Size([32, 49,49])
                # print("score.shape")
                '''公式第二步,score/根号下d 维度-维度不匹配的广播机制'''
                # 如果是y一个数，x所有元素除以y
                score = torch.div(score, np.sqrt(att_hid / head))  # [32,49,49]
                score = F.leaky_relu(score)
                '''公式第三步-score * v'''
                # score：torch.Size([32, 49,49]) image_value = torch.Size([32, 49, 32])
                attention = torch.tensordot(score, self_img_value, dims=([2], [1]))
                # print(attention.shape)  # torch.Size([32, 49, 32,32])
                attention = torch.stack([attention[i, :, i, :] for i in range(len(attention))])
                # print(attention.shape)  #   torch.Size([32, 49, 32])
                '''得出询问后的自己后的 att'''
                self_att_img = attention
                '''将注意力均值化-无法平均-也就是多个列表，对应的元素相加求平均在进行成为当前元素'''
                self_att_img21 = self.self_att_average(self_att_img)
                self_att_img22 = self.self_att_average(self_att_img)
                self_att_img23 = self.self_att_average(self_att_img)
                self_att_img24 = self.self_att_average(self_att_img)
                # 均值
                self_att_img2 = self_att_img21.add(self_att_img22).add(self_att_img23).add(self_att_img24)
                self_att_img2 = torch.div(self_att_img2, 4)
                '''均值后数据'''
                self_att_img2 = self.self_re_Dro(self_att_img2)
                '''将注意力后的数据相加:image_+self_att_img'''
                self_att_img = torch.add(in_AttSelf_query, self_att_img2)
                # print(self_att_img.shape) # [32,49,32]
                '''层标准化'''
                self_att_img = self.self_layer_norm(self_att_img)
                '''不能将他直接merge进去因为，维度不同，faltten之后进行输入'''
                # print(self_att_img.shape)# [32,49,32]
                in_AttSelf_query = self_att_img
                inp_AttSelf_key = self_att_img
        '''将后面的两维度flatten，变成二维'''
        self_att_img = torch.flatten(self_att_img, start_dim=1, end_dim=2)  # end_dim=-1
        self_att_img = self.self_faltten(self_att_img)
        '''此时向量的维度是正常的'''
        # print(self_att_img.shape)#[32,32]
        merge.append(self_att_img)
        # print("-" * 50, "结束融合", "-" * 50)
        # print(len(merge))#5
        '''共有5部分数据- 32*32 ,32-160'''
        '''text自注意力机制'''
        # print("-" * 50, "self_attention_img", "-" * 50)
        in_Self_TEXT_key = Tm  # torch.Size([32, 21, 768])
        in_Self_TEXT_query = Tm  # torch.Size([32, 21, 768])
        for layer in range(att_layers):
            self_text_att_img = []
            for _ in range(head):
                self_text_query = self.self_text_query(in_Self_TEXT_query)
                self_text_key = self.self_text_key(in_Self_TEXT_key)
                self_text_value = self.self_text_value(in_Self_TEXT_key)
                # torch.Size([32, 49, 32])  torch.Size([32, 49, 32])
                '''计算sccore Q*K-公式第一步'''
                score = torch.tensordot(self_text_query, self_text_key, dims=([2], [2]))  # torch.Size([32, 32, 21])
                # print(score.shape)  # torch.Size([32, 21,32, 21])
                '''改变合并方法'''
                score = torch.stack([score[i, :, i, :] for i in range(len(score))])
                # print(score.shape)  # torch.Size([32, 21,21])
                # print("score.shape")
                '''公式第二步,score/根号下d 维度-维度不匹配的广播机制'''
                # 如果是y一个数，x所有元素除以y
                score = torch.div(score, np.sqrt(att_hid / head))  # [32,49,49]
                score = F.leaky_relu(score)
                '''公式第三步-score * v'''
                # score：torch.Size([32, 21,21]) image_value = torch.Size([32, 21, 32])
                attention = torch.tensordot(score, self_text_value, dims=([2], [1]))
                # print(attention.shape)  # torch.Size([32, 49, 32,32])
                attention = torch.stack([attention[i, :, i, :] for i in range(len(attention))])
                # print(attention.shape)  #   torch.Size([32, 49, 32])
                '''得出询问后的自己后的 att'''
                self_text_att_img = attention
                '''将注意力均值化-无法平均-也就是多个列表，对应的元素相加求平均在进行成为当前元素'''
                self_text_att_img21 = self.self_text_att_average(self_text_att_img)
                self_text_att_img22 = self.self_text_att_average(self_text_att_img)
                self_text_att_img23 = self.self_text_att_average(self_text_att_img)
                self_text_att_img24 = self.self_text_att_average(self_text_att_img)
                # 均值
                self_text_att_img2 = self_text_att_img21.add(self_text_att_img22).add(self_text_att_img23).add(
                    self_text_att_img24)
                self_text_att_img2 = torch.div(self_text_att_img2, 4)
                '''均值后数据'''
                self_text_att_img2 = self.self_text_re_Dro(self_text_att_img2)
                '''将注意力后的数据相加:image_+self_att_img'''
                self_text_att_img = torch.add(self_text_query, self_text_att_img2)
                # print(self_att_img.shape) # [32,49,32]
                '''层标准化'''
                self_text_att_img = self.self_text_layer_norm(self_text_att_img)
                '''不能将他直接merge进去因为，维度不同，faltten之后进行输入'''
                # print(self_att_img.shape)# [32,12,32]
                in_Self_TEXT_query = self_text_att_img
                in_Self_TEXT_key = self_text_att_img
        '''将后面的两维度flatten，变成二维'''
        self_text_att_img = torch.flatten(self_text_att_img, start_dim=1, end_dim=2)  # end_dim=-1
        self_text_att_img = self.self_text_faltten(self_text_att_img)
        '''此时向量的维度是正常的'''
        # print(self_att_img.shape)#[32,32]
        merge.append(self_text_att_img)
        '''自注意力机制'''

        # print("-" * 50, "结束融合", "-" * 50)
        # print(len(merge))  # 6
        feature_merge = torch.cat(merge, dim=1)  # 32, 192
        feature_merge = self.merge_feature(feature_merge)
        # print(feature_merge.shape)#torch.Size([32, 32])
        class_output = self.class_classifier(feature_merge)
        reverse_feature = grad_reverse(feature_merge)#
        domain_output = self.domain_classifier(reverse_feature)
        return class_output, domain_output


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


def select(train, selec_indices):
    temp = []
    for i in range(len(train)):
        print("length is " + str(len(train[i])))
        print(i)
        # print(train[i])
        ele = list(train[i])
        temp.append([ele[i] for i in selec_indices])
    return temp


def make_weights_for_balanced_classes(event, nclasses=15):
    count = [0] * nclasses
    for item in event:
        count[item] += 1
    weight_per_class = [0.] * nclasses
    N = float(sum(count))
    for i in range(nclasses):
        weight_per_class[i] = N / float(count[i])
    weight = [0] * len(event)
    for idx, val in enumerate(event):
        weight[idx] = weight_per_class[val]
    return weight


def split_train_validation(train, percent):
    whole_len = len(train[0])

    train_indices = (sample(range(whole_len), int(whole_len * percent)))
    train_data = select(train, train_indices)
    print("train data size is " + str(len(train[3])))
    # print()

    validation = select(train, np.delete(range(len(train[0])), train_indices))
    print("validation size is " + str(len(validation[3])))
    print("train and validation data set has been splited")

    return train_data, validation


def main(args):
    print("-" * 50, "开始载入数据", "-" * 50)
    print('loading data')
    train, validation, test = load_data(args)
    test_id = test['post_id']
    '''通过Rumor——data进行找到数据存在的所有项'''
    train_dataset = Rumor_Data(train)

    validate_dataset = Rumor_Data(validation)

    test_dataset = Rumor_Data(test)

    # Data Loader (Input Pipeline)
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)
    print("-" * 50, "开始生成模型", "-" * 50)

    print('building model')
    model = CNN_Fusion(args)
    print("-" * 50, "模型结构", "-" * 50)
    print(model)
    if torch.cuda.is_available():
        print("CUDA OK")
        model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                                 lr=args.learning_rate, weight_decay=0.1)
    print("bach总数：", "训练集" + str(len(train_loader)), "验证集" + str(len(validate_loader)), "测试集" + str(len(test_loader)))
    print("-" * 50, "开始训练", "-" * 50)

    print("loader size " + str(len(train_loader)))
    best_validate_acc = 0.000
    best_test_acc = 0.000
    best_loss = 100
    best_validate_dir = ''
    best_list = [0, 0]

    print('training model')
    adversarial = True
    # Train the Model
    # 初始化 early_stopping 对象
    #初始化 early_stopping 对象
    patience = 20  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=True)  # 关于 EarlyStopping 的代码可先看博客后面的内容

    pictuTrainLoss = []
    pictuTrainACC = []
    pictuvaliLoss = []
    pictuvaliACC = []

    for epoch in range(args.num_epochs):

        p = float(epoch) / 100
        # lambd = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.001 / (1. + 10 * p) ** 0.75

        optimizer.lr = lr
        # rgs.lambd = lambd
        cost_vector = []
        class_cost_vector = []
        domain_cost_vector = []
        acc_vector = []
        valid_acc_vector = []
        test_acc_vector = []
        vali_cost_vector = []
        test_cost_vector = []

        for i, (train_data, train_labels, event_labels) in enumerate(train_loader):
            train_text, train_image, train_mask, train_labels, event_labels = \
                to_var(train_data[0]), to_var(train_data[1]), to_var(train_data[2]), \
                to_var(train_labels), to_var(event_labels)
            # start_time = time.time()
            # Forward + Backward + Optimize
            optimizer.zero_grad()

            class_outputs, domain_outputs = model(train_text, train_image, train_mask)

            # Fake or Real loss
            class_loss = criterion(class_outputs, train_labels)
            # Event Loss
            # Event Loss
            domain_loss = criterion(domain_outputs, event_labels)

            loss = class_loss- domain_loss
            loss.backward()
            optimizer.step()
            _, argmax = torch.max(class_outputs, 1)

            cross_entropy = True

            if True:
                accuracy = (train_labels == argmax.squeeze()).float().mean()
            else:
                _, labels = torch.max(train_labels, 1)
                accuracy = (labels.squeeze() == argmax.squeeze()).float().mean()

            class_cost_vector.append(class_loss.item())
            domain_cost_vector.append(domain_loss.item())

            cost_vector.append(loss.item())
            acc_vector.append(accuracy.item())

        model.eval()
        validate_acc_vector_temp = []
        for i, (validate_data, validate_labels, event_labels) in enumerate(validate_loader):
            validate_text, validate_image, validate_mask, validate_labels, event_labels = \
                to_var(validate_data[0]), to_var(validate_data[1]), to_var(validate_data[2]), \
                to_var(validate_labels), to_var(event_labels)
            validate_outputs,domain_outputs = model(validate_text, validate_image, validate_mask)
            _, validate_argmax = torch.max(validate_outputs, 1)
            vali_loss = criterion(validate_outputs, validate_labels)
            #domain_loss = criterion(domain_outputs, event_labels)
            #验证集的不参与反向梯度传播，搜易这个地方不同写出来
            # _, labels = torch.max(validate_labels, 1)
            validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            vali_cost_vector.append(vali_loss.item())
            # validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            validate_acc_vector_temp.append(validate_accuracy.item())
        validate_acc = np.mean(validate_acc_vector_temp)
        valid_acc_vector.append(validate_acc)
        '''gh'''
        validate_loss = np.mean(vali_cost_vector)
        valid_acc_vector.append(validate_acc)
        '''gh'''
        early_stopping(validate_loss, model)
        '''gh-tensorboard'''
        writer.add_scalar(tag='Loss/train_loss', scalar_value=np.mean(cost_vector), global_step=epoch + 1)
        writer.add_scalar(tag='Loss/val_loss', scalar_value=validate_loss.item(), global_step=epoch + 1)
        writer.add_scalar(tag='Accuracy/train_acc', scalar_value=np.mean(acc_vector), global_step=epoch + 1)
        writer.add_scalar(tag='Accuracy/val_acc', scalar_value=validate_acc.item(), global_step=epoch + 1)

        for name, param in model.named_parameters():
            writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch + 1)

        model.train()


        print('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, domain loss: %.4f, Train_Acc: %.4f, Validate_Loss: %.4f, Validate_Acc: %.4f.'
              % (
                  epoch + 1, args.num_epochs, np.mean(cost_vector), np.mean(class_cost_vector),
                  np.mean(domain_cost_vector),
                  np.mean(acc_vector), np.mean(vali_cost_vector),validate_acc))
        pictuTrainLoss.append(np.mean(cost_vector))
        pictuTrainACC.append(np.mean(acc_vector))
        pictuvaliACC.append(validate_acc)
        pictuvaliLoss.append(validate_loss)
        if validate_acc >= best_validate_acc:
            best_validate_acc = validate_acc
            if not os.path.exists(args.output_file):
                os.mkdir(args.output_file)

            best_validate_dir = args.output_file + 'best' + '.pkl'
            torch.save(model.state_dict(), best_validate_dir)
        # 若满足 early stopping 要求
        if early_stopping.early_stop:
           print("Early stopping")
            # 结束模型训练
           break
    print('TrainLoss ：',pictuTrainLoss)
    print('TrainACC ：', pictuTrainACC)
    print('ValiACC ：', pictuvaliACC)
    print('ValiLoss ：', pictuvaliLoss)

    print('testing model')
    model = CNN_Fusion(args)
    model.load_state_dict(torch.load(best_validate_dir))
    #    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    test_score = []
    test_pred = []
    test_true = []
    #all_logits = []  # 所有点
    #y_labels = []  # 标签名称
    with torch.no_grad():
        for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
            test_text, test_image, test_mask, test_labels = to_var(
                test_data[0]), to_var(test_data[1]), to_var(test_data[2]), to_var(test_labels)
            test_outputs = model(test_text, test_image, test_mask)
            _, test_argmax = torch.max(test_outputs, 1)
            # torch.max(a, 1): 返回每一行的最大值，且返回索引:_是索引（返回最大元素在各行的列索引）。
            #all_logits.append(test_outputs)
            #y_labels.append(test_argmax)
            if i == 0:
                test_score = to_np(test_outputs.squeeze())
                test_pred = to_np(test_argmax.squeeze())
                test_true = to_np(test_labels.squeeze())
            else:
                test_score = np.concatenate((test_score, to_np(test_outputs.squeeze())), axis=0)
                test_pred = np.concatenate((test_pred, to_np(test_argmax.squeeze())), axis=0)
                test_true = np.concatenate((test_true, to_np(test_labels.squeeze())), axis=0)

    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    test_recall = metrics.recall_score(test_true, test_pred, average='macro')
    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')
    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)
    print("Classification Acc: %.4f, AUC-ROC: %.4f"
          % (test_accuracy, test_aucroc))
    print("Classification report:\n%s\n"
          % (metrics.classification_report(test_true, test_pred)))
    print("Classification confusion matrix:\n%s\n"
          % (test_confusion_matrix))
    #all_logits = torch.cat(all_logits, dim=0)  # 31,147
    #y_labels = torch.cat(y_labels, dim=0)  # 31
    #writer.add_embedding(mat=all_logits,  # 所有点
    #                     metadata=y_labels,  # 标签名称  # 标签图片
    #                    )

def parse_arguments(parser):
    parser.add_argument('training_file', type=str, metavar='<training_file>', help='')
    # parser.add_argument('validation_file', type=str, metavar='<validation_file>', help='')
    parser.add_argument('testing_file', type=str, metavar='<testing_file>', help='')
    parser.add_argument('output_file', type=str, metavar='<output_file>', help='')

    parse.add_argument('--static', type=bool, default=True, help='')
    parser.add_argument('--sequence_length', type=int, default=28, help='')
    parser.add_argument('--class_num', type=int, default=2, help='')
    parser.add_argument('--hidden_dim', type=int, default=32, help='')
    parser.add_argument('--embed_dim', type=int, default=32, help='')
    parser.add_argument('--vocab_size', type=int, default=300, help='')
    parser.add_argument('--dropout', type=int, default=0.5, help='')
    parser.add_argument('--filter_num', type=int, default=5, help='')
    parser.add_argument('--lambd', type=int, default=1, help='')
    parser.add_argument('--text_only', type=bool, default=False, help='')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')

    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')

    #    args = parser.parse_args()
    return parser


def get_top_post(output, label, test_id, top_n=500):
    filter_output = []
    filter_id = []
    # print(test_id)
    # print(output)
    for i, l in enumerate(label):
        # print(np.argmax(output[i]))
        if np.argmax(output[i]) == l and int(l) == 1:
            filter_output.append(output[i][1])
            filter_id.append(test_id[i])

    filter_output = np.array(filter_output)

    top_n_indice = filter_output.argsort()[-top_n:][::-1]

    top_n_id = np.array(filter_id)[top_n_indice]
    top_n_id_dict = {}
    for i in top_n_id:
        top_n_id_dict[i] = True

    pickle.dump(top_n_id_dict, open("../Data/weibo/top_n_id.pickle", "wb"))

    return top_n_id


def word2vec(post, word_id_map, W):
    word_embedding = []
    mask = []
    # length = []

    for sentence in post:
        sen_embedding = []
        seq_len = len(sentence) - 1
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word_id_map[word])

        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)

        word_embedding.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
        # length.append(seq_len)
    return word_embedding, mask


def re_tokenize_sentence(flag):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_texts = []
    original_texts = flag['original_post']
    for sentence in original_texts:
        tokenized_text = tokenizer.encode(sentence)
        tokenized_texts.append(tokenized_text)
    flag['post_text'] = tokenized_texts


def get_all_text(train, validate, test):
    all_text = list(train['post_text']) + list(validate['post_text']) + list(test['post_text'])
    return all_text


def align_data(flag, args):
    text = []
    mask = []
    for sentence in flag['post_text']:
        sen_embedding = []
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):
            sen_embedding.append(word)

        while len(sen_embedding) < args.sequence_len:
            sen_embedding.append(0)

        text.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
    flag['post_text'] = text
    flag['mask'] = mask


def load_data(args):
    train, validate, test, event_num = process_data.get_data(args.text_only)
    args.event_num = event_num
    print("-" * 50, "预训练模型处理中文训练集得到词向量", "-" * 50)
    re_tokenize_sentence(train)
    print("-" * 50, "预训练模型处理中文验证集得到词向量", "-" * 50)
    re_tokenize_sentence(validate)
    print("-" * 50, "联合所有文本，找最长 max_len", "-" * 50)
    re_tokenize_sentence(test)
    all_text = get_all_text(train, validate, test)

    max_len = len(max(all_text, key=len))
    print("数据集文本max_len最长为：", max_len)

    args.sequence_len = max_len
    print("-" * 50, "对齐数据，填充mask：将句子变为统一长度", "-" * 50)

    align_data(train, args)
    align_data(validate, args)
    align_data(test, args)
    return train, validate, test


def transform(event):
    matrix = np.zeros([len(event), max(event) + 1])
    # print("Translate  shape is " + str(matrix))
    for i, l in enumerate(event):
        matrix[i, l] = 1.00
    return matrix


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    train = ''
    test = ''
    output = '/tmp/pycharm_project_169/src/weight/'
    args = parser.parse_args([train, test, output])

    main(args)


