import argparse
import time, os
from src.early_stopping import *
# 提取模型
import matplotlib.pyplot as plt
from src import process_data_weibo2 as process_data
import copy
import pickle as pickle
from random import sample
import torchvision
import torch
print(torch.version.cuda)
if (torch.cuda.is_available()):
    print("CUDA 存在")
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from src.trans_padding import Conv1d as conv1d
# 记录输出
import sys
#多模线性池化
from src.pytorch_compact_bilinear_pooling import  CountSketch, CompactBilinearPooling
class Logger(object):
    def __init__(self, fileN="Default.log"):
        self.terminal = sys.stdout
        self.log = open(fileN, "a")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
#sys.stdout = Logger("/root/autodl-tmp/shiyan/sci1.txt")  # 保存到D盘
# 可视化
#from tensorboardX import SummaryWriter
#writer = SummaryWriter('runs/multi_fusion')

from sklearn import metrics
from transformers import AutoConfig, TFAutoModel, AutoTokenizer, BertModel, BertTokenizer

#config = AutoConfig.from_pretrained('/tmp/pycharm_project_169/PreTrainModel/config.json')
MODEL = '/root/autodl-tmp/pre_trainmodel/ber-base-chinese'
MODEL_NAME = 'bert-base-chinese'
N_LABELS = 1
import warnings
warnings.filterwarnings("ignore")
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
        print('TEXT: %d, Image: %d, labe: %d, Event: %d'
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
    return ReverseLayerF.apply(x)


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
        convs = []
        self.filter_sizes = [2, 3, 4, 5]
        self.size_pool = 3
        self.drop_rate = 0.2
        self.final_hid = 32
        self.sequence_out = 3840
        # bert
        bert_model = BertModel.from_pretrained(MODEL, output_hidden_states=True)
        self.bert_hidden_size = args.bert_hidden_dim
        for param in bert_model.parameters():
            param.requires_grad = False
        self.bertModel = bert_model
        '''卷积'''
        self.dropout = nn.Dropout(args.dropout)
        # 4卷积
        self.convs4_2 = nn.Sequential(
            nn.Conv1d(self.sequence_out, 768, 2),#32 768 191
            nn.BatchNorm1d(768),##32 768 191
            nn.LeakyReLU(),#32 768 191
            nn.MaxPool1d(self.size_pool)##32 768 63
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
        # 2卷积
        self.l2_pool = 768
        self.convs2_1 = nn.Sequential(
            conv1d(self.l2_pool, 768, 3),#torch.Size([32, 768, 251])
            nn.BatchNorm1d(self.l2_pool),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.size_pool)#torch.Size([32, 768, 83])
        )
        self.convs2_2 = nn.Sequential(
            conv1d(self.l2_pool, 768, 3),
            nn.BatchNorm1d(self.l2_pool),
            nn.LeakyReLU(),
            nn.MaxPool1d(self.size_pool)
        )
        # text_append,长度不同
        self.text_flatten = 20736
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
        resnet_1 = torchvision.models.resnet50(pretrained=True)  # 1000
        resnet_1.fc = nn.Linear(2048, 2048)  # 重新定义最后一层
        for param in resnet_1.parameters():
            param.requires_grad = False
        # visual model
        resnet_3 = torchvision.models.resnet50(pretrained=True)
        for param in resnet_3.parameters():
            param.requires_grad = False
        # visual model-分类器取到最后倒数一层
        # resnet_1. =  torch.nn.Sequential(*list(resnet_1.children())[:-1])#提取最后一层了
        self.resnet_1 = resnet_1  # 2048
        # 视觉处理的取到倒数的含有区域的一层
        resnet_3 = torch.nn.Sequential(*list(resnet_3.children())[:-2])  # 提取最后一层了
        self.resnet_3 = resnet_3  # 2048*7*7
          # image_append
        self.image_append_layer = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(self.drop_rate),
            nn.Linear(1024, self.final_hid),
            nn.BatchNorm1d(self.final_hid),
            nn.LeakyReLU()
        )
        # region_image
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
        #文本自注意力机制-text_region
        # self_attention 32 21 768
        self.self_text_query = nn.Linear(768, int(self.att_hid / self.head))
        self.self_text_key = nn.Linear(768, int(self.att_hid / self.head))
        self.self_text_value = nn.Linear(768, int(self.att_hid / self.head))
        # soft同上
        # self_注意力均值化
        self.self_text_att_average = nn.Sequential(
            nn.Linear(32, self.final_hid),
            nn.BatchNorm1d(27),
            nn.ReLU()
        )
        # 均值后数句
        self.self_text_re_Dro = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(27),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )
        # 层标准化
        self.self_text_layer_norm = nn.LayerNorm([27, 32])
        # flatten_self
        self.self_text_faltten = nn.Sequential(
            nn.Linear(27 * 32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        #image
        self.multi = nn.Linear(768, 32)
        self.merge = nn.Linear(1024,32)
        self.mcb = CompactBilinearPooling(27, 49, 32)
        # 融合激活
        self.merge_feature = nn.Sequential(
            nn.Linear(160, self.final_hid),
            nn.Dropout(self.drop_rate),
        )
        # Class Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(self.hidden_size, 2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        # Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))

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

    def forward(self, text, image, mask):#torch.Size([32, 192]) torch.Size([32, 3, 224, 224]) torch.Size([32, 192])
        """GH"""
        # print(text.size())
        # print(image.size())
        out = self.bertModel(text)  # berts输出
        last_hidden_state = out[0]
        '''bert隐藏层四层'''
        all_hidden_state = out[2]  # bert模型的原因，因为，out【2】是所有层的输出
        '''将bert的后面四层堆叠'''
        sequence_output = torch.cat((all_hidden_state[-1], all_hidden_state[-2], all_hidden_state[-3], all_hidden_state[-4],all_hidden_state[-5]), 2)
        #print(sequence_output.shape)    #torch.Size([32, 192, 3840])
        # 由于torch卷积的原因，需要将max_len和词向量维度进行转变使用permute函数
        sequence_output = sequence_output.permute(0, 2, 1)  #torch.Size([32, 3840, 192])
        #print(sequence_output.shape)
        '''四卷积'''
        convs = []
        l_pool_2 = self.convs4_2(sequence_output)   #torch.Size([32, 768, 63])
        #print(l_pool_2.shape)
        convs.append(l_pool_2)
        l_pool_3 = self.convs4_3(sequence_output)   #torch.Size([32, 768, 63])
        #print(l_pool_3.shape)
        convs.append(l_pool_3)
        l_pool_4 = self.convs4_4(sequence_output)   #torch.Size([32, 768, 63])
        #print(l_pool_4.shape)
        convs.append(l_pool_4)
        l_pool_5 = self.convs4_5(sequence_output)   #torch.Size([32, 768, 62])
        #print(l_pool_5.shape)
        convs.append(l_pool_5)
        '''拼接4卷积-convs'''
        l2_pool = torch.cat(convs, dim=2)
        #print(l2_pool.shape)    #torch.Size([32, 768, 251])
        '''两卷积'''
        # 卷积 - 批标准化 - 激活- 池化
        l2_pool = self.convs2_1(l2_pool)    #torch.Size([32, 768, 83])
        #print(l2_pool.shape)
        l2_pool = self.convs2_2(l2_pool)    #torch.Size([32, 768, 27])

        #print(l2_pool.shape)
        '''卷积完成，将词向量的维度和max——len进行调换位置-TM：【batch，区域数，向量维度】'''
        Tm = l2_pool.permute(0, 2, 1)   #torch.Size([32, 27, 768])
        #print(Tm.shape)batch,chanel,dim
        '''展平,（batch 不展平）此时后面的两个维度交不交换没有必要性，因为后面要进行flatten操作'''
        text = torch.flatten(Tm, start_dim=1, end_dim=2)    #torch.Size([32, 20736])
        #print(text.shape)
        '''text双流1:tm'''
        #multi_text = self.multi(text)   #torch.Size([32, 1568])
        multi_text = self.multi(Tm)#32 27 768--32 27 32
        #print(multi_text.shape)
        multi_text = multi_text.permute(0,2,1)#batch,dim,chanel
        #print(multi_text.shape)
        merge = []
        '''text双流2:text_append[batch_size,32],直接拼接'''
        text_append = self.text_append_layer(text)
        #print(text_append.shape)    #torch.Size([32, 32])
        merge.append(text_append)

        '''image_双流1：image_append——倒数一层vgg'''
        image_1 = self.resnet_1(image)  #   torch.Size([32, 2048])
        #print(image_1.shape)
        image_append = self.image_append_layer(image_1) #torch.Size([32, 32])
        #print(image_append.shape)
        merge.append(image_append)
        '''image双柳2：区域image——倒数3层'''
        image_3 = self.resnet_3(image)    # torch.Size([32, 2048, 7, 7])
        #print(image_3.shape)
        image_3 = image_3.permute(0, 2, 3, 1)  # torch.Size([32, 7, 7, 2048])
        image_3 = torch.reshape(image_3, ((image_3.shape[0], -1, image_3.shape[-1])))
        #print(image_3.shape)#torch.Size([32, 49, 2048])
        image = self.region_image(image_3)#torch.Size([32, 49, 32])
        #print(image.shape)
        Im = image
        #multi_image = torch.flatten(Im, start_dim=1, end_dim=2)  #torch.Size([32, 1568])
        multi_image = image.permute(0,2,1)#32 32,49
        #print(multi_image.shape)
        '''多模线性池化块'''
        # print("-" * 50, "multi-model", "-" * 50)
        x = multi_text
        #print(x.shape)  # torch.Size([32, 32, 27])
        y = multi_image
        #print(y.shape)  # torch.Size([32, 32, 49])
        Merge_feture = self.mcb(x, y)
        #print(Merge_feture.shape)  # torch.Size([32, 32, 32])
        Merge_feture= torch.flatten(Merge_feture, start_dim=1, end_dim=2)
        Merge_feture = self.merge(Merge_feture)#
        #print(Merge_feture.shape)
        merge.append(Merge_feture)

        head = 1
        att_layers = 1
        att_hid = 32
        '''自注意力机制-Im-Im'''
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
        '''text自注意力机制'''
        in_Self_TEXT_key = Tm  #torch.Size([32, 27, 768])
        #print(in_Self_TEXT_key.shape)
        in_Self_TEXT_query = Tm  #
        for layer in range(att_layers):
            self_text_att_img = []
            for _ in range(head):
                self_text_query = self.self_text_query(in_Self_TEXT_query)##torch.Size([32, 27, 32])
                self_text_key = self.self_text_key(in_Self_TEXT_key)#torch.Size([32, 27, 32])
                self_text_value = self.self_text_value(in_Self_TEXT_key)#torch.Size([32, 27, 32])
                '''计算sccore Q*K-公式第一步'''
                score = torch.tensordot(self_text_query, self_text_key, dims=([2], [2]))  # torch.Size([32, 32, 21])
                #print(score.shape)  #torch.Size([32, 27, 32, 27])
                '''改变合并方法'''
                score = torch.stack([score[i, :, i, :] for i in range(len(score))])
                #print(score.shape)  #torch.Size([32, 27, 27])
                '''公式第二步,score/根号下d 维度-维度不匹配的广播机制'''
                # 如果是y一个数，x所有元素除以y
                score = torch.div(score, np.sqrt(att_hid / head))
                score = F.leaky_relu(score)
                '''公式第三步-score * v'''
                attention = torch.tensordot(score, self_text_value, dims=([2], [1]))
                attention = torch.stack([attention[i, :, i, :] for i in range(len(attention))])
                '''得出询问后的自己后的 att'''
                self_text_att_img = attention
                '''将注意力均值化-无法平均-也就是多个列表，对应的元素相加求平均在进行成为当前元素'''
                self_text_att21 = self.self_text_att_average(self_text_att_img)
                self_text_att22 = self.self_text_att_average(self_text_att_img)
                self_text_att23 = self.self_text_att_average(self_text_att_img)
                self_text_att24 = self.self_text_att_average(self_text_att_img)
                # 均值
                self_text_att2 = self_text_att21.add(self_text_att22).add(self_text_att23).add(self_text_att24)
                self_text_att2 = torch.div(self_text_att2, 4)
                '''均值后数据'''
                self_text_att2 = self.self_text_re_Dro(self_text_att2)
                '''将注意力后的数据相加:image_+self_att_img'''
                self_text_att = torch.add(self_text_query, self_text_att2) # [32,27,32]
                '''层标准化'''
                self_text_att = self.self_text_layer_norm(self_text_att)
                '''不能将他直接merge进去因为，维度不同，flatten之后进行输入'''
                in_Self_TEXT_query = self_text_att
                in_Self_TEXT_key = self_text_att
        '''将后面的两维度flatten，变成二维'''
        self_text_att = torch.flatten(self_text_att, start_dim=1, end_dim=2)  # end_dim=-1
        self_text_att = self.self_text_faltten(self_text_att)
        '''此时向量的维度是正常的'''
        #print(self_text_att.shape)  #torch.Size([32, 32])
        merge.append(self_text_att)
        #print("-" * 50, "结束融合", "-" * 50)
        '''共有5部分数据- 32*32 ,32-160'''
        feature_merge = torch.cat(merge, dim=1)  # 32, 160
        feature_merge = self.merge_feature(feature_merge)
        #print(feature_merge.shape)
        class_output = self.class_classifier(feature_merge)
        #print(class_output.shape)
        reverse_feature = grad_reverse(feature_merge)
        domain_output = self.domain_classifier(reverse_feature)
        #print(domain_output.shape)
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

    # Data Loader (Input Pipeline)，构造数据，定义data——set和batch
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True)

    validate_loader = DataLoader(dataset=validate_dataset,
                                 batch_size=args.batch_size,
                                 shuffle=False)
    # test也是按照batch送入的
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=False)
    print("-" * 50, "开始生成模型", "-" * 50)
    print('building model')
    model = CNN_Fusion(args)

    print("-" * 50, "打印模型", "-" * 50)
    '''summary-生成模型-浏览器'''
    print("-" * 50, "模型结构", "-" * 50)
    print(model)
    if torch.cuda.is_available():
        print("CUDA ok")
        model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                                 lr=args.learning_rate)
    print("bach总数：", "训练集" + str(len(train_loader)), "验证集" + str(len(validate_loader)), "测试集" + str(len(test_loader)))
    # 用数据总数、batch_size得到数据的共有几个数据
    iter_per_epoch = len(train_loader)
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
    ppp = 0
    # 初始化 early_stopping 对象
    patience = 20  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=True)  # 关于 EarlyStopping 的代码可先看博客后面的内容

    '''初始化设置寻找最优解'''
    pictuTrainLoss = []
    pictuTrainACC = []
    pictuvaliLoss = []
    pictuvaliACC = []

    for epoch in range(args.num_epochs):

        p = float(epoch) / 100
        # lambd = 2. / (1. + np.exp(-10. * p)) - 1
        #学习率衰减
        lr = 0.001 / (1. + 10 * p) ** 0.75
        optimizer.lr = lr
        # rgs.lambd = lambd
        start_time = time.time()
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
            # Forward + Backward + Optimize
            optimizer.zero_grad()

            class_outputs, domain_outputs = model(train_text, train_image, train_mask)
            # Fake or Real loss
            class_loss = criterion(class_outputs, train_labels.long())
            # Event Loss
            domain_loss = criterion(domain_outputs, event_labels.long())
            # loss = class_loss + domain_loss
            loss = class_loss - domain_loss
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
            # pictuvaliLoss.append(validate_loss)
        '''model.eval()是模型的某些特定层/部分的一种开关，这些层/部分在训练和推断（评估）期间的行为不同'''
        model.eval()
        validate_acc_vector_temp = []
        for i, (validate_data, validate_labels, event_labels) in enumerate(validate_loader):
            validate_text, validate_image, validate_mask, validate_labels, event_labels = \
                to_var(validate_data[0]), to_var(validate_data[1]), to_var(validate_data[2]), \
                to_var(validate_labels), to_var(event_labels)
            validate_outputs, domain_outputs = model(validate_text, validate_image, validate_mask)
            _, validate_argmax = torch.max(validate_outputs, 1)
            vali_loss = criterion(validate_outputs, validate_labels)
            # domain_loss = criterion(domain_outputs, event_labels)
            # _, labels = torch.max(validate_labels, 1)
            validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            vali_cost_vector.append(vali_loss.item())
            # validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            validate_acc_vector_temp.append(validate_accuracy.item())
        validate_acc = np.mean(validate_acc_vector_temp)
        valid_acc_vector.append(validate_acc)
        '''gh'''
        validate_loss = np.mean(vali_cost_vector)
        early_stopping(validate_loss, model)

        valid_acc_vector.append(validate_acc)

        '''gh-tensorboard'''
        #writer.add_scalar(tag='Loss/train_loss', scalar_value=np.mean(cost_vector), global_step=epoch + 1)
        #writer.add_scalar(tag='Loss/val_loss', scalar_value=validate_loss.item(), global_step=epoch + 1)
        #writer.add_scalar(tag='Accuracy/train_acc', scalar_value=np.mean(acc_vector), global_step=epoch + 1)
        #writer.add_scalar(tag='Accuracy/val_acc', scalar_value=validate_acc.item(), global_step=epoch + 1)

        #for name, param in model.named_parameters():
         #   writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch + 1)

        model.train()
        print('Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, domain loss: %.4f, Train_Acc: %.4f,  Validate_Loss: %.4f, Validate_Acc: %.4f.'
              % (
                  epoch + 1, args.num_epochs, np.mean(cost_vector), np.mean(class_cost_vector),
                  np.mean(domain_cost_vector),
                  np.mean(acc_vector),validate_loss, validate_acc))
        pictuTrainLoss.append(np.mean(cost_vector))
        pictuTrainACC.append(np.mean(acc_vector))
        pictuvaliACC.append(validate_acc)
        pictuvaliLoss.append(validate_loss)
        if validate_acc > best_validate_acc:
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
    print("-" * 50, "训练结果", "-" * 50)
    # Test the Model
    print("-"*50,"训练结果","-"*50)
    print('TrainLoss ：',pictuTrainLoss)
    print('TrainACC ：', pictuTrainACC)
    print('ValiACC ：', pictuvaliACC)
    print('ValiLoss ：', pictuvaliLoss)

    print('testing model')
    model = CNN_Fusion(args)
    model.load_state_dict(torch.load(best_validate_dir))
    if torch.cuda.is_available():
        model.cuda()
    # 已经model.eval
    '''TSEk可视化'''
    model.eval()
    #all_logits = []  # 所有点
    #y_labels = []  # 标签名称

    test_score = []
    test_pred = []
    test_true = []

    with torch.no_grad():
        for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
            test_text, test_image, test_mask, test_labels = to_var(
                test_data[0]), to_var(test_data[1]), to_var(test_data[2]), to_var(test_labels)
            test_outputs, domain_outputs = model(test_text, test_image, test_mask)  # logits
            _, test_argmax = torch.max(test_outputs, 1)
            # torch.max(a, 1): 返回每一行的最大值，且返回索引:_是索引（返回最大元素在各行的列索引）。
            #all_logits.append(test_outputs)#预测向量
            #y_labels.append(test_argmax)#真真实标签
            if i == 0:
                test_score = to_np(test_outputs.squeeze())
                test_pred = to_np(test_argmax.squeeze())
                test_true = to_np(test_labels.squeeze())
            else:
                test_score = np.concatenate((test_score, to_np(test_outputs)), axis=0)
                test_pred = np.concatenate((test_pred, to_np(test_argmax)), axis=0)
                test_true = np.concatenate((test_true, to_np(test_labels)), axis=0)

    #all_logits = torch.cat(all_logits, dim=0)  # 31,147
    #y_labels = torch.cat(y_labels, dim=0)  # 31
    test_accuracy = metrics.accuracy_score(test_true, test_pred)
    test_f1 = metrics.f1_score(test_true, test_pred, average='macro')
    test_precision = metrics.precision_score(test_true, test_pred, average='macro')
    test_recall = metrics.recall_score(test_true, test_pred, average='macro')
    test_score_convert = [x[1] for x in test_score]
    test_aucroc = metrics.roc_auc_score(test_true, test_score_convert, average='macro')
    test_confusion_matrix = metrics.confusion_matrix(test_true, test_pred)

    '''目的：all_logits[n*d]:32 * 147，竖着拼接，dim=0,默认
            y_labels[n] 竖着拼接'''
    print("Classification Acc: %.4f, AUC-ROC: %.4f"
          % (test_accuracy, test_aucroc))
    print("Classification report:\n%s\n"
          % (metrics.classification_report(test_true, test_pred)))
    print("Classification confusion matrix:\n%s\n"
          % (test_confusion_matrix))
    """    writer.add_embedding(mat=all_logits,  # 所有点
                         metadata=y_labels,  # 标签名称  # 标签图片
                         )"""


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
    # 100个epoch
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')
    #试试0.01
    parser.add_argument('--event_num', type=int, default=10, help='')
    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
    #    args = parser.parse_args()
    return parser


def get_top_post(output, label, test_id, top_n=500):
    filter_output = []
    filter_id = []
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
    tokenizer = BertTokenizer.from_pretrained(MODEL)
    tokenized_texts = []
    original_texts = flag['original_post']
    for sentence in original_texts:  # 数据中的每一个句子
        tokenized_text = tokenizer.encode(sentence)  # 编码
        tokenized_texts.append(tokenized_text)  #
    flag['post_text'] = tokenized_texts  # 覆盖


def get_all_text(train, validate, test):
    all_text = list(train['post_text']) + list(validate['post_text']) + list(test['post_text'])
    return all_text


# 对齐数据
def align_data(flag, args):
    text = []
    mask = []
    for sentence in flag['post_text']:  # 每一个句子拿出来
        sen_embedding = []  # 一个句子用一个list
        # 最长的max_len，初始化max——len
        mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
        # mask所有的数据都覆为1.0
        mask_seq[:len(sentence)] = 1.0
        for i, word in enumerate(sentence):  # 把句子中的每一个单词进行列举
            sen_embedding.append(word)  # 嵌入向量进行append
        '''就是不满足最大长度的向量，进行补0操作，一直循环补0，补到最大长度为止'''
        while len(sen_embedding) < args.sequence_len:  #
            sen_embedding.append(0)

        text.append(copy.deepcopy(sen_embedding))
        mask.append(copy.deepcopy(mask_seq))
    flag['post_text'] = text  # 重新赋值
    flag['mask'] = mask  # mask也赋值


def load_data(args):
    # 先判断是不是只有text
    train, validate, test = process_data.get_data(args.text_only)
    print("-" * 50, "预训练模型处理中文训练集得到词向量", "-" * 50)
    re_tokenize_sentence(train)  # {字典}
    print("-" * 50, "预训练模型处理中文验证集得到词向量", "-" * 50)
    re_tokenize_sentence(validate)
    print("-" * 50, "预训练模型处理中文测试集得到词向量", "-" * 50)
    re_tokenize_sentence(test)
    print("-" * 50, "联合所有文本，找最长 max_len", "-" * 50)
    all_text = get_all_text(train, validate, test)
    max_len = len(max(all_text, key=len))
    print("max_len最长为：", max_len)
    args.sequence_len = max_len  # 将数据的sequence——len覆为最大值
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
    output = '/tmp/pycharm_project_815/src/weight/'
    args = parser.parse_args([train, test, output])

    main(args)
