import os
import warnings

import cv2
from transformers import AutoConfig, TFAutoModel, AutoTokenizer, BertModel, BertTokenizer
from sklearn import metrics
from src.pytorch_compact_bilinear_pooling import CountSketch, CompactBilinearPooling
import sys
import pandas as pd
from src.trans_padding import Conv1d as conv1d
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable, Function
import torch.nn as nn
import argparse
import time
import os
from src.early_stopping import *
# 提取模型
from PIL import Image

from src import process_data_weibo2 as process_data
import copy
from random import sample
import torchvision
from torchvision import datasets, models, transforms
import torch

print(torch.version.cuda)
if (torch.cuda.is_available()):
    print("CUDA 存在")
# 记录输出
MODEL_NAME = 'bert-base-chinese'


class ReverseLayerF(Function):
    @staticmethod
    def forward(self, x):
        self.lambd = 1
        return x.view_as(x)

    @staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)


def grad_reverse(x):
    return ReverseLayerF.apply(x)


import process_data_weibo2 as process_data
from fusion_module import *
from src.early_stopping import *
from gate_module import *


class CNN_Fusion(nn.Module):
    def __init__(self, args):
        super(CNN_Fusion, self).__init__()
        self.args = args
        # self.layer_attention = LayerAttention

        # self.event_num = args.event_num
        #
        # vocab_size = args.vocab_size
        # emb_dim = args.embed_dim
        #
        # C = args.class_num
        self.hidden_size = args.hidden_dim
        # GGGGru
        # self.hidden_size_ = args.lambd
        # self.num_layers = args.lambd
        # self.batch_first = args.batch_first
        # self.bidirectional = args.bidirectional
        # self.lstm_size = args.embed_dim
        # self.social_size = 19
        self.share_fc = nn.Linear(self.hidden_size, self.hidden_size)

        from resnet_cbam import resnet50_cbam  # 使用哪个引入哪个即可

        self.resnet50_CBAM = resnet50_cbam(pretrained=True)
        # resnet50_CBAM.fc = nn.Linear(2048, 2048)  # 重新定义最后一层 为了方便调整维度
        for name, para in self.resnet50_CBAM.named_parameters():
            if "ca" or "sa" not in name:
                para.requires_grad_(False)

        # self.resnet50_CBAM = resnet50_CBAM  # 2048

        # bert
        # bert_att = NeuralNet()
        # for param in bert_att.parameters():
        #     param.requires_grad = False
        # self.bert_att = bert_att
        # self.GatedFusion = GatedFusion(32, 4, 0.0)
        self.GatedFusionGlobal = GatedFusionGlobal(32, 4, 0.0)
        self.GatedFusionGlobal2 = GatedFusionGlobal2(32, 4, 0.0)
        bert_model = BertModel.from_pretrained('bert-base-chinese', output_hidden_states=True)
        self.bert_hidden_size = args.bert_hidden_dim
        self.fc2 = nn.Linear(self.bert_hidden_size, self.hidden_size)  # 768*32

        for param in bert_model.parameters():
            param.requires_grad = False
        self.bertModel = bert_model
        self.drop_rate = 0.2
        self.att_hid = 32
        self.head = 1
        self.final_hid = 32
        self.self_text_att_average = nn.Sequential(
            nn.Linear(32, self.final_hid),
            nn.BatchNorm1d(192),
            nn.ReLU()
        )
        self.self_text_re_Dro = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )
        self.cross_text_re_Dro = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )
        self.cross_img_text_re_Dro = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(192),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )
        # 层标准化
        self.self_text_layer_norm = nn.LayerNorm([192, 32])
        self.self_layer_norm = nn.LayerNorm([192, 32])
        self.cross_layer_norm = nn.LayerNorm([64, 32])
        # flatten_self
        self.self_text_faltten = nn.Sequential(
            nn.Linear(192 * 32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.self_img_text_faltten = nn.Sequential(
            nn.Linear(192 * 32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.self_text_query = nn.Linear(768, int(self.att_hid / self.head))
        self.self_text_key = nn.Linear(768, int(self.att_hid / self.head))
        self.self_text_value = nn.Linear(768, int(self.att_hid / self.head))

        self.dropout = nn.Dropout(args.dropout)
        # GRU
        # self.bigru = GRU(self.bert_hidden_size, self.hidden_size * 2, self.num_layers, self.batch_first,
        #                  self.bidirectional)
        # IMAGE

        # attetion att_img
        self.img_dim = 32
        self.att_hid = 32
        self.head = 1
        self.cross_img_txt_query = nn.Linear(32, int(self.att_hid / self.head))
        self.cross_img_txt_key = nn.Linear(32, int(self.att_hid / self.head))
        self.cross_img_txt_value = nn.Linear(32, int(self.att_hid / self.head))

        self.cross_txt_img_query = nn.Linear(32, int(self.att_hid / self.head))
        self.cross_img_key = nn.Linear(32, int(self.att_hid / self.head))
        self.cross_img_value = nn.Linear(32, int(self.att_hid / self.head))

        self.self_att_average = nn.Sequential(
            nn.Linear(32, self.final_hid),
            nn.BatchNorm1d(192),
            nn.ReLU()
        )

        self.cross_img_att_average = nn.Sequential(
            nn.Linear(32, self.final_hid),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.self_img_text_att_average = nn.Sequential(
            nn.Linear(32, self.final_hid),
            nn.BatchNorm1d(192),
            nn.ReLU()
        )
        self.cross_img_re_Dro = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )

        self.self_img_faltten = nn.Sequential(
            nn.Linear(64 * 32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        # vgg
        # vgg_19 = torchvision.models.vgg19(pretrained=True)
        # for param in vgg_19.parameters():
        #     param.requires_grad = False
        # # visual model
        # num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.image_fc1 = nn.Linear(1000, self.hidden_size)
        self.image_cross_fc = nn.Linear(1000, 2048)
        # print(num_ftrs) 1000
        # self.vgg = vgg_19

        # resnet
        # Resnet_50 = torchvision.models.resnet50(pretrained=True)
        # num_ftrs2 = Resnet_50.fc.out_features
        # self.image_fc3 = nn.Linear(num_ftrs2, self.hidden_size)
        # for param in Resnet_50.parameters():
        #     param.requires_grad = False
        # self.resnet50 = Resnet_50
        # print(self.resnet50)
        self.BCELoss = nn.BCELoss()
        self.dropout = nn.Dropout()
        # self.image_fc1 = nn.Linear(num_ftrs,  512)
        # self.image_fc2 = nn.Linear(512, self.hidden_size)
        # self.image_adv = nn.Linear(self.hidden_size, int(self.hidden_size))
        # self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)
        # self.textcnn = nn.Sequential(
        #     nn.Conv1d(1768, 256, kernel_size=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=1),
        #     nn.Conv1d(256, 128, kernel_size=1),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=1),
        #     nn.Flatten(),
        #     nn.Linear(128, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 1)
        # )
        self.fc = nn.Linear(64, 128)
        # Class Classifier
        self.class_Multimodal_classifier = nn.Sequential()
        self.class_Multimodal_classifier.add_module('c_fc1', nn.Linear(11 * self.hidden_size, 2))
        self.class_Multimodal_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(3 * self.hidden_size, 2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        self.multi = nn.Linear(64, 192)

        self.multi_text = nn.Linear(192, 64)

        self.fc_out = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.01),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, text, image, ocr):
        head = 1
        att_layers = 1
        att_hid = 32
        # Image   第一章  维度为1000
        images = self.resnet50_CBAM(image)  # [32, 1000]->[32,2048]->[32,64,32] 用来做交叉注意力
        image = F.relu(self.image_fc1(images))  # [32, 1000]

        image_cross = self.image_cross_fc(images)  # [32, 2048]
        image_cross = torch.reshape(image_cross, (image_cross.shape[0], -1, 32))  # torch.Size([32, 64, 32])

        # image = F.relu(self.share_fc(image))
        # Text
        # last_hidden_state = torch.mean(self.bertModel(text)[0], dim=1, keepdim=False)
        # text = F.relu(self.fc2(last_hidden_state))
        out = self.bertModel(text, output_attentions=True)  # berts输出
        hidden_states = out[0]  # 最后一层的输出
        '''文本自注意力机制'''
        in_Self_TEXT_key = hidden_states
        in_Self_TEXT_query = hidden_states  ##torch.Size([32, 192, 768])
        for _ in range(head):
            self_text_query = self.self_text_query(in_Self_TEXT_query)  ##torch.Size([32, 192, 32])
            self_text_key = self.self_text_key(in_Self_TEXT_key)  # torch.Size([32, 192, 32])
            self_text_value = self.self_text_value(in_Self_TEXT_key)  # torch.Size([32, 192, 32])
            '''计算sccore Q*K-公式第一步'''
            score = torch.tensordot(self_text_query, self_text_key, dims=([2], [2]))
            score = torch.stack([score[i, :, i, :] for i in range(len(score))])
            # print(score.shape)  # torch.Size([32, 192, 192])
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
            '''将注意力后的数据相加:text_+self_att_text'''
            self_text_att = torch.add(self_text_query, self_text_att2)  # [32,192,32]
            '''层标准化'''
            self_text_att = self.self_text_layer_norm(self_text_att)
            '''不能将他直接merge进去因为，维度不同，flatten之后进行输入'''
            in_Self_TEXT_query = self_text_att
            in_Self_TEXT_key = self_text_att
        '''将后面的两维度flatten，变成二维'''
        # print(self_text_att.size()) [32,192,32]
        self_text_att1 = self_text_att
        self_text_att = torch.flatten(self_text_att, start_dim=1, end_dim=2)  # end_dim=-1
        self_text_att = self.self_text_faltten(self_text_att)
        '''此时向量的维度是正常的'''
        # print(self_text_att.shape)  # torch.Size([32, 32])
        text = self_text_att
        # text = F.relu(self.share_fc(text))

        '''______________________________交叉注意力机制__________________________________'''  # 做注意力机制是否必须是三维向量 其中第一维是batch_size  因为self_text_att是三维的
        # text_quary = self_text_att  # [32,192,32]
        # text_key = self_text_att
        # text_value = self_text_att
        #
        # image_quary = image_cross  # [32,64,32]
        # image_key = image_cross
        # image_value = image_cross

        #         维度变换
        #         text[32,32,192]
        #         image[32,32,64]
        cross_text_att1 = self_text_att1.permute(0, 2, 1)  # text[32,32,192]
        image_cross = image_cross.permute(0, 2, 1)  # image[32,32,64]

        '''视觉特征更新文本特征   视觉q  文本k v  视觉特征调整维度和文本相同'''
        image_quary = image_cross  # image[32,32,64]
        image_quary = self.multi(image_quary)  # 32 32 64--[32 32 192]
        image_quary = image_quary.permute(0, 2, 1)  # [32 192 32]
        text_key = cross_text_att1  # text[32,32,192]
        text_key = text_key.permute(0, 2, 1)  # [32 192 32]
        text_value = cross_text_att1  # text[32,32,192]
        text_value = text_value.permute(0, 2, 1)  # [32 192 32]
        for layer in range(att_layers):
            cross_att_img_text = []
            for _ in range(head):
                cross_img_txt_query = self.cross_img_txt_query(image_quary)  # 视觉q [32 192 32]
                cross_text_key = self.cross_img_txt_key(text_key)  # 文本k
                cross_text_value = self.cross_img_txt_value(text_value)  # 文本v
                # print(self_img_txt_query.shape)
                '''计算sccore Q*K-公式第一步'''
                score = torch.tensordot(cross_img_txt_query, cross_text_key, dims=([2], [2]))
                # print(score.shape)  # torch.Size([32, 192, 32, 192])
                '''改变合并方法'''
                score = torch.stack([score[i, :, i, :] for i in range(len(score))])
                # print(score.shape)  # torch.Size([32, 192, 192])
                # print("score.shape")
                '''公式第二步,score/根号下d 维度-维度不匹配的广播机制'''
                # 如果是y一个数，x所有元素除以y
                score = torch.div(score, np.sqrt(att_hid / head))
                score = F.leaky_relu(score)
                '''公式第三步-score * v'''
                attention = torch.tensordot(score, cross_text_value, dims=([2], [1]))
                # print(attention.shape)  # torch.Size([32, 192, 32, 32])
                attention = torch.stack([attention[i, :, i, :] for i in range(len(attention))])
                # print(attention.shape)  # torch.Size([32, 192, 32])
                '''得出询问后的自己后的 att'''
                cross_att_img = attention
                '''将注意力均值化-无法平均-也就是多个列表，对应的元素相加求平均在进行成为当前元素'''
                cross_att_img21 = self.self_img_text_att_average(cross_att_img)
                cross_att_img22 = self.self_img_text_att_average(cross_att_img)
                cross_att_img23 = self.self_img_text_att_average(cross_att_img)
                cross_att_img24 = self.self_img_text_att_average(cross_att_img)
                # 均值
                cross_att_img2 = cross_att_img21.add(cross_att_img22).add(cross_att_img23).add(cross_att_img24)
                cross_att_img2 = torch.div(cross_att_img2, 4)
                '''均值后数据'''
                cross_att_img2 = self.cross_img_text_re_Dro(cross_att_img2)
                '''将注意力后的数据相加:image_+self_att_img'''
                cross_att_img = torch.add(image_quary, cross_att_img2)
                # print(self_att_img.shape)  # [32,192,32]
                '''层标准化'''
                cross_att_img = self.self_layer_norm(cross_att_img)
                '''不能将他直接merge进去因为，维度不同，faltten之后进行输入'''
                # print(self_att_img.shape)  # [32,192,32]
                image_quary = cross_att_img
                text_key = cross_att_img
        cross_att_img = torch.flatten(cross_att_img, start_dim=1, end_dim=2)  # end_dim=-1
        cross_att_image_text = self.self_img_text_faltten(cross_att_img)  # 视觉更新的文本特征
        '''此时向量的维度是正常的'''
        # print(cross_att_image_text.shape)  # [32,32]

        '''文本特征更新视觉特征   文本q  视觉k v  文本特征调整维度和视觉相同'''
        text_query = cross_text_att1  # [32 32 192]
        text_query = self.multi_text(text_query)  # text[32,32,64]
        text_query = text_query.permute(0, 2, 1)  # [32 64 32]
        image_key = image_cross  # image[32,32,64]
        image_key = image_key.permute(0, 2, 1)  # [32 64 32]
        image_value = image_cross  # image[32,32,64]
        image_value = image_value.permute(0, 2, 1)  # [32 64 32]
        for layer in range(att_layers):
            cross_att_img_text = []
            for _ in range(head):
                cross_txt_img_query = self.cross_txt_img_query(text_query)  # 文本q [32 64 32]
                cross_img_key = self.cross_img_key(image_key)  # 视觉k
                cross_img_value = self.cross_img_value(image_value)  # 视觉v
                # print(self_img_txt_query.shape)
                '''计算sccore Q*K-公式第一步'''
                score = torch.tensordot(cross_txt_img_query, cross_img_key, dims=([2], [2]))
                # print(score.shape)  # torch.Size([32, 64, 32, 64])
                '''改变合并方法'''
                score = torch.stack([score[i, :, i, :] for i in range(len(score))])
                # print(score.shape)  # torch.Size([32, 64, 64])
                # print("score.shape")
                '''公式第二步,score/根号下d 维度-维度不匹配的广播机制'''
                # 如果是y一个数，x所有元素除以y
                score = torch.div(score, np.sqrt(att_hid / head))
                score = F.leaky_relu(score)
                '''公式第三步-score * v'''
                attention = torch.tensordot(score, cross_img_value, dims=([2], [1]))
                # print(attention.shape)  # torch.Size([32, 64, 32, 32])
                attention = torch.stack([attention[i, :, i, :] for i in range(len(attention))])
                # print(attention.shape)  # torch.Size([32, 64, 32])
                '''得出询问后的自己后的 att'''
                cross_img_att = attention
                '''将注意力均值化-无法平均-也就是多个列表，对应的元素相加求平均在进行成为当前元素'''
                cross_img21_att = self.cross_img_att_average(cross_img_att)
                cross_img22_att = self.cross_img_att_average(cross_img_att)
                cross_img23_att = self.cross_img_att_average(cross_img_att)
                cross_img24_att = self.cross_img_att_average(cross_img_att)
                # 均值
                cross_img2_att = cross_img21_att.add(cross_img22_att).add(cross_img23_att).add(cross_img24_att)
                cross_img2_att = torch.div(cross_img2_att, 4)
                '''均值后数据'''
                cross_img2_att = self.cross_img_re_Dro(cross_img2_att)
                '''将注意力后的数据相加:image_+self_att_img'''
                cross_img_att = torch.add(text_query, cross_img2_att)
                # print(self_att_img.shape)  # [32,64,32]
                '''层标准化'''
                cross_img_att = self.cross_layer_norm(cross_img_att)
                '''不能将他直接merge进去因为，维度不同，faltten之后进行输入'''
                # print(cross_img_att.shape)  # [32,64,32]
                text_query = cross_img_att
                image_key = cross_img_att
        cross_img_att = torch.flatten(cross_img_att, start_dim=1, end_dim=2)  # end_dim=-1
        cross_att_text_image = self.self_img_faltten(cross_img_att)
        last_hidden_state = torch.mean(self.bertModel(ocr.long())[0], dim=1, keepdim=False)
        ocr = F.relu(self.fc2(last_hidden_state))
        '''此时向量的维度是正常的'''

        # gate文本
        Multimodal_text = self.GatedFusionGlobal2(text, cross_att_text_image, ocr, False, "words")
        # gate图像
        Multimodal_image = self.GatedFusionGlobal2(cross_att_image_text, image, ocr, False, "words")

        Fm1 = torch.cat((Multimodal_text, Multimodal_image), 1)  # (32,128)
        # gate文本OCR
        Multimodal_ocr = self.GatedFusionGlobal2(ocr, cross_att_text_image, text, False, "words")
        # gate图像
        Multimodal_image_ = self.GatedFusionGlobal2(image, ocr, text, False, "words")

        Fm2 = torch.cat((Multimodal_ocr, Multimodal_image_), 1)  # (32,128)

        Multimodal = torch.cat((Fm1, Fm2), dim=-1)  # (32,256)

        score_1 = self.fc_out(Fm1)  # (32,1) #匹配分数
        score_2 = self.fc_out(Fm2)  # (32,1)
        # print(score_1.shape)
        avg_score = (score_1 + score_2) / 2

        distance = torch.ones_like(avg_score) - avg_score  # 不匹配分数
        # print(distance.shape)
        distance_ = torch.cat([distance, avg_score], dim=1)
        # print(distance_.shape)

        # weighted_features = F.softmax(score_1, dim=1)

        # weighted_features2 = F.softmax(score_2, dim=1)

        Multimodal_com = torch.mul(Multimodal, avg_score)

        # weighted_fm2 = torch.mul(Fm2, weighted_features2)

        # Multimodal = torch.cat((weighted_fm1, weighted_fm2), dim=-1)  # (32,256)

        Unimodal = torch.cat((text, image, ocr), dim=-1)

        Unimodal_com = torch.mul(Unimodal, distance)
        # torch.Size([32, 96])
        final = torch.cat((Multimodal_com, Unimodal_com), dim=-1)  # (32,352)

        final_output = self.class_Multimodal_classifier(final)

        # Multimodal_output = self.class_Multimodal_classifier(Multimodal)  # torch.Size([32, 2])

        # Unimodal_output = self.class_classifier(Unimodal)  # torch.Size([32, 2])

        return final_output, distance_


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def parse_arguments(parser):
    parser.add_argument('training_file', type=str,
                        metavar='<training_file>', help='')
    # parser.add_argument('validation_file', type=str, metavar='<validation_file>', help='')
    parser.add_argument('testing_file', type=str,
                        metavar='<testing_file>', help='')
    parser.add_argument('output_file', type=str,
                        metavar='<output_file>', help='')
    parser.add_argument('--static', type=bool, default=True, help='')
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
    # 试试0.01
    parser.add_argument('--event_num', type=int, default=10, help='')
    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
    #    args = parser.parse_args()
    return parser


"""提前加载模型"""
parse = argparse.ArgumentParser()
parser = parse_arguments(parse)
train = ''
test = ''
# output = '/tmp/pycharm_project_815/src/weight/'
output = '../Data/result/'
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
args = parser.parse_args([train, test, output])
print("----虚假新闻检测模型加载中-----")
best_validate_dir = '../Data/result/best.pkl'
model = CNN_Fusion(args)
model.eval()
model.load_state_dict(torch.load(best_validate_dir))
print("-----虚假新闻检测模型加载完毕----")
sequence_len = 192
ld = {0: '真实', 1: '虚假'}  # 对应的字典


def predict(text, img, ocr):
    '''text'''
    tokenized_text = tokenizer.encode(text)  # 编码
    sen_embedding = []  # 一个句子用一个list
    # # 最长的max_len，初始化max——len
    # mask_seq = np.zeros(sequence_len, dtype=np.float32)
    # # mask所有的数据都覆为1.0，其余的都为0
    # mask_seq[:len(text)] = 1.0
    for i, word in enumerate(tokenized_text):  # 把句子中的每一个单词进行列举
        sen_embedding.append(word)  # 嵌入向量进行append
    '''就是不满足最大长度的向量，进行补0操作，一直循环补0，补到最大长度为止'''
    while len(sen_embedding) < sequence_len:  #
        sen_embedding.append(0)
    print('原始文本:', text)
    # print("mask_seq:", mask_seq)
    print("sen_embedding:", sen_embedding)
    sen_embedding = torch.from_numpy(np.array(sen_embedding))
    if len(sen_embedding) >= 192:
        sen_embedding = sen_embedding[:192]
    sen_embedding = sen_embedding.unsqueeze(0)
    # mask_seq = torch.from_numpy(np.array(mask_seq))
    # mask_seq = mask_seq.unsqueeze(0)
    print('sentence的shape:', sen_embedding.size())
    # print('mask_seq的shape:', mask_seq.size())
    '''flask调用时覆盖'''

    '''OCR'''
    tokenized_ocr = tokenizer.encode(ocr)  # 编码
    sen_embedding_ocr = []  # 一个句子用一个list
    # 最长的max_len，初始化max——len
    # mask_seq_ocr = np.zeros(sequence_len, dtype=np.float32)
    # # mask所有的数据都覆为1.0，其余的都为0
    # mask_seq_ocr[:len(ocr)] = 1.0
    for i, word in enumerate(tokenized_ocr):  # 把句子中的每一个单词进行列举
        sen_embedding_ocr.append(word)  # 嵌入向量进行append
    '''就是不满足最大长度的向量，进行补0操作，一直循环补0，补到最大长度为止'''
    while len(sen_embedding_ocr) < sequence_len:  #
        sen_embedding_ocr.append(0)
    print('原始文本ocr:', ocr)
    # print("mask_seq_ocr:", mask_seq_ocr)
    print("sen_embedding_ocr:", sen_embedding_ocr)
    sen_embedding_ocr = torch.from_numpy(np.array(sen_embedding_ocr))
    if len(sen_embedding_ocr) >= 192:
        sen_embedding_ocr = sen_embedding_ocr[:192]
    sen_embedding_ocr = sen_embedding_ocr.unsqueeze(0)
    # mask_seq_ocr = torch.from_numpy(np.array(mask_seq_ocr))
    # mask_seq_ocr = mask_seq_ocr.unsqueeze(0)
    print('sentence_ocr的shape:', sen_embedding_ocr.size())
    # print('mask_seq_ocr的shape:', mask_seq_ocr.size())
    '''flask调用时覆盖'''

    img = img.unsqueeze(0)
    print("img的shape:", img.size())
    with torch.no_grad():
        print("----开始预测------")
        test_outputs, distance_ = model(
            sen_embedding, img, sen_embedding_ocr)  # logits
        _, predict = torch.max(test_outputs, 1)
    print("虚假新闻预测结果：", ld[predict.numpy().tolist()[0]],
          type(predict.numpy().tolist()))  # 输出对应的标签
    print("虚假新闻预测结果", predict)
    return ld[predict.numpy().tolist()[0]]
