import numpy as np
import argparse
import time, os, random

from torch.nn import GRU
from torch.utils.tensorboard import SummaryWriter

# import random
import process_data_weibo2 as process_data
import copy
import pickle as pickle
from random import sample
import torchvision
import torch
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ExponentialLR
import torch.nn as nn
from torch.autograd import Variable, Function
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from src.early_stopping import *
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn import metrics
from transformers import *
from Bert_att import LayerAttention
import warnings
# 导入模块
from transformers import RobertaModel, RobertaTokenizer
from resnet_cbam import resnet50_cbam
from fusion_module import *

from src.gate_module import GatedFusionGlobal

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seed_value = 2020  # 设定随机数种子

np.random.seed(seed_value)
random.seed(seed_value)
os.environ['PYTHONHASHSEED'] = str(seed_value)  # 为了禁止hash随机化，使得实验可复现。

torch.manual_seed(seed_value)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed_value)  # 为当前GPU设置随机种子（只用一块GPU）
torch.cuda.manual_seed_all(seed_value)  # 为所有GPU设置随机种子（多块GPU）

torch.backends.cudnn.deterministic = True


class Rumor_Data(Dataset):
    def __init__(self, dataset):
        self.text = torch.from_numpy(np.array(dataset['post_text']))
        self.image = list(dataset['image'])
        # self.social_context = torch.from_numpy(np.array(dataset['social_feature']))
        self.ocr = torch.from_numpy(np.array(dataset['ocr']).astype(float))

        self.mask = torch.from_numpy(np.array(dataset['mask']))
        self.label = torch.from_numpy(np.array(dataset['label']))
        self.event_label = torch.from_numpy(np.array(dataset['event_label']))
        print('TEXT: %d, Image: %d,ocr: %d, label: %d, Event: %d'
              % (len(self.text), len(self.image), len(self.ocr), len(self.label), len(self.event_label)))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return (self.text[idx], self.image[idx], self.ocr[idx], self.mask[idx]), self.label[idx], self.event_label[idx]


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


from resnet_cbam import resnet50_cbam  # 使用哪个引入哪个即可


class NeuralNet(nn.Module):
    def __init__(self, hidden_size=768, num_class=3):
        super(NeuralNet, self).__init__()

        self.config = BertConfig.from_pretrained('bert-base-chinese', num_labels=2)
        self.config.output_hidden_states = True  # 需要设置为true才输出
        self.bert = BertModel.from_pretrained('bert-base-chinese', config=self.config)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.weights = nn.Parameter(torch.rand(13, 1))
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, input_ids, input_mask, segment_ids):
        last_hidden_states, pool, all_hidden_states = self.bert(input_ids, token_type_ids=segment_ids,
                                                                attention_mask=input_mask)
        batch_size = input_ids.shape[0]
        ht_cls = torch.cat(all_hidden_states)[:, :1, :].view(
            13, batch_size, 1, 768)
        atten = torch.sum(ht_cls * self.weights.view(
            13, 1, 1, 1), dim=[1, 3])
        atten = F.softmax(atten.view(-1), dim=0)
        feature = torch.sum(ht_cls * atten.view(13, 1, 1, 1), dim=[0, 2])
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                h = self.fc(dropout(feature))
            else:
                h += self.fc(dropout(feature))
        h = h / len(self.dropouts)
        return h


# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(self, args):
        super(CNN_Fusion, self).__init__()
        self.args = args
        self.layer_attention = LayerAttention

        self.event_num = args.event_num
        #
        # vocab_size = args.vocab_size
        # emb_dim = args.embed_dim
        #
        # C = args.class_num
        self.hidden_size = args.hidden_dim
        # GGGGru
        self.hidden_size_ = args.lambd
        self.num_layers = args.lambd
        self.batch_first = args.batch_first
        self.bidirectional = args.bidirectional
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
        self.GatedFusion = GatedFusionGlobal(32, 4, 0.0)
        # bert
        # bert_att = NeuralNet()
        # for param in bert_att.parameters():
        #     param.requires_grad = False
        # self.bert_att = bert_att

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
        self.self_text_query = nn.Linear(768, int(self.att_hid / self.head))
        self.self_text_key = nn.Linear(768, int(self.att_hid / self.head))
        self.self_text_value = nn.Linear(768, int(self.att_hid / self.head))

        self.dropout = nn.Dropout(args.dropout)
        # GRU
        self.bigru = GRU(self.bert_hidden_size, self.hidden_size * 2, self.num_layers, self.batch_first,
                         self.bidirectional)
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
        vgg_19 = torchvision.models.vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False
        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)
        self.image_cross_fc = nn.Linear(num_ftrs, 2048)
        # print(num_ftrs) 1000
        self.vgg = vgg_19

        # resnet
        Resnet_50 = torchvision.models.resnet50(pretrained=True)
        num_ftrs2 = Resnet_50.fc.out_features
        self.image_fc3 = nn.Linear(num_ftrs2, self.hidden_size)
        for param in Resnet_50.parameters():
            param.requires_grad = False
        self.resnet50 = Resnet_50
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
        # Class Classifier
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(2 * self.hidden_size, 2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        self.multi = nn.Linear(64, 192)

        self.multi_text = nn.Linear(192, 64)

    def forward(self, text, image, ocr, mask):
        global cross_img_att
        head = 1
        att_layers = 1
        att_hid = 32
        # print(text.size())    torch.Size([32, 192])
        # print(image.size())   torch.Size([32, 3, 224, 224])

        # text1 = self.bertModel(text)[0][:, 0, :]
        # with torch.no_grad():
        #     image1 = self.resnet50(image)
        # # print(text1.size()) torch.Size([32, 768])
        # # print(image1.size()) torch.Size([32, 1000])
        # out1 = torch.cat([text1, image1], dim=1)
        # # print(out1.size()) torch.Size([32, 1768])
        # out1 = out1.view(-1, 1768)
        # out1 = out1.unsqueeze(1)  # 扩展维度 [32*1, 1, 1768]
        # out1 = out1.transpose(1, 2)  # 转置维度 [32, 1768, 1]
        # out1 = self.textcnn(out1)
        # out2 = torch.sigmoid(out1)

        # Image   第一章  维度为1000
        images = self.resnet50(image)  # [32, 1000]->[32,2048]->[32,64,32] 用来做交叉注意力
        image = F.relu(self.image_fc1(images))  # [32, 1000]

        image_cross = self.image_cross_fc(images)  # [32, 2048]
        image_cross = torch.reshape(image_cross, ((image_cross.shape[0], -1, 32)))  # torch.Size([32, 64, 32])

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
        text = F.relu(self.share_fc(text))

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
                cross_att_img21 = self.self_text_att_average(cross_att_img)
                cross_att_img22 = self.self_text_att_average(cross_att_img)
                cross_att_img23 = self.self_text_att_average(cross_att_img)
                cross_att_img24 = self.self_text_att_average(cross_att_img)
                # 均值
                cross_att_img2 = cross_att_img21.add(cross_att_img22).add(cross_att_img23).add(cross_att_img24)
                cross_att_img2 = torch.div(cross_att_img2, 4)
                '''均值后数据'''
                cross_att_img2 = self.cross_text_re_Dro(cross_att_img2)
                '''将注意力后的数据相加:image_+self_att_img'''
                cross_att_img = torch.add(cross_img_txt_query, cross_att_img2)
                # print(self_att_img.shape)  # [32,192,32]
                '''层标准化'''
                cross_att_img = self.self_layer_norm(cross_att_img)
                '''不能将他直接merge进去因为，维度不同，faltten之后进行输入'''
                # print(self_att_img.shape)  # [32,192,32]
                image_quary = cross_att_img
                text_key = cross_att_img
        cross_att_img = torch.flatten(cross_att_img, start_dim=1, end_dim=2)  # end_dim=-1
        cross_att_image_text = self.self_text_faltten(cross_att_img)  # 视觉更新的文本特征
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
        cross_att_text_image = self.self_img_faltten(cross_img_att)  # 文本更新的视觉特征
        '''此时向量的维度是正常的'''
        # print(cross_att_text_image.shape)  # [32,32]

        # GatedFusionFeature = self.GatedFusion(cross_att_text_image, cross_att_image_text, False)
        # print(GatedFusionFeature.shape)

        # OCR
        last_hidden_state = torch.mean(self.bertModel(ocr.long())[0], dim=1, keepdim=False)
        ocr = F.relu(self.fc2(last_hidden_state))

        # Fake or real   prediction
        text_image1 = torch.cat((cross_att_image_text, cross_att_text_image), 1)
        class_output = self.class_classifier(text_image1)

        # cos_sim
        ''' 计算了每个向量的L2范数，并对向量进行了归一化。
            这样做的目的是为了将每个向量映射到单位长度，从而消除向量长度对余弦相似度的影响。'''
        image_norm = torch.sqrt(torch.sum(torch.pow(image, 2), dim=1))  # [32]

        text_norm = torch.sqrt(torch.sum(torch.pow(text, 2), dim=1))
        '''计算图像和文本之间的点积,并使用image_norm和text_norm计算了它们的L2范数的乘积。
        最后，将点积除以L2范数的乘积得到余弦相似度,并使用(1 + cos_simi) / 2将其转换为距离度量。
        添加了一个较小的常数1e-8以避免分母为零的情况'''
        image_text = torch.sum(torch.mul(image, text), dim=1)

        '''代码使用了(1 + cos_simi) / 2的公式。
        这个公式的目的是将余弦相似度的范围从[-1,1]映射到[0,1]，以便使用距离度量表示相似度。'''
        cos_simi = (1 + (image_text / (image_norm * text_norm + 1e-8))) / 2

        '''计算相似度的补数，即不相似的距离。'''
        distance_ = torch.ones_like(cos_simi) - cos_simi

        '''将相似度和不相似的距离堆叠在一起，以便于后续的损失计算。'''
        distance = torch.stack([distance_, cos_simi], dim=1)  # [32, 2]

        # ocr_cos_sim
        ocr_norm = torch.sqrt(torch.sum(torch.pow(ocr, 2), dim=1))
        text_norm_ = torch.sqrt(torch.sum(torch.pow(text, 2), dim=1))

        ocr_text = torch.sum(torch.mul(ocr, text), dim=1)

        cos_simi_ocr = (1 + (ocr_text / (ocr_norm * text_norm_ + 1e-8))) / 2
        # print(cos_simi_ocr.size())torch.Size([32])
        distance_ocr = torch.ones_like(cos_simi_ocr) - cos_simi_ocr
        # print(distance_ocr.size())torch.Size([32])
        distance_ocr_ = torch.stack([distance_ocr, cos_simi_ocr], dim=1)
        # print(distance_ocr_.size())torch.Size([32, 2])

        # ocr_cos_sim   ocr—img
        ocr_norm_ = torch.sqrt(torch.sum(torch.pow(ocr, 2), dim=1))
        img_norm = torch.sqrt(torch.sum(torch.pow(image, 2), dim=1))
        ocr_text_ = torch.sum(torch.mul(ocr, image), dim=1)
        cos_simi_ocr_img = (1 + (ocr_text_ / (ocr_norm_ * img_norm + 1e-8))) / 2
        distance_ocr_im = torch.ones_like(cos_simi_ocr_img) - cos_simi_ocr_img
        distance_ocr_img = torch.stack([distance_ocr_im, cos_simi_ocr_img], dim=1)

        return class_output, distance, distance_ocr_, distance_ocr_img


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def to_np(x):
    return x.data.cpu().numpy()


# def select(train, selec_indices):
#     temp = []
#     for i in range(len(train)):
#         print("length is " + str(len(train[i])))
#         print(i)
#         # print(train[i])
#         ele = list(train[i])
#         temp.append([ele[i] for i in selec_indices])
#     return temp

#
# def make_weights_for_balanced_classes(event, nclasses=15):
#     count = [0] * nclasses
#     for item in event:
#         count[item] += 1
#     weight_per_class = [0.] * nclasses
#     N = float(sum(count))
#     for i in range(nclasses):
#         weight_per_class[i] = N / float(count[i])
#     weight = [0] * len(event)
#     for idx, val in enumerate(event):
#         weight[idx] = weight_per_class[val]
#     return weight


# def split_train_validation(train, percent):
#     whole_len = len(train[0])
#
#     train_indices = (sample(range(whole_len), int(whole_len * percent)))
#     train_data = select(train, train_indices)
#     print("train data size is " + str(len(train[3])))
#
#     validation = select(train, np.delete(range(len(train[0])), train_indices))
#     print("validation size is " + str(len(validation[3])))
#     print("train and validation data set has been splited")
#
#     return train_data, validation


def main(args):
    print('loading data')
    train, validation, test = load_data(args)
    test_id = test['post_id']

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

    print('building model')
    model = CNN_Fusion(args)

    if torch.cuda.is_available():
        print("CUDA")
        model.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    BCEloss = nn.BCELoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                                 lr=args.learning_rate)
    print("bach总数：", "训练集" + str(len(train_loader)), "验证集" + str(len(validate_loader)), "测试集" + str(len(test_loader)))
    iter_per_epoch = len(train_loader)
    print("loader size " + str(len(train_loader)))
    best_validate_acc = 0.000
    best_test_acc = 0.000
    best_loss = 100
    best_validate_dir = ''
    best_list = [0, 0]

    print('training model')
    adversarial = True
    # 初始化 early_stopping 对象
    patience = 20  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=True)  # 关于 EarlyStopping 的代码可先看博客后面的内容

    '''初始化设置寻找最优解'''
    pictuTrainLoss = []
    pictuTrainACC = []
    # pictuvaliLoss = []
    # pictuvaliACC = []

    # Train the Model
    for epoch in range(args.num_epochs):

        p = float(epoch) / 100
        # lambd = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.001 / (1. + 10 * p) ** 0.75

        optimizer.lr = lr
        # rgs.lambd = lambd
        start_time = time.time()
        cost_vector = []
        class_cost_vector = []
        domain_cost_vector = []
        cos_sim_cost_vector = []
        ocr_cos_sim_cost_vector = []
        ocr_text_cos_sim_cost_vector = []
        ocr_img_cos_sim_cost_vector = []
        acc_vector = []
        valid_acc_vector = []
        test_acc_vector = []
        vali_cost_vector = []
        test_cost_vector = []

        for i, (train_data, train_labels, event_labels) in enumerate(train_loader):
            train_text, train_image, train_ocr, train_mask, train_labels, event_labels = \
                to_var(train_data[0]), to_var(train_data[1]), to_var(train_data[2]), to_var(train_data[3]), \
                to_var(train_labels), to_var(event_labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()

            class_outputs, distance, distance_ocr_, distance_ocr_img = model(train_text, train_image,
                                                                             train_ocr,
                                                                             train_mask)

            # Fake or Real loss
            # print(class_outputs.shape)  torch.Size([32, 2])
            # print(domain_outputs.shape)
            # print(train_labels) tensor([1, 0, 1, 0, 0, 0, 1, 0, 0, ...], device='cuda:0', dtype=torch.int32)
            # print(event_labels)
            class_loss = criterion(class_outputs, train_labels.long())

            # Event Loss
            # domain_loss = criterion(domain_outputs, event_labels.long())
            # cos_sim_loss
            # print(distance.size())  #torch.Size([32, 2])
            # print(train_labels.size())  #torch.Size([32])
            # cos_sim_loss=-torch.mean(torch.sum(train_labels.long()* torch.log(distance), dim=1))

            cos_sim_loss = criterion(distance, train_labels.long())

            ocr_text_cos_sim_loss = criterion(distance_ocr_, train_labels.long())
            ocr_img_cos_sim_loss = criterion(distance_ocr_img, train_labels.long())
            # loss = class_loss + domain_lossC
            loss = 0.2 * cos_sim_loss + 0.6 * class_loss + 0.1 * ocr_text_cos_sim_loss + 0.1 * ocr_img_cos_sim_loss
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
            # domain_cost_vector.append(domain_loss.item())
            cos_sim_cost_vector.append(cos_sim_loss.item())
            ocr_text_cos_sim_cost_vector.append(ocr_text_cos_sim_loss.item())
            ocr_img_cos_sim_cost_vector.append(ocr_img_cos_sim_loss.item())
            cost_vector.append(loss.item())
            acc_vector.append(accuracy.item())
        '''model.eval()是模型的某些特定层/部分的一种开关，这些层/部分在训练和推断（评估）期间的行为不同'''
        model.eval()
        validate_acc_vector_temp = []
        for i, (validate_data, validate_labels, event_labels) in enumerate(validate_loader):
            validate_text, validate_image, validate_ocr, validate_mask, validate_labels, event_labels = \
                to_var(validate_data[0]), to_var(validate_data[1]), to_var(validate_data[2]), to_var(validate_data[2]), \
                to_var(validate_labels), to_var(event_labels)
            validate_outputs, distance, distance_ocr_, distance_ocr_img = model(validate_text, validate_image,
                                                                                validate_ocr, validate_mask)
            _, validate_argmax = torch.max(validate_outputs, 1)
            vali_loss = criterion(validate_outputs, validate_labels.long())
            # domain_loss = criterion(domain_outputs, event_labels)
            # _, labels = torch.max(validate_labels, 1)
            validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            vali_cost_vector.append(vali_loss.item())
            # validate_accuracy = (validate_labels == validate_argmax.squeeze()).float().mean()
            validate_acc_vector_temp.append(validate_accuracy.item())
        validate_acc = np.mean(validate_acc_vector_temp)
        valid_acc_vector.append(validate_acc)

        '''lbx'''
        validate_loss = np.mean(vali_cost_vector)
        early_stopping(validate_loss, model)

        model.train()
        print(
            'Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, cos_sim loss: %.4f, ocr loss: %.4f , Train_Acc: %.4f,  Validate_Acc: %.4f.'
            % (
                epoch + 1, args.num_epochs, np.mean(cost_vector), np.mean(class_cost_vector),
                np.mean(cos_sim_cost_vector),
                np.mean(ocr_cos_sim_cost_vector),
                np.mean(acc_vector), validate_acc))
        # ---------------------------------------画图-------------------------
        writer = SummaryWriter(log_dir="../Data/weibo_visual2")
        writer.add_scalar("train_acc", np.mean(acc_vector), epoch)
        writer.add_scalar("validate_acc", validate_acc, epoch)
        writer.add_scalar("loss", np.mean(cost_vector), epoch)
        writer.add_scalar("class_loss", np.mean(class_cost_vector), epoch)
        writer.add_scalar("cos_sim_loss", np.mean(cos_sim_cost_vector), epoch)
        writer.add_scalar("ocr_text_sim_loss", np.mean(ocr_text_cos_sim_cost_vector), epoch)
        writer.add_scalar("ocr_img_sim_loss", np.mean(ocr_img_cos_sim_cost_vector), epoch)

        pictuTrainLoss.append(np.mean(cost_vector))
        pictuTrainACC.append(np.mean(acc_vector))
        # pictuvaliACC.append(validate_acc)
        # pictuvaliLoss.append(validate_loss)
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

    # epochs = []
    # for i in range(args.num_epochs):
    #     epochs.append(i+1)

    # plt.subplot(1, 2, 1)
    # plt.plot(epochs, pictuTrainACC, color='r', label='acc')  # r表示红色
    # plt.xlabel('epochs')  # x轴表示
    # plt.ylabel('acc')  # y轴表示
    # plt.title("Train_Acc")  # 图标标题表示
    # plt.legend()  # 每条折线的label显示
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(epochs, pictuTrainLoss, color=(0, 0, 0), label='loss')  # 也可以用RGB值表示颜色
    # plt.xlabel('epochs')  # x轴表示
    # plt.ylabel('loss')  # y轴表示
    # plt.title("Train_Loss")  # 图标标题表示
    # plt.legend()  # 每条折线的label显示
    #
    # plt.savefig('../Data/photo/train.jpg')  # 保存图片，路径名为train.jpg
    # #plt.show()  # 显示图片

    # print('TrainLoss ：', pictuTrainLoss)
    # print('TrainACC ：', pictuTrainACC)
    # print('ValiACC ：', pictuvaliACC)
    # print('ValiLoss ：', pictuvaliLoss)

    # Test the Model
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
    for i, (test_data, test_labels, event_labels) in enumerate(test_loader):
        test_text, test_image, test_ocr, test_mask, test_labels = to_var(
            test_data[0]), to_var(test_data[1]), to_var(test_data[2]), to_var(test_data[3]), to_var(test_labels)
        test_outputs, distance, distance_ocr_, distance_ocr_img = model(test_text, test_image, test_ocr,
                                                                        test_mask)
        _, test_argmax = torch.max(test_outputs, 1)
        if i == 0:
            test_score = to_np(test_outputs.squeeze())
            test_pred = to_np(test_argmax.squeeze())
            test_true = to_np(test_labels.squeeze())
        else:
            test_score = np.concatenate((test_score, to_np(test_outputs)), axis=0)
            test_pred = np.concatenate((test_pred, to_np(test_argmax)), axis=0)
            test_true = np.concatenate((test_true, to_np(test_labels)), axis=0)

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
    parser.add_argument('--batch_first', type=bool, default=True, help='')
    parser.add_argument('--bidirectional', type=bool, default=True, help='')
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='')  # 原0.001
    parser.add_argument('--event_num', type=int, default=10, help='')
    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')

    #    args = parser.parse_args()
    return parser


# def get_top_post(output, label, test_id, top_n=500):
#     filter_output = []
#     filter_id = []
#     for i, l in enumerate(label):
#         # print(np.argmax(output[i]))
#         if np.argmax(output[i]) == l and int(l) == 1:
#             filter_output.append(output[i][1])
#             filter_id.append(test_id[i])
#
#     filter_output = np.array(filter_output)
#
#     top_n_indice = filter_output.argsort()[-top_n:][::-1]
#
#     top_n_id = np.array(filter_id)[top_n_indice]
#     top_n_id_dict = {}
#     for i in top_n_id:
#         top_n_id_dict[i] = True
#
#     pickle.dump(top_n_id_dict, open("../Data/weibo/top_n_id.pickle", "wb"))
#
#     return top_n_id


# def word2vec(post, word_id_map, W):
#     word_embedding = []
#     mask = []
#     # length = []
#
#     for sentence in post:
#         sen_embedding = []
#         seq_len = len(sentence) - 1
#         mask_seq = np.zeros(args.sequence_len, dtype=np.float32)
#         mask_seq[:len(sentence)] = 1.0
#         for i, word in enumerate(sentence):
#             sen_embedding.append(word_id_map[word])
#
#         while len(sen_embedding) < args.sequence_len:
#             sen_embedding.append(0)
#
#         word_embedding.append(copy.deepcopy(sen_embedding))
#         mask.append(copy.deepcopy(mask_seq))
#         # length.append(seq_len)
#     return word_embedding, mask


def re_tokenize_sentence(flag):
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    tokenized_texts = []
    original_texts = flag['original_post']
    ocr = flag['ocr']
    tokenized_ocrs = []
    for sentence in original_texts:
        tokenized_text = tokenizer.encode(sentence)
        tokenized_texts.append(tokenized_text)
    flag['post_text'] = tokenized_texts
    for sentences in ocr:
        tokenized_ocr = tokenizer.encode(sentences)
        tokenized_ocrs.append(tokenized_ocr)
    flag['ocr'] = tokenized_ocrs


def get_all_text(train, validate, test):
    all_text = list(train['post_text']) + list(validate['post_text']) + list(test['post_text'])
    return all_text


def align_data(flag, args):
    text = []
    ocr = []
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

    # for sentences in flag['ocr']:
    #     sentences=sentences[:10]
    #     print(sentences)
    #     sen_embeddings = []
    #     for i, words in enumerate(sentences):
    #         if len(sen_embeddings) < args.sequence_len:
    #             sen_embeddings.append(words)
    #             sen_embeddings.append(0)
    #         else:
    #             sen_embeddings.append(words)
    #
    #     ocr.append(copy.deepcopy(sen_embeddings))
    # flag['ocr'] = ocr

    for sentences in flag['ocr']:
        sentences = sentences[:args.sequence_len]
        sen_embeddings = []
        for i, words in enumerate(sentences):
            sen_embeddings.append(words)

        while len(sen_embeddings) < args.sequence_len:
            sen_embeddings.append(0)

        ocr.append(copy.deepcopy(sen_embeddings))

    flag['ocr'] = ocr


def load_data(args):
    train, validate, test = process_data.get_data(args.text_only)

    re_tokenize_sentence(train)
    re_tokenize_sentence(validate)
    re_tokenize_sentence(test)
    all_text = get_all_text(train, validate, test)
    max_len = len(max(all_text, key=len))
    print(max_len)
    args.sequence_len = max_len
    align_data(train, args)
    align_data(validate, args)
    align_data(test, args)
    print((np.array(train['ocr'])).shape)
    print((np.array(train['post_text'])).shape)
    return train, validate, test


#
# def transform(event):
#     matrix = np.zeros([len(event), max(event) + 1])
#     # print("Translate  shape is " + str(matrix))
#     for i, l in enumerate(event):
#         matrix[i, l] = 1.00
#     return matrix


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parser = parse_arguments(parse)
    train = ''
    test = ''
    output = '../Data/result/'
    args = parser.parse_args([train, test, output])

    main(args)
