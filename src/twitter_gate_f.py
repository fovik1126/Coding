import argparse
import copy
import os
import random
import warnings

import torchvision
from sklearn import metrics
from torch.autograd import Function
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from transformers import *

# import random
import process_twitter as process_data
from gate_module import *
from src.early_stopping import EarlyStopping

# from logger import Logger

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seed_value = 3407  # 设定随机数种子

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
        self.ocr = torch.from_numpy(np.array(dataset['ocr']).astype(float))
        # self.social_context = torch.from_numpy(np.array(dataset['social_feature']))
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


dtype = torch.FloatTensor


class BiLSTM(nn.Module):
    def __init__(self):
        self.n_class = 768
        self.n_hidden = 384
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=self.n_class, hidden_size=self.n_hidden, bidirectional=True)
        # fc
        self.fc = nn.Linear(self.n_hidden * 2, self.n_class)

    def forward(self, X):
        # X: [batch_size, max_len, n_class]
        batch_size = X.shape[0]
        input = X.transpose(0, 1)  # input : [max_len, batch_size, n_class]

        hidden_state = torch.randn(1 * 2, batch_size,
                                   self.n_hidden).cuda()  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.randn(1 * 2, batch_size,
                                 self.n_hidden).cuda()  # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs[-1]  # [batch_size, n_hidden * 2]
        model = self.fc(outputs)  # model : [batch_size, n_class]
        return model


# Neural Network Model (1 hidden layer)
class CNN_Fusion(nn.Module):
    def __init__(self, args):
        super(CNN_Fusion, self).__init__()
        self.args = args
        self.GatedFusionGlobal = GatedFusionGlobal(32, 4, 0.0)
        self.GatedFusionGlobal2 = GatedFusionGlobal2(32, 4, 0.0)
        self.event_num = args.event_num
        self.fc_out = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.01),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        vocab_size = args.vocab_size
        emb_dim = args.embed_dim

        C = args.class_num
        self.hidden_size = args.hidden_dim
        self.lstm_size = args.embed_dim
        self.social_size = 19
        self.final_hid = 32
        self.drop_rate = 0.2
        self.att_hid = 32
        self.head = 1
        # bert
        bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_hidden_size = args.bert_hidden_dim
        self.fc2 = nn.Linear(self.bert_hidden_size, self.hidden_size)

        for param in bert_model.parameters():
            param.requires_grad = False
        self.bertModel = bert_model
        self.self_text_att_average = nn.Sequential(
            nn.Linear(32, self.final_hid),
            nn.BatchNorm1d(87),
            nn.ReLU()
        )
        self.self_text_re_Dro = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(87),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )
        self.cross_text_re_Dro = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(87),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )
        self.cross_img_text_re_Dro = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(87),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )
        # 层标准化
        self.self_text_layer_norm = nn.LayerNorm([87, 32])
        self.self_layer_norm = nn.LayerNorm([87, 32])
        self.cross_layer_norm = nn.LayerNorm([64, 32])
        # flatten_self
        self.self_text_faltten = nn.Sequential(
            nn.Linear(87 * 32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.self_img_text_faltten = nn.Sequential(
            nn.Linear(87 * 32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.self_text_query = nn.Linear(768, int(self.att_hid / self.head))
        self.self_text_key = nn.Linear(768, int(self.att_hid / self.head))
        self.self_text_value = nn.Linear(768, int(self.att_hid / self.head))

        self.fc_out_com = nn.Linear(96, 256)

        self.dropout = nn.Dropout(args.dropout)

        # IMAGE
        # hidden_size = args.hidden_dim
        vgg_19 = torchvision.models.vgg19(pretrained=True)
        for param in vgg_19.parameters():
            param.requires_grad = False
        # visual model
        num_ftrs = vgg_19.classifier._modules['6'].out_features
        self.image_cross_fc = nn.Linear(num_ftrs, 2048)
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
            nn.BatchNorm1d(87),
            nn.ReLU()
        )

        self.cross_img_att_average = nn.Sequential(
            nn.Linear(32, self.final_hid),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        self.self_img_text_att_average = nn.Sequential(
            nn.Linear(32, self.final_hid),
            nn.BatchNorm1d(87),
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

        self.vgg = vgg_19
        self.image_fc1 = nn.Linear(num_ftrs, self.hidden_size)

        self.image_adv = nn.Linear(self.hidden_size, int(self.hidden_size))
        self.image_encoder = nn.Linear(self.hidden_size, self.hidden_size)

        # resnet

        Resnet_50 = torchvision.models.resnet50(pretrained=True)
        num_ftrs2 = Resnet_50.fc.out_features
        for param in Resnet_50.parameters():
            param.requires_grad = False
        self.resnet50 = Resnet_50
        self.image_fc3 = nn.Linear(num_ftrs2, self.hidden_size)
        # Class  Classifier
        self.class_Multimodal_classifier = nn.Sequential()
        self.class_Multimodal_classifier.add_module('c_fc1', nn.Linear(11 * self.hidden_size, 2))
        self.class_Multimodal_classifier.add_module('c_softmax', nn.Softmax(dim=1))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(3 * self.hidden_size, 2))
        self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))
        # self.class_classifier = nn.Sequential()
        # self.class_classifier.add_module('c_fc1', nn.Linear(2 * self.hidden_size, 2))
        # self.class_classifier.add_module('c_bn1', nn.BatchNorm2d(100))
        # self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        # self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        # self.class_classifier.add_module('c_fc2', nn.Linear(self.hidden_size, 2))
        # self.class_classifier.add_module('c_bn2', nn.BatchNorm2d(self.hidden_size))
        # self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        # self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        # self.class_classifier.add_module('c_softmax', nn.Softmax(dim=1))
        self.multi = nn.Linear(64, 87)

        self.multi_text = nn.Linear(87, 64)
        # Event Classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(2 * self.hidden_size, self.hidden_size))
        # self.domain_classifier.add_module('d_fc1', nn.Linear(self.hidden_size, self.hidden_size))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(self.hidden_size))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(self.hidden_size, self.event_num))
        self.domain_classifier.add_module('d_softmax', nn.Softmax(dim=1))
        from resnet_cbam import resnet50_cbam
        self.resnet50_cbam = resnet50_cbam(pretrained=True)
        for name, para in self.resnet50_cbam.named_parameters():
            if "ca" or "sa" not in name:
                para.requires_grad_(False)
        # def init_hidden(self, batch_size):
        #     # Before we've done anything, we dont have any hidden state.
        #     # Refer to the Pytorch documentation to see exactly
        #     # why they have this dimensionality.
        #     # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        #     return (to_var(torch.zeros(1, batch_size, self.lstm_size)),
        #             to_var(torch.zeros(1, batch_size, self.lstm_size)))
        #
        # def conv_and_pool(self, x, conv):
        #     x = F.relu(conv(x)).squeeze(3)  # (sample number,hidden_dim, length)
        #     # x = F.avg_pool1d(x, x.size(2)).squeeze(2)
        #     x = F.max_pool1d(x, x.size(2)).squeeze(2)
        #
        #     return x
        self.BiLSTM = BiLSTM().cuda()

        self.drop_rate = 0.2
        self.att_hid = 32
        self.head = 1
        self.final_hid = 32
        self.self_text_att_average = nn.Sequential(
            nn.Linear(32, self.final_hid),
            nn.BatchNorm1d(87),
            nn.ReLU()
        )
        self.self_text_re_Dro = nn.Sequential(
            nn.Linear(32, 32),
            nn.BatchNorm1d(87),
            nn.LeakyReLU(),
            nn.Dropout(self.drop_rate)
        )
        # 层标准化
        self.self_text_layer_norm = nn.LayerNorm([87, 32])
        # flatten_self
        self.self_text_faltten = nn.Sequential(
            nn.Linear(87 * 32, 32),
            nn.BatchNorm1d(32),
            nn.ReLU()
        )
        self.self_text_query = nn.Linear(768, int(self.att_hid / self.head))
        self.self_text_key = nn.Linear(768, int(self.att_hid / self.head))
        self.self_text_value = nn.Linear(768, int(self.att_hid / self.head))

    def forward(self, text, image, ocr, mask):
        # IMAGE
        head = 1
        att_layers = 1
        att_hid = 32
        images = self.resnet50_cbam(image)  # [N, 512]
        # image = self.image_fc1(image)
        image = F.relu(self.dropout(self.image_fc3(images)))

        image_cross = self.image_cross_fc(images)  # [32, 2048]
        image_cross = torch.reshape(image_cross, (image_cross.shape[0], -1, 32))
        # Text
        # text = self.bertModel(text)[0]  # torch.Size([32, 192, 768])

        # #text = self.BiLSTM.forward(text)  # torch.Size([32, 768])
        # last_hidden_state = torch.mean(self.bertModel(text)[0], dim=1, keepdim=False)
        # #text = self.fc2(last_hidden_state)
        # text = F.relu(self.dropout(self.fc2(last_hidden_state)))
        out = self.bertModel(text, output_attentions=True)  # berts输出
        hidden_states = out[0]  # 最后一层的输出
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
        self_text_att1 = self_text_att
        self_text_att = torch.flatten(self_text_att, start_dim=1, end_dim=2)  # end_dim=-1
        self_text_att = self.self_text_faltten(self_text_att)
        '''此时向量的维度是正常的'''
        text = self_text_att
        # print(self_text_att.shape)  # torch.Size([32, 32])
        '''______________________________交叉注意力机制__________________________________'''

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
        # text_image = torch.cat((text, image), 1)
        # OCR
        last_hidden_state = torch.mean(self.bertModel(ocr.long())[0], dim=1, keepdim=False)
        ocr = F.relu(self.dropout(self.fc2(last_hidden_state)))

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
#
#
# def split_train_validation(train, percent):
#     whole_len = len(train[0])
#
#     train_indices = (sample(range(whole_len), int(whole_len * percent)))
#     train_data = select(train, train_indices)
#     print("train data size is " + str(len(train[3])))
#     # print()
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
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, list(model.parameters())),
                                 lr=args.learning_rate, weight_decay=0.1)

    print("loader size " + str(len(train_loader)))
    best_validate_acc = 0.000
    best_test_acc = 0.000
    best_loss = 100
    best_validate_dir = ''
    best_list = [0, 0]

    print('training model')
    adversarial = True

    patience = 20  # 当验证集损失在连续20次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=True)  # 关于 EarlyStopping 的代码可先看博客后面的内容

    '''初始化设置寻找最优解'''
    pictuTrainLoss = []
    pictuTrainACC = []
    # Train the Model

    for epoch in range(args.num_epochs):

        p = float(epoch) / 100
        # lambd = 2. / (1. + np.exp(-10. * p)) - 1
        lr = 0.001 / (1. + 10 * p) ** 0.75
        cos_sim_cost_vector = []
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
        ocr_cos_sim_cost_vector = []
        ocr_text_cos_sim_cost_vector = []
        ocr_img_cos_sim_cost_vector = []
        for i, (train_data, train_labels, event_labels) in enumerate(train_loader):
            train_text, train_image, train_ocr, train_mask, train_labels, event_labels = \
                to_var(train_data[0]), to_var(train_data[1]), to_var(train_data[2]), to_var(train_data[3]), \
                to_var(train_labels), to_var(event_labels)
            # start_time = time.time()
            # Forward + Backward + Optimize
            optimizer.zero_grad()

            final_output, distance_ = model(train_text, train_image, train_ocr, train_mask)
            # Fake or Real loss
            # print(class_outputs.shape)  torch.Size([32, 2])
            # print(domain_outputs.shape)
            # print(train_labels) tensor([1, 0, 1, 0, 0, 0, 1, 0, 0, ...], device='cuda:0', dtype=torch.int32)
            # print(event_labels)
            # Multimodal_output Loss
            # class_loss = criterion(Multimodal_output, train_labels.long())
            class_loss = criterion(final_output, train_labels.long())
            # Unitimodal_output Loss
            domain_loss = criterion(distance_, train_labels.long())

            loss = 0.6 * class_loss + 0.4 * domain_loss  # 0.934
            # loss = class_loss + domain_loss #0.885
            # loss = class_loss + 0 * domain_loss #0.815
            # loss = 0 * class_loss + domain_loss #

            loss.backward()
            optimizer.step()
            _, argmax = torch.max(final_output, 1)

            cross_entropy = True

            if True:
                accuracy = (train_labels == argmax.squeeze()).float().mean()
            else:

                _, labels = torch.max(train_labels, 1)
                accuracy = (labels.squeeze() == argmax.squeeze()).float().mean()

            class_cost_vector.append(class_loss.item())
            domain_cost_vector.append(domain_loss.item())
            # cos_sim_cost_vector.append(cos_sim_loss.item())
            # ocr_text_cos_sim_cost_vector.append(ocr_text_cos_sim_loss.item())
            # ocr_img_cos_sim_cost_vector.append(ocr_img_cos_sim_loss.item())
            cost_vector.append(loss.item())
            acc_vector.append(accuracy.item())

        model.eval()
        validate_acc_vector_temp = []
        for i, (validate_data, validate_labels, event_labels) in enumerate(validate_loader):
            validate_text, validate_image, validate_ocr, validate_mask, validate_labels, event_labels = \
                to_var(validate_data[0]), to_var(validate_data[1]), to_var(validate_data[2]), to_var(validate_data[3]), \
                to_var(validate_labels), to_var(event_labels)
            validate_outputs, distance_ = model(validate_text, validate_image,
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

        print(
            'Epoch [%d/%d],  Loss: %.4f, Class Loss: %.4f, Train_Acc: %.4f,  Validate_Acc: %.4f.'
            % (
                epoch + 1, args.num_epochs, np.mean(cost_vector), np.mean(class_cost_vector),

                np.mean(acc_vector), validate_acc))
        # ---------------------------------------画图-------------------------
        writer = SummaryWriter(log_dir="../Data/weibo_visual2")
        writer.add_scalar("train_acc", np.mean(acc_vector), epoch)
        writer.add_scalar("validate_acc", validate_acc, epoch)
        writer.add_scalar("loss", np.mean(cost_vector), epoch)
        writer.add_scalar("class_loss", np.mean(class_cost_vector), epoch)
        # if validate_acc > best_validate_acc:
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
        test_outputs, distance_ = model(test_text, test_image, test_ocr,
                                        test_mask)
        _, test_argmax = torch.max(test_outputs, 1)
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
    # --------------------------------画图------------------------
    # writer.add_scalar("test_acc", test_accuracy, epoch)
    # writer.add_scalar("test_acc_roc", test_aucroc, epoch)
    writer.close()
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
    parser.add_argument('--d_iter', type=int, default=3, help='')
    parser.add_argument('--batch_size', type=int, default=32, help='')
    parser.add_argument('--num_epochs', type=int, default=100, help='')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='')
    parser.add_argument('--event_num', type=int, default=10, help='')

    parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')

    parser.add_argument('--unified_dim', type=int, default=768, help='')

    #    args = parser.parse_args()
    return parser


#
# def get_top_post(output, label, test_id, top_n=500):
#     filter_output = []
#     filter_id = []
#     # print(test_id)
#     # print(output)
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
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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
    train, validate, test, event_num = process_data.get_data(args.text_only)
    args.event_num = event_num
    re_tokenize_sentence(train)
    re_tokenize_sentence(validate)
    re_tokenize_sentence(test)
    all_text = get_all_text(train, validate, test)
    max_len = len(max(all_text, key=len))
    args.sequence_len = max_len
    align_data(train, args)
    align_data(validate, args)
    align_data(test, args)
    print((np.array(train['ocr'])).shape)
    print((np.array(train['post_text'])).shape)
    return train, validate, test


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
    output = '../Data/twitter/RESULT_text_image/'
    args = parser.parse_args([train, test, output])

    main(args)
