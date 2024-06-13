# '''1.包'''
from PIL import Image
from transformers import AutoConfig, TFAutoModel, AutoTokenizer, BertModel, BertTokenizer
from flask import Flask, request, render_template, flash, redirect, url_for
import numpy as np
import torch
from torchvision import datasets, models, transforms
import base64
import weibo_predict

# 2.初始化模型， 避免在函数内部初始化，耗时过长
# from src.sci import weibopredict
# MODEL_NAME = 'bert-base-chinese'
# MODEL_NAME = '/root/autodl-tmp/pretrain_model/bert-base-chinese'
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
# tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
# from src.test_good import  predict2weibo_
# from src.test_good import mbpam_decoder_predict


# 3.初始化flask
app = Flask(__name__)

# 4. 数据库存储
# from model import *

# 4.启动服务后会进到一个起始页
'''起始页'''


@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("index.html")


@app.route("/index", methods=["GET", "POST"])
def index():
    return render_template("index.html")



@app.route("/boot", methods=["GET", "POST"])
def localization():
    return render_template("boot.html")

@app.route("/login", methods=["GET", "POST"])
def login():

        username = request.form['username']
        password = request.form['password']
        print(password)
        if not username or not password:
            flash('Invalid input.')
            return redirect(url_for('login'))
        # user = User.query.filter_by(count=username).first()
        user = {
            'username': 'admin',
            'password': 'admin'
        }
        if not user:
            print("用户不存在")
            return redirect(url_for('index'))
        if username == user["username"] and user["password"] == password:
            # login_user(user)  # 登入用户
            print('Login success.')
            print("登录成功")
            return redirect(url_for('boot'))  # 重定向到主页

        # flash('Invalid username or password.')  # 如果验证失败，显示错误消息
        # return redirect(url_for('template.login'))  # 重定向回登录页面
        return render_template('boot.html')


@app.route("/unlogin", methods=["GET", "POST"])
def unlogin():
    return render_template("index.html")

# 5.上传数据
@app.route('/up_photo', methods=['GET', 'post'])
def up_photo():
        '''图像处理预先加载'''
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


        # 5.1获取文本内容，并预处理
        sentence = request.form.get('text')  # 传入表单对应输入字段的 name 值

        print("ssss:", sentence)
        tokenized_text = tokenizer.encode(sentence)  # 编码
        sequence_len = 192
        sen_embedding = []  # 一个句子用一个list
        mask_seq = np.zeros(sequence_len, dtype=np.float32)
        mask_seq[:len(sentence)] = 1.0  # mask所有的数据都覆为1.0
        for i, word in enumerate(tokenized_text):  # 把句子中的每一个单词进行列举
            sen_embedding.append(word)  # 嵌入向量进行append
        '''不满足最大长度的向量，进行补0操作，一直循环补0，补到最大长度为止'''
        while len(sen_embedding) < sequence_len:
            sen_embedding.append(0)
        print('原始文本', sentence)
        print(mask_seq)
        print(sen_embedding)
        sen_embedding = torch.from_numpy(np.array(sen_embedding))
        mask_seq = torch.from_numpy(np.array(mask_seq))
        print(sen_embedding)
        # 5.2获取图像内容，进行预处理
        img1 = request.files['photo']
        print(type(img1))
        img = Image.open(img1.stream)
        print(sentence, img)
        im = img.convert('RGB')
        '''图像处理成tensor'''
        im = data_transforms(im)
        print(im.shape)

        # 5.3获取OCR内容，进行预处理
        sentence_ocr = request.form.get('textOCR')  # 传入表单对应输入字段的 name 值
        print("ssss:", sentence_ocr)
        tokenized_text_ocr = tokenizer.encode(sentence_ocr)  # 编码
        sequence_len_ocr = 192
        sen_embedding_ocr = []  # 一个句子用一个list
        mask_seq_ocr = np.zeros(sequence_len_ocr, dtype=np.float32)
        mask_seq_ocr[:len(sentence_ocr)] = 1.0  # mask所有的数据都覆为1.0
        for i, word in enumerate(tokenized_text_ocr):  # 把句子中的每一个单词进行列举
            sen_embedding_ocr.append(word)  # 嵌入向量进行append
        '''不满足最大长度的向量，进行补0操作，一直循环补0，补到最大长度为止'''
        while len(sen_embedding_ocr) < sequence_len_ocr:
            sen_embedding_ocr.append(0)
        print('原始OCR', sentence_ocr)
        print(mask_seq_ocr)
        print(sen_embedding_ocr)
        sen_embedding_ocr = torch.from_numpy(np.array(sen_embedding_ocr))
        mask_seq_ocr = torch.from_numpy(np.array(mask_seq_ocr))
        print(sen_embedding_ocr)
        context = {
            'text': sentence,
            'tensor_mask': mask_seq,
            'tensor_text': sen_embedding,
            'image': img1,
            'tensor_image': im,
            'ocr': sentence_ocr,
            'tensor_mask_ocr': mask_seq_ocr,
            'tensor_ocr': sen_embedding_ocr,
        }
        # 将接受的到数据处理完成的送入到展示页面

        return render_template('process.html', context=context)


# 预测新闻
@app.route('/predict', methods=['GET', 'POST'])
def predict_news():

        sentence = request.form.get('text')  # 传入表单对应输入字段的 name 值
        sentence_ocr = request.form.get('textOCR')
        sentence_ocr = str(sentence_ocr)
        print('原始文本', sentence)
        print('原始文本_ocr', sentence_ocr)
        '''图像'''
        img1 = request.files['photo']

        # 存到数据库
        img2 = request.files['photo'].read()
        img3 = base64.b64encode(img2)
        # print(img3)
        print(type(img3))
        img = Image.open(img1.stream)
        print(sentence, img)
        im = img.convert('RGB')
        data_transforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])  # '''图像处理成tensor'''
        im = data_transforms(im)
        print(im.shape)
        # 将预处理完成的数据，送入模型进行检测。
        result = weibo_predict.predict(text=sentence, img=im, ocr=sentence_ocr)  # 会返回结果
        context = {
            'text': sentence,
            'image': img,
            'ocr': sentence_ocr,
            'result_label': result
        }
        print(result)

        # # 数据库存储
        # news_sql = news_(text=sentence, image=img3, result=result)
        # db.session.add(news_sql)  # 添加数据
        # db.session.commit()  # 数据提交

        return render_template('predict2.html', context=context)


# 将数据加载到html页面进行展示
# @app.route('/check_data', methods=['GET', 'post'])
# def check_data():
#     '''图像处理预先加载'''
#     print('ok')
#     all_news = news_.query.all()
#     print(all_news)
#     for i in all_news:
#         print(i.text)
#     print("数据")
#     context = all_news
#     return render_template('all_data.html', context=context)

# python3 -m flask run


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=6006, debug=True, use_reloader=False)
