import math

import torch
import torch.nn as nn


class LayerAttention(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(LayerAttention, self).__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_tensor):
        batch_size, seq_length, hidden_size = input_tensor.size()

        query = self.query(input_tensor)
        key = self.key(input_tensor)
        value = self.value(input_tensor)

        query = query.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        key = key.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)
        value = value.view(batch_size, seq_length, self.num_heads, self.head_size).transpose(1, 2)

        attention_scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_tensor = torch.matmul(attention_probs, value).transpose(1, 2).contiguous().view(
            batch_size, seq_length, hidden_size)

        attention_output = self.output(context_tensor)

        return attention_output, attention_probs

#
# import torch
# from transformers import BertModel
#
# # load pre-trained BERT model
# model = BertModel.from_pretrained('bert-base-uncased')
#
# # define attention module
# layer_attention = LayerAttention(hidden_size=768, num_heads=12)
#
# # define input text and tokenization
# text = "example input text"
# tokens = tokenizer(text, return_tensors='pt')
#
# # run BERT model and get hidden states
# import torch
#
# class BERTWithAttention(torch.nn.Module):
#     def __init__(self, bert_model):
#         super(BERTWithAttention, self).__init__()
#         self.bert_model = bert_model
#         self.attention_layer = torch.nn.Linear(self.bert_model.config.hidden_size, 1)
#
#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert_model(input_ids, attention_mask=attention_mask)
#         hidden_states = outputs.last_hidden_state
#         attention_weights = torch.softmax(self.attention_layer(hidden_states), dim=1)
#         context_vector = torch.sum(attention_weights * hidden_states, dim=1)
#         return context_vector
#
#
# import torch
# import torch.nn as nn
# from transformers import BertModel
#
#
# class BertAttentionFeatures(nn.Module):
#     def __init__(self, num_layers):
#         super().__init__()
#         self.num_layers = num_layers
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.attention_layers = nn.ModuleList([nn.MultiheadAttention(
#             embed_dim=self.bert.config.hidden_size,
#             num_heads=self.bert.config.num_attention_heads)
#             for _ in range(num_layers)])
#
#     def forward(self, input_ids, attention_mask):
#         # Get BERT model outputs
#         bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
#         sequence_output = bert_outputs.last_hidden_state  # shape: (batch_size, sequence_length, hidden_size)
#         # Initialize the attention layer outputs as the sequence output
#         attention_layer_outputs = [sequence_output]
#         # Loop through attention layers
#         for i in range(self.num_layers):
#             # Perform self-attention using the current layer's output as the input
#             attention_output, _ = self.attention_layers[i](
#                 query=attention_layer_outputs[-1],
#                 key=attention_layer_outputs[-1],
#                 value=attention_layer_outputs[-1])
#             # Add the attention layer's output to the list of layer outputs
#             attention_layer_outputs.append(attention_output)
#         # Return the outputs from all attention layers as features
#         features = torch.stack(attention_layer_outputs[1:],
#                                dim=1)  # shape: (batch_size, num_layers, sequence_length, hidden_size)
#         return features
# import torch
# from transformers import BertModel, BertTokenizer
#
# # 加载BERT模型和tokenizer
# model = BertModel.from_pretrained('bert-base-uncased')
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# # 添加注意力层
# class BertWithAttention(torch.nn.Module):
#     def __init__(self, bert_model):
#         super().__init__()
#         self.bert = bert_model
#         self.attention = torch.nn.Sequential(
#             torch.nn.Linear(768, 1),
#             torch.nn.Sigmoid()
#         )
#
#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids, attention_mask)
#         hidden_states = outputs[0]
#         attention_scores = self.attention(hidden_states)
#         attention_weights = attention_scores.squeeze(-1)
#         attention_weights = torch.nn.functional.softmax(attention_weights, dim=-1)
#         weighted_hidden_states = hidden_states * attention_weights.unsqueeze(-1)
#         weighted_sum = weighted_hidden_states.sum(dim=1)
#         return weighted_sum
#
# # 初始化带有注意力层的BERT模型
# model_with_attention = BertWithAttention(model)
#
# # 输入文本
# text = "This is an example sentence."
#
# # 对输入进行编码和嵌入
# inputs = tokenizer(text, return_tensors='pt')
#
# # 使用带有注意力层的BERT模型处理输入
# outputs = model_with_attention(**inputs)
#
# # 使用最终的特征表示执行下游任务
# # ...
