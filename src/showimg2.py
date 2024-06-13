import matplotlib.pyplot as plt
import numpy as np


#示例数据
x_values = ['Weibo', 'Twitter']

y_values1 = np.array([0.877, 0.901])
y_values2 = np.array([0.880, 0.905])
y_values3 = np.array([0.895, 0.912])
y_values5 = np.array([0.908, 0.934])
plt.rcParams['font.family'] = 'Times New Roman'
fig, ax1 = plt.subplots(1)  # 调整图的大小

bar_width = 0.2  # 调整柱子宽度

# 修改颜色
# color1 = '#1f78b4'  # 蓝色
# color2 = '#33a02c'  # 绿色
# color3 = '#e31a1c'  # 红色


color1 = '#e57373'  # 钢蓝色
color2 = '#00b050'  # 紫罗兰色
color3 = '#ffc000'  # 深绿色
color4 = '#5cbfea'  # 深海蓝色
color5 = '#ffc000'  # 紫色
color6 = '#c98eef'  # 橄榄绿色
color7 = '#f48978'  # 棕色


bars2 = ax1.bar(np.arange(len(x_values)) - 1.25 * bar_width, y_values1, width=bar_width,
                label='alignment',
                color=color1, edgecolor='black')
bars3 = ax1.bar(np.arange(len(x_values)) - 0.25 * bar_width, y_values2, width=bar_width,
                label='alignment+interactive', color=color2,
                edgecolor='black')

bars5 = ax1.bar(np.arange(len(x_values)) + 0.75 * bar_width, y_values3, width=bar_width, label='alignment+interactive+selective',
                color=color3, edgecolor='black')
bars6 = ax1.bar(np.arange(len(x_values)) + 1.75 * bar_width, y_values5, width=bar_width, label='Ours', color=color6,
                edgecolor='black')

# 调整两组柱状图的位置
bar_group_width = 1.8
ax1.set_xticks(np.arange(len(x_values)))
ax1.set_xticklabels(x_values, fontsize=14)

ax1.set_ylabel('Accuracy (%)', fontsize=16)
# ax1.set_xlabel('Weibo', fontsize=16)
# ax1.set_xticks([])
# 设置y轴刻度范围
ax1.set_ylim([0.85, 0.95])

# 添加数据标签
for bars, label in zip([ bars2, bars3, bars5, bars6],
                       ['alignment’',
                        'alignment+interactive', 'alignment+interactive+selective', 'Ours']):
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval * 100:.1f}', va='bottom', ha='center', fontsize=12,
                 color='black')

# 设置图例样式和位置
ax1.legend(loc='upper left', fontsize=10)
#
# 添加网格线
# plt.grid(True, axis='y', linestyle='--', alpha=0.7)
#
# 调整边框和背景颜色
# ax1.spines['top'].set_visible(True)
# ax1.spines['right'].set_visible(True)
# ax1.spines['bottom'].set_color('#DDDDDD')
# ax1.spines['left'].set_color('#DDDDDD')
# ax1.set_facecolor('#FFFFFF')
#
# 显示图表
plt.tight_layout()
plt.savefig('cn3.png', dpi=400, bbox_inches='tight')  # 降低dpi以减小文件大小
plt.show()
#
#
#
# x_values = ['Weibo', 'Twitter']
#
#
# y_values2 = np.array([0.896, 0.913])
# y_values3 = np.array([0.901, 0.925])
# y_values5 = np.array([0.908, 0.934])
# plt.rcParams['font.family'] = 'Times New Roman'
# fig, ax1 = plt.subplots(1)  # 调整图的大小
#
# bar_width = 0.2  # 调整柱子宽度
#
# # 修改颜色
# # color1 = '#1f78b4'  # 蓝色
# # color2 = '#33a02c'  # 绿色
# # color3 = '#e31a1c'  # 红色
#
#
# color1 = '#e57373'  # 钢蓝色
# color2 = '#00b050'  # 紫罗兰色
# color3 = '#ffc000'  # 深绿色
# color4 = '#5cbfea'  # 深海蓝色
# color5 = '#ffc000'  # 紫色
# color6 = '#c98eef'  # 橄榄绿色
# color7 = '#f48978'  # 棕色
# color8 = '#E29607'
#
#
# bars3 = ax1.bar(np.arange(len(x_values)) - 1.25 * bar_width, y_values2, width=bar_width,
#                 label='-w/o match scores', color=color8,
#                 edgecolor='black')
#
# bars5 = ax1.bar(np.arange(len(x_values)) - 0.25 * bar_width, y_values3, width=bar_width, label='-r/w cosine scores',
#                 color=color4, edgecolor='black')
# bars6 = ax1.bar(np.arange(len(x_values)) + 0.75 * bar_width, y_values5, width=bar_width, label='Ours',
#                 color=color6,
#                 edgecolor='black')
#
# # 调整两组柱状图的位置
# bar_group_width = 1.8
# ax1.set_xticks(np.arange(len(x_values)))
# ax1.set_xticklabels(x_values, fontsize=14)
#
# ax1.set_ylabel('Accuracy (%)', fontsize=16)
# # ax1.set_xlabel('Weibo', fontsize=16)
# # ax1.set_xticks([])
# # 设置y轴刻度范围
# ax1.set_ylim([0.85, 0.95])
#
# # 添加数据标签
# for bars, label in zip([ bars3, bars5, bars6],
#                        ['-w/o match scores', '-r/w cosine scores','Ours']):
#     for bar in bars:
#         yval = bar.get_height()
#         ax1.text(bar.get_x() + bar.get_width() / 2.0, yval, f'{yval * 100:.1f}', va='bottom', ha='center', fontsize=12,
#                  color='black')
#
# # 设置图例样式和位置
# ax1.legend(loc='upper left', fontsize=10)
#
# # 添加网格线
# # plt.grid(True, axis='y', linestyle='--', alpha=0.7)
#
# # 调整边框和背景颜色
# # ax1.spines['top'].set_visible(True)
# # ax1.spines['right'].set_visible(True)
# # ax1.spines['bottom'].set_color('#DDDDDD')
# # ax1.spines['left'].set_color('#DDDDDD')
# # ax1.set_facecolor('#FFFFFF')
#
# # 显示图表
# plt.tight_layout()
# plt.savefig('cn4.png', dpi=400, bbox_inches='tight')  # 降低dpi以减小文件大小
# plt.show()

