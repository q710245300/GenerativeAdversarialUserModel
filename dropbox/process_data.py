import numpy as np
import pickle
import pandas as pd
import argparse

# 定义要执行的数据集文件
cmd_opt = argparse.ArgumentParser(description='Argparser for data process')
cmd_opt.add_argument('-dataset', type=str, default="yelp", help='choose rsc, tb, or yelp')
cmd_args = cmd_opt.parse_args()
print(cmd_args)


# The format of processed data:
# data_behavior[user][0] is user_id
# data_behavior[user][1][t] is displayed list at time t
# data_behavior[user][2][t] is picked id at time t

filename = './'+cmd_args.dataset+'.txt'

# 只选取数据集中的[Time, is_click, session_new_index, tr_val_tst, item_new_index]这5cols
# usecols选择列，但是好像没有顺序而言，虽然是7, 6,但是最终结果还是按照13567排列的
raw_data = pd.read_csv(filename, sep='\t', usecols=[1, 3, 5, 7, 6], dtype={1: int, 3: int, 7: int, 5:int, 6:int})

# drop_duplicates: 将subset对应的重复列去重，默认保留第一个
raw_data.drop_duplicates(subset=['session_new_index','Time','item_new_index','is_click'], inplace=True)
# sort_values按照by进行排序, inplce=True,在原数据上修改
raw_data.sort_values(by='is_click',inplace=True)
# drop_duplicates: keep=’last‘保留最后一项，删除前面的重复项，默认是first
raw_data.drop_duplicates(keep='last', subset=['session_new_index','Time','item_new_index'], inplace=True)

# 2019-7-30 modify the error that dataframe has no attribute nuique()
size_user = raw_data['session_new_index'].nunique()
size_item = raw_data['item_new_index'].nunique()

# 按照session_new_index进行分组
data_user = raw_data.groupby(by='session_new_index')
# 建立了一个存放list的list，内部list的数量和user_size一样
data_behavior = [[] for _ in range(size_user)]

train_user = []
vali_user = []
test_user = []

sum_length = 0
event_count = 0

# size_user=user的数量
for user in range(size_user):
    data_behavior[user] = [[], [], []]
    data_behavior[user][0] = user
    # 实际上在这里user就代替了sessionId了，
    data_u = data_user.get_group(user)
    # 将user对应的是train或者是valid或者是test对应的标号取出
    split_tag = list(data_u['tr_val_tst'])[0]
    if split_tag == 0:
        train_user.append(user)
    elif split_tag == 1:
        vali_user.append(user)
    else:
        test_user.append(user)

    data_u_time = data_u.groupby(by='Time')
    # time_set就是由几种时间, 使用set能够去重
    time_set = list(set(data_u['Time']))
    # 排个序，小在前大在后
    time_set.sort()

    true_t = 0
    for t in range(len(time_set)):
        # 把时间一样的分成一组display_set
        display_set = data_u_time.get_group(time_set[t])
        event_count += 1
        sum_length += len(display_set)

        # 每个display_set中is_click等于1的为当前sessiond的点击商品
        data_behavior[user][1].append(list(display_set['item_new_index']))
        data_behavior[user][2].append(int(display_set[display_set.is_click==1]['item_new_index']))

# 生成对角矩阵
new_features = np.eye(size_item)

filename = './'+cmd_args.dataset+'.pkl'
file = open(filename, 'wb')
# pickle.dump, 将对象保存到pickle中, protocol是存储格式
pickle.dump(data_behavior, file, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(new_features, file, protocol=pickle.HIGHEST_PROTOCOL)
file.close()

filename = './'+cmd_args.dataset+'-split.pkl'
file = open(filename, 'wb')
pickle.dump(train_user, file, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(vali_user, file, protocol=pickle.HIGHEST_PROTOCOL)
pickle.dump(test_user, file, protocol=pickle.HIGHEST_PROTOCOL)
file.close()
