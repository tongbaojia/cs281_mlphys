import pandas as pd
from time import time
import numpy as np
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors
from collections import defaultdict
from tqdm import tqdm
import xgboost
import itertools

print 'started'
t0 = time()
df = pd.read_csv('../dark_matter_data/train.csv', index_col=0)
t1 = time()
print 'time to load', t1-t0

df_play = df[:1000000]

info_by_board = defaultdict(list)
for da in tqdm(df_play.values):
	info_by_board[da[-1]].append(da[:-1])

all_X = []; all_Y = [];
for k in tqdm(info_by_board):
	data = np.array(info_by_board[k])
	nbrs = NearestNeighbors(n_neighbors = 5, algorithm='ball_tree')
	nbrs.fit(data[:,1:4])
	for ix,indices in enumerate(nbrs.kneighbors(data[:,1:4])[1]):
		sum_of_chi2 = sum(data[indices][:,-2])
		position_vel = data[ix][1:7]
		x_vec = np.append(position_vel, sum_of_chi2)
		all_X.append(x_vec)
		all_Y.append(data[ix][-1])
all_X = np.array(all_X)
all_Y = np.array(all_Y)


train_X, test_X, train_Y, test_Y = train_test_split(all_X, all_Y, test_size=.3, random_state=35)

# max_depth_arr = [3,5,7,9]
# min_child_weight_arr = [3,5,7,9]
# gamma_arr = [0.0,.1,.2,.3,.4]
#for gamma in gamma_arr:
model = xgboost.XGBClassifier(max_depth=9, min_child_weight=5, learning_rate=.1, gamma=0, subsample=.8,
	colsample_bytree=.8, scale_pos_weight = 1)
#model = RandomForestClassifier(n_estimators=128, verbose=1, n_jobs = -1)
print 'fitting'
t0 = time()
model.fit(train_X, train_Y)
print time() - t0

preds_train = model.predict_proba(train_X)[:,1]
preds_test = model.predict_proba(test_X)[:,1]
train_score = roc_auc_score(train_Y, preds_train)
test_score = roc_auc_score(test_Y, preds_test)

print 'train score', train_score
print 'test_score', test_score
print ''


print 'started'
t0 = time()
df_submit = pd.read_csv('../dark_matter_data/test.csv', index_col=0)
t1 = time()
print 'time to load', t1-t0

X_submit = df_submit.values[:,:6]
preds_submit = model.predict_proba(X_submit)[:,1]

submission = pd.DataFrame(preds_submit, columns=['Prediction'])
submission.index.name = 'Id'
submission.to_csv('xgblocal4.csv', header=True)



