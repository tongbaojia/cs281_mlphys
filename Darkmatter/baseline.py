import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import seaborn
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection


BRICK_X = 124000
BRICK_Y = 99000
BRICK_Z = 75000
SAFE_M = 10000
dZ = 205

df = pd.read_csv('dark_matter_data/train.csv', index_col=0)

plt.figure(dpi = 80, figsize=(5,4))
first = plt.hist(df.chi2[df.signal == True], normed = True, label = 'signal')
second = plt.hist(df.chi2[df.signal == False], normed = True, label = 'background')
plt.title('chi2 distribution')
plt.legend()
plt.show()

def plot_bg_and_mc(pbg, pmc, id=0, step=1):
    df = pbg
    # mind the order!
    d0 = pd.DataFrame([
                df['Z'][::step],
                df['X'][::step],
                df['Y'][::step]],
                index=['z', 'x', 'y']).T
    numtracks = d0.shape[0]
    dd = pd.DataFrame([
            df['TX'][::step]*dZ,
            df['TY'][::step]*dZ],
            index=['x', 'y']).T
    dd.insert(loc=0, column='z', value=dZ)
    d1 = d0 + dd
    # print d0, d1
    C = plt.cm.Blues(0.5)
    lc_bg = Line3DCollection(list(zip(d0.values, d1.values)), colors=C, alpha=0.5, lw=2)

    df = pmc

    # mind the order!
    d0 = pd.DataFrame([
                df['Z'],
                df['X'],
                df['Y']],
                index=['z', 'x', 'y']).T
    numtracks = d0.shape[0]
    dd = pd.DataFrame([
            df['TX']*dZ,
            df['TY']*dZ],
            index=['x', 'y']).T
    dd.insert(loc=0, column='z', value=dZ)
    d1 = d0 + dd
    print(d1.shape)
    C = plt.cm.Reds(0.5)
    lc_mc = Line3DCollection(list(zip(d0.values, d1.values)), colors=C, alpha=0.9, lw=2)
    
    fig = plt.figure(figsize=(12,8))
    ax = fig.gca(projection='3d')
    ax.view_init(azim=-50, elev=10)
    ax.add_collection3d(lc_mc)
    ax.add_collection3d(lc_bg)
    
    # mind the order!
    ax.set_xlabel("z")
    ax.set_ylabel("x")
    ax.set_zlabel("y")  
    ax.set_xlim(0, BRICK_Z)
    ax.set_ylim(0, BRICK_X)
    ax.set_zlim(0, BRICK_Y)


plot_bg_and_mc(df[np.logical_and(df.brick_number == 1, df.signal == 0)], df[np.logical_and(df.brick_number == 1, df.signal == 1)])

train_df = df[df.signal == True].values[:, 1:8]
train_df = np.concatenate([train_df, df[df.signal == False].values[:150000, 1:8]])
len(train_df)
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth = 5, n_estimators = 100, n_jobs = -1)
print len(train_df)
clf.fit(train_df[:, :-1], np.array(train_df[:, -1], dtype = int))
df_test = pd.read_csv('dark_matter_data/test.csv', index_col=0)
df_test.head()
prediction = clf.predict_proba(df_test.values[:, :-1])[:, 1]
baseline = pd.DataFrame(prediction, columns=['Prediction'])
baseline.index.name = 'Id'
baseline.to_csv('baseline.csv', header=True)