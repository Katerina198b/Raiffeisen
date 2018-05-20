import pandas as pd
import datetime as dt
import numpy as np
from tqdm import tqdm_notebook
import collections
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import lightgbm as lgb
from sklearn import linear_model
import xgboost as xgb
from multiprocessing import Pool
from pandas.api.types import CategoricalDtype
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint
from sklearn import metrics

m=3

def get_centermost_point(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    return tuple(centermost_point)

def get_cluster_size(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    cd = max([great_circle(point, centroid).m for point in cluster])
    return cd

def clear_bad_symbols(x, symbols):
        lenght = len(x)
        for sym in symbols:
            x = x.replace(sym, "")

        while len(x) != lenght:
            lenght = len(x)
            for sym in symbols:
                x = x.replace(sym, "")
        return x
    
def pre(df):
        df = df[df.country.isin(["RUS", "RU"])]
        df.city = df.city.apply(lambda x: clear_bad_symbols(str(x).upper(), ["\t", "\n", " ", "-", "-", ".", ",", "(", ")"]))

        city_d = {'******': 'NZ', '************': 'NZ', '***********': 'NZ', 'NEZADAN': 'NZ',\
          'BIKOVO': 'BYKOVO', 'CHEREPOVEC': 'CHEREPOVETS', 'ELEKTROSTALE': 'ELEKTROSTAL',\
          'Ekaterinburg': 'EKATERINBURG', 'ILINSKIY': 'ILYINSKIY', 'KAZANE': 'KAZAN',\
          'Kazan': 'KAZAN ', 'LIPECK': 'LIPETSK', 'LIVNE': 'LIVNY',\
          'LYUBERCE': 'LYUBERTSY', 'LYUBERCY': 'LYUBERTSY', 'Lyuberci': 'LYUBERTSY', 'LYUBERCI': 'LYUBERTSY',\
          'Lyubercy': 'LYUBERTSY', 'MALAHOVKA': 'MALAKHOVKA', 'MO': 'MOSCOW REGION',\
          'MOSKVA': 'MOSCOW', 'MITYSHCHY': 'MYTISHCHI', 'MYTISCHI': 'MYTISHCHI', \
          'MYTISHI': 'MYTISHCHI', 'Moscow': 'MOSCOW', 'Moskva': 'MOSCOW','MOSKVA': 'MOSCOW',\
          'Mytischi': 'MYTISHCHI', 'N NOVGOROD': 'N.NOVGOROD', 'N-NOVGOROD': 'N.NOVGOROD',\
          'NNovgorod': 'NNOVGOROD',  'NIZHNIJNOVGO': 'NNOVGOROD',\
          'NIZHNIYNOVGO': 'NNOVGOROD', 'NOVOKUIBESHEV': 'NOVOKUYBYSHEV', 'NOVOKUIBYSHEV': 'NOVOKUYBYSHEV',\
          'NOVOROSSIISK': 'NOVOROSSIYSK ', 'NOVOROSSYSK': 'NOVOROSSIYSK ', 'NOVOSIBISRK': 'NOVOSIBIRSK',\
           'ODINCOVO': 'ODINTSOVO', 'ORYEL': 'OREL',\
          'PERME': 'PERM', 'PETROZAVODS': 'PETROZAVODSK', 'PODOLESK': 'PODOLSK',\
          'ROSTOVONDON': 'ROSTOVNADON',  'ROSTOVONDON': 'ROSTOVNADON',\
          'SAINTPETERSB': 'SANKTPETERBU', 'SERPUHOV': 'SERPUKHOV', 'SOLNETCHNOGOR': 'SOLNECHNOGORS',\
          'SPB': 'SANKTPETERBU', 'STPETERBURG': 'SANKTPETERBU', 'STPETERSBURG': 'SANKTPETERBU',\
          'STPETERBURG': 'SANKTPETERBU', 'STPETERSBURG': 'SANKTPETERBU', 'STPETERSBURG': 'SANKTPETERBU',\
            'TOGLIATTI': 'TOLYATTI',\
          'TOLEYATTI': 'TOLYATTI', 'TROICK': 'TROITSK', 'VELNOVGOROD': 'VELIKIYNOVGO',\
          'VELIKYNOVGOR': 'VELIKIYNOVGO', 'VOLJSKII': 'VOLZHSKIY', 'VORONEJ': 'VORONEZH',\
          'VSEVOLOJSK': 'VSEVOLOZHSK', 'Volgograd': 'VOLGOGRAD', 'YAROSLAVLE': 'YAROSLAVL',\
          'ZHUKOVSKY': 'ZHUKOVSKIY', 'Zelenograd': 'ZELENOGRAD', 'STAREIOSKOL': 'STARYYOSKOL',\
          'STARIYOSKOL': 'STARYYOSKOL'}
        df = df.replace({"city": city_d})

        return df
    
def preprocess(train):
    train = train[~train.atm_address_lat.isnull() | ~train.pos_address_lat.isnull()]
    train = pre(train)
    train = train[train.country.isin(['RUS', 'RU'])]
    
    temp = train[~train.atm_address.isnull() & train.atm_address_lat.isnull()].atm_address.unique()
    lat_mapping = train[train.atm_address.isin(temp)][~train[train.atm_address.isin(temp)].atm_address_lat.isnull()].groupby('atm_address').atm_address_lat.median().to_dict()
    train['atm_address_lat_upd'] = train[train.atm_address_lat.isnull()]['atm_address'].map(lat_mapping)
    train.atm_address_lat = train.atm_address_lat.fillna(train.atm_address_lat_upd)
    
    temp = train[~train.atm_address.isnull() & train.atm_address_lon.isnull()].atm_address.unique()
    lon_mapping = train[train.atm_address.isin(temp)][~train[train.atm_address.isin(temp)].atm_address_lon.isnull()].groupby('atm_address').atm_address_lon.median().to_dict()
    train['atm_address_lon_upd'] = train[train.atm_address_lon.isnull()]['atm_address'].map(lon_mapping)
    train.atm_address_lon = train.atm_address_lon.fillna(train.atm_address_lon_upd)
    
    temp = train.groupby(['terminal_id', 'city']).apply(lambda row: row.terminal_id + row.city).reset_index(level=[0,1])
    temp.columns = ['terminal_id', 'city', 'terminal_id_upd']
    temp = temp.drop_duplicates()
    train_df = train.merge(temp, on=['terminal_id', 'city'], how='left')
    
    train_df['atm_address_lat'] = train_df['terminal_id_upd'].map(train_df.groupby('terminal_id_upd')['atm_address_lat'].median())
    train_df['atm_address_lon'] = train_df['terminal_id_upd'].map(train_df.groupby('terminal_id_upd')['atm_address_lon'].median())
    
    train_df['terminal_id'] = train_df['terminal_id_upd'] + train_df['pos_address_lat'].apply(str)
    train_df['terminal_id'].fillna(train_df['terminal_id_upd'], inplace=True)
    
    return train_df

def parse(x):
        try:
            return dt.datetime.strptime(x, "%Y-%m-%d")
        except: return np.nan

def count_mean_dist(lat, lon, lat_list, lon_list):
    lat_list, lon_list = np.unique(list(zip(lat_list, lon_list)), axis=0)[:,0],\
                         np.unique(list(zip(lat_list, lon_list)), axis=0)[:,1]
    return np.nanmedian(np.sqrt((lat_list-lat)**2+(lon_list-lon)**2))


def count_max_dist(lat, lon, lat_list, lon_list):
    lat_list1, lon_list1 = np.unique(list(zip(lat_list, lon_list)), axis=0)[:,0],\
                         np.unique(list(zip(lat_list, lon_list)), axis=0)[:,1]
    return np.nanvar(np.sqrt((lat_list1-lat)**2+(lon_list1-lon)**2))

def count_min_dist(lat, lon, lat_list, lon_list):
    lat_list1, lon_list1 = np.unique(list(zip(lat_list, lon_list)), axis=0)[:,0],\
                         np.unique(list(zip(lat_list, lon_list)), axis=0)[:,1]
    temp = np.sqrt((lat_list1-lat)**2+(lon_list1-lon)**2)
    if len(temp[np.where(temp!=0)]) == 0: return 0
    return np.nanmin(temp[np.where(temp!=0)])

def dist_features(df):
    ldf = df.copy()
    mapping = df.groupby("customer_id").pos_address_lat.apply(lambda x: np.array(x)).to_dict()
    ldf["customer_lat"] = ldf.customer_id.map(mapping)
    mapping = df.groupby("customer_id").pos_address_lon.apply(lambda x: np.array(x)).to_dict()
    ldf["customer_lon"] = ldf.customer_id.map(mapping)
    df["dist_to_mean"] = \
    ldf[["pos_address_lat", "pos_address_lon", 
         "customer_lat", "customer_lon"]].progress_apply(lambda x: count_mean_dist(x.pos_address_lat,
                                                                          x.pos_address_lon,
                                                                          x.customer_lat,
                                                                          x.customer_lon), axis=1)
    df["dist_to_min"] = \
    ldf[["pos_address_lat", "pos_address_lon", 
         "customer_lat", "customer_lon"]].progress_apply(lambda x: count_min_dist(x.pos_address_lat,
                                                                          x.pos_address_lon,
                                                                          x.customer_lat,
                                                                          x.customer_lon), axis=1)
    df["dist_to_max"] = \
    ldf[["pos_address_lat", "pos_address_lon", 
         "customer_lat", "customer_lon"]].progress_apply(lambda x: count_max_dist(x.pos_address_lat,
                                                                          x.pos_address_lon,
                                                                          x.customer_lat,
                                                                          x.customer_lon), axis=1)
    return df

def count_number(lat, lon, lat_list, lon_list):

    lat_list, lon_list = np.unique(list(zip(lat_list, lon_list)), axis=0)[:,0],\
                         np.unique(list(zip(lat_list, lon_list)), axis=0)[:,1]
    
    temp = np.sqrt((lat_list-lat)**2+(lon_list-lon)**2) < 0.02
    return (sum(temp)-1)/len(lat_list)

def number_near_terminals(df):
    ldf = df.copy()
    mapping = df.groupby("customer_id").pos_address_lat.apply(lambda x: np.array(x)).to_dict()
    ldf["customer_lat"] = ldf.customer_id.map(mapping)
    mapping = df.groupby("customer_id").pos_address_lon.apply(lambda x: np.array(x)).to_dict()
    ldf["customer_lon"] = ldf.customer_id.map(mapping)
    df["near_terminals_ratio"] = \
    ldf[["pos_address_lat", "pos_address_lon", 
         "customer_lat", "customer_lon"]].progress_apply(lambda x: count_number(x.pos_address_lat,
                                                                          x.pos_address_lon,
                                                                          x.customer_lat,
                                                                          x.customer_lon), axis=1)
    return df

def get_target(df):
    df["y_home"] = (np.sqrt((df.pos_address_lat-df.home_add_lat)**2+(df.pos_address_lon-df.home_add_lon)**2) < .02).astype(int)
    df["y_work"] = (np.sqrt((df.pos_address_lat-df.work_add_lat)**2+(df.pos_address_lon-df.work_add_lon)**2) < .02).astype(int)
    return df

def get_dummies_mcc(df, top_mcc):
    mapping = {}
    all_mcc = df.mcc.unique()
    for mcc in all_mcc:
        if mcc in top_mcc:

            mapping[mcc] = mcc
        else:
            mapping[mcc] = "OTHER"
    df["mcc_for_dummies"] = df.mcc.map(mapping)
    df = pd.get_dummies(df, columns=["mcc_for_dummies"])

    return df

def ter_using(df):
    ldf = df.copy()
    ldf = ldf.merge(ldf.groupby(["customer_id", 'terminal_id']).size().reset_index(name='count_ters'), left_on=['customer_id','terminal_id'], right_on=['customer_id','terminal_id']) 
    return ldf

def ter_using_upd(df):
    ldf = df.copy()
    ldf["pos"] = ldf[["pos_address_lat", "pos_address_lon"]].apply(lambda x: (x.pos_address_lat, x.pos_address_lon) , axis=1)

    ldf = ldf.merge(ldf.groupby(["customer_id", 'pos']).size().reset_index(name='count_ters'), left_on=['customer_id','pos'], right_on=['customer_id','pos']) 
    return ldf

def ter_using_w(df):
    ldf = df.copy()
    ldf = ldf.merge(ldf[ldf.week_day.isin([5,6])].groupby(["customer_id", 'terminal_id']).size().reset_index(name='count_ters_w'), how='left', left_on=['customer_id','terminal_id'], right_on=['customer_id','terminal_id']) 
    ldf.count_ters_w.fillna(0, inplace=True)
    return ldf

def get_freq_place(df, mcc_codes):
    mapping_all = df.groupby("customer_id").size().to_dict()
    df["merchants"] = df.customer_id.map(mapping_all)
    for code in tqdm_notebook(mcc_codes):
        mapping_size = df[df.mcc==code].groupby("customer_id").size().to_dict()
        df[str(code)+"_freq"] = df.customer_id.map(mapping_size).fillna(0)
        df[str(code)+"_freq"] /= df["merchants"]
    return df

def number_equal_coordinates(df):
    mapping = df.groupby(["pos_address_lat", "pos_address_lon"]).size().sort_values(ascending=False).to_dict()
    df["number_of_equal"] = df.pos.map(mapping)
    return df

def make_pairs(df):
    df["pos"] = df[["pos_address_lat", "pos_address_lon"]].apply(lambda x: (x.pos_address_lat, x.pos_address_lon) , axis=1)
    return df

from math import *
def foo_max_size(lat_list):
    max_lat = lat_list.max()
    min_lat = lat_list.min()
    x = np.arange(round(min_lat, 3) - 1, round(max_lat, 3) + 1, 0.02)
    h = np.histogram(lat_list, bins=x)
    a = x[np.argmax(h[0])+1]
    return a


def max_size(df, prop):
    mapping = df.groupby("customer_id")[prop].apply(lambda x: foo_max_size(np.array(x)))
    df[prop + '_coord'] = df.customer_id.map(mapping)
    return df  


def mean_enc_train(df, col_name, typ, alpha=50, globalmean = 0.33):
    kf = KFold(n_splits=5,random_state=42)
    
    if typ == 'home':
        df[col_name + '_mean_home'] = df[col_name]
        for train_index, test_index in kf.split(df):
  
            nrows = df.iloc[train_index].groupby(col_name).size()
            means = df.iloc[train_index].groupby(col_name).y_home.agg('mean')
            #print(means[:10])
            #print(sum(means.isnull()))
            score = (np.multiply(means,nrows)  + globalmean*alpha) / (nrows+alpha)
            df[col_name + '_mean_home'].iloc[test_index] = df.iloc[test_index][col_name + '_mean_home'].map(score)
            #df.at[test_index, col_name + '_mean'] = df[col_name + '_mean'].map(score).iloc[test_index]
        df[col_name + '_mean_home'].fillna(globalmean, inplace=True)
    
    if typ == 'work':
        df[col_name + '_mean_work'] = df[col_name]
        for train_index, test_index in kf.split(df):
  
            nrows = df.iloc[train_index].groupby(col_name).size()
            means = df.iloc[train_index].groupby(col_name).y_work.agg('mean')
            #print(means[:10])
            #print(sum(means.isnull()))
            score = (np.multiply(means,nrows)  + globalmean*alpha) / (nrows+alpha)
            df[col_name + '_mean_work'].iloc[test_index] = df.iloc[test_index][col_name + '_mean_work'].map(score)
            #df.at[test_index, col_name + '_mean'] = df[col_name + '_mean'].map(score).iloc[test_index]
        df[col_name + '_mean_work'].fillna(globalmean, inplace=True)
    return df

def mean_enc_test(df, test_df, col_name, typ, alpha=50, globalmean = 0.33):
    kf = KFold(n_splits=5,random_state=42)
    
    if typ == 'home':
            test_df[col_name + '_mean_home'] = test_df[col_name]
            #print(test_df[col_name + '_mean'].value_counts())
            nrows = df.groupby(col_name).size()
            means = df.groupby(col_name).y_home.agg('mean')

            score = (np.multiply(means,nrows)  + globalmean*alpha) / (nrows+alpha)
            test_df[col_name + '_mean_home'] = test_df[col_name + '_mean_home'].map(score)
            test_df[col_name + '_mean_home'].fillna(globalmean, inplace=True)
    
    if typ == 'work':
            test_df[col_name + '_mean_work'] = test_df[col_name]
            #print(test_df[col_name + '_mean'].value_counts())
            nrows = df.groupby(col_name).size()
            means = df.groupby(col_name).y_work.agg('mean')

            score = (np.multiply(means,nrows)  + globalmean*alpha) / (nrows+alpha)
            test_df[col_name + '_mean_work'] = test_df[col_name + '_mean_work'].map(score)
            test_df[col_name + '_mean_work'].fillna(globalmean, inplace=True)
    return test_df

def fit(X_train, y_train, cols, num_boost_round=101, mt='xgb', typ="home", pr='gbdt', params=None):
    if typ=="home":
        good_merch = X_train.home_add_lat.dropna().index
        X_train = X_train.loc[good_merch]
        y_train = y_train.loc[good_merch]
    else:
        good_merch = X_train.work_add_lat.dropna().index
        X_train = X_train.loc[good_merch]
        y_train = y_train.loc[good_merch]
        
    if mt == 'xgb':
        dtrain = xgb.DMatrix(X_train[cols], y_train)
        params = {'eta': 0.1,
          'objective': 'binary:logistic',
          'eval_metric': ['auc'],
          'max_depth': 3}
        model = xgb.train(params, dtrain, num_boost_round=num_boost_round, verbose_eval=10)
        
    if mt == 'lr':
        model = linear_model.LogisticRegression(C=1e5)

        model.fit(X_train[cols], y_train)

    if mt == 'lgb':
        lgb_train = lgb.Dataset(X_train[cols], y_train)

        # specify your configurations as a dict
        if params == None:
            params = {
            'boosting_type': pr,
            'objective': 'binary',
            'metric': 'binary_logloss',
            'bagging_freq': 0,
            'colsample_bytree': '0.514',
            'learning_rate': '0.02',
            'feature_fraction': '0.625',
            'bagging_fraction': '0.844',
            'num_leaves': 60,
            'verbose': 1,
            #'is_unbalance':True
            }

        #print('Start training...')
        # train
        model = lgb.train(params,
                lgb_train,
                num_boost_round=430)

    return model





def evaluate(df, preds, typ):
    #print(df.shape)
    df = df[df.customer_id.isin(preds.index)]
    df = df[["customer_id",typ+"_add_lat", typ+"_add_lon"]].drop_duplicates()#
    customers = df.customer_id.value_counts()
    customers = customers[customers==1].index
    df = df[df.customer_id.isin(customers)]
    df = df.set_index("customer_id")
    res = np.array(df.loc[preds.index]) - np.array(preds)
    
    r = np.sum(np.sqrt(res[:, 0]**2+res[:, 1]**2) < .02)/res.shape[0]
    return r

def predict(model, X_test, cols, mt):
    if mt == 'xgb': proba = model.predict(xgb.DMatrix(X_test[cols]))
    if mt == 'lgb': proba = model.predict(X_test[cols], num_iteration=model.best_iteration)
    if mt == 'lr': proba = model.predict_proba(X_test[cols])

    X_test['proba'] = proba
    mapping = X_test.groupby("customer_id").proba.max()
    X_test["max_proba"] = X_test.customer_id.map(mapping)
    X_test = X_test.loc[X_test.proba==X_test.max_proba,]
    #return X_test
    lat = X_test.groupby("customer_id").center_lat.median()
    lon = X_test.groupby("customer_id").center_lon.median()
    return pd.merge(lat.to_frame("lat"), lon.to_frame("lon"), right_index=True, left_index=True)


def check_df(train_df, cols, rs=42, typ='home'):

#     folds_generator = GroupShuffleSplit(1,random_state=rs)
#     ind = folds_generator.split(train_df, train_df.y_home, groups=train_df.customer_id)
#     folds = []
#     for el, er in ind:
#         folds += [(el, er)]

#     temp = train_df.iloc[folds[0][0]].copy()
#     temp_test = train_df.iloc[folds[0][1], :].copy()
#     HOME_GM = train_df.y_home.mean()
#     WORK_GM = train_df.y_work.mean()
    
#     temp = mean_enc_train(temp.copy(), "count_ters", typ='home', globalmean=HOME_GM)
#     temp = mean_enc_train(temp.copy(), "mcc", typ='home', globalmean=HOME_GM)
#     temp = mean_enc_train(temp.copy(), "num_near_terminals", typ='home', globalmean=HOME_GM)
#     temp = mean_enc_train(temp.copy(), "week_day", typ='home', globalmean=HOME_GM)
#     temp = mean_enc_train(temp.copy(), "count_ters", typ='work', globalmean=WORK_GM)
#     temp = mean_enc_train(temp.copy(), "mcc", typ='work', globalmean=WORK_GM)
#     temp = mean_enc_train(temp.copy(), "num_near_terminals", typ='work', globalmean=WORK_GM)
#     temp = mean_enc_train(temp.copy(), "week_day", typ='work', globalmean=WORK_GM)



#     temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "count_ters", typ='home', globalmean=HOME_GM)
#     temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "mcc", typ='home', globalmean=HOME_GM)
#     temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "num_near_terminals", typ='home', globalmean=HOME_GM)
#     temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "week_day", typ='home', globalmean=HOME_GM)
#     temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "count_ters", typ='work', globalmean=WORK_GM)
#     temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "mcc", typ='work', globalmean=WORK_GM)
#     temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "num_near_terminals", typ='work', globalmean=WORK_GM)
#     temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "week_day", typ='work', globalmean=WORK_GM)
    
    
    cc = list(set(train_df.customer_id.values))
    np.random.seed(rs)
    test_cc = np.random.choice(cc, 2000, replace=False)

    #temp = train_df.iloc[folds[0][0]].copy()
    #temp_test = train_df.iloc[folds[0][1], :].copy()
    
    temp = train_df[~train_df.customer_id.isin(test_cc)].copy()
    temp_test = train_df[train_df.customer_id.isin(test_cc)].copy()
    
    if typ == 'home':
        model = fit(temp, temp.y_home, cols, typ="home", mt='lgb', pr='gbdt')
        preds = predict(model, temp_test, cols, mt='lgb')
        #print(preds.head())
        print(rs, ': ', evaluate(train_df.copy(), preds, "home"))
        
    if typ == 'work':
        model = fit(temp, temp.y_work, cols, typ="work", mt='lgb', pr='gbdt')
        preds = predict(model, temp_test, cols, mt='lgb')
        print(rs, ': ', evaluate(train_df.copy(), preds, "work"))
      
    return model
    
    
    
    

    
def calculate_cluster_features(df):
    clusters = []

    for customer_id, transactions in tqdm_notebook(df.groupby('customer_id')):
        for cluster_id, cluster_transactions in transactions.groupby('cluster_id'):
            if cluster_id == -1: continue
                
            cluster_uniq = len(cluster_transactions[['pos_address_lat', 'pos_address_lon']].drop_duplicates())
            cluster_median = cluster_transactions[['pos_address_lat', 'pos_address_lon']].median()
            cluster_mean_dist = cluster_transactions['cluster_mean_dist'].value_counts().index[0]
            cluster_mean_to_center = cluster_transactions['cluster_mean_to_center'].value_counts().index[0]
            
            cluster_max_dist = cluster_transactions['cluster_max_dist'].value_counts().index[0]
            cluster_max_to_center = cluster_transactions['cluster_max_to_center'].value_counts().index[0]
            
            cluster_mc_lat = cluster_transactions['pos_address_lat'].value_counts().index[0]
            cluster_mc_lon = cluster_transactions['pos_address_lon'].value_counts().index[0]
            
            center_lat, center_lon = cluster_transactions['cluster_center_point'].values[0]
            real_center_lat, real_center_lon = cluster_transactions['cluster_real_center'].values[0]
            amount_histogram = cluster_transactions.amount.round().value_counts(normalize=True)
            amount_histogram = amount_histogram.add_prefix('amount_hist_').to_dict()
            mcc_whitelist = [
                5411, 6011, 5814, 5812, 5499,
                5541, 5912, 4111, 5921, 5331,
                5691, 5261, 5977,
                -1
            ]

            mcc = cluster_transactions.mcc.copy()
            mcc.loc[~mcc.isin(mcc_whitelist)] = -1
            mcc_histogram = mcc.astype(CategoricalDtype(categories=mcc_whitelist)).value_counts(normalize=True)
            mcc_histogram = mcc_histogram.add_prefix('mcc_hist_').to_dict()
            day_histogram = cluster_transactions.transaction_date.dt.dayofweek.value_counts(normalize=True).add_prefix('day_hist_').to_dict()

            try:
                # pylint: disable=no-member
                area = ConvexHull(cluster_transactions[['pos_address_lat', 'pos_address_lon']]).area
            # pylint: disable=broad-except
            except Exception as _:
                area = 0

            # TODO AS: Might reconsider this later
            first_transaction = transactions.iloc[0]

            features = {
                'cluster_id': cluster_id,
                'customer_id': customer_id,
                'cluster_lat': cluster_median['pos_address_lat'],
                'cluster_lon': cluster_median['pos_address_lon'],
                'cluster_mean_dist': cluster_mean_dist,
                'cluster_mean_to_center': cluster_mean_to_center,
                'cluster_max_dist': cluster_max_dist,
                'cluster_max_to_center': cluster_max_to_center,
                #'home_add_lat': first_transaction['home_add_lat'],
                #'home_add_lon': first_transaction['home_add_lon'],
                #'work_add_lat': first_transaction['work_add_lat'],
                #'work_add_lon': first_transaction['work_add_lon'],
                'center_lat': center_lat,
                'center_lon': center_lon,
                'real_center_lat': real_center_lat,
                'real_center_lon': real_center_lon,
                'cluster_uniq': cluster_uniq,
                'cluster_mc_lat': cluster_mc_lat,
                'cluster_mc_lon': cluster_mc_lon,
                #'area': area,
                'transaction_ratio': len(cluster_transactions) / len(transactions),
                'amount_ratio': np.sum(np.exp(cluster_transactions.amount)) / np.sum(np.exp(transactions.amount)),
                'date_ratio': len(cluster_transactions.transaction_date.unique()) / len(transactions.transaction_date.unique()),
                'amount_hist_-2.0': 0,
                'amount_hist_-1.0': 0,
                'amount_hist_0.0': 0,
                'amount_hist_1.0': 0,
                'amount_hist_2.0': 0,
                'amount_hist_3.0': 0,
                'amount_hist_4.0': 0,
                'amount_hist_5.0': 0,
                'amount_hist_6.0': 0,
                **amount_histogram,
                'mcc_hist_5411': 0,
                'mcc_hist_6011': 0,
                'mcc_hist_5814': 0,
                'mcc_hist_5812': 0,
                'mcc_hist_5499': 0,
                'mcc_hist_4111': 0,
                'mcc_hist_5921': 0,
                'mcc_hist_5331': 0,
                'mcc_hist_5691': 0,
                'mcc_hist_5261': 0,
                'mcc_hist_5977': 0,
                'mcc_hist_-1': 0,
                **mcc_histogram,
                'day_hist_0': 0,
                'day_hist_1': 0,
                'day_hist_2': 0,
                'day_hist_3': 0,
                'day_hist_4': 0,
                'day_hist_5': 0,
                'day_hist_6': 0,
                **day_histogram
            }

            clusters.append(features)


    return pd.DataFrame(clusters)

def cluster(df):
    print('min samples: ', m)
    grouped_by_customer = df.groupby('customer_id', sort=False, as_index=False, group_keys=False)
    df['cluster_mean_dist'] = grouped_by_customer.progress_apply(get_mean_dist)
    df['cluster_mean_to_center'] = grouped_by_customer.progress_apply(get_mean_to_center)
    
    df['cluster_max_dist'] = grouped_by_customer.progress_apply(get_mean_dist)
    df['cluster_max_to_center'] = grouped_by_customer.progress_apply(get_mean_to_center)
    
    df['cluster_id'] = grouped_by_customer.progress_apply(get_cluster_ids)
    df['cluster_center_point'] = grouped_by_customer.progress_apply(get_cluster_centers)
    df['cluster_real_center'] = grouped_by_customer.progress_apply(get_real_center)
    return df

def get_centroid(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    return tuple(centroid)

def get_real_center(df):
    coords = df.as_matrix(columns=['pos_address_lat', 'pos_address_lon'])
    weights = df.as_matrix(columns=['count_ters'])
    #print('transactions: ', len(coords))
    kms_per_radian = 6371.0088
    epsilon = 1.5 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=m, algorithm='ball_tree', metric='haversine').fit(np.radians(coords), sample_weight=weights)
    num_clusters = len(set(db.labels_))
    #print(num_clusters)
    #print(np.where(db.labels_ == -1)[0].shape[0])
    if np.where(db.labels_ == -1)[0].shape[0] > 0:
        clusters = pd.Series([coords[db.labels_ == n] for n in range(-1,num_clusters-1)])
    else:
        clusters = pd.Series([coords[db.labels_ == n] for n in range(num_clusters)])
    #print(clusters)
    centermost_points = clusters.map(get_centroid)
    
    if np.where(db.labels_ == -1)[0].shape[0] > 0:
        #clusters = pd.Series([coords[db.labels_ == n] for n in range(-1,num_clusters-1)])
        return pd.Series(index=df.index, data=[centermost_points[j+1] for j in db.labels_])
    else:
        return pd.Series(index=df.index, data=[centermost_points[j] for j in db.labels_])
    
from scipy.spatial.distance import pdist
def get_pair_max(cluster):
    
    
    return np.max(pdist(cluster))

def get_max_dist(df):
    coords = df.as_matrix(columns=['pos_address_lat', 'pos_address_lon'])
    weights = df.as_matrix(columns=['count_ters'])
    #print('transactions: ', len(coords))
    kms_per_radian = 6371.0088
    epsilon = 1.5 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=m, algorithm='ball_tree', metric='haversine').fit(np.radians(coords), sample_weight=weights)
    num_clusters = len(set(db.labels_))
    #print(num_clusters)
    #print(np.where(db.labels_ == -1)[0].shape[0])
    if np.where(db.labels_ == -1)[0].shape[0] > 0:
        clusters = pd.Series([coords[db.labels_ == n] for n in range(-1,num_clusters-1)])
    else:
        clusters = pd.Series([coords[db.labels_ == n] for n in range(num_clusters)])
    #print(clusters)
    center_means = clusters.map(get_pair_max)
    
    if np.where(db.labels_ == -1)[0].shape[0] > 0:
        #clusters = pd.Series([coords[db.labels_ == n] for n in range(-1,num_clusters-1)])
        return pd.Series(index=df.index, data=[center_means[j+1] for j in db.labels_])
    else:
        return pd.Series(index=df.index, data=[center_means[j] for j in db.labels_])

from scipy.spatial.distance import pdist
def get_pair_mean(cluster):
    
    
    return np.mean(pdist(cluster))

def get_mean_dist(df):
    coords = df.as_matrix(columns=['pos_address_lat', 'pos_address_lon'])
    weights = df.as_matrix(columns=['count_ters'])
    #print('transactions: ', len(coords))
    kms_per_radian = 6371.0088
    epsilon = 1.5 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=m, algorithm='ball_tree', metric='haversine').fit(np.radians(coords), sample_weight=weights)
    num_clusters = len(set(db.labels_))
    #print(num_clusters)
    #print(np.where(db.labels_ == -1)[0].shape[0])
    if np.where(db.labels_ == -1)[0].shape[0] > 0:
        clusters = pd.Series([coords[db.labels_ == n] for n in range(-1,num_clusters-1)])
    else:
        clusters = pd.Series([coords[db.labels_ == n] for n in range(num_clusters)])
    #print(clusters)
    center_means = clusters.map(get_pair_mean)
    
    if np.where(db.labels_ == -1)[0].shape[0] > 0:
        #clusters = pd.Series([coords[db.labels_ == n] for n in range(-1,num_clusters-1)])
        return pd.Series(index=df.index, data=[center_means[j+1] for j in db.labels_])
    else:
        return pd.Series(index=df.index, data=[center_means[j] for j in db.labels_])

def get_center_max(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    
    return np.max(np.linalg.norm(cluster-centermost_point))

def get_max_to_center(df):
    coords = df.as_matrix(columns=['pos_address_lat', 'pos_address_lon'])
    weights = df.as_matrix(columns=['count_ters'])
    #print('transactions: ', len(coords))
    kms_per_radian = 6371.0088
    epsilon = 1.5 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=m, algorithm='ball_tree', metric='haversine').fit(np.radians(coords), sample_weight=weights)
    num_clusters = len(set(db.labels_))
    #print(num_clusters)
    #print(np.where(db.labels_ == -1)[0].shape[0])
    if np.where(db.labels_ == -1)[0].shape[0] > 0:
        clusters = pd.Series([coords[db.labels_ == n] for n in range(-1,num_clusters-1)])
    else:
        clusters = pd.Series([coords[db.labels_ == n] for n in range(num_clusters)])
    #print(clusters)
    center_means = clusters.map(get_center_max)
    
    if np.where(db.labels_ == -1)[0].shape[0] > 0:
        #clusters = pd.Series([coords[db.labels_ == n] for n in range(-1,num_clusters-1)])
        return pd.Series(index=df.index, data=[center_means[j+1] for j in db.labels_])
    else:
        return pd.Series(index=df.index, data=[center_means[j] for j in db.labels_])
    
def get_center_mean(cluster):
    centroid = (MultiPoint(cluster).centroid.x, MultiPoint(cluster).centroid.y)
    centermost_point = min(cluster, key=lambda point: great_circle(point, centroid).m)
    
    return np.mean(np.linalg.norm(cluster-centermost_point))

def get_mean_to_center(df):
    coords = df.as_matrix(columns=['pos_address_lat', 'pos_address_lon'])
    weights = df.as_matrix(columns=['count_ters'])
    #print('transactions: ', len(coords))
    kms_per_radian = 6371.0088
    epsilon = 1.5 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=m, algorithm='ball_tree', metric='haversine').fit(np.radians(coords), sample_weight=weights)
    num_clusters = len(set(db.labels_))
    #print(num_clusters)
    #print(np.where(db.labels_ == -1)[0].shape[0])
    if np.where(db.labels_ == -1)[0].shape[0] > 0:
        clusters = pd.Series([coords[db.labels_ == n] for n in range(-1,num_clusters-1)])
    else:
        clusters = pd.Series([coords[db.labels_ == n] for n in range(num_clusters)])
    #print(clusters)
    center_means = clusters.map(get_center_mean)
    
    if np.where(db.labels_ == -1)[0].shape[0] > 0:
        #clusters = pd.Series([coords[db.labels_ == n] for n in range(-1,num_clusters-1)])
        return pd.Series(index=df.index, data=[center_means[j+1] for j in db.labels_])
    else:
        return pd.Series(index=df.index, data=[center_means[j] for j in db.labels_])
 

def get_cluster_centers(df):
    coords = df.as_matrix(columns=['pos_address_lat', 'pos_address_lon'])
    weights = df.as_matrix(columns=['count_ters'])
    #print('transactions: ', len(coords))
    kms_per_radian = 6371.0088
    epsilon = 1.5 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=m, algorithm='ball_tree', metric='haversine').fit(np.radians(coords), sample_weight=weights)
    num_clusters = len(set(db.labels_))
    #print(num_clusters)
    #print(np.where(db.labels_ == -1)[0].shape[0])
    if np.where(db.labels_ == -1)[0].shape[0] > 0:
        clusters = pd.Series([coords[db.labels_ == n] for n in range(-1,num_clusters-1)])
    else:
        clusters = pd.Series([coords[db.labels_ == n] for n in range(num_clusters)])
    #print(clusters)
    centermost_points = clusters.map(get_centermost_point)
    
    if np.where(db.labels_ == -1)[0].shape[0] > 0:
        #clusters = pd.Series([coords[db.labels_ == n] for n in range(-1,num_clusters-1)])
        return pd.Series(index=df.index, data=[centermost_points[j+1] for j in db.labels_])
    else:
        return pd.Series(index=df.index, data=[centermost_points[j] for j in db.labels_])
    
    
def get_cluster_ids(df):
    coords = df.as_matrix(columns=['pos_address_lat', 'pos_address_lon'])
    weights = df.as_matrix(columns=['count_ters'])
    #print('transactions: ', len(coords))
    kms_per_radian = 6371.0088
    epsilon = 1.5 / kms_per_radian
    db = DBSCAN(eps=epsilon, min_samples=m, algorithm='ball_tree', metric='haversine').fit(np.radians(coords), sample_weight=weights)
    return pd.Series(index=df.index, data=db.labels_)




    
