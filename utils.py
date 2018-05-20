import pandas as pd
import datetime as dt
import numpy as np
from tqdm import tqdm_notebook
import collections
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import lightgbm as lgb
from sklearn import linear_model
from multiprocessing import Pool

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
    ## только там где координаты известны
    train = train[~train.atm_address_lat.isnull() | ~train.pos_address_lat.isnull()]
    train = pre(train)
    ##  страна россия
    train = train[train.country.isin(['RUS', 'RU'])]
    
    ##  усреднение координат для одинакого адреса
    temp = train[~train.atm_address.isnull() & train.atm_address_lat.isnull()].atm_address.unique()
    lat_mapping = train[train.atm_address.isin(temp)][~train[train.atm_address.isin(temp)].atm_address_lat.isnull()].groupby('atm_address').atm_address_lat.median().to_dict()
    train['atm_address_lat_upd'] = train[train.atm_address_lat.isnull()]['atm_address'].map(lat_mapping)
    train.atm_address_lat = train.atm_address_lat.fillna(train.atm_address_lat_upd)
    
    temp = train[~train.atm_address.isnull() & train.atm_address_lon.isnull()].atm_address.unique()
    lon_mapping = train[train.atm_address.isin(temp)][~train[train.atm_address.isin(temp)].atm_address_lon.isnull()].groupby('atm_address').atm_address_lon.median().to_dict()
    train['atm_address_lon_upd'] = train[train.atm_address_lon.isnull()]['atm_address'].map(lon_mapping)
    train.atm_address_lon = train.atm_address_lon.fillna(train.atm_address_lon_upd)
    
    ##  уникальный id для терминалов перебежчиков
    temp = train.groupby(['terminal_id', 'city']).apply(lambda row: row.terminal_id + row.city).reset_index(level=[0,1])
    temp.columns = ['terminal_id', 'city', 'terminal_id_upd']
    temp = temp.drop_duplicates()
    train_df = train.merge(temp, on=['terminal_id', 'city'], how='left')
    
     ## усреднение координат
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
    return (sum(temp)-1)

##   бцдем считать что ерминал рядом если адрес терминала назоися не дальше 0.2 от адреса пользователя
def number_near_terminals(df):
    ldf = df.copy()
    mapping = df.groupby("customer_id").pos_address_lat.apply(lambda x: np.array(x)).to_dict()
    ldf["customer_lat"] = ldf.customer_id.map(mapping)
    mapping = df.groupby("customer_id").pos_address_lon.apply(lambda x: np.array(x)).to_dict()
    ldf["customer_lon"] = ldf.customer_id.map(mapping)
    df["num_near_terminals"] = \
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


def ter_using_upd_w(df):
    ldf = df.copy()
    ldf["pos"] = ldf[["pos_address_lat", "pos_address_lon"]].apply(lambda x: (x.pos_address_lat, x.pos_address_lon) , axis=1)

    ldf = ldf.merge(ldf[ldf.week_day.isin([5,6])].groupby(["customer_id", 'pos']).size().reset_index(name='count_ters_w'), left_on=['customer_id','pos'], right_on=['customer_id','pos'], how='left',) 
    
    ldf.count_ters_w.fillna(0, inplace=True)
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



def predict(model, X_test, cols, mt):
    if mt == 'xgb': proba = model.predict(xgb.DMatrix(X_test[cols]))
    if mt == 'lgb': proba = model.predict(X_test[cols], num_iteration=model.best_iteration)
    if mt == 'lr': proba = model.predict_proba(X_test[cols])

    X_test['proba'] = proba
    mapping = X_test.groupby("customer_id").proba.max()
    X_test["max_proba"] = X_test.customer_id.map(mapping)
    X_test = X_test.loc[X_test.proba==X_test.max_proba]
    #return X_test
    lat = X_test.groupby("customer_id").pos_address_lat.median()
    lon = X_test.groupby("customer_id").pos_address_lon.median()
    return pd.merge(lat.to_frame("lat"), lon.to_frame("lon"), right_index=True, left_index=True)


def evaluate(df, preds, typ):
    df = df[df.customer_id.isin(preds.index)]
    df = df[["customer_id",typ+"_add_lat", typ+"_add_lon"]].drop_duplicates()#
    customers = df.customer_id.value_counts()
    customers = customers[customers==1].index
    df = df[df.customer_id.isin(customers)]
    df = df.set_index("customer_id")
    res = np.array(df.loc[preds.index]) - np.array(preds)
    r = np.sum(np.sqrt(res[:, 0]**2+res[:, 1]**2) < .02)/res.shape[0]

    return r



def check_df(train_df, cols, rs=42, typ='home'):

    #folds_generator = GroupShuffleSplit(1,random_state=rs)
    #ind = folds_generator.split(train_df, train_df.y_home, groups=train_df.customer_id)
    #folds = []
    #for el, er in ind:
    #    folds += [(el, er)]
    np.random.seed(rs)
    cc = sorted(list(set(train_df.customer_id.values)))
    test_cc = np.random.choice(cc, 2000, replace=False)

    #temp = train_df.iloc[folds[0][0]].copy()
    #temp_test = train_df.iloc[folds[0][1], :].copy()
    
    temp = train_df[~train_df.customer_id.isin(test_cc)].copy()
    temp_test = train_df[train_df.customer_id.isin(test_cc)].copy()
    
    #print(temp.customer_id.nunique(), temp_test.customer_id.nunique())
    HOME_GM = train_df.y_home.mean()
    WORK_GM = train_df.y_work.mean()
    
    temp = mean_enc_train(temp.copy(), "count_ters", typ='home', globalmean=HOME_GM)
    temp = mean_enc_train(temp.copy(), "mcc", typ='home', globalmean=HOME_GM)
    temp = mean_enc_train(temp.copy(), "num_near_terminals", typ='home', globalmean=HOME_GM)
    temp = mean_enc_train(temp.copy(), "week_day", typ='home', globalmean=HOME_GM)
    temp = mean_enc_train(temp.copy(), "count_ters", typ='work', globalmean=WORK_GM)
    temp = mean_enc_train(temp.copy(), "mcc", typ='work', globalmean=WORK_GM)
    temp = mean_enc_train(temp.copy(), "num_near_terminals", typ='work', globalmean=WORK_GM)
    temp = mean_enc_train(temp.copy(), "week_day", typ='work', globalmean=WORK_GM)
    
    temp = mean_enc_train(temp.copy(), "transaction_date", typ='home', globalmean=HOME_GM)
    temp = mean_enc_train(temp.copy(), "transaction_date", typ='work', globalmean=WORK_GM)


    temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "count_ters", typ='home', globalmean=HOME_GM)
    temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "mcc", typ='home', globalmean=HOME_GM)
    temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "num_near_terminals", typ='home', globalmean=HOME_GM)
    temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "week_day", typ='home', globalmean=HOME_GM)
    temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "count_ters", typ='work', globalmean=WORK_GM)
    temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "mcc", typ='work', globalmean=WORK_GM)
    temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "num_near_terminals", typ='work', globalmean=WORK_GM)
    temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "week_day", typ='work', globalmean=WORK_GM)
    
    temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "transaction_date", typ='home', globalmean=HOME_GM)
    temp_test = mean_enc_test(temp.copy(), temp_test.copy(), "transaction_date", typ='work', globalmean=WORK_GM)

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
    