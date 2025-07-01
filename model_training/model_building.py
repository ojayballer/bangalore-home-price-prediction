import pandas as pd
import numpy as np
import pickle 
import json
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
def read_csv(filepath):
    df=pd.read_csv(filepath)
    return df

def clean_data(df):
    df1=df.drop(columns=df[['society','availability','balcony']],axis='columns')
    df1['bath']=df1['bath'].fillna(np.nanmedian(df1['bath']))
    df1=df1.dropna(subset=['size','location'])
    df1['bedroom']=df1['size'].apply(lambda x: int(x.split(' ')[0]))
    df1=df1.drop(columns=df1[['size']],axis='columns')
    return df1

def convert_sqft(df1) :
    def convert(x):
        try:   
          if isinstance(x,str)  and '-'  in x  :
            return (float(x.split('-')[0])+(float(x.split('-')[1]))/2)
          return float(x.split()[0])
        except:
          return None
            
    df1['total_sqft']=df1['total_sqft'].apply(convert)
    df1=df1.dropna(subset=['total_sqft'])                               
    df2=df1.copy()
    return df2

def feat_engr(df2):
    df2['price_per_sqft']=df2['price']*100000/df2['total_sqft']
    location_stat = df2.groupby('location')['location'].agg('count')
    df2['location']=df2['location'].apply(lambda x:'others' if(location_stat[x]<=10) else x)
    df3=df2.copy()
    return df3

def outlier_removal(df3):
    df3=df3[~(df3.total_sqft/df3.bedroom<300)]
    new_df=pd.DataFrame()
    for key,subdf in df3.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        std=np.std(subdf.price_per_sqft)
        df_reduced=subdf[(subdf.price_per_sqft<=(m+std))&(subdf.price_per_sqft>(m-std))]
        new_df=pd.concat([new_df,df_reduced],ignore_index=True)
    df4=new_df.copy()
    return df4

def remove_bedroom_outliers(df4):
    exclude=np.array([])
    for location,location_df in df4.groupby('location'):
        bedroom_stats={}#for each location,create a dictionary to store mean,std and count(bedroom count)
        for bedroom,bedroom_df in location_df.groupby('bedroom'):
             bedroom_stats[bedroom] = {'mean':np.mean(bedroom_df.price_per_sqft),
                                     'std':np.std(bedroom_df.price_per_sqft),
                                     'count':bedroom_df.shape[0]
                 }
        for bedroom,bedroom_df in location_df.groupby('bedroom'):
            stats=bedroom_stats.get(bedroom-1)
            if stats and stats['count']>5:
                exclude=np.append(exclude,bedroom_df[bedroom_df.price_per_sqft<stats['mean']].index.values)
    df4= df4.drop(exclude,axis='index') 
    df4=df4[df4.bath<(df4.bedroom+2)] 
    df4=df4.drop(columns=df4[['price_per_sqft']],axis='columns')
    return df4

def encode_labels_and_prepare_features(df5):
    dummies=pd.get_dummies(df5[['area_type','location']],drop_first=True)
    df5=pd.concat([df5,dummies],axis='columns')
    df5=df5.drop(columns=['area_type','location'],axis='columns')
    Y=df5.price
    X=df5.drop(columns=df5[['price']],axis='columns')
    df5.columns.str.lower()
    return  X,Y
    
def find_best_model_using_GridSearchCV(X,Y):
    model_params={ 'lin_reg':{
        'model':LinearRegression(),
        'params':{
                  }},
                'lasso':{'model':Lasso(),
                         'params':{'alpha':[1,2],'selection':
                             ['random','cyclic']}},
                'decision_tree':{
                    'model':DecisionTreeRegressor(),
                    'params':{
                    'criterion':['squared_error','friedman_mse'],
                    'splitter':['best','random']
                    }} }
    scores=[]
    cv=ShuffleSplit(n_splits=5,test_size=0.2,random_state=10)
    for model_name,mp in model_params.items():
        clf=GridSearchCV(mp['model'],mp['params'],cv=cv,return_train_score=False)
        clf.fit(X,Y)
        scores.append({'model':model_name,'best_score':clf.best_score_,
                       'best_params':clf.best_params_})
    return pd.DataFrame(scores,columns=['model','best_params','best_score']) #we use the model with the best score to train our model

def train_model(X,Y):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=10)
    model=LinearRegression()
    model.fit(X_train,Y_train)
    return model

def predict_prices(model,X,location,total_sqft,bath,bedroom,area_type):
    X.columns=X.columns.str.lower()
    x=np.zeros(len(X.columns))
    x[0]=total_sqft
    x[1]=bath
    x[2]=bedroom
    loc_col=f"location_{location.strip().lower()}"
    if loc_col in X.columns:
        loc_index=np.where(X.columns==loc_col)[0][0]
        x[loc_index]=1
        
    area_col=f"area_type_{area_type.strip().lower()}"
    if area_col in X.columns:
        area_index=np.where(X.columns==area_col)[0][0]
        x[area_index]=1
    return model.predict([x])[0]

def save_model(df8,X):
    with open('banglore_home_prices_model.pickle','wb') as f:
        pickle.dump(df8,f)
        
    columns={'data_columns':[col.lower() for col in X.columns]
             }
    with open("columns.json","w") as f:
        f.write(json.dumps(columns))
    return "model  saved successfully!"

def main():
   filepath="C:\\Users\Omojire\Downloads\\Bengaluru_House_Data.csv"
   df=read_csv(filepath)
   df1=clean_data(df) 
   df2=convert_sqft(df1)
   df3=feat_engr(df2)
   df4=outlier_removal(df3)
   df5=remove_bedroom_outliers(df4)
   X,Y=encode_labels_and_prepare_features(df5)
   df7=find_best_model_using_GridSearchCV(X,Y)
   df8=train_model(X,Y)
   
   #let's predict the prices of some properties to test the accuracy 
   price_predicted=predict_prices(df8, X, '1st phase jp nagar', 1000, 3, 3, 'super built-up  area')
   print(predict_prices(df8, X, 'electronic city phase ii', 1000, 3, 3, 'super built-up  area'))

   print(f"Predicted price(in Lakhs) : {price_predicted}")
   s=save_model(df8,X)
main()
