from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import joblib
import os
import pandas as pd

pickle_folder = 'pickle'
dataset_folder = 'dataset'

dirpath = os.path.dirname(__file__)
filepath_cleaned_data = os.path.join(dirpath, pickle_folder, 'data.pkl')
open(filepath_cleaned_data, 'r')

filepath_user_final_rating = os.path.join(dirpath, pickle_folder, 'user_final_rating.pkl')
open(filepath_user_final_rating, 'r')

filepath_final_model = os.path.join(dirpath, pickle_folder, 'xgb_tuned3.pkl')
open(filepath_final_model, 'r')

filepath_raw_dataset = os.path.join(dirpath, dataset_folder, 'sample30.csv')
open(filepath_raw_dataset, 'r')

filepath_tfidf_features = os.path.join(dirpath, pickle_folder, 'tf_idf_vectorized_features.pkl')
open(filepath_tfidf_features, 'r')

class SentimentBasedProductRecommendationSystem:
    def __init__(self):
        print(dirpath)
        self.data = joblib.load(filepath_cleaned_data)
        self.user_final_rating = joblib.load(filepath_user_final_rating)
        self.final_xgboost_model = joblib.load(filepath_final_model)
        self.raw_data = pd.read_csv(filepath_raw_dataset)

        self.raw_data['reviews_didPurchase'].fillna(False,inplace=True)
        self.raw_data['reviews_doRecommend'].fillna(False,inplace=True)
        self.raw_data['reviews_title'].fillna('',inplace=True)
        self.raw_data['manufacturer'].fillna('',inplace=True)
        self.raw_data['reviews_username'].fillna('',inplace=True)
        self.raw_data = self.raw_data[self.raw_data['user_sentiment'].notna()]

        self.data = pd.concat([self.raw_data[['id', 'name', 'brand', 'categories', 'manufacturer']], self.data], axis=1)

   
    def recommendProducts(self, user_name):
        items = self.user_final_rating.loc[user_name].sort_values(ascending=False)[0:20].index
        features = joblib.load(filepath_tfidf_features)
        vectorizer = TfidfVectorizer(vocabulary = features)
        df_prediction=self.data[self.data.id.isin(items)]
        X = vectorizer.fit_transform(df_prediction['Review'].values.astype(str))
        df_prediction=df_prediction[['id']]
        df_prediction['prediction'] = self.final_xgboost_model.predict(X)
        df_prediction['prediction'] = df_prediction['prediction'].map({'Positive':1,'Negative':-1})
        df_prediction=df_prediction.groupby('id').sum()
        df_prediction['positive_reviews']=df_prediction.apply(lambda x: 0.0 if sum(x) == 0 else x['prediction']/sum(x), axis=1)
        product_recommendations=df_prediction.sort_values('positive_reviews', ascending=False).iloc[:5,:].index
        self.data_final = self.data[self.data.id.isin(product_recommendations)][['id', 'name', 'brand',
                              'categories', 'manufacturer']].drop_duplicates()

        return self.data_final.to_html(index=False)
    
