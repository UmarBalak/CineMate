import pandas as pd
import joblib
from sklearn.neighbors import NearestNeighbors

# Load the dataset
df2 = pd.read_csv('netflix_8k_cf.csv', encoding='latin1')

# Load the pre-trained models
tfidf_vectorizer2 = joblib.load('Models/tfidf_netflix_8k.joblib')
knn_model2 = joblib.load('Models/knn_model_netflix_8k.joblib')
assert isinstance(knn_model2, NearestNeighbors), "knn_model is not a NearestNeighbors instance"



list2 = [4, 121, 3452, 3129, 33, 3541, 2711, 2665, 3117, 2349]