import pickle
from lightfm import cross_validation
import numpy as np
from recommenders.datasets import movielens
from lightfm.data import Dataset
import pandas as pd
import itertools
from recommenders.models.lightfm.lightfm_utils import (
    track_model_metrics,compare_metric)
import matplotlib.pyplot as plt
import seaborn as sns

K=10
# percentage of data used for testing
TEST_PERCENTAGE = 0.25
# seed for pseudonumber generations
SEEDNO = 42
# Select MovieLens data size
MOVIELENS_DATA_SIZE = '100k'
# no of epochs to fit model
NO_EPOCHS = 20
# no of threads to fit model
NO_THREADS = 32


data = movielens.load_pandas_df(
    size=MOVIELENS_DATA_SIZE,
    genres_col='genre',
    header=["userID", "itemID", "rating"]
)

model2 = pickle.load(open("model.pkl", "rb"))
model1 = pickle.load(open("model1.pkl", "rb"))

dataset = Dataset()
dataset.fit(users=data['userID'],
            items=data['itemID'])

(interactions, weights) = dataset.build_interactions(data.iloc[:, 0:3].values)

train_interactions, test_interactions = cross_validation.random_train_test_split(
    interactions, test_percentage=TEST_PERCENTAGE,
    random_state=np.random.RandomState(SEEDNO))

#-----------------------------------------------------------------------------------#

# split the genre based on the separator
movie_genre = [x.split('|') for x in data['genre']]

# retrieve the all the unique genres in the data
all_movie_genre = sorted(list(set(itertools.chain.from_iterable(movie_genre))))

user_feature_URL = 'ml-100k/u.user'
user_data = pd.read_table(user_feature_URL,
              sep='|', header=None)
user_data.columns = ['userID','age','gender','occupation','zipcode']

# merging user feature with existing data
new_data = data.merge(user_data[['userID','occupation']], left_on='userID', right_on='userID')

# retrieve all the unique occupations in the data
all_occupations = sorted(list(set(new_data['occupation'])))

dataset2 = Dataset()
dataset2.fit(data['userID'],
            data['itemID'],
            item_features=all_movie_genre,
            user_features=all_occupations)
(interactions2, weights2) = dataset2.build_interactions(data.iloc[:, 0:3].values)
train_interactions2, test_interactions2 = cross_validation.random_train_test_split(
    interactions2, test_percentage=TEST_PERCENTAGE,
    random_state=np.random.RandomState(SEEDNO))

item_features = dataset2.build_item_features(
    (x, y) for x,y in zip(data.itemID, movie_genre))
user_features = dataset2.build_user_features(
    (x, [y]) for x,y in zip(new_data.userID, new_data['occupation']))
#----------------------------------------------------------------------------------------#

output1, _ = track_model_metrics(model=model1, train_interactions=train_interactions,
                              test_interactions=test_interactions, k=K,
                              no_epochs=NO_EPOCHS, no_threads=NO_THREADS)

output2, _ = track_model_metrics(model=model2, train_interactions=train_interactions2,
                              test_interactions=test_interactions2, k=K,
                              no_epochs=NO_EPOCHS, no_threads=NO_THREADS,
                              item_features=item_features,
                              user_features=user_features)

for i in ['Precision', 'Recall']:
    sns.set_palette("Set2")
    plt.figure()
    sns.scatterplot(x="epoch", y="value", hue='data',
                data=compare_metric(df_list = [output1, output2], metric=i)
               ).set_title(f'{i} comparison using test set');
    plt.show()

