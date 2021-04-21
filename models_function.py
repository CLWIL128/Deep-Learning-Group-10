# Some functions are defined

# Author: Shangsi Ren
# Date: 2021/4/20

#----------------------------------------------------------------------------------------#
import numpy as np
import pandas as pd
import re
import pickle
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from collections import Counter
from tensorflow.python.ops import math_ops

#----------------------------------------------------------------------------------------#
#### For Data preprocessing.

# UserID, Occupation and MovieID remain unchanged.
# Gender field: Need to convert ‘F’ and ‘M’ to 0 and 1.
# Age field: to be converted into 7 consecutive numbers 0 to 6.
# Genres field: It is a classification field and needs to be converted into a number. First convert the categories in Genres into a dictionary of strings to numbers, and then convert the Genres field of each movie into a list of numbers, because some movies are a combination of multiple Genres.
# Title field: The processing method is the same as the Genres field. First, create a text-to-number dictionary, and then convert the description in the Title into a list of numbers. In addition, the year in the Title also needs to be removed.
# The Genres and Title fields need to be uniform in length so that they can be easily processed in the neural network. The blank part is filled with the number corresponding to ‘< PAD >’.
def load_data():  # Load Dataset from File
    
    #Read User data
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']
    users = pd.read_table('./ml-1m/users.dat', sep='::', header=None, names=users_title, engine = 'python')
    users = users.filter(regex='UserID|Gender|Age|JobID')
    users_orig = users.values
    #Change gender and age in User data
    gender_map = {'F':0, 'M':1}
    users['Gender'] = users['Gender'].map(gender_map)

    age_map = {val:ii for ii,val in enumerate(set(users['Age']))}
    users['Age'] = users['Age'].map(age_map)

    #Read Movie data set
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table('./ml-1m/movies.dat', sep='::', header=None, names=movies_title, engine = 'python')
    movies_orig = movies.values
    #Remove the year in the Title
    pattern = re.compile(r'^(.*)\((\d+)\)$')

    title_map = {val:pattern.match(val).group(1) for ii,val in enumerate(set(movies['Title']))}
    movies['Title'] = movies['Title'].map(title_map)

    #Movie type to digital dictionary
    genres_set = set()
    for val in movies['Genres'].str.split('|'):
        genres_set.update(val)

    genres_set.add('<PAD>')
    genres2int = {val:ii for ii, val in enumerate(genres_set)}

    #Convert the movie type to a list of equal length numbers, the length is 18
    genres_map = {val:[genres2int[row] for row in val.split('|')] for ii,val in enumerate(set(movies['Genres']))}

    for key in genres_map:
        for cnt in range(max(genres2int.values()) - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt,genres2int['<PAD>'])
    
    movies['Genres'] = movies['Genres'].map(genres_map)

    #Movie Title to Digital Dictionary
    title_set = set()
    for val in movies['Title'].str.split():
        title_set.update(val)
    
    title_set.add('<PAD>')
    title2int = {val:ii for ii, val in enumerate(title_set)}

    #Convert the movie title into a list of equal length numbers, the length is 15
    title_count = 15
    title_map = {val:[title2int[row] for row in val.split()] for ii,val in enumerate(set(movies['Title']))}
    
    for key in title_map:
        for cnt in range(title_count - len(title_map[key])):
            title_map[key].insert(len(title_map[key]) + cnt,title2int['<PAD>'])
    
    movies['Title'] = movies['Title'].map(title_map)

    #Read the rating data set
    ratings_title = ['UserID','MovieID', 'ratings', 'timestamps']
    ratings = pd.read_table('./ml-1m/ratings.dat', sep='::', header=None, names=ratings_title, engine = 'python')
    ratings = ratings.filter(regex='UserID|MovieID|ratings')

    #Combine three tables
    data = pd.merge(pd.merge(ratings, users), movies)
    
    #Divide the data into two tables, X and y
    target_fields = ['ratings']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]
    
    features = features_pd.values
    targets_values = targets_pd.values
    
    return title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig


#----------------------------------------------------------------------------------------#
#### For define input.

#Define placeholders for input
def get_inputs():
    uid = tf.compat.v1.placeholder(tf.int32, [None, 1], name="uid")
    user_gender = tf.compat.v1.placeholder(tf.int32, [None, 1], name="user_gender")
    user_age = tf.compat.v1.placeholder(tf.int32, [None, 1], name="user_age")
    user_job = tf.compat.v1.placeholder(tf.int32, [None, 1], name="user_job")
    
    movie_id = tf.compat.v1.placeholder(tf.int32, [None, 1], name="movie_id")
    movie_categories = tf.compat.v1.placeholder(tf.int32, [None, 18], name="movie_categories")
    movie_titles = tf.compat.v1.placeholder(tf.int32, [None, 15], name="movie_titles")
    targets = tf.compat.v1.placeholder(tf.int32, [None, 1], name="targets")
    LearningRate = tf.compat.v1.placeholder(tf.float32, name = "LearningRate")
    dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name = "dropout_keep_prob")
    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, LearningRate, dropout_keep_prob


#----------------------------------------------------------------------------------------#
#### For Building a neural network.

# Reference: https://zhuanlan.zhihu.com/p/32078473

# Define the embedding matrix of User
def get_user_embedding(uid, user_gender, user_age, user_job, uid_max, embed_dim, gender_max, age_max, job_max):
    with tf.name_scope("user_embedding"):
        uid_embed_matrix = tf.Variable(tf.compat.v1.random_uniform([uid_max, embed_dim], -1, 1), name = "uid_embed_matrix")
        uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name = "uid_embed_layer")
    
        gender_embed_matrix = tf.Variable(tf.compat.v1.random_uniform([gender_max, embed_dim // 2], -1, 1), name= "gender_embed_matrix")
        gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name = "gender_embed_layer")
        
        age_embed_matrix = tf.Variable(tf.compat.v1.random_uniform([age_max, embed_dim // 2], -1, 1), name="age_embed_matrix")
        age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name="age_embed_layer")
        
        job_embed_matrix = tf.Variable(tf.compat.v1.random_uniform([job_max, embed_dim // 2], -1, 1), name = "job_embed_matrix")
        job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name = "job_embed_layer")
    return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer


# Fully connect User's embedding matrix together to generate User characteristics
def get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer, embed_dim):
    with tf.name_scope("user_fc"):
        # First layer fully connected
        uid_fc_layer = tf.compat.v1.layers.dense(uid_embed_layer, embed_dim, name = "uid_fc_layer", activation=tf.nn.relu)
        gender_fc_layer = tf.compat.v1.layers.dense(gender_embed_layer, embed_dim, name = "gender_fc_layer", activation=tf.nn.relu)
        age_fc_layer = tf.compat.v1.layers.dense(age_embed_layer, embed_dim, name ="age_fc_layer", activation=tf.nn.relu)
        job_fc_layer = tf.compat.v1.layers.dense(job_embed_layer, embed_dim, name = "job_fc_layer", activation=tf.nn.relu)
        
        # The second layer is fully connected
        user_combine_layer = tf.concat([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer], 2)  #(?, 1, 128)
        user_combine_layer = tf.contrib.layers.fully_connected(user_combine_layer, 200, tf.tanh)  #(?, 1, 200)
    
        user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200])
    return user_combine_layer, user_combine_layer_flat


# Define the embedding matrix of Movie ID
def get_movie_id_embed_layer(movie_id, embed_dim, movie_id_max):
    with tf.name_scope("movie_embedding"):
        movie_id_embed_matrix = tf.Variable(tf.random_uniform([movie_id_max, embed_dim], -1, 1), name = "movie_id_embed_matrix")
        movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id, name = "movie_id_embed_layer")
    return movie_id_embed_layer


# Add multiple embedding vectors of movie type
def get_movie_categories_layers(movie_categories, movie_categories_max, embed_dim, combiner):
    with tf.name_scope("movie_categories_layers"):
        movie_categories_embed_matrix = tf.Variable(tf.random_uniform([movie_categories_max, embed_dim], -1, 1), name = "movie_categories_embed_matrix")
        movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, movie_categories, name = "movie_categories_embed_layer")
        if combiner == "sum":
            movie_categories_embed_layer = tf.reduce_sum(movie_categories_embed_layer, axis=1, keep_dims=True)
    return movie_categories_embed_layer


# Realization of Movie Title's Text Convolutional Network
def get_movie_cnn_layer(movie_titles, movie_title_max, embed_dim, filter_num, window_sizes, sentences_size, dropout_keep_prob):
    # Get the embedding vector of each word corresponding to the movie name from the embedding matrix
    with tf.name_scope("movie_embedding"):
        movie_title_embed_matrix = tf.Variable(tf.random_uniform([movie_title_max, embed_dim], -1, 1), name = "movie_title_embed_matrix")
        movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles, name = "movie_title_embed_layer")
        movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)
    
    # Use convolution kernels of different sizes for the text embedding layer for convolution and maximum pooling
    pool_layer_lst = []
    for window_size in window_sizes:
        with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):
            filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num],stddev=0.1),name = "filter_weights")
            filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")
            
            conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand, filter_weights, [1,1,1,1], padding="VALID", name="conv_layer")
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer,filter_bias), name ="relu_layer")
            
            maxpool_layer = tf.nn.max_pool(relu_layer, [1,sentences_size - window_size + 1 ,1,1], [1,1,1,1], padding="VALID", name="maxpool_layer")
            pool_layer_lst.append(maxpool_layer)

    #Dropout层
    with tf.name_scope("pool_dropout"):
        pool_layer = tf.concat(pool_layer_lst, 3, name ="pool_layer")
        max_num = len(window_sizes) * filter_num
        pool_layer_flat = tf.reshape(pool_layer , [-1, 1, max_num], name = "pool_layer_flat")
    
        dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name = "dropout_layer")
    return pool_layer_flat, dropout_layer


# Connect all layers of Movie together
def get_movie_feature_layer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer, embed_dim):
    with tf.name_scope("movie_fc"):
        # First layer fully connected
        movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, embed_dim, name = "movie_id_fc_layer", activation=tf.nn.relu)
        movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer, embed_dim, name = "movie_categories_fc_layer", activation=tf.nn.relu)
    
        # The second layer is fully connected
        movie_combine_layer = tf.concat([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)  #(?, 1, 96)
        movie_combine_layer = tf.contrib.layers.fully_connected(movie_combine_layer, 200, tf.tanh)  #(?, 1, 200)
    
        movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])
    return movie_combine_layer, movie_combine_layer_flat


# Get batch
def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]
        
        
#----------------------------------------------------------------------------------------#
#### For Recommend movies.        


# The idea is to calculate the cosine similarity between the feature vector of the current movie and the feature matrix of the entire movie, 
# and take the top_k with the largest similarity. 
def recommend_same_type_movie(movie_id_val, load_dir, movie_matrics, movies_orig, movieid2idx, top_k = 20):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)
        
        norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keep_dims=True))
        normalized_movie_matrics = movie_matrics / norm_movie_matrics

        # Recommend movies of the same type
        probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(normalized_movie_matrics))
        sim = (probs_similarity.eval())
        
        print("The original movie is：{}".format(movies_orig[movieid2idx[movie_id_val]]))
        print("The following are recommendations：")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        while len(results) != 5:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in (results):
            print(movies_orig[val])
        return results


# The idea is to use the user feature vector and the movie feature matrix to calculate the ratings of all movies, 
# take the top_k with the highest rating, and also add some random selection parts.
def recommend_your_favorite_movie(user_id_val, load_dir, users_matrics, movie_matrics, movies_orig, top_k = 10):

    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # Recommend movies you like
        probs_embeddings = (users_matrics[user_id_val-1]).reshape([1, 200])

        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))
        sim = (probs_similarity.eval())
    
        print("The following are recommendations：")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        while len(results) != 5:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in (results):
            print(movies_orig[val])

        return results
    
    
# First, select top_k individuals who like a certain movie, and obtain the user feature vectors of these individuals.
# Then calculate the ratings of these people for all the movies
# Choose the movie with the highest rating for everyone as a recommendation
# Random selection
import random
def recommend_other_favorite_movie(movie_id_val, load_dir, movie_matrics, movieid2idx, users_matrics, movies_orig, users_orig, top_k = 20):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        probs_movie_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])
        probs_user_favorite_similarity = tf.matmul(probs_movie_embeddings, tf.transpose(users_matrics))
        favorite_user_id = np.argsort(probs_user_favorite_similarity.eval())[0][-top_k:]
    
        print("The original movie is：{}".format(movies_orig[movieid2idx[movie_id_val]]))
        
        print("The people who like to watch this movie are：{}".format(users_orig[favorite_user_id-1]))
        probs_users_embeddings = (users_matrics[favorite_user_id-1]).reshape([-1, 200])
        probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(movie_matrics))
        sim = (probs_similarity.eval())

        p = np.argmax(sim, 1)
        print("People who like to watch this movie also like to watch：")

        results = set()
        while len(results) != 5:
            c = p[random.randrange(top_k)]
            results.add(c)
        for val in (results):
            print(movies_orig[val])
        return results
    
    
    
#----------------------------------------------------------------------------------------#    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
