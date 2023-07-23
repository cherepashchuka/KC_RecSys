import os
import psutil
import json
from fastapi import FastAPI, HTTPException
from datetime import datetime
from data_funcs import load_features, get_exp_group, process_features
from model_funcs import load_model, get_recs
from loguru import logger
from schema import Response

config = json.load(open(file="config.json", encoding="utf-8"))
app = FastAPI()

logger.info('loading features')

logger.info('loading all_users_features')
all_users_features = load_features(config['query_users'])  # loading features for all users from postgresql db

logger.info('loading all_posts_features TEST')
all_posts_features_t = load_features(config['query_posts_t']).drop('index', axis=1)  # loading prepared new features for test model

logger.info('loading all_posts_features CONTROL')
all_posts_features_c = load_features(config['query_posts_c'])  # loading prepared features for control model

# in order not to make another request, just take the columns we need from the already downloaded data
content = all_posts_features_c[['post_id', 'text', 'topic']]

all_posts_features_c = all_posts_features_c.drop(['text', 'index'], axis=1)

logger.info('loading all_liked_posts')
all_liked_posts = load_features(config['query_liked_posts'])  # loading all posts that have been liked

logger.info('loading models')
model_test = load_model('test')
model_control = load_model('control')

logger.info(f"memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024} MB")


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 10) -> Response:
    # let's show the requester a more informative error if the user with such a user_id does not exist
    if id not in all_users_features['user_id'].values:
        raise HTTPException(404, 'user does not exist')

    exp_group = get_exp_group(id)

    # from the previously loaded data form a pandas dataframe for prediction and select a model based on the user group
    if exp_group == 'control':
        model = model_control
        user_posts = process_features(id, time, all_users_features, all_posts_features_c, all_liked_posts)
    elif exp_group == 'test':
        model = model_test
        user_posts = process_features(id, time, all_users_features, all_posts_features_t, all_liked_posts)
    else:
        raise ValueError('unknown group')

    logger.info(f"memory usage: {psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024} MB")

    return get_recs(limit, user_posts, content, model, exp_group)
