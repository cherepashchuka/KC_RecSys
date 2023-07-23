"""
Support functions for working with models:
get_model_path - assigning a path depending on the type of model: test or control
load_models - loading test and control models
get_recs - get recommendations for specific user
"""

import json
import pandas as pd
from schema import PostGet, Response
from catboost import CatBoostClassifier
from loguru import logger

config = json.load(open(file="config.json", encoding="utf-8"))


def get_model_path(type: str) -> str:
    """
    :param type: model type - control or test
    :return: string with path to the required model
    """
    if type == 'control':
        MODEL_PATH = config['control_model_path']
    elif type == 'test':
        MODEL_PATH = config['test_model_path']
    else:
        MODEL_PATH = config['control_model_path']
    return MODEL_PATH


def load_model(type: str):
    """
    :param type: model type - control or test
    :return: loaded catboost model
    """
    model_path = get_model_path(type)
    loaded_model = CatBoostClassifier()
    loaded_model.load_model(model_path)
    return loaded_model


def get_recs(limit: int, user_posts: pd.DataFrame, content: pd.DataFrame, model, exp_group: str) -> Response:
    """
    :param limit: number of recommendations
    :param user_posts: combined pandas dataframe with users and posts
    :param content: all available posts
    :param model: loaded model
    :param exp_group: user test group
    :return: requested number of recommendations
    """
    logger.info("predicting")
    user_posts['predict'] = model.predict_proba(user_posts)[:, 1]
    user_posts = user_posts.sort_values('predict')[-limit:].index
    recs = []

    logger.info("forming a response")
    for i in user_posts:
        item = PostGet(**{'id': i[1], 'text': content[content['post_id'] == i[1]]['text'].values[0],
                          'topic': content[content['post_id'] == i[1]]['topic'].values[0]})
        recs.append(item)
    return Response(**{'exp_group': exp_group, 'recommendations': recs})
