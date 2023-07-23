"""
Support functions for working with data:
load_features - loading large amount of data from sql db (Postgres in this project)
get_exp_group - assigning exp group to users
process_features - brings all the features into the desired form for the model
"""

import pandas as pd
import hashlib
import json
from sqlalchemy import create_engine
from datetime import datetime
from loguru import logger

config = json.load(open(file="config.json", encoding="utf-8"))


def load_features(query: str) -> pd.DataFrame:
    """
    :param query: sql-query for downloading your data
    :return: pandas dataframe with data which was received after execute your sql query
    """
    engine = create_engine(config['database_url'])
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=config['chunksize']):
        chunks.append(chunk_dataframe)
        logger.info(f"got chunk: {len(chunk_dataframe)}")
    conn.close()
    return pd.concat(chunks, ignore_index=True)


def get_exp_group(id: int) -> str:
    """
    :param id: id of the user to whom we want to assign the group
    :return: assigned user group
    """
    value_str = str(id) + config['salt']
    percent = int(hashlib.md5(value_str.encode()).hexdigest(), 16) % 100
    if percent < config['percent_1']:
        return "control"
    elif percent < config['percent_2']:
        return "test"
    return "unknown"


def process_features(
        id: int, time: datetime, all_users_features: pd.DataFrame, all_posts_features: pd.DataFrame,
        all_liked_posts: pd.DataFrame
) -> pd.DataFrame:
    logger.info(f"user_id: {id}")
    logger.info("reading features")
    user_features = all_users_features[all_users_features['user_id'] == id]

    logger.info("zipping features")
    user_posts = all_posts_features.assign(**dict(zip(user_features.columns, user_features.values[0])))

    logger.info("add time info")
    user_posts.insert(loc=0, column='month', value=time.month)
    user_posts.insert(loc=0, column='hour', value=time.hour)

    logger.info("deleting liked posts")
    user_liked_posts = all_liked_posts[all_liked_posts['user_id'] == id]['post_id'].values
    user_posts = user_posts[~user_posts.index.isin(user_liked_posts)]

    user_posts = user_posts.set_index(['user_id', 'post_id'])
    logger.info("final user_posts done")
    return user_posts
