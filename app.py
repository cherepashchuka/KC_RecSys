import os
import pandas as pd
from typing import List
from catboost import CatBoostClassifier
from fastapi import FastAPI
from schema import PostGet, Response 
from datetime import datetime
from sqlalchemy import create_engine
from loguru import logger
import hashlib

SALT = 'justwmodel'


app = FastAPI()


def load_features(query: str) -> pd.DataFrame:
    CHUNKSIZE = 200000
    DATABASE_URL = "###"
    engine = create_engine(DATABASE_URL)
    conn = engine.connect().execution_options(stream_results=True)
    chunks = []
    for chunk_dataframe in pd.read_sql(query, conn, chunksize=CHUNKSIZE):
        chunks.append(chunk_dataframe)
    conn.close()
    return pd.concat(chunks, ignore_index=True)
    

def get_exp_group(id: int) -> str:
    value_str = str(id) + SALT
    percent = int(hashlib.md5(value_str.encode()).hexdigest(), 16) % 100
    if percent < 50:
        return "control"
    elif percent < 100:
        return "test"
    return "unknown"
    
def get_model_path(path: str, type: str) -> str:
    if os.environ.get("IS_LMS") == "1":
        if type == 'control':
            MODEL_PATH = '/workdir/user_input/model_control'
        elif type == 'test':
            MODEL_PATH = '/workdir/user_input/model_test'
    else:
        MODEL_PATH = path
    return MODEL_PATH
    
def load_models():
    model_path_control = get_model_path("final","control")
    model_path_test = get_model_path("final", "test")

    from catboost import CatBoostClassifier

    model_c = CatBoostClassifier()
    model_c.load_model(model_path_control)
    
    model_t = CatBoostClassifier()
    model_t.load_model(model_path_test)

    return model_c, model_t
    
    
def get_recs(id, time, limit, all_users_features, all_posts_features, all_liked_posts, content, model, exp_group):
    user_features = all_users_features[all_users_features['user_id'] == id]
    user_posts = all_posts_features.assign(**dict(zip(user_features.columns, user_features.values[0])))
    user_posts.insert(loc=0, column='month', value=time.month)
    user_posts.insert(loc=0, column='hour', value=time.hour)
    user_liked_posts = all_liked_posts[all_liked_posts['user_id'] == id]['post_id'].values
    user_posts = user_posts[~user_posts.index.isin(user_liked_posts)]
    user_posts = user_posts.set_index(['user_id', 'post_id'])
    user_posts['predict'] = model.predict_proba(user_posts)[:, 1]
    user_posts = user_posts.sort_values('predict')[-limit:].index
    recs = []
    for i in user_posts:
        item = PostGet(**{'id': i[1], 'text': content[content['post_id'] == i[1]]['text'].values[0],
                          'topic': content[content['post_id'] == i[1]]['topic'].values[0]})
        recs.append(item)
    return Response(**{'exp_group': exp_group, 'recommendations': recs})
    
all_users_features = load_features("SELECT * FROM public.user_data")

all_posts_features_t = load_features("SELECT * FROM public.a_cherepaschuk_post_features_v4_main_lesson_22")

all_posts_features_t = all_posts_features_t.drop('index', axis=1)

all_posts_features_c = load_features("select * from public.a_cherepaschuk_post_features_main_lesson_22")

content = all_posts_features_c[['post_id', 'text', 'topic']]

all_posts_features_c = all_posts_features_c.drop(['text', 'index'], axis=1)

all_liked_posts = load_features("select distinct post_id, user_id from public.feed_data where action = 'like'")


model_control, model_test = load_models()


@app.get("/post/recommendations/", response_model=Response)
def recommended_posts(
        id: int,
        time: datetime,
        limit: int = 10) -> Response:
    
    exp_group = get_exp_group(id)
    
    if exp_group == 'control':
        return get_recs(id, time, limit, all_users_features, all_posts_features_c, all_liked_posts, content, model_control, exp_group)
    elif exp_group == 'test':
        return get_recs(id, time, limit, all_users_features, all_posts_features_t, all_liked_posts, content, model_test, exp_group)
    else:
        raise ValueError('unknown group')