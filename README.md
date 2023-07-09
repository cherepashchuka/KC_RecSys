# KC_RecSys
Recommendation system for the social network of students
HitRate@5 > 0.6, average service time <= 0.5 sec, less than 4 GB RAM per launch

1 * Postgres, pandas, sklearn, catboost

model_training_v1.ipynb - first model with smaller number of components PCA, but additional features from users actions
ROC-AUC: 0.69
ACCURACY: 0.86
HitRate@5: 0.605

2 * Postgres, pandas, sklearn, catboost

model_training_v2.ipynb - second model with features from PCA and more number of components(100)
ROC-AUC: 0.65
ACCURACY: 0.86
HitRate@5: 0.616

3 * Postgres, pandas, sklearn, catboost, pytorch, bert

get_embeddings.ipynb - get embeddings from posts

model_training_v3.ipynb - third model with embeddings from bert model

ROC-AUC: 0.62
ACCURACY: 0.88
HitRate@5: 0.576

4 * Postgres, pandas, FastAPI

app.py - service for working with the recommendation system via API. Implemented the possibility of A/B testing of new models and dividing users into test and control groups
