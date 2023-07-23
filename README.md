# KC_RecSys
Recommendation system for the social network of students
HitRate@5 > 0.6, average service time <= 0.5 sec, less than 4 GB RAM per launch

## 1 * model training\model_training_v1.ipynb

First model with smaller number of components PCA, but additional features from users actions

Postgres, pandas, sklearn, catboost

**ROC-AUC:** 0.69\
**ACCURACY:** 0.86\
**HitRate@5:** 0.605

## 2 * model training\model_training_v2.ipynb

Second model with features from PCA and more number of components(100)

Postgres, pandas, sklearn, catboost

ROC-AUC: 0.65\
ACCURACY: 0.86\
HitRate@5: 0.616

## 3 * model training\model_training_v3.ipynb, get_embeddings.ipynb

Third model with embeddings from bert model

Postgres, pandas, sklearn, catboost, pytorch, bert

ROC-AUC: 0.62\
ACCURACY: 0.88\
HitRate@5: 0.576

## 4 * app.py

Service for working with the recommendation system via API. Implemented the possibility of A/B testing of new models and dividing users into test and control groups

Postgres, pandas, FastAPI, pydantic
```
.
├── config.json             # Config file with constants such a database url, salt and models path
├── data_funcs.py           # Support functions for working with data
├── model_funcs.py          # Support functions for working with models
├── schema.py               # Response model schema
└── app.py                  # Main service
```
