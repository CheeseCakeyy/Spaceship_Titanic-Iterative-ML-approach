#This iteration will cover usage of LightGBM model which uses leaf wise growth unlike XGBoost which use level wise growth 
'''The advantage this model provides over XGBoost is that: 1)leaf-wise growth = faster loss reduction 
                                                           2)Much faster on larger datasets(the one we are using is quite large too)
                                                           3)Better handling of categorical features(No need for one-hot encoding)'''

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score
from sklearn.preprocessing import OneHotEncoder


train_path = "data/train.csv"
df = pd.read_csv(train_path)

df.info()

#--------------------
'''Feature Construction'''
#--------------------
#Old features from iter(1,2):
df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split('/',expand=True) 
df = df.drop('Cabin',axis=1)
df["Total_spend"] = df[["RoomService","ShoppingMall", "FoodCourt", "Spa", "VRDeck"]].sum(axis=1)
df["Has_spent"] = (df["Total_spend"] > 0).astype(int)
df['Luxury_ratio'] = df[["Spa","VRDeck"]].sum(axis=1) / (df["Total_spend"] +1) 
df["Food_ratio"] = df["FoodCourt"] / (df["Total_spend"] +1)
df["Group"] = df["PassengerId"].str.split("_").str[0]
df["Group_size"] = df.groupby("Group")["Group"].transform("count")
df["Is_solo"] = (df["Group_size"] == 1).astype(int)

#since we are using LGBM which doesnt need one-hot-encoding we'll have to change the dtype of categorical features to 'category'
cat_cols = ['HomePlanet', 'Deck','Cabin_num', 'Side','Destination']
bool_cols = ['CryoSleep','VIP','Has_spent','Is_solo']
df[bool_cols] = df[bool_cols].astype('bool')
df[cat_cols] = df[cat_cols].astype('category')
print(df.head())

#dropping useless columns now
useless_cols = ['PassengerId','Name','Group']
df = df.drop(columns = useless_cols)
df.info()

#seperating Label and features
X = df.drop("Transported",axis=1)
y = df["Transported"]


#--------------
'''Preprocessing Pipelines'''
#--------------
num_cols = ['Age','RoomService','FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'Total_spend', 'Luxury_ratio','Food_ratio',"Group_size"] #impute with median 
cat_cols = ['HomePlanet', 'Deck','Cabin_num', 'Side','Destination'] #impute with most_frequent and apply onehotencoding
bool_cols = ['CryoSleep','VIP','Has_spent','Is_solo'] #impute with most_frequent 


num_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='median'))
])
cat_pipe = Pipeline([
    ('impute',SimpleImputer(strategy='most_frequent')),
        ('encode',OneHotEncoder(handle_unknown='ignore'))

])

preprocessor = ColumnTransformer(
    transformers=[
        ('num',num_pipe,num_cols),
        ('cat',cat_pipe,cat_cols),
    ],
    remainder='passthrough'
)


#--------------
'''LGBM'''
#--------------
model = LGBMClassifier(
                objective="binary",
                boosting_type="gbdt",
                random_state=42,
                n_jobs=-1,
                max_depth=6,
                learning_rate=0.1,
                )


lgbm_pipeline = Pipeline([
    ('prep',preprocessor),
    ('model',model)
])


cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
score = cross_val_score(
    lgbm_pipeline,
    X,
    y,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

print("LGBM CV scores: ", score)
print("LGBM CV mean:", score.mean())
print("LGBM CV std :", score.std())

'''LGBM CV scores:  [0.8119609  0.80103508 0.8119609  0.81933257 0.79401611]
LGBM CV mean: 0.8076611096810395
LGBM CV std : 0.008980974822494991'''

#------------
'''Iteration 4 Submission '''
#------------
test_path = "data/test.csv"
test_df = pd.read_csv(test_path)

#creating features we created in train df
test_df[["Deck", "Cabin_num", "Side"]] = test_df["Cabin"].str.split('/',expand=True) 
test_df = test_df.drop('Cabin',axis=1)
test_df["Total_spend"] = test_df[["RoomService","ShoppingMall", "FoodCourt", "Spa", "VRDeck"]].sum(axis=1)
test_df["Has_spent"] = (test_df["Total_spend"] > 0).astype(int)
test_df['Luxury_ratio'] = test_df[["Spa","VRDeck"]].sum(axis=1) / (test_df["Total_spend"] +1) 
test_df["Food_ratio"] = test_df["FoodCourt"] / (test_df["Total_spend"] +1)
test_df["Group"] = test_df["PassengerId"].str.split("_").str[0]
test_df["Group_size"] = test_df.groupby("Group")["Group"].transform("count")
test_df["Is_solo"] = (test_df["Group_size"] == 1).astype(int)
test_df[bool_cols] = test_df[bool_cols].astype('bool')
test_df[cat_cols] = test_df[cat_cols].astype('category')
useless_cols = ['PassengerId','Name','Group']
X_testdataset = test_df.drop(columns = useless_cols)

#predicting on xgb_pipeline
lgbm_pipeline.fit(X,y)
y_pred = lgbm_pipeline.predict(X_testdataset)

submission = pd.DataFrame({
    "PassengerId": test_df["PassengerId"],
    'Transported': y_pred
})


# submission.to_csv('submission_iter(4).csv',index=False) #0.80734 best so far 293/2292 

