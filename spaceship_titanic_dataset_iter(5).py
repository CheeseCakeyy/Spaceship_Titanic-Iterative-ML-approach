#This iteration is a try at increasing the accuracy on the test data through usage of the Gradient boosting algorithm 'Catboost'
#Catboost is good at dealing with categorical features and might yield good results 
#Apparantly catboost also does missing value handing :   Treats NaN as a separate direction
#                                                        Learns whether missing values should go left or right
#                                                        Does this independently per split


import pandas as pd 
from catboost import CatBoostClassifier,Pool,cv
from sklearn.model_selection import train_test_split


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

#handling null values outside of pipeline since catboost partially provide support to sklearn tags for that reason there wont be a preprocessing pipeline
cat_cols = ['HomePlanet', 'Deck','Cabin_num', 'Side','Destination']
bool_cols = ['CryoSleep','VIP','Has_spent','Is_solo']
df[bool_cols] = df[bool_cols].astype('Int64')
df[cat_cols] = df[cat_cols].fillna('Missing')
df[cat_cols] = df[cat_cols].astype('category')

useless_cols = ['PassengerId','Name','Group']
df = df.drop(columns = useless_cols)
df.info()

#seperating the target and features 
X = df.drop('Transported',axis=1)
y = df['Transported']

#splittin in trainvalidatin splits
X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)


#--------------
'''Catboost'''
#--------------
model = CatBoostClassifier(
    depth=6,             # tree depth (controls complexity)
    learning_rate=0.05,  # smaller LR + more iterations = better generalization
    iterations=1200,     # more trees
    l2_leaf_reg=3,       # L2 regularization to reduce overfitting
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_state=42,
    verbose=100
)

model.fit(
    X_train,
    y_train,
    cat_features=cat_cols,
    eval_set=(X_val,y_val)
)
#The result image explaination:
'''The resut is in form: 
                        100: learn: 0.8123382  test: 0.8073606  best: 0.8073606 (98)

100(iteration number/ tree count so far), learn(accuracy on training set), test(validation set accuracy), best(best val accuracy seen so far and at what iteration),total/remaining(estimated remaining time)

Our final output explainatin: 
                        Out of 1500 iterations the best iteration with a val accuracy of 81.6 was at iter(824)
                        what the model did after that was it discarded the trees after 824 since no gain was seen in val accuracy 
                        so final model = best generalization point
                        we automatically avoid overfitting
 '''


#---------------------
'''Iteration 5 Submission'''
#---------------------
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

test_df[bool_cols] = test_df[bool_cols].astype('Int64')
test_df[cat_cols] = test_df[cat_cols].fillna('Missing')
test_df[cat_cols] = test_df[cat_cols].astype('category')

useless_cols = ['PassengerId','Name','Group',]
X_test = test_df.drop(columns = useless_cols)

model.fit(X,y,cat_features=cat_cols)
y_pred = model.predict(X_test)

submission = pd.DataFrame({
    'PassengerId' : test_df['PassengerId'],
    'Transported' : y_pred
})


submission.to_csv('submission_iter(5).csv',index=False)  #0.8024 less than LGBM which was 0.80734

'''When changing algorithms stops giving gains, the problem is no longer the model — it’s the representation.'''
