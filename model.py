import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import math
cleaned_data = pd.read_csv('cleaned_data.csv')
# Ordinal feature encoding
df = cleaned_data.copy()
#encode = ['EDUCATION','MARRIAGE']
#target = 'default payment next month'
features_response = df.columns.tolist()
items_to_remove = ['ID', 'SEX', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
                   'EDUCATION_CAT', 'graduate school', 'high school', 'none',
                   'others', 'university']
features_response = [item for item in features_response if item not in items_to_remove]
#features_response = ['LIMIT_BAL','EDUCATION','MARRIAGE','AGE','PAY_1','BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6','PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6','default payment next month']

'''for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]'''

#X = df.drop('default payment next month', axis=1)
#Y = df['default payment next month']
#df['species'] = df['species'].apply(target_encode)
from sklearn.model_selection import train_test_split

X = df[features_response[:-1]].values
y = df['default payment next month'].values

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=24)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression()

log_reg.fit(X_train, y_train)
import pickle
pickle.dump(log_reg, open('credit_card_clf2.pkl', 'wb'))

loaded_model = pickle.load(open('credit_card_clf2.pkl', 'rb'))
result = loaded_model.score(X_test, y_test)
print(result)

