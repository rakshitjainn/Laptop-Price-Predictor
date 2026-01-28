import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('laptop_data_cleaned.csv')
X = df.drop(columns=['Price'])
y = np.log(df['Price']) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=2)

categorical_cols = X.select_dtypes(include=['object']).columns

step1 = ColumnTransformer(transformers=[
    ('col_tnf', OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore'), categorical_cols)
], remainder='passthrough')

step2 = RandomForestRegressor(n_estimators=100,
                              random_state=3,
                              max_samples=0.5,
                              max_features=0.75,
                              max_depth=15)

pipe = Pipeline([
    ('step1', step1),
    ('step2', step2)
])

pipe.fit(X_train, y_train)

pickle.dump(pipe, open('LaptopPriceModel.pkl', 'wb'))