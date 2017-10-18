from io import StringIO
import pandas as pd

csv_data = '''A,B,C,d
            1.0,2.0,3.0,4.0
            5.0,6.0,,8.0
            0.0,11.0,12.0,'''

'''csv_data = unicode(csv_data, 'utf-8')'''
df = pd.read_csv(StringIO(csv_data))
print(df)

df.isnull().sum()
print(df.values)

print(df.dropna())

print(df.dropna(axis=1))

print(df.dropna(how='all'))

print(df.dropna(thresh=4))

print(df.dropna(subset=['C']))

from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr = imr.fit(df)
imputed_data = imr.transform(df.values)
print(df.values)
print('-------------------------------------------')
print(imputed_data)

df = pd.DataFrame([['green', 'M', 10.1, 'class1'],
                   ['red', 'L', 13.5, 'class2'],
                   ['blue', 'XL', 15.3, 'class1']])

df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

size_mapping = {
    'XL':3,
    'L':2,
    'M':1
    }

df['size'] = df['size'].map(size_mapping)
print(df)

import numpy as np

class_mapping = {label:idx for idx,label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)

df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)

inv_class_mapping = {v:k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)

from sklearn.preprocessing import LabelEncoder

class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)
print(class_le.inverse_transform(y))

X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:,0] = color_le.fit_transform(X[:,0])
print(X)

from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features=[0])
print(ohe.fit_transform(X).toarray())
print(X)
print(ohe.fit_transform(X))
print(pd.get_dummies(df[['price', 'color', 'size']]))
