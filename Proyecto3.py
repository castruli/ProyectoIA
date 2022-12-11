# -*- coding: utf-8 -*-
"""
Created on Tue Nov 29 23:40:03 2022

@author: Cristobal
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from sklearn.neural_network import MLPClassifier

col_names = ['buying','maint','doors','persons','lug_boot','safety','class']
df = pd.read_csv("car.data",names=col_names)

buying_label = {ni:n for n,ni in enumerate(set(df['buying']))}
maint_label = {ni:n for n,ni in enumerate(set(df['maint']))}
doors_label = {ni:n for n,ni in enumerate(set(df['doors']))}
persons_label = {ni:n for n,ni in enumerate(set(df['persons']))}
lug_boot_label = {ni:n for n,ni in enumerate(set(df['lug_boot']))}
safety_label = {ni:n for n,ni in enumerate(set(df['safety']))}
class_label = {ni:n for n,ni in enumerate(set(df['class']))}

df1=df

df1['buying'] = df1['buying'].map(buying_label)

df1['maint'] = df1['maint'].map(maint_label)
df1['doors'] = df1['doors'].map(doors_label)
df1['persons'] = df1['persons'].map(persons_label)
df1['lug_boot'] = df1['lug_boot'].map(lug_boot_label)
df1['buying'] = df1['buying'].map(buying_label)
df1['safety'] = df1['safety'].map(safety_label)
df1['class'] = df1['class'].map(class_label)

lb = LabelEncoder()
df2 = df

for i in df2.columns:
    df2[i]=lb.fit_transform(df2[i])

df1.dtypes
plt.figure(figsize=(10,6))
sns.heatmap(df1.corr(),annot=True)

Xfeatures = df1[['buying','maint','doors','persons','lug_boot','safety']]
ylabels = df1['class']

X_train, X_test, Y_train, Y_test = train_test_split(Xfeatures,ylabels, test_size=0.2, random_state=42)

logic = LogisticRegression()
logic.fit(X_train, Y_train)

print("Accuracy: ", accuracy_score(Y_test, logic.predict(X_test)))

naive = MultinomialNB()
naive.fit(X_train, Y_train)

print("Accuracy: ", accuracy_score(Y_test, naive.predict(X_test)))

neuro = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes = (5,2), random_state=1)
neuro.fit(X_train,Y_train)

print("Accuracy: ", accuracy_score(Y_test, neuro.predict(X_test)))

tree = DecisionTreeClassifier()
tree.fit(X_train,Y_train)

print("Accuracy: ", accuracy_score(Y_test, tree.predict(X_test)))







    