# -*- coding: utf-8 -*-

pip install h2o
pip install adjustText
pip install bayesian-optimization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
plt.style.use('ggplot')
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import tensorflow as tf
import keras
from keras import layers, optimizers
from keras.models import Model, load_model
import math
from adjustText import adjust_text

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn import linear_model
import xgboost as xgb
from sklearn.ensemble import StackingRegressor
from bayes_opt import BayesianOptimization

data = pd.read_csv('stroke.csv')
data = data.loc[:, ['Year', 'Municipality', 'stroke']]

data = data.rename(columns={'stroke': 'target'})
n_mun = np.unique(data.Municipality.values).shape[0]

"""# Feature Engineering"""

data['target_lag1'] = data['target'].shift(n_mun) # Lagged Variables
data['target_lag2'] = data['target'].shift(n_mun*2)
data = data.fillna(-1)
train = data[data.Year < 2021]
test = data[data.Year == 2021]

data.head(20)

"""## Create Embeddings for Municipalities"""

to_embed = ['Year', 'Municipality']

le = LabelEncoder()
yy = le.fit(train[to_embed[0]]).transform(train[to_embed[0]])
le2 = LabelEncoder()
mun = le2.fit(train[to_embed[1]]).transform(train[to_embed[1]])

y_train = train[['target']].values

inputs = []
outputs = []
for c in to_embed:
    num_unique_vals = int(train[c].nunique())
    embed_dim = int(min(np.ceil(num_unique_vals/2), 50))
    inp = layers.Input(shape = (1, ))
    out = layers.Embedding(num_unique_vals, embed_dim, name = c )(inp)
    out = layers.Reshape(target_shape=(embed_dim, ))(out)
    inputs.append(inp)
    outputs.append(out)

x = layers.Concatenate()(outputs)
x = layers.Dense(128, activation = 'relu')(x)
x = layers.Dropout(.3)(x)
x = layers.Dense(64, activation = 'relu')(x)
x = layers.Dropout(.2)(x)
y = layers.Dense(1, activation = 'relu')(x)

model = Model(inputs = inputs, outputs = y)
model.summary()

tf.keras.utils.plot_model(model)

l_r = 0.001
EPOCHS = 100
BATCH_SIZE = 256
model.compile(loss = tf.keras.losses.MeanSquaredError(),
              optimizer = tf.keras.optimizers.legacy.Adam(learning_rate = l_r))
model.fit([yy, mun],
          y_train,
          epochs = EPOCHS,
          batch_size = BATCH_SIZE)

emb_yy = model.layers[2].get_weights()[0]
emb_mun = model.layers[3].get_weights()[0]
yys = list(le.inverse_transform(range(len(np.unique(yy)))))
muns = list(le2.inverse_transform(range(len(np.unique(mun)))))

kmeans_ = KMeans(random_state=3, n_clusters=5)
tsne = TSNE(init='pca', random_state=120, method='exact')
Y = tsne.fit_transform(emb_mun)
kmeans = kmeans_.fit(Y)

cmap = plt.cm.get_cmap('tab20', len(np.unique(kmeans.labels_)))

plt.figure(figsize=(15,10))
plt.scatter(-Y[:, 0], -Y[:, 1], c=cmap(kmeans.labels_))

texts = []
for i, txt in enumerate(muns):
    texts.append(plt.text(-Y[i, 0], -Y[i, 1], txt))

adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

plt.show()

col_names2 = ['Embs_Mun_' + str(i) for i in range(emb_mun.shape[1])]
df_embs = pd.DataFrame(emb_mun, columns=col_names2)

df_embs['Municipality'] = le2.inverse_transform(range(n_mun))

train_f = pd.merge(train, df_embs)
test_f = pd.merge(test, df_embs)

y_train = train_f.loc[:, 'target'].values
y_test = test_f.loc[:, 'target'].values
X_train = train_f.drop(['Year', 'Municipality', 'target'], axis = 1)
X_test = test_f.drop(['Year', 'Municipality', 'target'], axis = 1)

"""# Modelling with Stacking"""

def rf_cv(n_estimators, min_samples_split, max_features):
    model = RandomForestRegressor(
        n_estimators=int(n_estimators),
        min_samples_split=int(min_samples_split),
        max_features=max_features,
        random_state=42
    )
    return cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()

def gb_cv(n_estimators, learning_rate, max_depth):
    model = GradientBoostingRegressor(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        random_state=42
    )
    return cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()

def xgb_cv(n_estimators, learning_rate, max_depth, gamma, colsample_bytree):
    model = xgb.XGBRegressor(
        n_estimators=int(n_estimators),
        learning_rate=learning_rate,
        max_depth=int(max_depth),
        gamma=gamma,
        colsample_bytree=colsample_bytree,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    return cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()

def mlp_cv(hidden_layer_sizes, alpha, learning_rate_init):
    hidden_layer_sizes = int(hidden_layer_sizes)
    model = MLPRegressor(
        hidden_layer_sizes=(hidden_layer_sizes,),
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=2000,
        random_state=42
    )
    return cross_val_score(model, X_train, y_train, cv=5, scoring='neg_mean_squared_error').mean()

#Random Forest
rf_bo = BayesianOptimization(
    rf_cv,
    {'n_estimators': (10, 200),
     'min_samples_split': (2, 10),
     'max_features': (0.1, 0.999)}
)
rf_bo.maximize(init_points=10, n_iter=40)

#Gradient Boosting
gb_bo = BayesianOptimization(
    gb_cv,
    {'n_estimators': (10, 150),
     'learning_rate': (0.01, 1),
     'max_depth': (1, 10)}
)
gb_bo.maximize(init_points=10, n_iter=40)

#XGBoost
xgb_bo = BayesianOptimization(
    xgb_cv,
    {'n_estimators': (10, 200),
     'learning_rate': (0.01, 1),
     'max_depth': (1, 10),
     'gamma': (0, 5),
     'colsample_bytree': (0.1, 0.999)}
)
xgb_bo.maximize(init_points=10, n_iter=40)

#MLP
mlp_bo = BayesianOptimization(
    mlp_cv,
    {'hidden_layer_sizes': (10, 200),
     'alpha': (1e-5, 1e-1),
     'learning_rate_init': (1e-5, 1e-1)}
)
mlp_bo.maximize(init_points=10, n_iter=40)

#Best Hyperparams
rf_best_params = rf_bo.max['params']
gb_best_params = gb_bo.max['params']
xgb_best_params = xgb_bo.max['params']
mlp_best_params = mlp_bo.max['params']

#Random Forest
rf_best_model = RandomForestRegressor(
    n_estimators=int(rf_best_params['n_estimators']),
    min_samples_split=int(rf_best_params['min_samples_split']),
    max_features=rf_best_params['max_features'],
    random_state=42
)

#Gradient Boosting
gb_best_model = GradientBoostingRegressor(
    n_estimators=int(gb_best_params['n_estimators']),
    learning_rate=gb_best_params['learning_rate'],
    max_depth=int(gb_best_params['max_depth']),
    random_state=42
)

#XGBoost
xgb_best_model = xgb.XGBRegressor(
    n_estimators=int(xgb_best_params['n_estimators']),
    learning_rate=xgb_best_params['learning_rate'],
    max_depth=int(xgb_best_params['max_depth']),
    gamma=xgb_best_params['gamma'],
    colsample_bytree=xgb_best_params['colsample_bytree'],
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

#MLP
mlp_best_model = MLPRegressor(
    hidden_layer_sizes=(int(mlp_best_params['hidden_layer_sizes']),),
    alpha=mlp_best_params['alpha'],
    learning_rate_init=mlp_best_params['learning_rate_init'],
    max_iter=2000,
    random_state=42
)

estimators = [
    ('rf', rf_best_model),
    ('gb', gb_best_model),
    ('xgb', xgb_best_model),
    ('mlp', mlp_best_model)
]
stack = StackingRegressor(
    estimators=estimators,
    final_estimator=linear_model.LinearRegression()
)
stack.fit(X_train, y_train)
preds_fin = stack.predict(X_test) # Final Predictions

def show_bars(base_naive, mae_naive, mse_naive, 
             baise_ema, mae_ema, mse_ema, 
             inla_chisq, mae_inla, mse_inla, 
             ml_chisq, mae_ml, mse_ml):
    models = ['Chisq', 'MAE', 'MSE']
    bar_width = 1
    index = np.arange(4)
    fig, axs = plt.subplots( 1, len(models),figsize=(10, 10))

    naive_data = [base_naive, mae_naive, mse_naive]
    ema_data = [baise_ema, mae_ema, mse_ema]
    inla_data = [inla_chisq, mae_inla, mse_inla]
    ml_data = [ml_chisq, mae_ml, mse_ml]
    colors = ['#1f77b4', '#2ca02c', '#f7c06c', '#d62728']
    labels = ['Naive', 'EMA', 'INLA', 'ML']

    for i, ax in enumerate(axs):
        ax.bar(index[0], naive_data[i], bar_width, color=colors[0], alpha=0.7, label=labels[0])
        ax.bar(index[1], ema_data[i], bar_width, color=colors[1], alpha=0.7, label=labels[1])
        ax.bar(index[2], inla_data[i], bar_width, color=colors[2], alpha=0.7, label=labels[2])
        ax.bar(index[3], ml_data[i], bar_width, color=colors[3], alpha=0.7, label=labels[3])

        ax.set_ylabel(models[i])
        ax.grid(False)
        ax.legend(loc='upper right')
        ax.set_xticks(index)
        ax.set_xticklabels(labels)

    plt.xlabel('Models')
    plt.tight_layout()
    plt.show()

show_bars()
