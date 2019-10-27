import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

# 単純なモデルを作り、残渣プロットおよびRMSEスコアを表示させる

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#デフォルトのスタイルをアレンジ
common_style={'font-family': 'Comic Sans MS', 'textAlign': 'center', 'margin': '0 auto'}

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

df = pd.read_csv('housing_data.csv')
X_train = df.iloc[:, :-1]
y_train = df.iloc[:, -1]

rmse_scores = []
r2_scores = []

# クロスバリデーションでモデルを評価する
kf = KFold(n_splits=4, shuffle=True, random_state=0)
for tr_idx, val_idx in kf.split(X_train):
    x_tr, x_val = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    y_tr, y_val = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    # 学習の実行
    lr = LinearRegression()
    lr.fit(x_tr, y_tr)

    y_val_pred = lr.predict(x_val)
    y_tr_pred = lr.predict(x_tr)

    rmse_score = np.sqrt(mean_squared_error(y_val, y_val_pred))
    rmse_scores.append(rmse_score)

    r2_score = r2_score(y_val, y_val_pred)
    r2_scores.append(r2_score)

# 各foldのスコア平均
avg_rmse_score = np.mean(rmse_scores)
avg_r2_score = np.mean(r2_scores)

app.layout = html.Div(
    html.Div([
        html.H1('Dash Machine Learning Application'),
        # 空白を加える
        html.Br(),
        html.H5(f'Average RMSE Score of Linear Regression model is {avg_rmse_score}'),
        html.H5(f'R2 score is {avg_r2_score}'),
        dcc.Graph(id='residual-plot',
                  figure={
                      'data': [go.Scatter(
                          x=y_tr_pred,
                          y=y_tr_pred - y_tr,
                          mode='markers',
                          opacity=0.7,
                          marker={
                              'size': 10,
                              'line': {'width': 0.5, 'color': 'white'}
                          },
                          name='train data'
                      ),
                          go.Scatter(
                              x=y_val_pred,
                              y=y_val_pred - y_val,
                              mode='markers',
                              opacity=0.7,
                              marker={
                                  'size': 10,
                                  'line': {'width': 0.5, 'color': 'white'}
                              },
                              name='test data'
                          )
                      ],
                      'layout':
                      # ここはarray([])じゃダメ
                          go.Layout(
                              title='Residual Plot of Median House Price',
                              xaxis={'title': 'Predicted Values'},
                              yaxis={'title': 'Residuals'},
                          )
                  },
                  style={'margin': '0px 100px'}
                  )
    ]),
    style=common_style
)

if __name__ == '__main__':
    app.run_server(debug=True)