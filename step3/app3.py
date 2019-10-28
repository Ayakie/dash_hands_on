import base64
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

# やること：モデルを選択するドロップダウンを作って、コールバックにより選択したモデルを学習させる

#デフォルトのスタイルをアレンジ
common_style={'font-family': 'Comic Sans MS', 'textAlign': 'center', 'margin': '0 auto'}

app = dash.Dash(__name__)

# 予測に用いるモデルとインスタンスを定義した辞書
models = {'Linear Regression': LinearRegression(),
          'Random Forest Regressor': RandomForestRegressor()}

# 一つ上の階層のディレクトリであることに注意
df = pd.read_csv('../housing_data.csv')

app.layout = html.Div(
    html.Div([
        html.H1('Dash Machine Learning Application'),
        # 空白を加える
        html.Br(),

        # モデルを選択するドロップダウンを追加する
        dcc.Dropdown(
            id='model-dropdown',
            options=[{'label': k, 'value': k} for k in models.keys()],
            value='Linear Regression'
        ),

        html.H3(id='rmse-sentence'),
        html.H3(id='r2-sentence'),

        dcc.Graph(id='residual-plot',
                  style={'margin': '0px 100px'})
    ]),
    style=common_style
)

#  モデルを学習させる関数を別途用意する
def modeling_function(key):
    '''
    :param key: 辞書のキー
    :return: rmse, r2スコア, 残渣プロットに必要な予測値
    '''
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
        model = models[key]
        model.fit(x_tr, y_tr)

        y_val_pred = model.predict(x_val)
        y_tr_pred = model.predict(x_tr)

        rmse_score = np.sqrt(mean_squared_error(y_val, y_val_pred))
        rmse_scores.append(rmse_score)

        # 左の変数はr2_scoreにしないように
        r2_score_ = r2_score(y_val, y_val_pred)
        r2_scores.append(r2_score_)

    # 各foldのスコア平均
    avg_rmse_score = np.mean(rmse_scores)
    avg_r2_score = np.mean(r2_scores)

    return avg_rmse_score, avg_r2_score, y_val_pred, y_val, y_tr_pred, y_tr

# ドロップダウンで選択したモデリングで学習し、スコアと残渣プロットを返す
@app.callback([Output('rmse-sentence', 'children'),
               Output('r2-sentence', 'children'),
               Output('residual-plot', 'figure')
               ],
              [Input('model-dropdown', 'value')]
              )
def update_result(model_name):

    avg_rmse_score = modeling_function(model_name)[0]
    avg_r2_score = modeling_function(model_name)[1]
    y_val_pred = modeling_function(model_name)[2]
    y_val = modeling_function(model_name)[3]
    y_tr_pred = modeling_function(model_name)[4]
    y_tr = modeling_function(model_name)[5]

    figure = {
        'data': [
            go.Scatter(
                x = y_tr_pred,
                y = y_tr_pred-y_tr,
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
                y=y_val_pred-y_val,
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 10,
                    'line': {'width': 0.5, 'color': 'white'}
                },
                name='test data'
            )
        ],
        'layout': go.Layout(
            title='Residual Plot of Median House Price',
            xaxis={'title': 'Predicted Values'},
            yaxis={'title': 'Residuals'}
        )
    }
    return [
        html.H5(f'Average RMSE Score of {model_name} is {avg_rmse_score}'),
        html.H5(f'R2 score is {avg_r2_score}'),
        figure
    ]

if __name__ == '__main__':
    app.run_server(debug=True)