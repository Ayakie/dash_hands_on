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

# やること：読み込んだデータをtrainデータとして読み込むように連携(Callback)させる

#デフォルトのスタイルをアレンジ
common_style={'font-family': 'Comic Sans MS', 'textAlign': 'center', 'margin': '0 auto'}

app = dash.Dash(__name__)

# callbackがlayoutに含まれていない時の例外を避ける
app.config.suppress_callback_exceptions = True

# 予測に用いるモデルとインスタンスを定義した辞書
models = {'Linear Regression': LinearRegression(),
          'Random Forest Regressor': RandomForestRegressor()}

# df = pd.read_csv('housing_data.csv')

app.layout = html.Div(
    html.Div([
        html.H1('Dash Machine Learning Application'),
        # 空白を加える
        html.Br(),

        # ファイルアップロードの部分を作る
        dcc.Upload(
            id='upload-data',
            children=html.Div([
                'Drag and Drop or ',
                html.A('Select Files')
            ]),
            style={
                'width': '60%',
                'height': '60px',
                'lineHeight': '60px',
                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '0 auto'
            },
            # Allow multiple files to be uploaded
            # # これないとおそらくエラーになる
            multiple=True
        ),
        html.Br(),

        # アップロードしたファイルをデータテーブルとして表示させるところ
        html.Div(

             children=[
                 dash_table.DataTable(
                     id='output-data-upload',
                     column_selectable='multi',
                     fixed_rows={'headers': True, 'data': 0},
                     style_table={
                         'overflowX': 'scroll',
                         'overflowY': 'scroll',
                         'maxHeight': '250px'
                     },
                     style_header={
                         'fontWeight': 'bold',
                         'textAlign': 'center'}
                 )
             ],
             style={
                 'height': '300px',

             }),
        html.Br(),

        # モデルを選択するドロップダウン
        dcc.Dropdown(
            id='model-dropdown',
            options=[{'label': k, 'value': k} for k in models.keys()],
            value='Linear Regression'
        ),

        html.Div(
            id='rmse-sentence',
            # children=[f'Average RMSE Score of Linear Regression model is {avg_rmse_score}']
        ),
        html.Div(
            id='r2-sentence',
            # children=[f'R2 score is {avg_r2_score}']
        ),
        dcc.Graph(id='residual-plot',
                  style={'margin': '0px 100px'})
    ]),
    style=common_style
)

# アップロードしたファイルをデータフレームとして読み込むところ
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])

    data_ = df.to_dict('records')
    columns_ = [{'name': i, 'id': i} for i in df.columns]

    return [data_, columns_]


@app.callback([Output('output-data-upload', 'data'),
               Output('output-data-upload', 'columns')],
              [Input('upload-data', 'contents')],
              [State('upload-data', 'filename')])
def update_output(list_of_contents, list_of_names):
    # ファイルがない時の自動コールバックを防ぐ
    if list_of_contents is None:
        raise dash.exceptions.PreventUpdate

    contents = [parse_contents(c, n) for c, n in zip(list_of_contents, list_of_names)]

    return [contents[0][0], contents[0][1]]


# モデルを学習させる関数を別途用意する
def modeling_function(key, dict_data):
    '''
    :param key, data: 辞書のキー, データテーブルの中身(list of dict)
    :return: rmse, r2スコア, 残渣プロットに必要な予測値
    '''
    df = pd.DataFrame(data=dict_data)
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
              [Input('model-dropdown', 'value'),
               Input('output-data-upload', 'data')],
               # [State('output-data-upload', 'children'),
               #  ],
              # [State('upload-data', 'filename')]
              )
def update_result(model_name, dict_data):
    if dict_data is None:
        raise dash.exceptions.PreventUpdate

    avg_rmse_score = modeling_function(model_name, dict_data)[0]
    avg_r2_score = modeling_function(model_name, dict_data)[1]
    y_val_pred = modeling_function(model_name, dict_data)[2]
    y_val = modeling_function(model_name, dict_data)[3]
    y_tr_pred = modeling_function(model_name, dict_data)[4]
    y_tr = modeling_function(model_name, dict_data)[5]

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
        # [html.H5(f'Average RMSE Score of {model_name} is {avg_rmse_score}'),
        # html.H5(f'R2 score is {avg_r2_score}'),
        # figure]


if __name__ == '__main__':
    app.run_server(debug=True)