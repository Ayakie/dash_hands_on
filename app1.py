import base64
import io

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table
import dash_bootstrap_components as dbc

import pandas as pd
import numpy as np
import plotly.graph_objs as go

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

# やること：pyファイルのみでWebページを出力させる→ CSSを書いて(common_style)文字や配置をいい感じにする

# 全体に適用させるスタイルをアレンジ: フォントは各自好きなの設定してみてもいいかも.(https://w3g.jp/sample/css/font-family#japanese)
# 欧文はフォントが豊富で楽しいので今回のアプリも英語で記述しています
common_style = {'font-family': 'Comic Sans MS', 'textAlign': 'center', 'margin': '0 auto'}

# アプリの実態(インスタンス)を定義
app = dash.Dash(__name__)

# アプリの見た目の記述
app.layout = html.Div([
    # タイトル
    html.H1('Dash Machine Learning Application'),

    # 空白を加える
    html.Br(),

    # アップロードの部分を作る
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
        }
    )
],
    # ここで全体Divのスタイルを反映
    style=common_style
)

# 仮に外部からファイルをインポートした際に勝手に中身が実行されないようにするおまじない
if __name__ == '__main__':
    app.run_server(debug=True)
