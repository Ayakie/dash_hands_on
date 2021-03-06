# dash_hands_on
研究室内向けのDashを使ったWebアプリケーションハンズオン

### 解説ブログ
https://wimper-1996.hatenablog.com/entry/2019/10/28/dash_machine_learning1

### 用いるデータについて
参考文献：[第二版] Python機械学習プログラミング
達人データサイエンティストによる理論と実践
https://raw.githubusercontent.com/rasbt/python-machine-learning-book-2nd-edition/master/code/ch10/housing.data.txt

- CRIM: 犯罪発生率(人口単位)
- ZN: 25,000平方フィート以上の住宅区画の割合
- INDUS: 非小売業の土地面積の割合(人口単位)
- CHAS: チャールズ川沿いかどうか(チャールズ川沿いであれば1, そうでなければ0)
- NOX: 窒素酸化物の濃度(pphm単位)
- RM: 一戸あたりの平均部屋数
- AGE: 1940年よりも前に建てられた家屋の割合
- DIS: ボストンの主な5つの雇用圏までの重み付き距離
- RAD: 幹線道路へのアクセス指数
- TAX: 10,000ドルあたりの所得税率
- PTRATIO: 教師一人当たりの生徒の数(人口単位)
- B: 1000(Bk-0.63)<sup>2</sup>として計算(Bkはアフリカ系アメリカ人居住者の割合(人口単位))
- LSTAT: 低所得者の割合
- MEDV(= ターゲット変数): 住宅価格の中央値(単位1,000ドル) 

### 流れ
Step1. pyファイルのみでWebページを出力させてみる→ CSSを書いて(common_style)文字や配置をいい感じにする<br>
Step2. 単純なモデルを作り、残渣プロットおよび予測スコアを表示させる<br>
Step3. モデルを選択するドロップダウンを作って、選択したモデルに応じて出力結果が変わるような動的なページを作る<br>
Step4. ファイルをアップロードし、データフレームとして表示させる<br>
Step5. 読み込んだデータをtrainデータとして読み込むように連携(Callback)させる<br>
app6: おまけ

![Screen Recording 2019-11-06 at 1 56 57 mov](https://user-images.githubusercontent.com/49149391/68228815-6bcdb280-0039-11ea-9612-3e52a5af0f4b.gif)
