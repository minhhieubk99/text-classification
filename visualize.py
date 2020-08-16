import pandas as pd
import numpy as np
import plotly.graph_objects as go


df = pd.read_excel('./statistic.xlsx', sheet_name='TF-IDF Vector - alpha thay doi ')

val_acc = np.around(df["val_acc"], decimals=6)
train_acc = np.around(df["train_acc"], decimals=6)


N = 19
x = np.linspace(1, N, N)

fig = go.Figure()

fig.add_trace(go.Scatter(x=x, y=val_acc,
                         mode='lines',
                         name='val_acc'))
fig.add_trace(go.Scatter(x=x, y=train_acc,
                         mode='lines', 
                         name='train_acc'))

fig.update_layout(title='TF-IDF Vector - alpha thay doi - min_df =1',
                  xaxis_title='min_df',
                  yaxis_title='acc')

fig.show()