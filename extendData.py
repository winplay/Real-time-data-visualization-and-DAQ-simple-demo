import dash
from dash import html
from dash import dcc
import numpy as np

from dash.dependencies import Input, Output

# Example data (a circle).
resolution = 20
t = np.linspace(0, np.pi * 2, resolution)
x, y = np.cos(t), np.sin(t)
# Example app.
figure = dict(data=[{'x': [], 'y': []}], layout=dict(xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1])))
app = dash.Dash(__name__, update_title=None)  # remove "Updating..." from title
app.layout = html.Div([dcc.Graph(id='graph', figure=figure), dcc.Interval(id="interval",n_intervals=0)])


@app.callback(Output('graph', 'extendData'), [Input('interval', 'n_intervals')])
def update_data(n_intervals):
    index = n_intervals % resolution
    # tuple is (dict of new data, target trace index, number of points to keep)
    return [dict(x=[[x[index]]], y=[[y[index]]]), [0], 10]


if __name__ == '__main__':
    app.run_server(debug=True)