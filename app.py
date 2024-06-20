# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input
import dash_daq as daq
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import dash_bootstrap_components as dbc
import numpy as np


# Initialize the app - incorporate a Dash Bootstrap theme
external_stylesheets = [dbc.themes.CERULEAN]
app = Dash(__name__)#, external_stylesheets=external_stylesheets)

###----------------生成右边栏----------------###
#####--------------生成第一栏--------------#####
def Generate_r_c_row1():
    '''
    生成右边第一栏
    包含我们所感兴趣的结果
    '''
    return dbc.Row(id='r_c_row1',children=[]
    )

#####--------------生成第二栏--------------#####

def Generate_r_c_row2():
    """
    生成右边第二栏
    包含图表
    """
    return dbc.Row(id='r_c_row2',children=[
        dcc.Graph(figure=px.line(x=[0],y=[0]), id='time_phi_graph'),
        dcc.Graph(figure=px.line(x=[0],y=[0]), id='frequency_phi_graph') 
        ])

#####--------------生成第三栏--------------#####
def Generate_cps_phi_graph():
    """
    生成图表
    """
    fig_cps_phi = go.Figure()
    fig_cps_phi.add_trace(go.Scatter(
        x=[],
        y=[],
        xaxis='x',
        yaxis='y',
        mode='markers',
        marker=dict(
            color='rgba(0,64,128,1)',
            size=3
        )))
    fig_cps_phi.update_layout(margin={'l': 5, 'r': 5, 'b': 5, 't': 5, 'pad': 0})
    return fig_cps_phi


def Generate_r_c_row3():
    """
    生成右边第三栏
    包含结果随时间变化的趋势
    """
    return dbc.Row(id='r_c_row3',children=[
        daq.BooleanSwitch(id='flag_cps_std', on=False,label="开启统计",labelPosition="top"),
        dcc.Graph(figure=Generate_cps_phi_graph(), id='cps_phi_graph')
    ])

# App layout
app.layout = dbc.Container([
    
    dcc.Interval(id='interval_component', interval=2*1000, n_intervals=0),
    
    dbc.Row(id='Title_app_name',children=[
        html.Div('General Application for real-time data visualization and DAQ',className='ten columns offset-by-one')
    ]),

    dbc.Row(id='main windows',children=[
        dbc.Col(id='left columns',width='auto',children=[
            dbc.RadioItems(options=[{"label": x, "value": x} for x in ['pop', 'lifeExp', 'gdpPercap']],
                       value='lifeExp',
                       inline=True,
                       id='radio-buttons-final')
        ]),

        dbc.Col(id='right columns',width='auto',children=[
            Generate_r_c_row1(),
            Generate_r_c_row2(),
            Generate_r_c_row3()
        ]),
    ]),

], fluid=True)

cps_std=[]
# Add controls to build the interaction
@callback(
    [Output(component_id='time_phi_graph', component_property='extendData'),
    Output(component_id='frequency_phi_graph', component_property='extendData'),
    Output(component_id='cps_phi_graph', component_property='extendData')],
    Input(component_id='interval_component', component_property='n_intervals'),
    Input(component_id='flag_cps_std', component_property='on')
)
def update_graph(n,flag_cps_std):
    data=np.random.random(20)
    data_time=np.arange(data.size)
    freq_data=np.abs(np.fft.rfft(data))
    freq=np.fft.rfftfreq(data.size,d=1/data.size)
    time_fig = [dict( x=[data_time],y=[data]), [0], data.size]
    freq_fig=[dict( x=[freq],y=[freq_data]), [0], freq.size]
    if flag_cps_std:
        cps_fig=[dict( y=[[np.std(data)]]), [0], 10]
    else:
        cps_fig=[dict(x=[[0]], y=[[0]]), [0], 10]
    return [time_fig, freq_fig, cps_fig]

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
