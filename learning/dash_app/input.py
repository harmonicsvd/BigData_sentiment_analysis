import dash
import dash_bootstrap_components as dbc
from dash import html, Input, Output, dcc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Initialize the Dash app with a Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Sample data for sentiment trend
data = {
    'time': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'sentiment': [0] * 100
}
df = pd.DataFrame(data)

# Define the layout of the app
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                html.H1("Sentiment Analysis Input", className="text-center mt-4")
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Input(id="input-box", placeholder="Enter a Reddit comment...", type="text", className="mt-3")
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id='speedometer', className="mt-3")
                ),
                dbc.Col(
                    dcc.Graph(id='sentiment-trend', className="mt-3")
                ),
            ],
            className="mt-3"
        ),
        dbc.Row(
            dbc.Col(
                html.Div(id="output-div", className="mt-3")
            )
        ),
    ],
    fluid=True,
)

# Define callback to update the output based on input value
@app.callback(
    Output("output-div", "children"),
    Output("speedometer", "figure"),
    Output("sentiment-trend", "figure"),
    Input("input-box", "value")
)
def update_output(value):
    if value:
        # For demonstration, we'll use the length of the input value as a mock sentiment score
        sentiment_value = len(value) % 500  # Example sentiment value logic
        sentiment_trend = df['sentiment'].tolist()
        sentiment_trend.append(sentiment_value)
        df['sentiment'] = sentiment_trend[-100:]  # Keep last 100 values
        
        fig_speedometer = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentiment"}
        ))
        
        fig_trend = px.line(df, x='time', y='sentiment', title='Sentiment Trend')
        
        return f"You entered: {value}", fig_speedometer, fig_trend
    else:
        # Default gauge chart when no input is provided
        fig_speedometer = go.Figure(go.Indicator(
            mode="gauge+number",
            value=0,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Sentiment"}
        ))

        fig_trend = px.line(df, x='time', y='sentiment', title='Sentiment Trend')
        
        return "Enter a comment to see the result here.", fig_speedometer, fig_trend

# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
