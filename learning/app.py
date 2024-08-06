from flask import Flask
import dash
import dash_bootstrap_components as dbc
from dash import html, Input, Output, dcc, State
import plotly.graph_objects as go
import requests

# Initialize the Flask app
app = Flask(__name__)

# Initialize the Dash app
dash_app = dash.Dash(
    __name__,
    server=app,
    routes_pathname_prefix='/dashboard/',
    external_stylesheets=[dbc.themes.LUX]
)

# Initialize default figures for the speedometer and history chart
def create_default_speedometer():
    return go.Figure(go.Indicator(
        mode="gauge+number",
        value=0,
        title={'text': "Sentiment Score"},
        gauge={
            'axis': {'range': [0, 2], 'tickvals': [0, 1, 2], 'ticktext': ["Negative", "Neutral", "Positive"]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, 1], 'color': "red"},
                {'range': [1, 2], 'color': "yellow"},
                {'range': [2, 3], 'color': "green"}
            ]
        }
    ))

def create_default_history_chart():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[], y=[], mode='lines+markers', name='Sentiment Score'))
    fig.update_layout(
        title="History of Sentiment Analysis",
        xaxis_title="Prediction Iteration",
        yaxis_title="Sentiment Score",
        template="plotly_white"
    )
    return fig

# Layout of the Dash app
dash_app.layout = dbc.Container(
    [
        dbc.NavbarSimple(
            brand="Sentiment Analysis Dashboard-Team A3",
            color="primary",
            dark=True,
            className="mb-4"
        ),
        dbc.Row(
            dbc.Col(
                [
                    html.H3("Enter Comment:", className="text-center mb-3"),
                    dbc.Input(id="input-box", placeholder="Enter a Reddit comment...", type="text", className="mb-3"),
                    dbc.Button("Submit", id="submit-button", color="primary", className="me-2"),
                    dbc.Button("Stop", id="stop-button", color="danger", className="ms-2"),
                    html.Div(id="output-div", className="mt-3")
                ],
                width=6, className="offset-md-3 text-center"
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    dcc.Graph(id='speedometer', figure=create_default_speedometer(), className="mt-3"),
                    width=6
                ),
                dbc.Col(
                    dcc.Graph(id='history-chart', figure=create_default_history_chart(), className="mt-3"),
                    width=6
                )
            ]
        ),
        dcc.Interval(
            id='interval-component',
            interval=1*1500,  # Update every second
            n_intervals=0
        )
    ],
    fluid=True,
)

# Define callback to update the output based on input value
@dash_app.callback(
    [Output("output-div", "children"),
     Output("speedometer", "figure"),
     Output("history-chart", "figure"),
     Output("input-box", "value"),
     Output("submit-button", "n_clicks")],
    [Input("submit-button", "n_clicks"),
     Input("stop-button", "n_clicks"),
     Input("interval-component", "n_intervals")],
    [State("input-box", "value")]
)
def update_output(submit_n_clicks, stop_n_clicks, n_intervals, value):
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if triggered_id == "stop-button":
        return "", create_default_speedometer(), create_default_history_chart(), "", None

    if submit_n_clicks and value:
        # Send the input text to the server's prediction endpoint
        response = requests.post("http://localhost:5001/predict", json={"text": value})
        if response.status_code == 200:
            data = response.json()
            sentiment_category = data['sentiment']
            sentiment_value = data['value']
        else:
            sentiment_category = "Error"
            sentiment_value = -1

        # Fetch updated history
        response = requests.get("http://localhost:5001/iterations")
        history_data = response.json()
        history_sentiment_values = history_data.get(value, [])

        # gauge chart with sentiment categories
        fig_speedometer = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_value,
            title={'text': "Sentiment Score"},
            gauge={
                'axis': {'range': [0, 2], 'tickvals': [0, 1, 2], 'ticktext': ["Negative", "Neutral", "Positive"]},
                'bar': {'color': "black"},
                'steps': [
                    {'range': [0, 1], 'color': "red"},
                    {'range': [1, 2], 'color': "yellow"},
                    {'range': [2, 3], 'color': "green"}
                ]
            }
        ))

        # history line chart
        fig_history = go.Figure()
        fig_history.add_trace(go.Scatter(x=list(range(1, len(history_sentiment_values) + 1)), y=history_sentiment_values,
                                         mode='lines+markers', name='Sentiment Score'))
        fig_history.update_layout(
            title="History of Sentiment Analysis",
            xaxis_title="Prediction Iteration",
            yaxis_title="Sentiment Score",
            template="plotly_white"
        )

        return f"Sentiment: {sentiment_category}", fig_speedometer, fig_history, dash.no_update, dash.no_update

    if value:
        # Fetch updated history periodically
        response = requests.get("http://localhost:5001/iterations")
        if response.status_code == 200:
            history_data = response.json()
            history_sentiment_values = history_data.get(value, [])

            # Create history line chart
            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(x=list(range(1, len(history_sentiment_values) + 1)), y=history_sentiment_values,
                                             mode='lines+markers', name='Sentiment Score'))
            fig_history.update_layout(
                title="History of Sentiment Analysis",
                xaxis_title="Prediction Iteration",
                yaxis_title="Sentiment Score",
                template="plotly_white"
            )

            return dash.no_update, dash.no_update, fig_history, dash.no_update, dash.no_update

    return "", create_default_speedometer(), create_default_history_chart(), dash.no_update, dash.no_update

# Run the Dash app
if __name__ == "__main__":
    app.run(debug=True, port=5000, host='0.0.0.0')
