import plotly.graph_objects as go

class Visualizer:
    @staticmethod
    def create_price_optimization_plot(prices, revenues):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=prices, y=revenues, mode='lines+markers', name='Revenue'))
        fig.update_layout(
            title='Price Optimization',
            xaxis_title='Price',
            yaxis_title='Revenue',
            template='plotly_white'
        )
        return fig

    @staticmethod
    def create_competition_analysis(optimal_price, competition_prices):
        fig = go.Figure()
        fig.add_trace(go.Box(y=competition_prices, name='Competition Prices'))
        fig.add_trace(go.Scatter(x=[1], y=[optimal_price], mode='markers', marker=dict(color='red', size=12), name='Optimal Price'))
        fig.update_layout(
            title='Competition Analysis',
            yaxis_title='Price',
            template='plotly_white'
        )
        return fig
