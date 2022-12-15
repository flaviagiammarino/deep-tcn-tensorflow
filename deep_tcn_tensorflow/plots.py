import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot(df, quantiles):

    '''
    Plot the target time series and the predicted quantiles.

    Parameters:
    __________________________________
    df: pd.DataFrame.
        Data frame with target time series and predicted quantiles.

    quantiles: list.
        Quantiles of target time series which have been predicted.

    Returns:
    __________________________________
    fig: go.Figure.
        Line chart of target time series and predicted quantiles, one subplot for each target.
    '''
    
    # get the number of predicted quantiles
    n_quantiles = len(quantiles)

    # get the number of targets
    n_targets = int((df.shape[1] - 1) / (n_quantiles + 1))
 
    # plot the predicted quantiles for each target
    fig = make_subplots(
        subplot_titles=['Target ' + str(i + 1) for i in range(n_targets)],
        vertical_spacing=0.15,
        rows=n_targets,
        cols=1
    )

    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=60, b=60, l=30, r=30),
        font=dict(
            color='#1b1f24',
            size=8,
        ),
        legend=dict(
            traceorder='normal',
            font=dict(
                color='#1b1f24',
                size=10,
            ),
            x=0,
            y=-0.1,
            orientation='h'
        ),
    )

    fig.update_annotations(
        font=dict(
            color='#1b1f24',
            size=12,
        )
    )

    for i in range(n_targets):

        fig.add_trace(
            go.Scatter(
                x=df['time_idx'],
                y=df['target_' + str(i + 1)],
                name='Actual',
                legendgroup='Actual',
                showlegend=True if i == 0 else False,
                mode='lines',
                line=dict(
                    color='#afb8c1',
                    width=1
                )
            ),
            row=i + 1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=df['time_idx'],
                y=df['target_' + str(i + 1) + '_0.5'],
                name='Median',
                legendgroup='Median',
                showlegend=True if i == 0 else False,
                mode='lines',
                line=dict(
                    width=1,
                    color='rgba(9, 105, 218, 0.5)',
                ),
            ),
            row=i + 1,
            col=1
        )

        for j in range(n_quantiles // 2):

            fig.add_trace(
                go.Scatter(
                    x=df['time_idx'],
                    y=df['target_' + str(i + 1) + '_' + str(quantiles[- (j + 1)])],
                    name='q' + format(quantiles[j], '.1%') + ' - ' + 'q' + format(quantiles[- (j + 1)], '.1%'),
                    legendgroup='q' + format(quantiles[j], '.1%') + ' - ' + 'q' + format(quantiles[- (j + 1)], '.1%'),
                    showlegend=False,
                    mode='lines',
                    line=dict(
                        color='rgba(9, 105, 218, ' + str(0.1 * (j + 1))  + ')',
                        width=0.1
                    )
                ),
                row=i + 1,
                col=1
            )

            fig.add_trace(
                go.Scatter(
                    x=df['time_idx'],
                    y=df['target_' + str(i + 1) + '_' + str(quantiles[j])],
                    name='q' + format(quantiles[j], '.1%') + ' - ' + 'q' + format(quantiles[- (j + 1)], '.1%'),
                    legendgroup='q' + format(quantiles[j], '.1%') + ' - ' + 'q' + format(quantiles[- (j + 1)], '.1%'),
                    showlegend=True if i == 0 else False,
                    mode='lines',
                    fill='tonexty',
                    fillcolor='rgba(9, 105, 218, ' + str(0.1 * (j + 1)) + ')',
                    line=dict(
                        color='rgba(9, 105, 218, ' + str(0.1 * (j + 1)) + ')',
                        width=0.1,
                    ),
                ),
                row=i + 1,
                col=1
            )

        fig.update_xaxes(
            title='Time',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            row=i + 1,
            col=1
        )

        fig.update_yaxes(
            title='Value',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            zeroline=False,
            row=i + 1,
            col=1
        )

    return fig
