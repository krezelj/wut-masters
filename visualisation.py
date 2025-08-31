import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

import plotly.graph_objects as go
import seaborn as sns

import numpy as np

def plot_training(
        data: list[tuple[list, list]],
        colors: list[str],
        labels: list[str],
        xlabel: str = "Model Iteration",
        ylabel: str = "Ratio of Wins",
        # title: str = "Title",
        width: int = 1000,
        height: int = 500,
        # font_family: str = "Times New Roman",
        font_size: int = 16):

    fig = go.Figure()

    for series, color, label in zip(data, colors, labels):
        fig.add_trace(go.Scatter(
            x=series[0], y=series[1],
            mode='lines',
            name=label,
            line=dict(color=color)
        ))

    # fig.add_shape(
    #     type="line",
    #     x0=0, x1=1, xref="paper",
    #     y0=0, y1=0, yref="y",
    #     line=dict(
    #         color="#000000",
    #         width=1,
    #         dash="dash"
    #     )
    # )

    fig.update_layout(
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        legend_title="Model",
        template="plotly_white",
        width=width,
        height=height,
        font=dict(
            size=font_size,
        ),
        yaxis=dict(range=[0, 1]),
        margin=dict(l=80, r=40, t=20, b=60)
    )

    return fig

def plot_tournament_matrix(matrix, labels, indices = None, include_selfplay = False):
    indices = indices if indices else np.arange(matrix.shape[0])
    matrix = matrix[np.ix_(indices, indices)]
    if not include_selfplay:
        np.fill_diagonal(matrix, np.nan)
    
    plt.figure(figsize=(6, 6))

    colors = ["#5555cc", "#FFFFFF", "#cc5555"]  # Blue → White → Red
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors)
    norm = TwoSlopeNorm(vmin=0, vcenter=0.45, vmax=1)
    ax = sns.heatmap(matrix, annot=True, fmt=".2f", norm=norm, linewidth=1, cmap=custom_cmap, cbar=False, annot_kws={"fontsize": 12})

    ax.set_xticks(np.arange(len(labels)) + 0.5)
    ax.set_xticklabels(labels, rotation=45, ha="left")

    ax.set_yticks(np.arange(len(labels)) + 0.5)
    ax.set_yticklabels(labels, rotation=0)

    ax.xaxis.tick_top()
    ax.tick_params(top=True, bottom=False)  # Only show ticks on top
    ax.xaxis.set_label_position('top')      # Optional: move x-axis label too

    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    ax.set_xlabel("White Player", fontsize=16)
    ax.set_ylabel("Black Player", fontsize=16)

    # plt.savefig("win_matrix.svg")
    # plt.show()
    return ax