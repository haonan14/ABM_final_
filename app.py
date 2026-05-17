"""Interactive visualization for the Scientific Search ABM.
Usage: solara run app.py"""

import numpy as np
import solara
from matplotlib.figure import Figure
from mesa.visualization import SolaraViz, make_plot_component, Slider
from mesa.visualization.utils import update_counter

from model import ScientificSearchModel
from agents import Scientist


model_params = {
    "n_agents": Slider("Number of Scientists", value=100, min=20, max=200, step=10),
    "n_topics": Slider("Number of Topics", value=200, min=100, max=400, step=50),
    "recognition_bias": Slider("Recognition Bias", value=0.5, min=0.0, max=1.0, step=0.1),
    "aspiration_bonus": Slider("Aspiration Bonus", value=0.4, min=0.0, max=0.8, step=0.1),
    "domestication_rate": Slider("Domestication Rate", value=0.03, min=0.01, max=0.10, step=0.01),
    "rep_gain": Slider("Reputation Gain", value=0.01, min=0.005, max=0.03, step=0.005),
    "turnover_rate": Slider("Turnover Rate", value=0.05, min=0.0, max=0.15, step=0.01),
    "max_steps": Slider("Max Steps", value=200, min=50, max=500, step=50),
    "seed": 42,
}

# Two custom panels:
#   1. Landscape view: topics as dots positioned by (recognizability, EP),
#      sized by agent count — shows the "flow" from safe zone to frontier
#   2. Agent scatter: each scientist plotted by (rbc, intrinsic_conservatism),
#      colored by which type of topic they chose — shows who explores


@solara.component
def LandscapeView(model):
    """Topics positioned by recognizability (x) and EP (y), sized by agent count."""
    update_counter.get()
    G = model.landscape

    # Count agents per topic this step
    agent_counts = {}
    for a in model.agents_by_type[Scientist]:
        if a.selected_topic is not None:
            agent_counts[a.selected_topic] = agent_counts.get(a.selected_topic, 0) + 1

    type_colors = {"established": "#6baed6", "recognizable": "#fd8d3c", "radical": "#e34a33"}

    fig = Figure(figsize=(6, 4.5))
    ax = fig.add_subplot()

    # Position topics by INITIAL values so dots don't drift with domestication
    for n, d in G.nodes(data=True):
        count = agent_counts.get(n, 0)
        color = type_colors[d["node_type"]]
        size = 8 + count * 25
        alpha = 0.15 if count == 0 else 0.8
        ax.scatter(
            model._initial_recog[n], model._initial_ep[n],
            s=size, c=color, alpha=alpha, edgecolors="none",
        )

    # Legend
    for label, color in type_colors.items():
        ax.scatter([], [], c=color, s=40, label=label, alpha=0.8)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.8)

    ax.set_xlabel("Recognizability", fontsize=8)
    ax.set_ylabel("Epistemic Potential", fontsize=8)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Knowledge Landscape (dot size = agent count)", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()

    solara.FigureMatplotlib(fig, format="png", bbox_inches="tight")


@solara.component
def AgentScatter(model):
    """Each scientist plotted by rbc (x) vs intrinsic_conservatism (y),
    colored by which topic type they selected this step."""
    update_counter.get()
    G = model.landscape
    type_colors = {"established": "#6baed6", "recognizable": "#fd8d3c", "radical": "#e34a33"}

    rbcs, ics, colors = [], [], []
    for a in model.agents_by_type[Scientist]:
        rbcs.append(a.risk_bearing_capacity)
        ics.append(a.intrinsic_conservatism)
        if a.selected_topic is not None:
            ttype = G.nodes[a.selected_topic]["node_type"]
        else:
            ttype = "established"
        colors.append(type_colors[ttype])

    fig = Figure(figsize=(6, 4.5))
    ax = fig.add_subplot()
    ax.scatter(rbcs, ics, c=colors, s=18, alpha=0.6, edgecolors="none")

    for label, color in type_colors.items():
        ax.scatter([], [], c=color, s=40, label=label, alpha=0.8)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.8)

    ax.set_xlabel("Risk-Bearing Capacity", fontsize=8)
    ax.set_ylabel("Intrinsic Conservatism", fontsize=8)
    ax.set_xlim(-0.05, 0.7)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Agent Strategies (color = topic type chosen)", fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()

    solara.FigureMatplotlib(fig, format="png", bbox_inches="tight")


# DV time-series plots (CC and RER use smoothed versions to reduce noise)
conservatism_plot = make_plot_component(
    {"CC_Smoothed": "tab:blue"}, backend="matplotlib")
exploration_plot = make_plot_component(
    {"RER_Smoothed": "tab:red"}, backend="matplotlib")
inequality_plot = make_plot_component(
    {"Innovation_Inequality": "tab:orange"}, backend="matplotlib")
domestication_plot = make_plot_component(
    {"Domestication_Progress": "tab:green"}, backend="matplotlib")

page = SolaraViz(
    model=ScientificSearchModel(),
    components=[
        LandscapeView, AgentScatter,
        conservatism_plot, exploration_plot,
        inequality_plot, domestication_plot,
    ],
    model_params=model_params,
    name="Scientific Search ABM",
)
