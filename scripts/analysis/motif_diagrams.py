import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import numpy as np  


def degree_in_motif():
    """
    Degree-in motif: one node with in-degree 4.
    Construction:
      - Target node T
      - Two senders: S1, S2
      - Edges: S1->T (2x), S2->T (2x)
      => in-degree(T) = 2 + 2 = 4, fan-in(T) = 2
    """
    G = nx.MultiDiGraph()
    T = "T"
    senders = ["S1", "S2"]

    G.add_node(T)
    G.add_nodes_from(senders)

    # Two parallel edges from S1 to T
    for _ in range(2):
        G.add_edge("S1", T)
    # Two parallel edges from S2 to T
    for _ in range(2):
        G.add_edge("S2", T)

    # Layout: senders on the left, target on the right
    pos = {
        "S1": (-1,  0.5),
        "S2": (-1, -0.5),
        "T":  (0.8, 0),
    }

    title = "Degree-in = 4"
    highlight_nodes = {"T"}  # node whose degree we refer to
    return G, pos, title, highlight_nodes


def degree_out_motif():
    """
    Degree-out motif: one node with out-degree 4.
    Construction:
      - Source node S
      - Two receivers: R1, R2
      - Edges: S->R1 (2x), S->R2 (2x)
      => out-degree(S) = 2 + 2 = 4, fan-out(S) = 2
    """
    G = nx.MultiDiGraph()
    S = "S"
    receivers = ["R1", "R2"]

    G.add_node(S)
    G.add_nodes_from(receivers)

    # Two parallel edges from S to R1
    for _ in range(2):
        G.add_edge(S, "R1")
    # Two parallel edges from S to R2
    for _ in range(2):
        G.add_edge(S, "R2")

    # Layout: source on the left, receivers on the right
    pos = {
        "S":  (-0.8, 0),
        "R1": (1,  0.5),
        "R2": (1, -0.5),
    }

    title = "Degree-out = 4"
    highlight_nodes = {"S"}
    return G, pos, title, highlight_nodes


def fan_in_motif():
    """
    Fan-in motif: many unique senders, one receiver.
    Here fan-in(C) = 3 (three distinct neighbors).
    """
    G = nx.MultiDiGraph()
    center = "C"
    senders = ["F1", "F2", "F3"]
    G.add_node(center)
    G.add_nodes_from(senders)

    for s in senders:
        G.add_edge(s, center)

    pos = {
        "F1": (-1,  1),
        "F2": (-1,  0),
        "F3": (-1, -1),
        "C":  (0.8, 0),
    }
    title = "Fan-in = 3"
    highlight_nodes = {"C"}
    return G, pos, title, highlight_nodes


def fan_out_motif():
    """
    Fan-out motif: one sender, many unique receivers.
    Here fan-out(S) = 3 (three distinct neighbors).
    """
    G = nx.MultiDiGraph()
    source = "F"
    receivers = ["R1", "R2", "R3"]
    G.add_node(source)
    G.add_nodes_from(receivers)

    for r in receivers:
        G.add_edge(source, r)

    pos = {
        "F":  (-0.8, 0),
        "R1": (1,  1),
        "R2": (1,  0),
        "R3": (1, -1),
    }
    title = "Fan-out = 3"
    highlight_nodes = {"F"}
    return G, pos, title, highlight_nodes


def scatter_gather_motif():
    """
    Scatter–gather pattern:
        Src -> {I1, I2, I3} -> Snk
    Highlight both Src and Snk as the key endpoints.
    """
    G = nx.MultiDiGraph()
    source = "Src"
    sink = "Snk"
    intermediates = ["I1", "I2", "I3"]

    G.add_node(source)
    G.add_node(sink)
    G.add_nodes_from(intermediates)

    # Scatter
    for i in intermediates:
        G.add_edge(source, i)

    # Gather
    for i in intermediates:
        G.add_edge(i, sink)

    pos = {
        "Src": (-1.2, 0),
        "I1": (0,  0.8),
        "I2": (0,  0.0),
        "I3": (0, -0.8),
        "Snk": (1.2, 0),
    }
    title = "Scatter–gather"
    highlight_nodes = {"Src", "Snk"}
    return G, pos, title, highlight_nodes


def biclique_motif():
    """
    Directed biclique K_{2,3}:
    Two sources -> three sinks, all-to-all.
    Highlight the left side as senders.
    """
    G = nx.MultiDiGraph()
    left = ["L1", "L2"]
    right = ["R1", "R2", "R3"]

    G.add_nodes_from(left + right)

    for l in left:
        for r in right:
            G.add_edge(l, r)

    pos = {
        "L1": (-1,  0.5),
        "L2": (-1, -0.5),
        "R1": (1,  1),
        "R2": (1,  0),
        "R3": (1, -1),
    }
    title = "Directed biclique"
    highlight_nodes = set(left)
    return G, pos, title, highlight_nodes


def cycle_motif(length):
    """
    Directed cycle of given length ℓ (2 ≤ ℓ ≤ 6).
    For ℓ = 2, create an explicit two-node, two-edge cycle C1<->C2
    with a simple horizontal layout that works nicely with arcs.
    """
    G = nx.MultiDiGraph()

    if length == 2:
        nodes = ["C1", "C2"]
        G.add_nodes_from(nodes)
        # Exactly one edge in each direction
        G.add_edge("C1", "C2")
        G.add_edge("C2", "C1")

        pos = {
            "C1": (-0.7, 0.0),
            "C2": (0.7, 0.0),
        }
    else:
        nodes = [f"C{i+1}" for i in range(length)]
        G.add_nodes_from(nodes)
        for i in range(length):
            G.add_edge(nodes[i], nodes[(i + 1) % length])

        import math
        pos = {}
        radius = 1.0
        for i, n in enumerate(nodes):
            angle = 2 * math.pi * i / length
            pos[n] = (radius * math.cos(angle), radius * math.sin(angle))

    title = f"Directed cycle (ℓ={length})"
    highlight_nodes = set(G.nodes())
    return G, pos, title, highlight_nodes



def draw_motif(ax, G, pos, title, highlight_nodes=None):
    """
    Draw a motif on the given axis.

    Special case:
      - If the graph is exactly a 2-cycle C1<->C2, draw two explicit arcs:
        * arc on top:  C1 -> C2
        * arc on bottom: C2 -> C1

    General case:
      - Use parallel-arc logic for MultiDiGraph edges (degree motifs, etc.).
    """
    if highlight_nodes is None:
        highlight_nodes = set()

    # Nodes
    node_colors = [
        "dimgray" if n in highlight_nodes else "white"
        for n in G.nodes()
    ]

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=700,
        node_color=node_colors,
        edgecolors="black",
        linewidths=1.0,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)

    # Detect explicit 2-cycle C1<->C2
    nodes_set = set(G.nodes())
    edges_set = {(u, v) for u, v in G.edges()}
    is_two_cycle = (
        nodes_set == {"C1", "C2"}
        and edges_set == {("C1", "C2"), ("C2", "C1")}
    )

    if is_two_cycle:
        # Use the same rad for both directions
        # (direction reversal flips the curvature, so same rad => opposite sides)
        cycle_rad = 0.6

        # C1 -> C2  (one side, e.g. top)
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=[("C1", "C2")],
            arrows=True,
            arrowstyle='-|>',
            min_source_margin=10,
            min_target_margin=10,
            connectionstyle=f"arc3,rad={cycle_rad}",
            width=1.2,
        )

        # C2 -> C1  (opposite side, e.g. bottom)
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edgelist=[("C2", "C1")],
            arrows=True,
            arrowstyle='-|>',
            min_source_margin=10,
            min_target_margin=10,
            connectionstyle=f"arc3,rad={cycle_rad}",  # <-- SAME sign!
            width=1.2,
        )
    else:
        # Generic edge drawing (degree motifs, biclique, longer cycles)
        edge_groups = defaultdict(list)
        if isinstance(G, nx.MultiDiGraph):
            for u, v, k in G.edges(keys=True):
                edge_groups[(u, v)].append((u, v))
        else:
            for u, v in G.edges():
                edge_groups[(u, v)].append((u, v))

        for (u, v), edges in edge_groups.items():
            m = len(edges)
            if m == 1:
                nx.draw_networkx_edges(
                    G, pos, ax=ax,
                    edgelist=[(u, v)],
                    arrows=True,
                    arrowstyle='-|>',
                    min_source_margin=10,
                    min_target_margin=10,
                )
            else:
                rads = np.linspace(-0.25, 0.25, m)
                for rad in rads:
                    nx.draw_networkx_edges(
                        G, pos, ax=ax,
                        edgelist=[(u, v)],
                        arrows=True,
                        arrowstyle='-|>',
                        min_source_margin=10,
                        min_target_margin=10,
                        connectionstyle=f"arc3,rad={rad}",
                    )

    # Center and square the layout on this axis
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0

    span = max(x_max - x_min, y_max - y_min)
    margin = 0.6

    half = span / 2.0 + margin
    ax.set_xlim(x_center - half, x_center + half)
    ax.set_ylim(y_center - half, y_center + half)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(title, fontsize=10)
    ax.set_axis_off()



def main():
    # Build all 11 motifs
    motif_builders = [
        degree_in_motif,         # 1
        degree_out_motif,        # 2
        fan_in_motif,            # 3
        fan_out_motif,           # 4
        scatter_gather_motif,    # 5
        biclique_motif,          # 6
        lambda: cycle_motif(2),  # 7
        lambda: cycle_motif(3),  # 8
        lambda: cycle_motif(4),  # 9
        lambda: cycle_motif(5),  # 10
        lambda: cycle_motif(6),  # 11
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    axes = axes.flatten()

    for i, builder in enumerate(motif_builders):
        G, pos, title, highlight_nodes = builder()
        draw_motif(axes[i], G, pos, title, highlight_nodes)

    # Hide the unused last axis (12th subplot)
    axes[-1].axis("off")

    fig.suptitle("Money Laundering Motifs (Degree-based & Higher-order)", fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    main()
