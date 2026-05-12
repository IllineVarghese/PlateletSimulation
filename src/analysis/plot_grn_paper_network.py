from pathlib import Path
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import networkx as nx


GRAPHML_FILE = Path("data/networks/platelet_squad_like_complex.graphml")
OUTPUT_DIR = Path("results/analysis/grn_paper_style")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "platelet_grn_paper_network.png"


def load_graphml_edges(path):
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}

    tree = ET.parse(path)
    root = tree.getroot()

    G = nx.DiGraph()

    for node in root.findall(".//g:node", ns):
        node_id = node.attrib["id"]
        G.add_node(node_id)

    for edge in root.findall(".//g:edge", ns):
        src = edge.attrib["source"]
        dst = edge.attrib["target"]

        weight = 1.0
        for data in edge.findall("g:data", ns):
            if data.attrib.get("key") == "weight":
                weight = float(data.text)

        G.add_edge(src, dst, weight=weight)

    return G


def node_category(node):
    if node.startswith("In"):
        return "input"
    if node.startswith("Out"):
        return "output"
    if "Activation" in node or "Program" in node:
        return "program"
    if any(k in node for k in ["ADP", "TXA2", "Ca2", "PKC", "Rap1", "PI3K", "PLC", "vWF", "GPVI"]):
        return "signaling"
    return "intermediate"


def main():
    G = load_graphml_edges(GRAPHML_FILE)

    pos = nx.spring_layout(
        G,
        seed=7,
        k=1.2,
        iterations=300,
        weight="weight"
    )

    node_colors = []
    node_sizes = []

    for node in G.nodes:
        cat = node_category(node)

        if cat == "input":
            node_colors.append("#9bd3e5")   # light blue
            node_sizes.append(1700)
        elif cat == "output":
            node_colors.append("#78df7c")   # green
            node_sizes.append(1800)
        elif cat == "program":
            node_colors.append("#ffd98c")   # light orange
            node_sizes.append(1700)
        elif cat == "signaling":
            node_colors.append("#dddddd")   # grey
            node_sizes.append(1450)
        else:
            node_colors.append("#ffffff")   # white
            node_sizes.append(1200)

    positive_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] >= 0]
    negative_edges = [(u, v) for u, v, d in G.edges(data=True) if d["weight"] < 0]

    positive_widths = [1.0 + abs(G[u][v]["weight"]) * 1.2 for u, v in positive_edges]
    negative_widths = [1.0 + abs(G[u][v]["weight"]) * 1.2 for u, v in negative_edges]

    plt.figure(figsize=(22, 14))
    ax = plt.gca()
    ax.set_title(
        "Platelet GRN interaction network: mechanosensing, chemical feedback, adhesion, secretion, morphology",
        fontsize=18,
        pad=20,
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=positive_edges,
        edge_color="#2ca25f",
        width=positive_widths,
        arrows=True,
        arrowsize=18,
        alpha=0.75,
        connectionstyle="arc3,rad=0.08",
    )

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=negative_edges,
        edge_color="#de2d26",
        width=negative_widths,
        arrows=True,
        arrowsize=18,
        alpha=0.75,
        connectionstyle="arc3,rad=-0.08",
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=node_colors,
        node_size=node_sizes,
        edgecolors="#333333",
        linewidths=1.2,
    )

    labels = {}
    for node in G.nodes:
        labels[node] = node.replace("_", "\n")

    nx.draw_networkx_labels(
        G,
        pos,
        labels=labels,
        font_size=8,
        font_weight="bold",
    )

    edge_labels = {
        (u, v): f"{d['weight']:.2g}"
        for u, v, d in G.edges(data=True)
        if abs(d["weight"]) >= 1.2 or d["weight"] < 0
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7,
        label_pos=0.55,
    )

    legend_items = [
        ("Inputs / sensors", "#9bd3e5"),
        ("Signaling nodes", "#dddddd"),
        ("Behavior programs", "#ffd98c"),
        ("Outputs / actuators", "#78df7c"),
    ]

    for i, (label, color) in enumerate(legend_items):
        ax.scatter([], [], s=300, c=color, edgecolors="#333333", label=label)

    ax.plot([], [], color="#2ca25f", linewidth=3, label="Activating edge")
    ax.plot([], [], color="#de2d26", linewidth=3, label="Inhibitory edge")

    ax.legend(loc="lower left", fontsize=11, frameon=True)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved paper-style GRN network figure to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()