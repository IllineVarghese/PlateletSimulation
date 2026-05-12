import os
import xml.etree.ElementTree as ET
import networkx as nx
import matplotlib.pyplot as plt


GRAPHML_PATH = "data/networks/platelet_squad_like_complex.graphml"
OUTPUT_DIR = "results/analysis/grn_network"


def load_graphml_edges(path):
    ns = {"g": "http://graphml.graphdrawing.org/xmlns"}
    tree = ET.parse(path)
    root = tree.getroot()
    graph = root.find("g:graph", ns)

    G = nx.DiGraph()

    for node in graph.findall("g:node", ns):
        node_id = node.attrib["id"]
        G.add_node(node_id)

    for edge in graph.findall("g:edge", ns):
        src = edge.attrib["source"]
        dst = edge.attrib["target"]

        weight = 1.0
        for data in edge.findall("g:data", ns):
            if data.text:
                weight = float(data.text.strip())

        G.add_edge(src, dst, weight=weight)

    return G


def node_color(node):
    if node.startswith("In"):
        return "lightblue"
    if node.startswith("Out"):
        return "lightgreen"
    if "Activation" in node:
        return "orange"
    if "Program" in node:
        return "wheat"
    return "lightgray"


def draw_network(G, output_path, title):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    pos = {
        "InCollisionImpulse": (-5.0, 1.8),
        "InShearStress": (-5.0, 0.6),
        "InChemicalConcentration": (-5.0, -1.4),

        "vWF_GPIb_ShearSensing": (-3.5, 1.2),
        "GPVI_CollagenSignal": (-3.5, 2.2),
        "Mechanosensitive_Ca2_Entry": (-3.5, 0.2),

        "ADP_P2Y12_Receptor": (-3.5, -1.2),
        "Thromboxane_TXA2_Signal": (-3.5, -2.2),
        "PI3K_Akt_Pathway": (-2.0, -1.2),

        "PLCgamma2": (-2.0, 1.5),
        "IP3_Ca2_Signaling": (-0.7, 0.9),
        "PKC": (0.5, 0.4),
        "Rap1": (0.5, -0.8),
        "PlateletActivation": (1.8, 0.0),

        "TalinKindlin_Activation": (3.2, 1.2),
        "Integrin_alphaIIb_beta3": (4.6, 1.2),
        "AdhesionProgram": (6.0, 1.2),

        "DenseGranuleSecretion": (3.2, -0.6),
        "ADP_TXA2_Feedback": (4.6, -0.6),
        "SecretionProgram": (6.0, -0.6),

        "RhoA_ROCK_Cytoskeleton": (3.2, -2.0),
        "ActinRemodeling": (4.6, -2.0),
        "MorphologyProgram": (6.0, -2.0),

        "OutStickiness": (7.6, 1.2),
        "OutSecretionRate": (7.6, -0.6),
        "OutMorphologyChange": (7.6, -2.0),
    }

    # fallback layout if a node is missing from manual layout
    fallback = nx.spring_layout(G, seed=2)
    for node in G.nodes:
        if node not in pos:
            pos[node] = fallback[node]

    colors = [node_color(n) for n in G.nodes]
    weights = [G[u][v]["weight"] for u, v in G.edges]
    widths = [1.2 + 1.2 * abs(w) for w in weights]

    plt.figure(figsize=(13, 7))

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=colors,
        node_size=2600,
        edgecolors="black",
        linewidths=1.0,
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8,
        font_weight="bold",
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="-|>",
        arrowsize=20,
        width=widths,
        connectionstyle="arc3,rad=0.05",
        edge_color="black",
    )

    edge_labels = {
        (u, v): f"{G[u][v]['weight']:.1f}"
        for u, v in G.edges
    }

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=8,
        label_pos=0.55,
    )

    plt.title(title, fontsize=14)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=250)
    plt.close()


def draw_pathway_highlights(G):
    # Full network
    draw_network(
        G,
        f"{OUTPUT_DIR}/grn_full_network.png",
        "SQUAD-inspired platelet GRN: mechanosensing, chemical feedback, adhesion, secretion, morphology",
    )

    # Mechanical pathway
    mech_edges = [
        ("InShearStress", "vWF_GPIb_ShearSensing"),
        ("InCollisionImpulse", "GPVI_CollagenSignal"),
        ("InShearStress", "Mechanosensitive_Ca2_Entry"),
        ("vWF_GPIb_ShearSensing", "PLCgamma2"),
        ("GPVI_CollagenSignal", "PLCgamma2"),
        ("Mechanosensitive_Ca2_Entry", "IP3_Ca2_Signaling"),
        ("PLCgamma2", "IP3_Ca2_Signaling"),
        ("IP3_Ca2_Signaling", "PKC"),
        ("PKC", "PlateletActivation"),
        ("PlateletActivation", "Rap1"),
        ("Rap1", "TalinKindlin_Activation"),
        ("TalinKindlin_Activation", "Integrin_alphaIIb_beta3"),
        ("Integrin_alphaIIb_beta3", "AdhesionProgram"),
        ("AdhesionProgram", "OutStickiness"),
        ("PlateletActivation", "RhoA_ROCK_Cytoskeleton"),
        ("RhoA_ROCK_Cytoskeleton", "ActinRemodeling"),
        ("ActinRemodeling", "MorphologyProgram"),
        ("MorphologyProgram", "OutMorphologyChange"),
    ]

    G_mech = nx.DiGraph()
    G_mech.add_nodes_from(G.nodes)
    for edge in mech_edges:
        if G.has_edge(*edge):
            G_mech.add_edge(edge[0], edge[1], weight=G[edge[0]][edge[1]]["weight"])

    draw_network(
        G_mech,
        f"{OUTPUT_DIR}/grn_mechanical_pathway.png",
        "Mechanical pathway: shear/collision sensing to adhesion and morphology",
    )

    # Chemical feedback pathway
    chem_edges = [
        ("InChemicalConcentration", "ADP_P2Y12_Receptor"),
        ("InChemicalConcentration", "Thromboxane_TXA2_Signal"),
        ("ADP_P2Y12_Receptor", "PI3K_Akt_Pathway"),
        ("PI3K_Akt_Pathway", "Rap1"),
        ("Thromboxane_TXA2_Signal", "PKC"),
        ("Rap1", "PlateletActivation"),
        ("PKC", "PlateletActivation"),
        ("PlateletActivation", "DenseGranuleSecretion"),
        ("DenseGranuleSecretion", "ADP_TXA2_Feedback"),
        ("ADP_TXA2_Feedback", "SecretionProgram"),
        ("SecretionProgram", "OutSecretionRate"),
        ("ADP_TXA2_Feedback", "ADP_P2Y12_Receptor"),
        ("PlateletActivation", "Rap1"),
        ("Rap1", "TalinKindlin_Activation"),
        ("TalinKindlin_Activation", "Integrin_alphaIIb_beta3"),
        ("Integrin_alphaIIb_beta3", "AdhesionProgram"),
        ("AdhesionProgram", "OutStickiness"),
    ]

    G_chem = nx.DiGraph()
    G_chem.add_nodes_from(G.nodes)
    for edge in chem_edges:
        if G.has_edge(*edge):
            G_chem.add_edge(edge[0], edge[1], weight=G[edge[0]][edge[1]]["weight"])

    draw_network(
        G_chem,
        f"{OUTPUT_DIR}/grn_chemical_feedback_pathway.png",
        "Chemical feedback pathway: ADP/TXA2 signaling to secretion and adhesion",
    )


def main():
    G = load_graphml_edges(GRAPHML_PATH)
    draw_pathway_highlights(G)
    print(f"Saved GRN network figures to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()