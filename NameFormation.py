#!/usr/bin/env python3

"""
Multi-Agent Name Formation Control
===================================

N = 20 agents
Communication graph = connected Erdős–Rényi graph
Letter generation = font -> binary glyph -> skeletonization
Formation controller = graph-based formation + pinning
Assignment = Hungarian algorithm
Output = MP4 animation

Usage
-----
Interactive:
    python formation_name.py

Command line:
    python formation_name.py Yashwanth

Requirements
------------
numpy
matplotlib
scipy
networkx
pillow
scikit-image
"""

# ============================================================
# IMPORTS
# ============================================================

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib import font_manager

import networkx as nx

from PIL import Image, ImageDraw, ImageFont

from scipy.optimize import linear_sum_assignment

from skimage.morphology import skeletonize


# ============================================================
# GLOBAL PARAMETERS
# ============================================================

N = 20

# Erdős–Rényi probability
P_EDGE = 0.30

# ------------------------------------------------------------
# Simulation
# ------------------------------------------------------------

DT = 0.02

# Maximum simulation time for each letter
MAX_LETTER_TIME = 4.0

# Time for which completed letter is displayed
HOLD_TIME = 0.8

# ------------------------------------------------------------
# Formation controller
# ------------------------------------------------------------

# Relative formation gain
K_FORM = 18.0

# Pinning gain
K_PIN = 12.0

# Velocity damping
K_DAMP = 8.0

# ------------------------------------------------------------
# Letter generation
# ------------------------------------------------------------

IMAGE_SIZE = 1000

# Bold font gives better recognition with only 20 agents
FONT_SIZE = 850

# Desired physical dimensions of letters
LETTER_WIDTH = 4.0
LETTER_HEIGHT = 5.0

# Minimum distance between selected points
# relative to normalized letter coordinates
MIN_POINT_DISTANCE = 0.20

# ------------------------------------------------------------
# Workspace
# ------------------------------------------------------------

WORKSPACE_X = (-8.0, 8.0)
WORKSPACE_Y = (-6.0, 6.0)

# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42

rng = np.random.default_rng(SEED)

OUTPUT_DIR = Path("T://IITH//Courses//Multi-Agent-Systems//Codes//Assignment_3//Formationoutput")


# ============================================================
# FONT
# ============================================================

def find_bold_font():
    """
    Find a suitable bold TrueType font.
    """

    preferred = [
        "DejaVuSans-Bold.ttf",
        "LiberationSans-Bold.ttf",
        "Arial Bold.ttf",
    ]

    fonts = font_manager.findSystemFonts(
        fontpaths=None,
        fontext="ttf"
    )

    for wanted in preferred:

        for font in fonts:

            if Path(font).name.lower() == wanted.lower():

                return font

    # fallback
    return font_manager.findfont("DejaVu Sans")


# ============================================================
# LETTER SKELETON
# ============================================================

def render_letter_mask(letter):
    """
    Render a capital letter as a binary image.
    """

    letter = letter.upper()

    font_path = find_bold_font()

    font = ImageFont.truetype(
        font_path,
        FONT_SIZE
    )

    image = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        color=0
    )

    draw = ImageDraw.Draw(image)

    # Bounding box
    bbox = draw.textbbox(
        (0, 0),
        letter,
        font=font,
        stroke_width=0
    )

    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    x = (IMAGE_SIZE - width) // 2 - bbox[0]
    y = (IMAGE_SIZE - height) // 2 - bbox[1]

    draw.text(
        (x, y),
        letter,
        font=font,
        fill=255
    )

    array = np.asarray(image)

    mask = array > 128

    return mask


def skeleton_points(letter):
    """
    Generate a SINGLE-LAYER centerline representation
    of a capital letter.

    This is the key improvement over the previous code.
    """

    mask = render_letter_mask(letter)

    # Skeletonize filled glyph
    skeleton = skeletonize(mask)

    ys, xs = np.where(skeleton)

    if len(xs) < N:
        raise RuntimeError(
            f"Skeleton for '{letter}' has only "
            f"{len(xs)} pixels."
        )

    points = np.column_stack(
        [
            xs.astype(float),
            -ys.astype(float)
        ]
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    center = points.mean(axis=0)

    points -= center

    width = np.ptp(points[:, 0])
    height = np.ptp(points[:, 1])

    if width <= 0 or height <= 0:

        raise RuntimeError(
            f"Invalid skeleton for letter '{letter}'."
        )

    points[:, 0] /= width
    points[:, 1] /= height

    # Scale
    points[:, 0] *= LETTER_WIDTH
    points[:, 1] *= LETTER_HEIGHT

    return points


# ============================================================
# FARTHER-POINT SAMPLING
# ============================================================

def farthest_point_sampling(points, n):
    """
    Select n well-separated points from the skeleton.

    Unlike random pixel selection, this keeps the agents
    distributed along the complete letter.
    """

    if len(points) <= n:

        return points.copy()

    selected = []

    # First point = point closest to centroid
    centroid = points.mean(axis=0)

    first = np.argmin(
        np.linalg.norm(
            points - centroid,
            axis=1
        )
    )

    selected.append(first)

    distances = np.linalg.norm(
        points - points[first],
        axis=1
    )

    for _ in range(1, n):

        idx = np.argmax(distances)

        selected.append(idx)

        new_distances = np.linalg.norm(
            points - points[idx],
            axis=1
        )

        distances = np.minimum(
            distances,
            new_distances
        )

    return points[selected]


# ============================================================
# IMPROVE POINT DISTRIBUTION
# ============================================================

def generate_letter_points(letter):
    """
    Generate exactly N points representing a letter.
    """

    points = skeleton_points(letter)

    points = farthest_point_sampling(
        points,
        N
    )

    # Re-center
    points -= points.mean(axis=0)

    # Scale again so all letters occupy comparable area
    width = np.ptp(points[:, 0])
    height = np.ptp(points[:, 1])

    if width > 0:
        points[:, 0] *= LETTER_WIDTH / width

    if height > 0:
        points[:, 1] *= LETTER_HEIGHT / height

    # Re-center after scaling
    points -= points.mean(axis=0)

    return points


# ============================================================
# HUNGARIAN ASSIGNMENT
# ============================================================

def assign_agents(x, desired):
    """
    Assign current agents to desired letter positions
    using the minimum-cost Hungarian assignment.
    """

    cost = np.sum(
        (
            x[:, None, :]
            -
            desired[None, :, :]
        ) ** 2,
        axis=2
    )

    rows, cols = linear_sum_assignment(cost)

    assigned = np.zeros_like(desired)

    for r, c in zip(rows, cols):

        assigned[r] = desired[c]

    return assigned


# ============================================================
# GRAPH GENERATION
# ============================================================

def generate_connected_graph(n, p):
    """
    Generate connected G(n,p).
    """

    attempts = 0

    while True:

        attempts += 1

        G = nx.erdos_renyi_graph(
            n,
            p,
            seed=int(
                rng.integers(
                    0,
                    2**31 - 1
                )
            )
        )

        if nx.is_connected(G):

            return G

        if attempts > 10000:

            raise RuntimeError(
                "Could not generate a connected graph."
            )


# ============================================================
# METROPOLIS WEIGHT MATRIX
# ============================================================

def metropolis_matrix(G):
    """
    Symmetric doubly-stochastic Metropolis matrix.
    """

    W = np.zeros(
        (N, N)
    )

    degree = dict(
        G.degree()
    )

    for i, j in G.edges():

        wij = 1.0 / (
            1.0
            +
            max(
                degree[i],
                degree[j]
            )
        )

        W[i, j] = wij
        W[j, i] = wij

    for i in range(N):

        W[i, i] = (
            1.0
            -
            np.sum(W[i])
        )

    return W


# ============================================================
# FAST GRAPH FORMATION CONTROLLER
# ============================================================

def formation_controller(
    x,
    v,
    desired,
    W
):
    """
    Graph-based formation controller.

    u_i =
        -K_FORM * relative formation error
        -K_PIN  * local pinning error
        -K_DAMP * velocity

    The relative term preserves the desired formation geometry.

    The pinning term dramatically accelerates convergence
    to the desired absolute formation.

    Each agent uses its own desired position and communicates
    only with graph neighbors.
    """

    # --------------------------------------------------------
    # Relative formation error
    #
    # r_i = sum_j W_ij[
    #       (x_i-x_j)-(p_i-p_j)
    #       ]
    # --------------------------------------------------------

    relative_error = np.zeros_like(x)

    for i in range(N):

        for j in range(N):

            if i == j:
                continue

            if W[i, j] > 0:

                relative_error[i] += (
                    W[i, j]
                    *
                    (
                        (x[i] - x[j])
                        -
                        (desired[i] - desired[j])
                    )
                )

    # --------------------------------------------------------
    # Local pinning error
    # --------------------------------------------------------

    pinning_error = (
        x - desired
    )

    # --------------------------------------------------------
    # Control input
    # --------------------------------------------------------

    u = (
        -K_FORM * relative_error
        -K_PIN * pinning_error
        -K_DAMP * v
    )

    return u


# ============================================================
# DRAW FRAME
# ============================================================

def draw_frame(
    ax,
    x,
    desired,
    G,
    letter,
    formation_number,
    total_letters,
    error,
    name
):

    ax.clear()

    # Desired formation
    ax.scatter(
        desired[:, 0],
        desired[:, 1],
        marker="x",
        s=90,
        linewidths=2.0,
        label="Desired"
    )

    # Agents
    ax.scatter(
        x[:, 0],
        x[:, 1],
        s=85,
        label="Agents"
    )

    # Communication graph
    for i, j in G.edges():

        ax.plot(
            [
                x[i, 0],
                x[j, 0]
            ],
            [
                x[i, 1],
                x[j, 1]
            ],
            linewidth=0.7,
            alpha=0.25
        )

    # Agent IDs
    for i in range(N):

        ax.text(
            x[i, 0],
            x[i, 1],
            str(i + 1),
            fontsize=7,
            ha="center",
            va="center"
        )

    ax.set_xlim(
        WORKSPACE_X
    )

    ax.set_ylim(
        WORKSPACE_Y
    )

    ax.set_aspect(
        "equal"
    )

    ax.grid(
        True,
        alpha=0
    )

    ax.set_xlabel(
        r"$x$"
    )

    ax.set_ylabel(
        r"$y$"
    )

    # ax.set_title(
    #     f"Name Formation: "
    #     f"{formation_number}/{total_letters}   "
    #     f"Letter = {letter}   "
    #     f"Error = {error:.4f}"
    # )
    ax.set_title(
            f"Name: {name}  |"
            # f"Letter = {letter}   "
            f"Error = {error:.4f}"
    )
        
    ax.legend(
        loc="upper right"
    )


# ============================================================
# MOVE TO LETTER
# ============================================================

def move_to_letter(
    x,
    v,
    desired,
    G,
    W,
    letter,
    formation_number,
    total_letters,
    writer,
    ax,
    name
):

    # --------------------------------------------------------
    # Assign agents to letter points
    # --------------------------------------------------------

    desired = assign_agents(
        x,
        desired
    )

    n_steps = int(
        MAX_LETTER_TIME / DT
    )

    convergence_count = 0

    required_consecutive_frames = int(
        0.25 / DT
    )

    for step in range(n_steps):

        # ----------------------------------------------------
        # Controller
        # ----------------------------------------------------

        u = formation_controller(
            x,
            v,
            desired,
            W
        )

        # ----------------------------------------------------
        # Dynamics
        # ----------------------------------------------------

        v += u * DT

        x += v * DT

        # ----------------------------------------------------
        # Error
        # ----------------------------------------------------

        error = np.sqrt(
            np.mean(
                np.sum(
                    (x - desired) ** 2,
                    axis=1
                )
            )
        )

        # ----------------------------------------------------
        # Animation
        # ----------------------------------------------------

        draw_frame(
            ax=ax,
            x=x,
            desired=desired,
            G=G,
            letter=letter,
            formation_number=formation_number,
            total_letters=total_letters,
            error=error,
            name=name
        )

        writer.grab_frame()

        # ----------------------------------------------------
        # Convergence
        # ----------------------------------------------------

        if error < 0.08:

            convergence_count += 1

        else:

            convergence_count = 0

        if (
            convergence_count
            >=
            required_consecutive_frames
        ):

            break

    # --------------------------------------------------------
    # Hold completed formation
    # --------------------------------------------------------

    hold_steps = int(
        HOLD_TIME / DT
    )

    for _ in range(hold_steps):

        draw_frame(
            ax=ax,
            x=x,
            desired=desired,
            G=G,
            letter=letter,
            formation_number=formation_number,
            total_letters=total_letters,
            error=error,
            name=name
        )

        writer.grab_frame()

    return x, v, desired


# ============================================================
# SAVE LETTER PREVIEWS
# ============================================================

def save_letter_preview(
    letter,
    points
):

    fig, ax = plt.subplots(
        figsize=(5, 5)
    )

    ax.scatter(
        points[:, 0],
        points[:, 1],
        s=80
    )

    for i, p in enumerate(points):

        ax.text(
            p[0],
            p[1],
            str(i + 1),
            fontsize=7,
            ha="center",
            va="center"
        )

    ax.set_aspect(
        "equal"
    )

    ax.grid(
        True,
        alpha=0.2
    )

    ax.set_title(
        f"20-Agent Formation: {letter}"
    )

    ax.set_xlim(
        -3,
        3
    )

    ax.set_ylim(
        -3.5,
        3.5
    )

    fig.savefig(
        OUTPUT_DIR /
        f"letter_{letter}.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Name input
    # --------------------------------------------------------

    if len(sys.argv) > 1:

        name = " ".join(
            sys.argv[1:]
        )

    else:

        name = input(
            "Enter the name to display: "
        )

    # Keep alphabetic characters only
    name = "".join(
        c
        for c in name.upper()
        if c.isalpha()
    )

    if not name:

        raise ValueError(
            "Please enter a valid alphabetic name."
        )

    print()
    print("=" * 70)
    print("MULTI-AGENT NAME FORMATION")
    print("=" * 70)
    print(
        f"Name                : {name}"
    )
    print(
        f"Number of agents    : {N}"
    )
    print(
        f"ER probability      : {P_EDGE}"
    )
    print(
        f"Time step           : {DT}"
    )
    print(
        f"Formation gain      : {K_FORM}"
    )
    print(
        f"Pinning gain        : {K_PIN}"
    )
    print(
        f"Damping gain        : {K_DAMP}"
    )
    print()

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Communication graph
    # --------------------------------------------------------

    print(
        "Generating connected "
        "Erdős–Rényi graph..."
    )

    G = generate_connected_graph(
        N,
        P_EDGE
    )

    W = metropolis_matrix(
        G
    )

    print(
        f"Edges               : "
        f"{G.number_of_edges()}"
    )

    print(
        f"Connected           : "
        f"{nx.is_connected(G)}"
    )

    # --------------------------------------------------------
    # Algebraic connectivity
    # --------------------------------------------------------

    L = nx.laplacian_matrix(
        G
    ).toarray()

    eigenvalues = np.linalg.eigvalsh(
        L
    )

    lambda_2 = eigenvalues[1]

    print(
        f"lambda_2(L)         : "
        f"{lambda_2:.5f}"
    )

    # --------------------------------------------------------
    # Save graph image
    # --------------------------------------------------------

    fig_graph, ax_graph = plt.subplots(
        figsize=(8, 7)
    )

    graph_pos = nx.spring_layout(
        G,
        seed=SEED
    )

    nx.draw_networkx(
        G,
        pos=graph_pos,
        ax=ax_graph,
        node_size=600,
        with_labels=True
    )

    ax_graph.set_title(
        f"Erdős–Rényi Communication Graph\n"
        f"N={N}, p={P_EDGE}, "
        f"$\\lambda_2={lambda_2:.4f}$"
    )

    fig_graph.savefig(
        OUTPUT_DIR /
        "communication_graph.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(
        fig_graph
    )

    # --------------------------------------------------------
    # Initial agent positions
    # --------------------------------------------------------

    x = np.column_stack(
        [
            rng.uniform(
                WORKSPACE_X[0],
                WORKSPACE_X[1],
                N
            ),
            rng.uniform(
                WORKSPACE_Y[0],
                WORKSPACE_Y[1],
                N
            )
        ]
    )

    initial_positions = x.copy()

    v = np.zeros(
        (N, 2)
    )

    # --------------------------------------------------------
    # Generate letter formations
    # --------------------------------------------------------

    print()
    print(
        "Generating single-layer "
        "letter skeletons..."
    )

    formations = {}

    for letter in sorted(
        set(name)
    ):

        print(
            f"  {letter}"
        )

        formations[letter] = (
            generate_letter_points(
                letter
            )
        )

        save_letter_preview(
            letter,
            formations[letter]
        )

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    video_file = (
        OUTPUT_DIR /
        f"{name.lower()}_formation.mp4"
    )

    print()
    print(
        f"Recording: {video_file}"
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    writer = FFMpegWriter(
        fps=30,
        metadata={
            "title":
                f"Multi-Agent Formation - {name}",
            "artist":
                "Formation Control"
        },
        bitrate=5000
    )

    # --------------------------------------------------------
    # Video
    # --------------------------------------------------------

    with writer.saving(
        fig,
        str(video_file),
        dpi=120
    ):

        # ----------------------------------------------------
        # Initial configuration
        # ----------------------------------------------------

        initial_frames = int(
            1.0 / DT
        )

        for _ in range(
            initial_frames
        ):

            ax.clear()

            ax.scatter(
                x[:, 0],
                x[:, 1],
                s=85
            )

            for i, j in G.edges():

                ax.plot(
                    [
                        x[i, 0],
                        x[j, 0]
                    ],
                    [
                        x[i, 1],
                        x[j, 1]
                    ],
                    linewidth=0.7,
                    alpha=0.25
                )

            for i in range(N):

                ax.text(
                    x[i, 0],
                    x[i, 1],
                    str(i + 1),
                    fontsize=7,
                    ha="center",
                    va="center"
                )

            ax.set_xlim(
                WORKSPACE_X
            )

            ax.set_ylim(
                WORKSPACE_Y
            )

            ax.set_aspect(
                "equal"
            )

            ax.grid(
                True,
                alpha=0.2
            )

            ax.set_title(
                f"Initial Random Configuration\n"
                f"Name: {name}"
            )

            writer.grab_frame()

        # ----------------------------------------------------
        # Letter sequence
        # ----------------------------------------------------

        for k, letter in enumerate(
            name,
            start=1
        ):

            print(
                f"Forming letter "
                f"{k}/{len(name)}: {letter}"
            )

            target = formations[
                letter
            ].copy()

            x, v, target = (
                move_to_letter(
                    x=x,
                    v=v,
                    desired=target,
                    G=G,
                    W=W,
                    letter=letter,
                    formation_number=k,
                    total_letters=len(name),
                    writer=writer,
                    ax=ax,
                    name=name
                )
            )

    plt.close(fig)

    # --------------------------------------------------------
    # Save data
    # --------------------------------------------------------

    np.save(
        OUTPUT_DIR /
        "initial_positions.npy",
        initial_positions
    )

    nx.write_gexf(
        G,
        OUTPUT_DIR /
        "communication_graph.gexf"
    )

    # --------------------------------------------------------
    # Save summary
    # --------------------------------------------------------

    with open(
        OUTPUT_DIR /
        "simulation_info.txt",
        "w"
    ) as f:

        f.write(
            "MULTI-AGENT NAME FORMATION\n"
        )

        f.write(
            "===========================\n\n"
        )

        f.write(
            f"Name: {name}\n"
        )

        f.write(
            f"N: {N}\n"
        )

        f.write(
            f"P_EDGE: {P_EDGE}\n"
        )

        f.write(
            f"Edges: {G.number_of_edges()}\n"
        )

        f.write(
            f"Connected: "
            f"{nx.is_connected(G)}\n"
        )

        f.write(
            f"lambda_2(L): "
            f"{lambda_2:.8f}\n"
        )

        f.write(
            f"DT: {DT}\n"
        )

        f.write(
            f"K_FORM: {K_FORM}\n"
        )

        f.write(
            f"K_PIN: {K_PIN}\n"
        )

        f.write(
            f"K_DAMP: {K_DAMP}\n"
        )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SIMULATION COMPLETE")
    print("=" * 70)

    print(
        f"Video: {video_file}"
    )

    print(
        "Letter previews: "
        f"{OUTPUT_DIR}/letter_*.png"
    )

    print(
        f"Graph: "
        f"{OUTPUT_DIR}/communication_graph.png"
    )

    print()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()