"""
fig1_taxonomy.py
================

Generates Figure 1 for the STREAM-BSG paper: a visual taxonomy of the five B2B
fraud topologies, each rendered as a small graph illustrating the structural
pattern.

Output: figures/fig1_taxonomy.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np

# IEEE conference paper style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'lines.linewidth': 1.0,
})

# Colors (IEEE-friendly, print-safe)
COL_BUYER     = '#3B7DDC'   # blue
COL_SUPPLIER  = '#E89500'   # orange
COL_INVOICE   = '#888888'   # gray
COL_SHELL     = '#D7263D'   # red (for shell/anomalous nodes)
COL_NORMAL    = '#2A9D8F'   # green (for normal/legitimate edges)
COL_FRAUD     = '#D7263D'   # red (for fraud edges)


def draw_node(ax, x, y, color, edgecolor='black', label='', shape='circle',
              size=0.15, linewidth=1.0, linestyle='solid', fontsize=8,
              text_offset=(0, -0.28), text_color='black'):
    """Draw a node with a label below it."""
    if shape == 'circle':
        patch = Circle((x, y), size, facecolor=color, edgecolor=edgecolor,
                       linewidth=linewidth, linestyle=linestyle, zorder=3)
    elif shape == 'square':
        patch = mpatches.Rectangle((x - size, y - size), 2 * size, 2 * size,
                                    facecolor=color, edgecolor=edgecolor,
                                    linewidth=linewidth, linestyle=linestyle,
                                    zorder=3)
    else:
        raise ValueError(f"Unknown shape: {shape}")
    ax.add_patch(patch)
    if label:
        ax.text(x + text_offset[0], y + text_offset[1], label,
                ha='center', va='center', fontsize=fontsize, color=text_color,
                zorder=4)


def draw_edge(ax, x1, y1, x2, y2, color='black', linewidth=1.0,
              linestyle='solid', arrow=True, label='', label_offset=(0, 0.15),
              fontsize=7, label_color='black'):
    """Draw a directed edge with optional label."""
    if arrow:
        patch = FancyArrowPatch((x1, y1), (x2, y2),
                                arrowstyle='-|>', mutation_scale=12,
                                color=color, linewidth=linewidth,
                                linestyle=linestyle, zorder=2,
                                shrinkA=10, shrinkB=10)
    else:
        patch = Line2D([x1, x2], [y1, y2], color=color, linewidth=linewidth,
                       linestyle=linestyle, zorder=2)
    ax.add_patch(patch) if arrow else ax.add_line(patch)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label,
                ha='center', va='center', fontsize=fontsize,
                color=label_color, zorder=4)


def setup_subplot_axes(ax, title):
    """Standard subplot setup."""
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=8.5, pad=4)


# ----------------------------------------------------------------------------
# Topology drawings
# ----------------------------------------------------------------------------

def draw_vendor_injection(ax):
    """T1: New supplier appears, single buyer pays, supplier disappears."""
    setup_subplot_axes(ax, '(a) Vendor Injection')
    # Buyer
    draw_node(ax, -0.6, 0, COL_BUYER, label='Buyer', shape='circle',
              text_offset=(0, -0.32))
    # Shell supplier (dashed border, red) - place label below the square node
    draw_node(ax, 0.6, 0, COL_SHELL, label='Shell supplier',
              edgecolor=COL_SHELL, linewidth=1.5, linestyle='dashed',
              shape='square', text_color=COL_SHELL,
              text_offset=(0, -0.38))
    # Concentrated payment edge
    draw_edge(ax, -0.6, 0, 0.6, 0, color=COL_FRAUD, linewidth=2.0,
              label='\\$\\$\\$ payment', label_offset=(0, 0.18),
              label_color=COL_FRAUD)
    # Annotation
    ax.text(0, -0.95, 'New supplier, concentrated payment',
            ha='center', va='center', fontsize=7, style='italic')


def draw_invoice_cycling(ax):
    """T2: Same buyer-supplier pair, duplicate invoices in short window."""
    setup_subplot_axes(ax, '(b) Invoice Cycling')
    # Buyer
    draw_node(ax, -0.7, 0, COL_BUYER, label='Buyer', shape='circle')
    # Supplier
    draw_node(ax, 0.7, 0, COL_SUPPLIER, label='Supplier', shape='square')
    # Two duplicate invoices
    draw_node(ax, 0, 0.45, COL_INVOICE, label='$I_1$', shape='circle',
              size=0.10, text_offset=(0.22, 0))
    draw_node(ax, 0, -0.45, COL_INVOICE, label='$I_2$', shape='circle',
              size=0.10, text_offset=(0.22, 0))
    # Edges
    draw_edge(ax, -0.7, 0, 0, 0.45, arrow=False, color=COL_FRAUD,
              linewidth=1.5)
    draw_edge(ax, 0, 0.45, 0.7, 0, arrow=False, color=COL_FRAUD,
              linewidth=1.5)
    draw_edge(ax, -0.7, 0, 0, -0.45, arrow=False, color=COL_FRAUD,
              linewidth=1.5, linestyle='dashed')
    draw_edge(ax, 0, -0.45, 0.7, 0, arrow=False, color=COL_FRAUD,
              linewidth=1.5, linestyle='dashed')
    # Annotation
    ax.text(0, -0.95, 'Duplicate invoices within $\\Delta t$',
            ha='center', va='center', fontsize=7, style='italic')


def draw_payment_term_manipulation(ax):
    """T3: Established edge sees abrupt payment term shift."""
    setup_subplot_axes(ax, '(c) Payment-Term Manipulation')
    # Buyer
    draw_node(ax, -0.6, 0, COL_BUYER, label='Buyer', shape='circle')
    # Supplier
    draw_node(ax, 0.6, 0, COL_SUPPLIER, label='Supplier', shape='square')
    # Edge with term shift
    draw_edge(ax, -0.6, 0, 0.6, 0, color=COL_FRAUD, linewidth=2.0,
              label='', label_offset=(0, 0.15))
    # Term annotation with arrow
    ax.text(0, 0.30, 'NET-30  $\\rightarrow$  NET-1',
            ha='center', va='center', fontsize=8, color=COL_FRAUD,
            fontweight='bold')
    # Small time line below
    ax.annotate('', xy=(0.7, -0.5), xytext=(-0.7, -0.5),
                arrowprops=dict(arrowstyle='->', color='gray', lw=0.7))
    ax.text(-0.7, -0.65, 'history', ha='left', va='top', fontsize=6.5,
            color='gray')
    ax.text(0.7, -0.65, 'recent', ha='right', va='top', fontsize=6.5,
            color='gray')
    # Annotation
    ax.text(0, -0.95, 'Anomalous term shift on edge',
            ha='center', va='center', fontsize=7, style='italic')


def draw_shell_supplier_ring(ax):
    """T4: Buyer connected to multiple shell suppliers."""
    setup_subplot_axes(ax, '(d) Shell-Supplier Ring')
    # Center buyer
    draw_node(ax, 0, 0, COL_BUYER, label='Buyer', shape='circle',
              text_offset=(0.30, -0.05))
    # Three shell suppliers
    angles = [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]
    radius = 0.72
    pos = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    labels = ['$S_1$', '$S_2$', '$S_3$']
    text_offsets = [(0, 0.25), (-0.22, -0.05), (0.22, -0.05)]
    for (x, y), lbl, toff in zip(pos, labels, text_offsets):
        draw_node(ax, x, y, COL_SHELL, label=lbl,
                  edgecolor=COL_SHELL, linewidth=1.5, linestyle='dashed',
                  shape='square', size=0.13, text_color=COL_SHELL,
                  text_offset=toff)
        draw_edge(ax, 0, 0, x, y, color=COL_FRAUD, linewidth=1.2,
                  linestyle='dashed', arrow=False)
    # Optional inter-supplier edges (cycle hint)
    for i in range(3):
        x1, y1 = pos[i]
        x2, y2 = pos[(i + 1) % 3]
        draw_edge(ax, x1, y1, x2, y2, color=COL_SHELL, linewidth=0.6,
                  linestyle='dotted', arrow=False)
    # Annotation
    ax.text(0, -1.10, 'Coordinated low-history suppliers',
            ha='center', va='center', fontsize=7, style='italic')


def draw_wire_redirection(ax):
    """T5: Established edge sees bank-attribute change + high-value payment."""
    setup_subplot_axes(ax, '(e) Wire Redirection (BEC)')
    # Buyer
    draw_node(ax, -0.7, 0, COL_BUYER, label='Buyer', shape='circle',
              text_offset=(0, -0.32))
    # Supplier (legitimate, but attribute changed)
    draw_node(ax, 0.7, 0, COL_SUPPLIER, label='Supplier', shape='square',
              edgecolor=COL_FRAUD, linewidth=1.5,
              text_offset=(0, -0.32))
    # Edge with attribute change marker
    draw_edge(ax, -0.7, 0, 0.7, 0, color=COL_FRAUD, linewidth=2.0,
              label='', label_offset=(0, 0))
    # Bank attribute change label - place above the edge
    ax.text(0, 0.40, 'bank: A  $\\rightarrow$  B',
            ha='center', va='center', fontsize=8, color=COL_FRAUD,
            fontweight='bold')
    # High-value payment label - place below the edge but above node labels
    ax.text(0, 0.13, 'high-value \\$\\$\\$',
            ha='center', va='center', fontsize=7.5, color=COL_FRAUD,
            style='italic')
    # Annotation
    ax.text(0, -0.95, 'Bank change $+$ payment spike',
            ha='center', va='center', fontsize=7, style='italic')


def draw_legend(ax):
    """Legend panel."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Legend', fontsize=8.5, pad=4)

    items = [
        (Circle((0.15, 0.85), 0.05, facecolor=COL_BUYER, edgecolor='black'),
         'Buyer node'),
        (mpatches.Rectangle((0.10, 0.62), 0.10, 0.10,
                            facecolor=COL_SUPPLIER, edgecolor='black'),
         'Supplier node'),
        (Circle((0.15, 0.45), 0.04, facecolor=COL_INVOICE, edgecolor='black'),
         'Invoice node'),
        (mpatches.Rectangle((0.10, 0.22), 0.10, 0.10,
                            facecolor=COL_SHELL, edgecolor=COL_SHELL,
                            linestyle='dashed'),
         'Shell / anomalous'),
    ]
    for patch, txt in items:
        ax.add_patch(patch)
    # Labels
    ax.text(0.28, 0.85, 'Buyer node', va='center', fontsize=8)
    ax.text(0.28, 0.67, 'Supplier node', va='center', fontsize=8)
    ax.text(0.28, 0.47, 'Invoice node', va='center', fontsize=8)
    ax.text(0.28, 0.27, 'Shell / anomalous', va='center', fontsize=8)

    # Edge styles
    ax.plot([0.10, 0.22], [0.10, 0.10], color=COL_FRAUD, linewidth=2.0)
    ax.text(0.28, 0.10, 'Fraud-signal edge', va='center', fontsize=8)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.04,
                         wspace=0.10, hspace=0.30)

    draw_vendor_injection(axes[0, 0])
    draw_invoice_cycling(axes[0, 1])
    draw_payment_term_manipulation(axes[0, 2])
    draw_shell_supplier_ring(axes[1, 0])
    draw_wire_redirection(axes[1, 1])
    draw_legend(axes[1, 2])

    import os
    os.makedirs('figures', exist_ok=True)
    out_pdf = 'figures/fig1_taxonomy.pdf'
    out_png = 'figures/fig1_taxonomy.png'
    fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0.05)
    fig.savefig(out_png, bbox_inches='tight', pad_inches=0.05, dpi=300)
    print(f"Wrote {out_pdf} and {out_png}")


if __name__ == '__main__':
    main()
