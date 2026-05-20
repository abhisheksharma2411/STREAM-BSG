"""
fig2_architecture.py
====================

Generates Figure 2 for the STREAM-BSG paper: the system architecture diagram
showing the streaming pipeline from event ingestion through to decision output.

Output: figures/fig2_architecture.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import os

# IEEE conference paper style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 8,
    'axes.linewidth': 0.5,
    'lines.linewidth': 1.0,
})

# Colors
COL_INGEST    = '#D6E9F8'   # light blue (ingestion)
COL_PROCESS   = '#F8E4B8'   # light gold (stream processing)
COL_STATE     = '#F5C6CB'   # light red (state stores)
COL_FEATURE   = '#D4EDDA'   # light green (features)
COL_ML        = '#E0CCEE'   # light purple (ML)
COL_OUTPUT    = '#FCE9E9'   # very light pink (output)
COL_BORDER    = '#333333'
COL_ARROW     = '#444444'
COL_LATENCY   = '#A52020'


def draw_box(ax, x, y, w, h, text, subtext=None, color='white', fontsize=8,
             subtext_fontsize=7, border_color=COL_BORDER, border_width=0.8,
             boxstyle='round,pad=0.02'):
    """Draw a labeled rounded box."""
    patch = FancyBboxPatch((x - w/2, y - h/2), w, h, boxstyle=boxstyle,
                           linewidth=border_width, edgecolor=border_color,
                           facecolor=color, zorder=2)
    ax.add_patch(patch)
    if subtext:
        ax.text(x, y + 0.07, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', zorder=3)
        ax.text(x, y - 0.10, subtext, ha='center', va='center',
                fontsize=subtext_fontsize, style='italic',
                color='#555555', zorder=3)
    else:
        ax.text(x, y, text, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, label='', label_offset=(0, 0.08),
               color=COL_ARROW, linewidth=1.2, label_color=COL_LATENCY,
               fontsize=6.5, style='solid'):
    """Draw a directed arrow between two points with an optional label."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                            arrowstyle='-|>', mutation_scale=14,
                            color=color, linewidth=linewidth,
                            linestyle=style, zorder=1)
    ax.add_patch(arrow)
    if label:
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx + label_offset[0], my + label_offset[1], label,
                ha='center', va='center', fontsize=fontsize,
                color=label_color, zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(7.5, 3.7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.2)
    ax.set_aspect('auto')
    ax.axis('off')

    # ===== Pipeline boxes =====
    box_h = 0.7
    box_w = 1.7

    # Main horizontal pipeline (top row), y = 3.3
    y_main = 3.3

    # 1. Transaction events (input)
    draw_box(ax, 1.0, y_main, box_w, box_h,
             'Transaction\nEvents', color=COL_INGEST,
             border_color='#2A6FAB')

    # 2. Kafka
    draw_box(ax, 3.0, y_main, box_w, box_h,
             'Kafka', subtext='message bus', color=COL_INGEST,
             border_color='#2A6FAB')

    # 3. Flink stream processor
    draw_box(ax, 5.0, y_main, box_w, box_h,
             'Flink', subtext='stream proc', color=COL_PROCESS,
             border_color='#B07900')

    # 4. Feature Computation
    draw_box(ax, 7.5, y_main, box_w + 0.2, box_h,
             'Feature', subtext='computation (47 feats)',
             color=COL_FEATURE, border_color='#2E7D32',
             subtext_fontsize=6.5)

    # 5. Feature Store
    draw_box(ax, 10.0, y_main, box_w, box_h,
             'Feature\nStore', color=COL_FEATURE,
             border_color='#2E7D32')

    # 6. XGBoost Inference
    draw_box(ax, 12.0, y_main, box_w, box_h,
             'XGBoost', subtext='inference',
             color=COL_ML, border_color='#5E3D85')

    # 7. Decision output (separate, below right-end)
    draw_box(ax, 12.5, 1.5, box_w + 0.5, box_h,
             'Decision', subtext='approve / block / review',
             color=COL_OUTPUT, border_color='#A52020',
             subtext_fontsize=6.5)

    # ===== Sidecar: streaming graph state (Redis) - below Flink =====
    draw_box(ax, 5.0, 1.5, box_w + 0.3, box_h + 0.1,
             'Redis', subtext='streaming graph state\n(90-day window)',
             color=COL_STATE, border_color='#A52020',
             subtext_fontsize=6.5)

    # ===== Arrows in the main pipeline =====
    arrow_label_offset_y = 0.22

    draw_arrow(ax, 1.85, y_main, 2.15, y_main)
    draw_arrow(ax, 3.85, y_main, 4.15, y_main,
               label='~2 ms', label_offset=(0, arrow_label_offset_y))
    draw_arrow(ax, 5.85, y_main, 6.55, y_main,
               label='~15 ms', label_offset=(0, arrow_label_offset_y))
    draw_arrow(ax, 8.55, y_main, 9.15, y_main,
               label='~8 ms', label_offset=(0, arrow_label_offset_y))
    draw_arrow(ax, 10.85, y_main, 11.15, y_main,
               label='~12 ms', label_offset=(0, arrow_label_offset_y))

    # Arrow from XGBoost (top right) down to Decision
    draw_arrow(ax, 12.0, y_main - 0.4, 12.4, 1.90,
               label='~3 ms', label_offset=(0.22, 0))

    # Bidirectional arrows: Flink <--> Redis (state read/write)
    draw_arrow(ax, 5.0, y_main - 0.4, 5.0, 1.90,
               style='dashed')
    draw_arrow(ax, 4.7, 1.90, 4.7, y_main - 0.4,
               style='dashed',
               label='~5 ms\nstate r/w',
               label_offset=(-0.55, 0.05))

    # Arrow from Feature computation down to Redis
    draw_arrow(ax, 7.5, y_main - 0.4, 5.9, 1.90,
               style='dashed', color='#888888', linewidth=0.9)

    # ===== Legend / colour key (bottom) =====
    legend_y = 0.3
    legend_items = [
        (COL_INGEST,  'Ingestion'),
        (COL_PROCESS, 'Stream proc'),
        (COL_STATE,   'State'),
        (COL_FEATURE, 'Features'),
        (COL_ML,      'ML'),
        (COL_OUTPUT,  'Decision'),
    ]
    legend_x_start = 1.5
    legend_spacing = 1.95
    for i, (col, lbl) in enumerate(legend_items):
        x = legend_x_start + i * legend_spacing
        ax.add_patch(Rectangle((x, legend_y), 0.35, 0.25,
                               facecolor=col, edgecolor=COL_BORDER,
                               linewidth=0.5))
        ax.text(x + 0.45, legend_y + 0.125, lbl, fontsize=7,
                va='center', ha='left')

    # Title / subtitle
    ax.text(7.0, 4.7, 'STREAM-BSG pipeline ($p99$ latency $<$ 100 ms)',
            ha='center', va='center', fontsize=9, fontweight='bold')

    os.makedirs('figures', exist_ok=True)
    out_pdf = 'figures/fig2_architecture.pdf'
    out_png = 'figures/fig2_architecture.png'
    fig.savefig(out_pdf, bbox_inches='tight', pad_inches=0.08)
    fig.savefig(out_png, bbox_inches='tight', pad_inches=0.08, dpi=300)
    print(f"Wrote {out_pdf} and {out_png}")


if __name__ == '__main__':
    main()
