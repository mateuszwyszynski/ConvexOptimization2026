"""Interactive demo: how an affine map A x + b acts on a convex set.

Left panel  — image:     C  ->  A . C + b           (forward direction).
Right panel — preimage:  S  ->  { x : A x + b in S }.

Both panels share the same A, b, and shape parameters; on the left the shape
is interpreted as C in the domain, on the right as S in the codomain.

The preimage view is the picture that directly motivates why f(A x + b) is
convex when f is: the epigraph of g(x) = f(A x + b) is the preimage of
epi(f) under the affine map (x, t) -> (A x + b, t).

Run with:  uv run streamlit run set_transformation.py
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

st.set_page_config(layout="wide", page_title="Affine map & convex sets")
st.title("Affine precomposition on convex sets")
st.caption(
    "Same A, b on both sides.  Left: image of C under x ↦ A x + b.  "
    "Right: preimage of S under the same map."
)

with st.sidebar:
    st.header("Affine map  A x + b")
    c1, c2 = st.columns(2)
    a11 = c1.slider("a₁₁", -3.0, 3.0, 1.0, 0.05)
    a12 = c2.slider("a₁₂", -3.0, 3.0, 0.0, 0.05)
    a21 = c1.slider("a₂₁", -3.0, 3.0, 0.0, 0.05)
    a22 = c2.slider("a₂₂", -3.0, 3.0, 1.0, 0.05)
    b1 = c1.slider("b₁", -3.0, 3.0, 0.0, 0.05)
    b2 = c2.slider("b₂", -3.0, 3.0, 0.0, 0.05)

    st.header("Convex set")
    shape = st.selectbox("Shape", ["Ellipse", "L¹ ball", "L² ball", "L∞ ball"])
    if shape == "Ellipse":
        cx = st.slider("center x", -3.0, 3.0, 0.0, 0.05)
        cy = st.slider("center y", -3.0, 3.0, 0.0, 0.05)
        rx = st.slider("semi-axis x", 0.1, 3.0, 1.0, 0.05)
        ry = st.slider("semi-axis y", 0.1, 3.0, 0.5, 0.05)
        angle_deg = st.slider("rotation (deg)", -180, 180, 0, 1)
    else:
        radius = st.slider("radius", 0.1, 3.0, 1.0, 0.05)

A = np.array([[a11, a12], [a21, a22]])
b = np.array([b1, b2])


def boundary() -> np.ndarray:
    """Closed polyline tracing the boundary of the chosen set."""
    if shape == "Ellipse":
        t = np.linspace(0, 2 * np.pi, 256)
        pts = np.stack([rx * np.cos(t), ry * np.sin(t)], axis=1)
        c, s = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
        R = np.array([[c, -s], [s, c]])
        return pts @ R.T + np.array([cx, cy])
    if shape == "L² ball":
        t = np.linspace(0, 2 * np.pi, 256)
        return np.stack([radius * np.cos(t), radius * np.sin(t)], axis=1)
    if shape == "L¹ ball":
        r = radius
        return np.array([[r, 0], [0, r], [-r, 0], [0, -r], [r, 0]])
    # L∞ ball
    r = radius
    return np.array([[r, r], [-r, r], [-r, -r], [r, -r], [r, r]])


P = boundary()

# Image of C under x ↦ A x + b
image = P @ A.T + b

# Preimage of S: x = A^{-1} (s - b) for each s on the boundary of S.
det = np.linalg.det(A)
preimage_ok = abs(det) > 1e-6
preimage = (P - b) @ np.linalg.inv(A).T if preimage_ok else None


def setup_ax(ax, title: str, lim: float) -> None:
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.axhline(0, color="black", lw=0.5)
    ax.axvline(0, color="black", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(title)


# Shared viewport across both panels for honest visual comparison.
all_pts = [P, image] + ([preimage] if preimage_ok else [])
lim = max(3.0, np.max(np.abs(np.vstack(all_pts))) * 1.15)

left, right = st.columns(2)

with left:
    fig, ax = plt.subplots(figsize=(5, 5))
    setup_ax(ax, "Image:  C  →  A·C + b", lim)
    ax.fill(P[:, 0], P[:, 1], color="C0", alpha=0.15, label="C")
    ax.plot(P[:, 0], P[:, 1], color="C0", lw=1.0)
    ax.fill(image[:, 0], image[:, 1], color="C0", alpha=0.55, label="A·C + b")
    ax.plot(image[:, 0], image[:, 1], color="C0", lw=1.5)
    ax.legend(loc="upper right")
    st.pyplot(fig)

with right:
    fig, ax = plt.subplots(figsize=(5, 5))
    setup_ax(ax, "Preimage:  S  →  { x : A x + b ∈ S }", lim)
    ax.fill(P[:, 0], P[:, 1], color="C3", alpha=0.15, label="S")
    ax.plot(P[:, 0], P[:, 1], color="C3", lw=1.0)
    if preimage_ok:
        ax.fill(preimage[:, 0], preimage[:, 1], color="C3", alpha=0.55, label="preimage")
        ax.plot(preimage[:, 0], preimage[:, 1], color="C3", lw=1.5)
    else:
        ax.text(
            0,
            0,
            "A is singular —\npreimage is unbounded",
            ha="center",
            va="center",
            fontsize=11,
            color="gray",
        )
    ax.legend(loc="upper right")
    st.pyplot(fig)

st.markdown("---")
st.markdown(
    f"$A = \\begin{{bmatrix}} {a11:+.2f} & {a12:+.2f} \\\\ {a21:+.2f} & {a22:+.2f} "
    f"\\end{{bmatrix}}, \\quad "
    f"b = \\begin{{bmatrix}} {b1:+.2f} \\\\ {b2:+.2f} \\end{{bmatrix}}, \\quad "
    f"\\det A = {det:+.3f}$"
)
if not preimage_ok:
    st.warning("A is (nearly) singular — preimage panel disabled.")

with st.expander("Why this connects to f(A x + b) being convex"):
    st.markdown(
        "Let g(x) = f(A x + b).  The epigraph of g is\n\n"
        "$$\\mathrm{epi}\\,g = \\{(x, t) : f(A x + b) \\le t\\}"
        " = \\{(x, t) : (A x + b, t) \\in \\mathrm{epi}\\,f\\},$$\n\n"
        "i.e. the **preimage** of epi(f) under the affine map "
        "$(x, t) \\mapsto (A x + b, t)$.  Preimages of convex sets under affine maps "
        "are convex (right panel), so epi(g) is convex — i.e. g is convex."
    )
