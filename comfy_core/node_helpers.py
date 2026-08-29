"""Minimal node_helpers shim for headless MiniMax H3 execution.

Only ``conditioning_set_values`` is referenced by the vendored H3 nodes.
This mirrors ComfyUI's implementation (append transformer_options values onto
each conditioning tensor's metadata dict).
"""


def conditioning_set_values(conditioning, values):
    c = []
    for t in conditioning:
        n = [t[0], t[1].copy()]
        n[1].update(values)
        c.append(n)
    return c
