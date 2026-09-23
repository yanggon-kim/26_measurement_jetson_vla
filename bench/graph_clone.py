"""Detach a CUDA-graph replay's outputs from its static buffers.

Shared by bench/opt_ladder.py (the LeRobot models, torch 2.8 in ~/venvs/pi0) and
bench/groot_phases.py (GR00T, torch 2.10 in ~/venvs/groot), which is why it lives in its
own module importing nothing but torch: the two run in different environments.

This is the single defect that cost the most in this study. `reduce-overhead` reuses one
static output buffer per compiled callable, so any callable invoked more than once per
inference has its earlier results overwritten before the caller consumes them. It surfaced
three separate times, each looking like a different bug:

  * pi0's embed_image, called once per camera -> camera 1's embedding silently became
    camera 3's (no exception at all; the numbers were merely wrong),
  * pg.forward returning a DynamicCache whose key_cache/value_cache lists alias the graph,
  * GR00T's language_model.forward -> "Error: accessing tensor output of CUDAGraphs that
    has been overwritten by a subsequent run".

Cloning walks containers AND HF cache objects, because a bare tensor .clone() fixed only
the first of the three.
"""
import torch


def clone_graph_outputs(r):
    """Detach a CUDA-graph replay's outputs from its static buffers.

    reduce-overhead reuses one set of output tensors per graph, so a callable invoked more
    than once per inference has its earlier results overwritten before the caller reads
    them. That is exactly what `embed_image` (once per camera) and `pg.forward` (once for
    the prefix, then once per denoising step) do, and both raised

        RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten
                      by a subsequent run

    A flat `.clone()` covered only a bare tensor, so it fixed embed_image and left
    pg.forward failing: pg.forward returns a tuple holding a list of embeddings and a
    cache. This walks tuples, lists and dicts and clones every tensor it finds, leaving
    anything else alone. The copies are inside the rung's own timing, so the number stays
    honest about what the workaround costs.
    """
    if torch.is_tensor(r):
        return r.clone()
    if isinstance(r, tuple):
        return tuple(clone_graph_outputs(x) for x in r)
    if isinstance(r, list):
        return [clone_graph_outputs(x) for x in r]
    if isinstance(r, dict):
        # HF's ModelOutput (BaseModelOutputWithPast and friends) SUBCLASSES OrderedDict.
        # Rebuilding it as a plain dict silently destroys the class, and the caller then
        # dies on attribute access: GR00T's vlm_graphs failed with "'dict' object has no
        # attribute 'last_hidden_state'" AFTER this clone was supposed to have fixed it.
        # Clone in place for any subclass so the type survives.
        if type(r) is dict:
            return {k: clone_graph_outputs(v) for k, v in r.items()}
        for k in list(r.keys()):
            r[k] = clone_graph_outputs(r[k])
        return r
    # HF cache objects (DynamicCache and friends) are neither tensor nor container, and
    # pg.forward returns one: `prefix_output.past_key_values` (modeling_pi0.py:475). Its
    # key/value tensors ARE graph outputs, so passing the object through untouched leaves
    # every later read aliasing the static buffers. Clone the lists in place.
    for attr in ("key_cache", "value_cache", "keys", "values"):
        if isinstance(getattr(r, attr, None), list):
            setattr(r, attr, [clone_graph_outputs(x) for x in getattr(r, attr)])
    if hasattr(r, "layers") and isinstance(getattr(r, "layers", None), list):
        for lay in r.layers:
            for attr in ("keys", "values"):
                if torch.is_tensor(getattr(lay, attr, None)):
                    setattr(lay, attr, getattr(lay, attr).clone())
    return r
