"""Environment compatibility shims (pykeen 1.10.1 + torch 2.8).

torch 2.6+ changed the default of ``torch.load`` to ``weights_only=True``, but
pykeen 1.10.1's dataset-cache loader calls ``torch.load`` without that kwarg,
and its cached triples-factory binaries contain a ``pathlib.PosixPath`` global
that the weights-only unpickler refuses. On this trusted local machine we
restore the previous default for the whole process. Import this module FIRST
in any script that loads pykeen datasets or models.
"""

import torch

_orig_torch_load = torch.load


def _load_any(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _load_any
