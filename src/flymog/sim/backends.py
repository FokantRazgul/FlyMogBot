"""Device selection and sparse matrix-multiply backends.

The brief requires the simulation to run on CUDA, MPS and CPU through one
interface. The two backends differ only in how they evaluate ``W @ x`` for a
sparse connectome matrix ``W`` and a dense batch of states ``x``:

* :class:`CsrBackend` uses ``torch.sparse_csr``. This is the fast path on CUDA.
* :class:`GatherBackend` uses ``gather`` plus ``index_add_``. It needs no sparse
  kernel support at all, which matters on MPS where sparse CSR matmul is not
  reliably available.

Both are benchmarked by ``flymog bench-sim`` and must agree numerically; the
test suite checks them against a dense reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import torch

BackendKind = Literal["auto", "csr", "gather"]
DeviceKind = Literal["cuda", "mps", "cpu"]


def available_devices() -> dict[str, bool]:
    """Report which accelerators this machine actually offers."""
    mps_backend = getattr(torch.backends, "mps", None)
    return {
        "cuda": torch.cuda.is_available(),
        "mps": bool(mps_backend is not None and mps_backend.is_available()),
        "cpu": True,
    }


def select_device(preferred: str | None = None) -> torch.device:
    """Pick a device, preferring CUDA, then MPS, then CPU.

    An explicit ``preferred`` value is honoured only if that device is really
    available; otherwise we fall back rather than crash at the first kernel.
    """
    avail = available_devices()
    if preferred and preferred != "auto":
        if avail.get(preferred, False):
            return torch.device(preferred)
        raise ValueError(f"device {preferred!r} requested but not available: {avail}")
    for name in ("cuda", "mps", "cpu"):
        if avail[name]:
            return torch.device(name)
    return torch.device("cpu")


def resolve_backend_kind(kind: BackendKind, device: torch.device) -> Literal["csr", "gather"]:
    """Resolve ``auto`` to a concrete backend for this device.

    CSR sparse matmul is the fast path on CUDA and is fine on CPU. On MPS it is
    not dependable, so we use the gather path there.
    """
    if kind != "auto":
        return kind
    return "gather" if device.type == "mps" else "csr"


class SparseMatmul(Protocol):
    """Computes ``W @ x`` for a fixed sparse ``W`` of shape ``[n_post, n_pre]``."""

    shape: tuple[int, int]

    def __call__(self, x: torch.Tensor) -> torch.Tensor: ...


@dataclass
class CsrBackend:
    """Sparse CSR matmul. Fast where a sparse kernel exists."""

    matrix: torch.Tensor
    shape: tuple[int, int]
    name: str = "csr"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sparse.mm(self.matrix, x)


@dataclass
class GatherBackend:
    """Sparse matmul via gather and ``index_add_``.

    Materialising ``values[:, None] * x[col]`` costs ``nnz * batch`` elements, so
    the product is accumulated in chunks over the non-zeros to bound peak memory
    regardless of how large the connectome is.
    """

    row: torch.Tensor
    col: torch.Tensor
    values: torch.Tensor
    shape: tuple[int, int]
    max_chunk_elements: int = 16_000_000
    name: str = "gather"

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        n_post = self.shape[0]
        batch = x.shape[1]
        out = torch.zeros((n_post, batch), dtype=x.dtype, device=x.device)
        nnz = self.values.shape[0]
        chunk = max(1, self.max_chunk_elements // max(1, batch))
        for start in range(0, nnz, chunk):
            stop = min(start + chunk, nnz)
            contrib = x.index_select(0, self.col[start:stop])
            contrib = contrib * self.values[start:stop].unsqueeze(1)
            out.index_add_(0, self.row[start:stop], contrib)
        return out


def build_backend(
    row: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    shape: tuple[int, int],
    *,
    kind: BackendKind = "auto",
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> SparseMatmul:
    """Build a sparse matmul backend from COO triplets.

    ``row`` indexes the postsynaptic neuron, ``col`` the presynaptic one, so the
    resulting operator maps presynaptic activity to postsynaptic input.
    """
    device = device or select_device()
    resolved = resolve_backend_kind(kind, device)

    row = row.to(device=device, dtype=torch.int64)
    col = col.to(device=device, dtype=torch.int64)
    values = values.to(device=device, dtype=dtype)

    if resolved == "gather":
        # Sorting by row keeps index_add_ writes local, which helps cache reuse.
        order = torch.argsort(row)
        return GatherBackend(row=row[order], col=col[order], values=values[order], shape=shape)

    sparse_coo = torch.sparse_coo_tensor(
        torch.stack([row, col]), values, size=shape, device=device
    ).coalesce()
    return CsrBackend(matrix=sparse_coo.to_sparse_csr(), shape=shape)


def dense_reference(
    row: torch.Tensor,
    col: torch.Tensor,
    values: torch.Tensor,
    shape: tuple[int, int],
    x: torch.Tensor,
) -> torch.Tensor:
    """Dense ground truth for ``W @ x``, used by tests only. Never call on the
    real connectome: the dense matrix would be hundreds of gigabytes."""
    dense = torch.zeros(shape, dtype=x.dtype, device=x.device)
    dense.index_put_(
        (row.to(torch.int64), col.to(torch.int64)), values.to(x.dtype), accumulate=True
    )
    return dense @ x
