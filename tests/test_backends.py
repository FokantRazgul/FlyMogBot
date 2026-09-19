"""Both sparse backends must agree with a dense reference and with each other."""

from __future__ import annotations

import pytest
import torch

from flymog.sim.backends import (
    build_backend,
    dense_reference,
    resolve_backend_kind,
    select_device,
)


@pytest.mark.parametrize("kind", ["csr", "gather"])
def test_backend_matches_dense_reference(kind, cpu):
    torch.manual_seed(0)
    n_post, n_pre, nnz, batch = 64, 48, 500, 5
    row = torch.randint(0, n_post, (nnz,))
    col = torch.randint(0, n_pre, (nnz,))
    values = torch.randn(nnz)
    x = torch.randn(n_pre, batch)

    expected = dense_reference(row, col, values, (n_post, n_pre), x)
    backend = build_backend(row, col, values, (n_post, n_pre), kind=kind, device=cpu)
    torch.testing.assert_close(backend(x), expected, rtol=1e-5, atol=1e-5)


def test_backends_agree_with_each_other(cpu):
    torch.manual_seed(1)
    n, nnz, batch = 100, 900, 7
    row = torch.randint(0, n, (nnz,))
    col = torch.randint(0, n, (nnz,))
    values = torch.randn(nnz)
    x = torch.randn(n, batch)

    csr = build_backend(row, col, values, (n, n), kind="csr", device=cpu)
    gather = build_backend(row, col, values, (n, n), kind="gather", device=cpu)
    torch.testing.assert_close(csr(x), gather(x), rtol=1e-5, atol=1e-5)


def test_duplicate_edges_accumulate(cpu):
    """Two synapses between the same pair must sum, not overwrite."""
    row = torch.tensor([0, 0])
    col = torch.tensor([1, 1])
    values = torch.tensor([2.0, 3.0])
    x = torch.ones(2, 1)
    for kind in ("csr", "gather"):
        backend = build_backend(row, col, values, (2, 2), kind=kind, device=cpu)
        assert backend(x)[0, 0].item() == pytest.approx(5.0)


def test_gather_chunking_does_not_change_result(cpu):
    """Chunked accumulation must be independent of the chunk size."""
    torch.manual_seed(2)
    n, nnz, batch = 50, 400, 4
    row = torch.randint(0, n, (nnz,))
    col = torch.randint(0, n, (nnz,))
    values = torch.randn(nnz)
    x = torch.randn(n, batch)

    whole = build_backend(row, col, values, (n, n), kind="gather", device=cpu)
    chunked = build_backend(row, col, values, (n, n), kind="gather", device=cpu)
    chunked.max_chunk_elements = 16  # forces many small chunks
    torch.testing.assert_close(whole(x), chunked(x), rtol=1e-6, atol=1e-6)


def test_auto_backend_avoids_sparse_kernels_on_mps():
    assert resolve_backend_kind("auto", torch.device("mps")) == "gather"
    assert resolve_backend_kind("auto", torch.device("cuda")) == "csr"
    assert resolve_backend_kind("auto", torch.device("cpu")) == "csr"


def test_select_device_rejects_unavailable_request():
    if torch.cuda.is_available():
        pytest.skip("this machine has CUDA, so the request would succeed")
    with pytest.raises(ValueError, match="not available"):
        select_device("cuda")
