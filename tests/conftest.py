"""Shared fixtures. Nothing here touches the network or the real connectome."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from flymog.config import ConnectomeConfig, LifConfig, load_sim_config


@pytest.fixture(scope="session")
def sim_cfg():
    return load_sim_config()


@pytest.fixture(scope="session")
def lif_cfg(sim_cfg) -> LifConfig:
    return sim_cfg.lif


@pytest.fixture(scope="session")
def connectome_cfg(sim_cfg) -> ConnectomeConfig:
    return sim_cfg.connectome


@pytest.fixture
def cpu() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def small_surrogate(connectome_cfg):
    from flymog.data.download import make_surrogate_connectome

    return make_surrogate_connectome(400, 2000, seed=7, cfg=connectome_cfg)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)
