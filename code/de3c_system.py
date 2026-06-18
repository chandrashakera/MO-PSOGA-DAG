"""
de3c_system.py
Device-Edge-Cloud (DE3C) three-tier system model.

System parameters replicated exactly from Wang et al. [2024] Table 1
(doi:10.1007/s10586-024-04851-3) to enable direct comparison.

Node indexing convention (matches paper Eq. 1):
  0 ... D-1            : user devices
  D ... D+E-1          : edge servers
  D+E ... D+E+V-1     : cloud servers
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class Node:
    """A computing node in the DE3C system."""
    idx: int               # global index in [0, D+E+V)
    node_type: str         # 'device' | 'edge' | 'cloud'
    capacity_ghz: float    # g_i  — total GHz (cores × speed)
    transfer_mbps: float   # r_i  — network transfer rate (Mbps)
    # Device-only power params (Wang et al. 2024, Section 3.1)
    power_compute: float = 0.0   # p_i^c  W/(GHz·s)
    power_transmit: float = 0.0  # p_i^r  W/MB
    # Edge-only power params (Katal et al. 2023 linear model)
    power_max: float = 0.0       # P_max (W)
    power_idle: float = 0.0      # P_idle (W)
    # Cloud-only pricing
    price_per_sec: float = 0.0   # h_i  monetary units/s


@dataclass
class Connectivity:
    """
    a_{i,j} = 1 if ES s_i covers device s_j.
    Represented as adjacency: es_covers[i] = set of device indices covered.
    (Wang et al. 2024, constraint Eq. 5 in paper)
    """
    es_covers: List[set] = field(default_factory=list)

    def edge_covers_device(self, es_local_idx: int, dev_idx: int) -> bool:
        return dev_idx in self.es_covers[es_local_idx]


@dataclass
class TaskAssignment:
    """
    b_{i,k} = 1 if task t_k is launched by device s_i.
    assignment: (T,) array, launched_by[k] = device index.
    (Wang et al. 2024, constraint Eq. 4 in paper)
    """
    launched_by: np.ndarray  # shape (T,), device index (0-indexed global)


class DE3CSystem:
    """
    DE3C three-tier computing system.

    Parameters follow Wang et al. [2024] Table 1.
    Each device is connected to exactly ONE edge server.
    All devices can reach all cloud servers via Internet.
    """

    def __init__(self, D: int, E: int, V: int, rng: np.random.Generator):
        self.D = D
        self.E = E
        self.V = V
        self.nodes: List[Node] = []
        self._generate(rng)

    def _generate(self, rng: np.random.Generator):
        """Generate nodes with parameters from Wang et al. [2024] Table 1."""
        # ── Devices ──────────────────────────────────────────────────────────
        for i in range(self.D):
            cores = int(rng.integers(2, 9))                       # 2-8 cores
            speed = rng.uniform(1.8, 2.5)                         # GHz/core
            self.nodes.append(Node(
                idx=i,
                node_type='device',
                capacity_ghz=cores * speed,
                transfer_mbps=0.0,       # devices don't receive; they send
                power_compute=0.5,       # p_i^c  W/(GHz·s)  (Wang 2024)
                power_transmit=0.1,      # p_i^r  W/MB        (Wang 2024)
            ))

        # ── Edge servers ─────────────────────────────────────────────────────
        for i in range(self.E):
            cores = int(rng.integers(4, 33))                      # 4-32 cores
            speed = rng.uniform(1.8, 3.0)                         # GHz/core
            rate = max(1.0, rng.normal(100.0, 20.0))              # 100±20 Mbps
            self.nodes.append(Node(
                idx=self.D + i,
                node_type='edge',
                capacity_ghz=cores * speed,
                transfer_mbps=rate,
                power_max=200.0,    # P_max (Katal et al. 2023)
                power_idle=100.0,   # P_idle
            ))

        # ── Cloud servers ─────────────────────────────────────────────────────
        for i in range(self.V):
            cores = int(rng.integers(1, 9))                       # 1-8 cores
            speed = rng.uniform(1.8, 3.0)                         # GHz/core
            rate = max(0.1, rng.normal(10.0, 2.0))               # 10±2 Mbps
            price = rng.uniform(0.01, 0.05)                       # h_i $/s
            self.nodes.append(Node(
                idx=self.D + self.E + i,
                node_type='cloud',
                capacity_ghz=cores * speed,
                transfer_mbps=rate,
                price_per_sec=price,
            ))

        # ── ES-device connectivity (a_{i,j}) ─────────────────────────────────
        # Each device is assigned to exactly one ES (Wang et al. 2024).
        # Additionally, all devices in a geographic cluster share one ES.
        # Simple model: devices assigned round-robin to ESs.
        self.device_to_es = np.zeros(self.D, dtype=int)
        for d in range(self.D):
            self.device_to_es[d] = d % self.E           # ES local index

        self.connectivity = Connectivity(
            es_covers=[
                {d for d in range(self.D) if d % self.E == e}
                for e in range(self.E)
            ]
        )

    # ── Accessors ──────────────────────────────────────────────────────────────

    def get_node(self, global_idx: int) -> Node:
        return self.nodes[global_idx]

    def is_device(self, idx: int) -> bool:
        return idx < self.D

    def is_edge(self, idx: int) -> bool:
        return self.D <= idx < self.D + self.E

    def is_cloud(self, idx: int) -> bool:
        return self.D + self.E <= idx < self.D + self.E + self.V

    def es_local_idx(self, global_idx: int) -> int:
        """Convert global ES index to local ES index (0-based among ESs)."""
        assert self.is_edge(global_idx)
        return global_idx - self.D

    def feasible_nodes_for_task(self, task_device: int) -> list:
        """
        Return list of global node indices that can legally process a task
        launched by device `task_device`.

        Constraints (paper Eqs. 4-5):
          - Local execution: only device `task_device`
          - Edge offload:    only ESs covering `task_device`
          - Cloud offload:   all cloud servers
        """
        feasible = [task_device]   # local device
        # edge servers that cover this device
        es_local = self.device_to_es[task_device]
        feasible.append(self.D + es_local)
        # all cloud servers
        feasible.extend(range(self.D + self.E, self.D + self.E + self.V))
        return feasible

    @property
    def total_nodes(self) -> int:
        return self.D + self.E + self.V

    def __repr__(self) -> str:
        return (f"DE3CSystem(D={self.D}, E={self.E}, V={self.V}, "
                f"total_nodes={self.total_nodes})")
