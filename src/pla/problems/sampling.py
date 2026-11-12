"""Collocation points: scrambled Sobol points in a box and on its boundary."""
from __future__ import annotations

import torch
from torch import Tensor


def sobol(n: int, dim: int, seed: int) -> Tensor:
    """``n`` scrambled Sobol points in the unit cube ``[0, 1]^dim`` (double precision)."""
    engine = torch.quasirandom.SobolEngine(dim, scramble=True, seed=seed)
    return engine.draw(n, dtype=torch.float64)


def box_boundary(n: int, dim: int, seed: int) -> tuple[Tensor, Tensor]:
    """``n`` points on the boundary of the unit cube and their outward unit normals.

    The points are divided evenly among the ``2 * dim`` faces, and every face
    receives its own Sobol sequence (seed ``seed + face``). In one dimension
    the two faces are the end points of the interval.
    """
    n_faces = 2 * dim
    points, normals = [], []
    for face in range(n_faces):
        count = n // n_faces + (1 if face < n % n_faces else 0)
        axis, side = face // 2, float(face % 2)
        block = torch.zeros(count, dim, dtype=torch.float64)
        if dim > 1:
            others = [c for c in range(dim) if c != axis]
            block[:, others] = sobol(count, dim - 1, seed + face) if count > 0 else block[:, others]
        block[:, axis] = side
        normal = torch.zeros(count, dim, dtype=torch.float64)
        normal[:, axis] = 2.0 * side - 1.0
        points.append(block)
        normals.append(normal)
    return torch.cat(points), torch.cat(normals)
