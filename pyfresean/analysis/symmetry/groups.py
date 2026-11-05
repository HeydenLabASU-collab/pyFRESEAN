"""Validate symmetry copy groups and map motif slots to atomgroup indices."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence, Union

import numpy as np

if TYPE_CHECKING:
    from MDAnalysis.core.groups import AtomGroup


GroupInput = Union["AtomGroup", Sequence[int], np.ndarray]


def symmetry_order_from_label(label: str) -> int:
    """Map ``C3`` / ``C6`` to cyclic order."""
    key = label.strip().upper()
    mapping = {"C3": 3, "C6": 6}
    if key not in mapping:
        raise ValueError(
            f"symmetry must be one of {sorted(mapping)}, got {label!r}"
        )
    return mapping[key]


def _group_indices(group: GroupInput, label: str) -> np.ndarray:
    if hasattr(group, "indices"):
        return np.asarray(group.indices, dtype=np.intp)
    return np.asarray(group, dtype=np.intp)


def resolve_partner_map(
    atomgroup: "AtomGroup",
    symmetry_groups: Sequence[GroupInput],
    order: int,
) -> np.ndarray:
    """
    Build ``partner_map`` with shape ``(m, order)``.

    Row ``k`` lists ``atomgroup`` indices (0 … n_atoms-1) for motif slot ``k``
    on copies ``0 … order-1``.
    """
    if len(symmetry_groups) != order:
        raise ValueError(
            f"symmetry {order} requires {order} groups, got {len(symmetry_groups)}"
        )

    sizes = [len(_group_indices(g, "group")) for g in symmetry_groups]
    if not sizes or not all(s == sizes[0] for s in sizes):
        raise ValueError(
            "symmetry_groups must have equal length; got sizes "
            f"{sizes}"
        )
    m = sizes[0]

    ag_ix_to_pos = {
        int(atom.ix): i for i, atom in enumerate(atomgroup.atoms)
    }
    n_atoms = atomgroup.n_atoms
    partner_map = np.empty((m, order), dtype=np.intp)

    seen: set[int] = set()
    for copy_j, group in enumerate(symmetry_groups):
        indices = _group_indices(group, f"symmetry_groups[{copy_j}]")
        if indices.size != m:
            raise ValueError("symmetry group length mismatch")
        for k, uix in enumerate(indices):
            pos = ag_ix_to_pos.get(int(uix))
            if pos is None:
                raise ValueError(
                    f"atom index {uix} in symmetry_groups[{copy_j}] "
                    f"is not in the FRESEAN selection ({n_atoms} atoms)"
                )
            if pos in seen:
                raise ValueError(
                    f"atom position {pos} appears in more than one symmetry group"
                )
            partner_map[k, copy_j] = pos
            seen.add(pos)

    expected = set(range(n_atoms))
    if seen != expected:
        missing = expected - seen
        extra = seen - expected
        raise ValueError(
            "symmetry_groups must partition the analysis atomgroup exactly; "
            f"missing positions {sorted(missing)[:8]}…, "
            f"extra {sorted(extra)[:8]}…"
        )

    return partner_map
