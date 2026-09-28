#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Leave-one-out attribution with permutation-based localisation.

For each counterfactual unit with n matched span pairs:

    X = {x_1, ..., x_n}
    Y = {y_1, ..., y_n}

Full cross-condition separation:

    D_full = mean_{a,b} d(x_a, y_b)

Leave-one-pair-out separation:

    D_-i

Observed local influence:

    I_i = D_full - D_-i

Positive I_i indicates that removing the matched pair reduces the
cross-condition separation.

Permutation calibration preserves analytical-unit membership while
breaking the observed local correspondence.

No model names, cue labels, file paths, or candidate p thresholds are
hard-coded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd



# Multiple-testing utility


def bh_adjust(
    p_values,
) -> np.ndarray:

    p = np.asarray(
        p_values,
        dtype=float,
    )

    output = np.full_like(
        p,
        np.nan,
        dtype=float,
    )

    valid = np.isfinite(p)

    if not valid.any():
        return output

    values = p[valid]

    m = len(values)

    order = np.argsort(
        values
    )

    ranked = values[order]

    adjusted = (
        ranked
        * m
        /
        np.arange(
            1,
            m + 1,
            dtype=float,
        )
    )

    adjusted = (
        np.minimum.accumulate(
            adjusted[::-1]
        )[::-1]
    )

    adjusted = np.clip(
        adjusted,
        0,
        1,
    )

    restored = np.empty(
        m,
        dtype=float,
    )

    restored[order] = adjusted

    output[valid] = restored

    return output


def empirical_p(
    null_sorted: np.ndarray,
    observed: np.ndarray,
):
    """
    One-sided high-tail empirical p with +1 correction.
    """

    null_sorted = np.asarray(
        null_sorted,
        dtype=float,
    )

    observed = np.asarray(
        observed,
        dtype=float,
    )

    positions = np.searchsorted(
        null_sorted,
        observed,
        side="left",
    )

    count_ge = (
        len(null_sorted)
        - positions
    )

    return (
        1
        + count_ge
    ) / (
        len(null_sorted)
        + 1
    )



# Embeddings and distance matrices


def build_embedding_lookup(
    embeddings: np.ndarray,
    embedding_index: pd.DataFrame,
):

    return {
        str(span_id):
            embeddings[int(index)]

        for span_id, index in zip(
            embedding_index[
                "span_id"
            ],
            embedding_index[
                "embedding_index"
            ],
        )
    }


def normalise_rows(
    matrix: np.ndarray,
):

    matrix = np.asarray(
        matrix,
        dtype=np.float64,
    )

    norms = np.linalg.norm(
        matrix,
        axis=1,
        keepdims=True,
    )

    norms[norms == 0] = 1

    return matrix / norms


def cross_distance_matrix(
    X: np.ndarray,
    Y: np.ndarray,
    *,
    metric: str = "cosine",
):

    X = np.asarray(
        X,
        dtype=np.float64,
    )

    Y = np.asarray(
        Y,
        dtype=np.float64,
    )

    if metric == "cosine":

        return (
            1
            -
            normalise_rows(X)
            @
            normalise_rows(Y).T
        )

    if metric == "euclidean":

        x2 = np.sum(
            X * X,
            axis=1,
        )[:, None]

        y2 = np.sum(
            Y * Y,
            axis=1,
        )[None, :]

        squared = np.maximum(
            x2
            + y2
            - 2 * X @ Y.T,
            0,
        )

        return np.sqrt(
            squared
        )

    raise ValueError(
        "metric must be "
        "'cosine' or 'euclidean'"
    )



# LOO influence


@dataclass
class UnitLOO:

    unit_id: Any

    row_indices: np.ndarray

    distance_matrix: np.ndarray

    full_distance: float

    influence_matrix: np.ndarray

    @property
    def n(self):

        return int(
            self.distance_matrix
            .shape[0]
        )


def unit_loo(
    *,
    unit_id,
    row_indices,
    distance_matrix,
):
    """
    Construct I(i,j) for every possible ref/cmp removal pair.

    The observed Hungarian correspondence is represented by the diagonal.
    """

    D = np.asarray(
        distance_matrix,
        dtype=np.float64,
    )

    n, m = D.shape

    if n != m:

        raise ValueError(
            "LOO requires equal "
            "matched-set sizes."
        )

    if n < 2:

        raise ValueError(
            "At least two matched "
            "pairs are required."
        )

    total = D.sum()

    full_distance = (
        total
        /
        (n * n)
    )

    row_sum = D.sum(
        axis=1
    )

    col_sum = D.sum(
        axis=0
    )

    # Removing ref i and cmp j
    remaining_total = (
        total
        -
        row_sum[:, None]
        -
        col_sum[None, :]
        +
        D
    )

    D_minus = (
        remaining_total
        /
        ((n - 1) ** 2)
    )

    influence = (
        full_distance
        -
        D_minus
    )

    return UnitLOO(

        unit_id=unit_id,

        row_indices=np.asarray(
            row_indices,
            dtype=int,
        ),

        distance_matrix=D,

        full_distance=float(
            full_distance
        ),

        influence_matrix=influence,
    )


def build_unit_objects(
    matched: pd.DataFrame,
    embedding_lookup: Mapping[
        str,
        np.ndarray,
    ],
    *,
    unit_col: str,
    ref_col: str = "span_id_ref",
    cmp_col: str = "span_id_cmp",
    metric: str = "cosine",
    min_matched: int = 2,
):

    output = {}

    for unit, group in matched.groupby(
        unit_col,
        sort=False,
        dropna=False,
    ):

        if len(group) < min_matched:
            continue

        X = np.vstack([
            embedding_lookup[str(x)]
            for x in group[ref_col]
        ])

        Y = np.vstack([
            embedding_lookup[str(x)]
            for x in group[cmp_col]
        ])

        distances = (
            cross_distance_matrix(
                X,
                Y,
                metric=metric,
            )
        )

        output[unit] = unit_loo(

            unit_id=unit,

            row_indices=(
                group.index
                .to_numpy()
            ),

            distance_matrix=distances,
        )

    return output



# Permutation null


def draw_permutation_null(
    units: Sequence[UnitLOO],
    *,
    B: int = 5000,
    seed: int = 20260916,
):
    """
    Structure-preserving pooled null.

    For each Monte-Carlo draw:

    1. sample a unit proportional to n matched spans;
    2. permute comparison positions within that unit;
    3. sample one reference position;
    4. record I(i, pi(i)).
    """

    if B < 1:
        raise ValueError(
            "B must be positive."
        )

    if not units:

        return np.array(
            [],
            dtype=float,
        )

    rng = np.random.default_rng(
        seed
    )

    weights = np.asarray(
        [
            unit.n
            for unit in units
        ],
        dtype=float,
    )

    weights /= weights.sum()

    null = np.empty(
        B,
        dtype=float,
    )

    for b in range(B):

        unit = units[
            int(
                rng.choice(
                    len(units),
                    p=weights,
                )
            )
        ]

        permutation = rng.permutation(
            unit.n
        )

        i = int(
            rng.integers(
                0,
                unit.n,
            )
        )

        j = int(
            permutation[i]
        )

        null[b] = (
            unit
            .influence_matrix[
                i,
                j,
            ]
        )

    return null



# Main localisation API


def localise_aligned_pairs(
    alignment: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_index: pd.DataFrame,
    *,
    unit_col: str,
    cell_cols: Sequence[str] = (),
    B: int = 5000,
    seed: int = 20260916,
    metric: str = "cosine",
    status_col: str = "alignment_status",
    ref_col: str = "span_id_ref",
    cmp_col: str = "span_id_cmp",
    min_matched: int = 2,
):
    """
    Calculate LOO attribution and permutation calibration.

    cell_cols may, for example, define independent null distributions
    by model × cue type × contrast, but the library imposes no specific
    experimental structure.
    """

    if status_col in alignment:

        matched = alignment[
            alignment[
                status_col
            ].eq("matched")
        ].copy()

    else:

        matched = alignment.copy()

    matched = matched[
        matched[ref_col].notna()
        &
        matched[cmp_col].notna()
    ].reset_index(
        drop=True
    )

    lookup = build_embedding_lookup(
        embeddings,
        embedding_index,
    )

    output = matched.copy()

    prefix = metric

    output[
        f"loo_status_{prefix}"
    ] = "insufficient_matches"

    output[
        f"unit_full_cross_{prefix}_distance"
    ] = np.nan

    output[
        f"observed_{prefix}_loo_influence"
    ] = np.nan

    output[
        f"perm_null_mean_{prefix}"
    ] = np.nan

    output[
        f"perm_null_sd_{prefix}"
    ] = np.nan

    output[
        f"p_perm_{prefix}"
    ] = np.nan

    output[
        f"z_perm_{prefix}"
    ] = np.nan

    if cell_cols:

        cells = output.groupby(
            list(cell_cols),
            sort=False,
            dropna=False,
        )

    else:

        cells = [((), output)]

    null_tables = []

    for cell_i, (
        cell_key,
        cell_data,
    ) in enumerate(cells):

        if not isinstance(
            cell_key,
            tuple,
        ):
            cell_key = (
                cell_key,
            )

        unit_objects = (
            build_unit_objects(
                cell_data,
                lookup,
                unit_col=unit_col,
                ref_col=ref_col,
                cmp_col=cmp_col,
                metric=metric,
                min_matched=min_matched,
            )
        )

        if not unit_objects:
            continue

        observed_rows = []
        observed_values = []

        for unit in (
            unit_objects.values()
        ):

            observed = np.diag(
                unit.influence_matrix
            )

            for local_i, row_i in (
                enumerate(
                    unit.row_indices
                )
            ):

                output.loc[
                    row_i,
                    f"loo_status_{prefix}",
                ] = "analyzable"

                output.loc[
                    row_i,
                    f"unit_full_cross_{prefix}_distance",
                ] = unit.full_distance

                output.loc[
                    row_i,
                    f"observed_{prefix}_loo_influence",
                ] = observed[local_i]

                observed_rows.append(
                    int(row_i)
                )

                observed_values.append(
                    float(
                        observed[
                            local_i
                        ]
                    )
                )

        null = draw_permutation_null(

            list(
                unit_objects.values()
            ),

            B=B,

            seed=(
                seed
                +
                cell_i
                * 100003
            ),
        )

        null_sorted = np.sort(
            null
        )

        null_mean = float(
            np.mean(null)
        )

        null_sd = float(
            np.std(
                null,
                ddof=1,
            )
        )

        observed_values = np.asarray(
            observed_values,
            dtype=float,
        )

        p_values = empirical_p(
            null_sorted,
            observed_values,
        )

        if (
            np.isfinite(null_sd)
            and
            null_sd > 0
        ):

            z_values = (
                observed_values
                -
                null_mean
            ) / null_sd

        else:

            z_values = np.full(
                len(observed_values),
                np.nan,
            )

        for (
            row_i,
            p_value,
            z_value,
        ) in zip(
            observed_rows,
            p_values,
            z_values,
        ):

            output.loc[
                row_i,
                f"perm_null_mean_{prefix}",
            ] = null_mean

            output.loc[
                row_i,
                f"perm_null_sd_{prefix}",
            ] = null_sd

            output.loc[
                row_i,
                f"p_perm_{prefix}",
            ] = p_value

            output.loc[
                row_i,
                f"z_perm_{prefix}",
            ] = z_value

        null_table = pd.DataFrame(
            {
                "draw":
                    np.arange(
                        1,
                        B + 1,
                    ),

                f"null_{prefix}_loo_influence":
                    null,
            }
        )

        for col, value in zip(
            cell_cols,
            cell_key,
        ):

            null_table[col] = value

        null_tables.append(
            null_table
        )

    p_col = (
        f"p_perm_{prefix}"
    )

    output[
        f"q_bh_global_{prefix}"
    ] = bh_adjust(
        pd.to_numeric(
            output[p_col],
            errors="coerce",
        )
    )

    output[
        f"q_bh_cell_{prefix}"
    ] = np.nan

    if cell_cols:

        groups = output.groupby(
            list(cell_cols),
            dropna=False,
        ).groups

        for indices in (
            groups.values()
        ):

            indices = list(
                indices
            )

            output.loc[
                indices,
                f"q_bh_cell_{prefix}",
            ] = bh_adjust(
                pd.to_numeric(
                    output.loc[
                        indices,
                        p_col,
                    ],
                    errors="coerce",
                )
            )

    else:

        output[
            f"q_bh_cell_{prefix}"
        ] = output[
            f"q_bh_global_{prefix}"
        ]

    if null_tables:

        null_output = pd.concat(
            null_tables,
            ignore_index=True,
        )

    else:

        null_output = (
            pd.DataFrame()
        )

    return (
        output,
        null_output,
    )


def select_localised_pairs(
    localisation: pd.DataFrame,
    *,
    p_col: str,
    threshold: float,
):
    """
    Convenience function for constructing a user-defined localised /
    high-influence candidate subset.
    """

    p = pd.to_numeric(
        localisation[p_col],
        errors="coerce",
    )

    return localisation[
        p.notna()
        &
        (p < threshold)
    ].copy()
