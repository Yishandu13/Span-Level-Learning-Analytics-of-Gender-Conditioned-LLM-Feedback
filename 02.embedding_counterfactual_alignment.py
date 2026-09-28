#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Embedding representation and counterfactual span alignment.

This module provides reusable utilities for:

1. representing frozen spans with an embedding model;
2. exact-text deduplication for API efficiency;
3. retaining a stable span_id -> embedding_index map;
4. constructing cosine-similarity matrices;
5. partial one-to-one Hungarian matching;
6. threshold calibration from human-reviewed matches.

No local paths, model names, experimental conditions, identity tokens,
or fixed similarity thresholds are hard-coded.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Sequence
import time

import numpy as np
import pandas as pd

from scipy.optimize import (
    linear_sum_assignment,
)



# Embedding representation


def clean_embedding_text(
    text: Any,
) -> str:

    return " ".join(
        str(text)
        .replace("\r", " ")
        .replace("\n", " ")
        .split()
    )


def batches(
    indices: Sequence[int],
    batch_size: int,
):

    for start in range(
        0,
        len(indices),
        batch_size,
    ):

        yield indices[
            start:
            start + batch_size
        ]


def embed_texts_openai(
    texts: Sequence[str],
    *,
    model: str = "text-embedding-3-large",
    dimensions: int | None = 3072,
    batch_size: int = 128,
    max_retries: int = 8,
    client=None,
) -> np.ndarray:
    """
    Embed a sequence of texts using the OpenAI Embeddings API.

    Authentication should be supplied through the environment rather
    than stored in source code.
    """

    if client is None:

        try:
            from openai import OpenAI

        except ImportError as exc:

            raise ImportError(
                "Install the OpenAI SDK: "
                "pip install openai"
            ) from exc

        client = OpenAI()

    clean_texts = [
        clean_embedding_text(text)
        for text in texts
    ]

    if any(
        text == ""
        for text in clean_texts
    ):
        raise ValueError(
            "Empty embedding text detected."
        )

    outputs = []

    ids = list(
        range(len(clean_texts))
    )

    for batch_ids in batches(
        ids,
        batch_size,
    ):

        payload = [
            clean_texts[i]
            for i in batch_ids
        ]

        last_error = None

        for attempt in range(
            max_retries + 1
        ):

            try:

                kwargs = {
                    "model": model,
                    "input": payload,
                }

                if dimensions is not None:
                    kwargs[
                        "dimensions"
                    ] = dimensions

                response = (
                    client
                    .embeddings
                    .create(**kwargs)
                )

                array = np.asarray(
                    [
                        item.embedding
                        for item
                        in response.data
                    ],
                    dtype=np.float32,
                )

                outputs.append(array)

                last_error = None

                break

            except Exception as exc:

                last_error = exc

                if attempt >= max_retries:
                    break

                time.sleep(
                    min(
                        60,
                        2 ** attempt,
                    )
                )

        if last_error is not None:
            raise last_error

    if not outputs:

        return np.empty(
            (
                0,
                dimensions or 0,
            ),
            dtype=np.float32,
        )

    return np.vstack(outputs)


def embed_span_table(
    spans: pd.DataFrame,
    *,
    span_id_col: str = "span_id",
    text_col: str = "span_text",
    model: str = "text-embedding-3-large",
    dimensions: int | None = 3072,
    batch_size: int = 128,
    max_retries: int = 8,
    text_transform: Callable[
        [str],
        str
    ] | None = None,
    client=None,
):
    """
    Embed frozen spans with exact-text deduplication.

    text_transform may optionally be used for a sensitivity representation,
    such as masking selected identity tokens.
    """

    required = {
        span_id_col,
        text_col,
    }

    missing = (
        required
        - set(spans.columns)
    )

    if missing:
        raise KeyError(
            f"Missing columns: {sorted(missing)}"
        )

    if (
        spans[span_id_col]
        .isna()
        .any()
        or
        spans[span_id_col]
        .duplicated()
        .any()
    ):
        raise ValueError(
            "span IDs must be unique "
            "and non-missing."
        )

    transform = (
        text_transform
        if text_transform is not None
        else lambda x: x
    )

    embedding_text = (
        spans[text_col]
        .astype(str)
        .map(transform)
        .map(clean_embedding_text)
    )

    codes, unique_values = (
        pd.factorize(
            embedding_text,
            sort=False,
        )
    )

    unique_texts = [
        str(value)
        for value in unique_values
    ]

    embeddings = embed_texts_openai(
        unique_texts,
        model=model,
        dimensions=dimensions,
        batch_size=batch_size,
        max_retries=max_retries,
        client=client,
    )

    embedding_index = pd.DataFrame(
        {
            "span_id":
                spans[
                    span_id_col
                ]
                .astype(str)
                .to_numpy(),

            "embedding_index":
                codes.astype(
                    np.int64
                ),
        }
    )

    return (
        embeddings,
        embedding_index,
    )



# Similarity


def l2_normalise(
    matrix: np.ndarray,
) -> np.ndarray:

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


def cosine_similarity_matrix(
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:

    return (
        l2_normalise(left)
        @
        l2_normalise(right).T
    )



# Partial Hungarian matching


def partial_hungarian(
    similarity: np.ndarray,
    threshold: float,
):
    """
    Globally optimal partial one-to-one matching.

    Benefit of a real-real edge:

        similarity - threshold

    scipy minimises cost, therefore:

        real-real cost = threshold - similarity
        unmatched cost = 0

    Real endpoints may therefore match dedicated dummy nodes whenever
    no sufficiently beneficial real-real assignment is available.
    """

    similarity = np.asarray(
        similarity,
        dtype=np.float64,
    )

    m, n = similarity.shape

    if m == 0 and n == 0:
        return [], [], []

    if m == 0:
        return [], [], list(range(n))

    if n == 0:
        return [], list(range(m)), []

    N = m + n

    BIG = 1e6

    cost = np.full(
        (N, N),
        BIG,
        dtype=np.float64,
    )

    cost[:m, :n] = (
        threshold
        - similarity
    )

    # Dedicated dummy columns
    for i in range(m):
        cost[
            i,
            n + i
        ] = 0

    # Dedicated dummy rows
    for j in range(n):
        cost[
            m + j,
            j
        ] = 0

    # Dummy-dummy
    cost[
        m:,
        n:
    ] = 0

    row_ind, col_ind = (
        linear_sum_assignment(
            cost
        )
    )

    matches = []

    matched_left = set()
    matched_right = set()

    for row, col in zip(
        row_ind,
        col_ind,
    ):

        if (
            row < m
            and
            col < n
            and
            similarity[row, col]
            >= threshold - 1e-12
        ):

            matches.append(
                (
                    int(row),
                    int(col),
                )
            )

            matched_left.add(
                int(row)
            )

            matched_right.add(
                int(col)
            )

    left_unmatched = [
        i
        for i in range(m)
        if i not in matched_left
    ]

    right_unmatched = [
        j
        for j in range(n)
        if j not in matched_right
    ]

    return (
        matches,
        left_unmatched,
        right_unmatched,
    )



# Embedding lookup


def build_embedding_lookup(
    embeddings: np.ndarray,
    embedding_index: pd.DataFrame,
):

    required = {
        "span_id",
        "embedding_index",
    }

    missing = (
        required
        - set(
            embedding_index.columns
        )
    )

    if missing:
        raise KeyError(
            f"Missing embedding-index "
            f"columns: {sorted(missing)}"
        )

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



# Counterfactual alignment


def align_counterfactual_table(
    spans: pd.DataFrame,
    contrasts: pd.DataFrame,
    embeddings: np.ndarray,
    embedding_index: pd.DataFrame,
    *,
    unit_col: str,
    condition_col: str,
    contrast_unit_col: str,
    reference_condition_col: str,
    comparison_condition_col: str,
    threshold: float,
    span_id_col: str = "span_id",
    order_col: str = "span_index",
    contrast_label_col: str | None = "contrast",
    paired_metadata_cols: Sequence[str] = (),
):
    """
    Align the two conditions of each counterfactual unit.

    The contrast table should specify:

        analytical unit
        reference condition
        comparison condition

    No particular experimental labels are assumed.
    """

    lookup = build_embedding_lookup(
        embeddings,
        embedding_index,
    )

    records = []

    contrasts = (
        contrasts
        .reset_index(drop=True)
    )

    for contrast_i, contrast_row in (
        contrasts.iterrows()
    ):

        unit = contrast_row[
            contrast_unit_col
        ]

        ref_condition = contrast_row[
            reference_condition_col
        ]

        cmp_condition = contrast_row[
            comparison_condition_col
        ]

        ref = spans[
            (
                spans[unit_col]
                == unit
            )
            &
            (
                spans[condition_col]
                == ref_condition
            )
        ].copy()

        cmp = spans[
            (
                spans[unit_col]
                == unit
            )
            &
            (
                spans[condition_col]
                == cmp_condition
            )
        ].copy()

        if order_col in ref:
            ref = ref.sort_values(
                order_col
            )

        if order_col in cmp:
            cmp = cmp.sort_values(
                order_col
            )

        ref = ref.reset_index(
            drop=True
        )

        cmp = cmp.reset_index(
            drop=True
        )

        if len(ref):

            X = np.vstack([
                lookup[str(span_id)]
                for span_id
                in ref[span_id_col]
            ])

        else:

            X = np.empty(
                (
                    0,
                    embeddings.shape[1],
                )
            )

        if len(cmp):

            Y = np.vstack([
                lookup[str(span_id)]
                for span_id
                in cmp[span_id_col]
            ])

        else:

            Y = np.empty(
                (
                    0,
                    embeddings.shape[1],
                )
            )

        similarity = (
            cosine_similarity_matrix(
                X,
                Y,
            )
            if len(ref) and len(cmp)
            else np.empty(
                (
                    len(ref),
                    len(cmp),
                )
            )
        )

        (
            matches,
            ref_unmatched,
            cmp_unmatched,
        ) = partial_hungarian(
            similarity,
            threshold,
        )

        if (
            contrast_label_col
            and
            contrast_label_col
            in contrast_row.index
        ):

            contrast_label = (
                contrast_row[
                    contrast_label_col
                ]
            )

        else:

            contrast_label = (
                f"{ref_condition}"
                f"_vs_"
                f"{cmp_condition}"
            )

        for match_i, (
            i,
            j,
        ) in enumerate(
            matches,
            start=1,
        ):

            record = {

                "alignment_id":
                    (
                        f"{contrast_i}"
                        f"__m{match_i:03d}"
                    ),

                "counterfactual_unit_id":
                    unit,

                "reference_condition":
                    ref_condition,

                "comparison_condition":
                    cmp_condition,

                "contrast":
                    contrast_label,

                "alignment_status":
                    "matched",

                "match_threshold":
                    float(threshold),

                "span_id_ref":
                    str(
                        ref.loc[
                            i,
                            span_id_col,
                        ]
                    ),

                "span_id_cmp":
                    str(
                        cmp.loc[
                            j,
                            span_id_col,
                        ]
                    ),

                "cosine_similarity":
                    float(
                        similarity[i, j]
                    ),

                "cosine_distance":
                    float(
                        1
                        - similarity[i, j]
                    ),
            }

            for col in (
                paired_metadata_cols
            ):

                if col in ref.columns:

                    record[
                        f"{col}_ref"
                    ] = ref.loc[
                        i,
                        col,
                    ]

                if col in cmp.columns:

                    record[
                        f"{col}_cmp"
                    ] = cmp.loc[
                        j,
                        col,
                    ]

            records.append(record)

        for i in ref_unmatched:

            records.append(
                {
                    "counterfactual_unit_id":
                        unit,

                    "reference_condition":
                        ref_condition,

                    "comparison_condition":
                        cmp_condition,

                    "contrast":
                        contrast_label,

                    "alignment_status":
                        "ref_only",

                    "match_threshold":
                        float(threshold),

                    "span_id_ref":
                        str(
                            ref.loc[
                                i,
                                span_id_col,
                            ]
                        ),

                    "span_id_cmp":
                        np.nan,
                }
            )

        for j in cmp_unmatched:

            records.append(
                {
                    "counterfactual_unit_id":
                        unit,

                    "reference_condition":
                        ref_condition,

                    "comparison_condition":
                        cmp_condition,

                    "contrast":
                        contrast_label,

                    "alignment_status":
                        "cmp_only",

                    "match_threshold":
                        float(threshold),

                    "span_id_ref":
                        np.nan,

                    "span_id_cmp":
                        str(
                            cmp.loc[
                                j,
                                span_id_col,
                            ]
                        ),
                }
            )

    return pd.DataFrame(
        records
    )



# Human threshold calibration


def calibrate_threshold(
    reviewed_pairs: pd.DataFrame,
    thresholds: Sequence[float],
    *,
    similarity_col: str = "cosine_similarity",
    human_label_col: str = "human_should_match",
):
    """
    Evaluate candidate similarity thresholds against human judgements.

    Ranking:
        1. F1
        2. precision
        3. recall
        4. higher threshold
    """

    similarity = pd.to_numeric(
        reviewed_pairs[
            similarity_col
        ],
        errors="coerce",
    )

    target = pd.to_numeric(
        reviewed_pairs[
            human_label_col
        ],
        errors="coerce",
    )

    valid = (
        similarity.notna()
        &
        target.isin([0, 1])
    )

    similarity = (
        similarity[valid]
        .to_numpy(float)
    )

    target = (
        target[valid]
        .to_numpy(int)
    )

    results = []

    for threshold in sorted(
        set(
            float(t)
            for t in thresholds
        )
    ):

        prediction = (
            similarity
            >= threshold
        ).astype(int)

        TP = int(
            (
                (prediction == 1)
                &
                (target == 1)
            ).sum()
        )

        FP = int(
            (
                (prediction == 1)
                &
                (target == 0)
            ).sum()
        )

        FN = int(
            (
                (prediction == 0)
                &
                (target == 1)
            ).sum()
        )

        TN = int(
            (
                (prediction == 0)
                &
                (target == 0)
            ).sum()
        )

        precision = (
            TP / (TP + FP)
            if TP + FP
            else np.nan
        )

        recall = (
            TP / (TP + FN)
            if TP + FN
            else np.nan
        )

        if (
            np.isfinite(precision)
            and
            np.isfinite(recall)
            and
            precision + recall > 0
        ):

            f1 = (
                2
                * precision
                * recall
                /
                (
                    precision
                    + recall
                )
            )

        else:

            f1 = np.nan

        results.append(
            {
                "threshold":
                    threshold,

                "TP": TP,
                "FP": FP,
                "FN": FN,
                "TN": TN,

                "precision":
                    precision,

                "recall":
                    recall,

                "f1":
                    f1,
            }
        )

    output = pd.DataFrame(
        results
    )

    ranking = output.sort_values(
        [
            "f1",
            "precision",
            "recall",
            "threshold",
        ],
        ascending=[
            False,
            False,
            False,
            False,
        ],
        na_position="last",
    )

    output["selected"] = False

    if len(ranking):

        best = ranking.iloc[0][
            "threshold"
        ]

        output.loc[
            output[
                "threshold"
            ].eq(best),
            "selected",
        ] = True

    return output
