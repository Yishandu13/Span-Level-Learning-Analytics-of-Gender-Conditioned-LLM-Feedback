#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Statistical and pedagogical characterisation of localised pairs


The input is an already-localised matched counterfactual span-pair table.

Statistical design
------------------
Each pedagogical / linguistic annotation category is represented as a
binary feature-presence indicator.

- Native binary variables remain 0/1.
- Nominal labels are converted to category-specific one-vs-rest indicators.
- Paired cross-condition differences are tested using McNemar's test.
- Exact two-sided McNemar is used for sparse discordant cells.
- Asymptotic McNemar without continuity correction is used when the number
  of discordant pairs is sufficiently large.

Reported:
- counts
- prevalence
- percentage-point difference
- discordant counts n10 / n01
- two-sided McNemar p

Also included:
- A-fixed analyses
- A-fixed-by-specific-module analyses
- selected-vs-analyzable enrichment ratios
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from scipy.stats import (
    binomtest,
    chi2
)


@dataclass(frozen=True)
class CharacterisationConfig:

    exact_if_discordant_below: int = 25

    metadata_prefixes: tuple[str, ...] = (
        "A_",
        "B_",
        "C_",
        "D_",
        "E_",
        "I1_",
        "I2_",
        "J_"
    )

    group_cols: tuple[str, ...] = (
        "model",
        "cue_type",
        "contrast"
    )


def clean_binary(
    series: pd.Series
) -> pd.Series:

    values = pd.to_numeric(
        series,
        errors="coerce"
    )

    return values.where(
        values.isin(
            [0, 1]
        )
    )


def is_binary_pair(
    ref: pd.Series,
    cmp: pd.Series
) -> bool:

    values = pd.concat(
        [
            pd.to_numeric(
                ref,
                errors="coerce"
            ),

            pd.to_numeric(
                cmp,
                errors="coerce"
            )
        ]
    ).dropna()

    if len(values) == 0:
        return False

    return set(
        values.unique()
    ).issubset(
        {0, 1}
    )


def mcnemar_test(
    ref: pd.Series,
    cmp: pd.Series,
    *,
    exact_if_discordant_below: int = 25
) -> dict:

    ref = clean_binary(
        ref
    )

    cmp = clean_binary(
        cmp
    )

    valid = (
        ref.notna()
        & cmp.notna()
    )

    ref = (
        ref[
            valid
        ]
        .astype(int)
    )

    cmp = (
        cmp[
            valid
        ]
        .astype(int)
    )

    n00 = int(
        (
            (ref == 0)
            & (cmp == 0)
        ).sum()
    )

    n01 = int(
        (
            (ref == 0)
            & (cmp == 1)
        ).sum()
    )

    n10 = int(
        (
            (ref == 1)
            & (cmp == 0)
        ).sum()
    )

    n11 = int(
        (
            (ref == 1)
            & (cmp == 1)
        ).sum()
    )

    discordant = (
        n01
        + n10
    )

    if discordant == 0:

        p_value = 1.0
        statistic = 0.0
        method = "exact"

    elif (
        discordant
        < exact_if_discordant_below
    ):

        p_value = float(
            binomtest(
                n01,
                n=discordant,
                p=0.5,
                alternative="two-sided"
            ).pvalue
        )

        statistic = np.nan

        method = (
            "exact"
        )

    else:

        statistic = (
            (n01 - n10) ** 2
            / discordant
        )

        p_value = float(
            chi2.sf(
                statistic,
                df=1
            )
        )

        method = (
            "asymptotic_"
            "no_continuity_correction"
        )

    n = len(
        ref
    )

    ref_prevalence = (
        float(
            ref.mean()
        )
        if n
        else np.nan
    )

    cmp_prevalence = (
        float(
            cmp.mean()
        )
        if n
        else np.nan
    )

    return {
        "n_valid_pairs":
            n,

        "ref_count":
            int(
                ref.sum()
            ),

        "cmp_count":
            int(
                cmp.sum()
            ),

        "ref_prevalence":
            ref_prevalence,

        "cmp_prevalence":
            cmp_prevalence,

        "delta_pp_cmp_minus_ref":
            (
                cmp_prevalence
                - ref_prevalence
            ) * 100,

        "n00":
            n00,

        "n01_ref0_cmp1":
            n01,

        "n10_ref1_cmp0":
            n10,

        "n11":
            n11,

        "n_discordant":
            discordant,

        "mcnemar_method":
            method,

        "mcnemar_statistic":
            statistic,

        "mcnemar_p":
            p_value
    }


def paired_feature_bases(
    data: pd.DataFrame,
    *,
    prefixes: Sequence[str]
) -> list[str]:

    ref = {
        column[:-4]

        for column
        in data.columns

        if (
            column.endswith(
                "_ref"
            )
            and
            column.startswith(
                tuple(
                    prefixes
                )
            )
        )
    }

    cmp = {
        column[:-4]

        for column
        in data.columns

        if (
            column.endswith(
                "_cmp"
            )
            and
            column.startswith(
                tuple(
                    prefixes
                )
            )
        )
    }

    return sorted(
        ref & cmp
    )


def feature_indicators(
    group: pd.DataFrame,
    feature: str
):

    ref_col = (
        f"{feature}_ref"
    )

    cmp_col = (
        f"{feature}_cmp"
    )

    ref = group[
        ref_col
    ]

    cmp = group[
        cmp_col
    ]

    if is_binary_pair(
        ref,
        cmp
    ):

        yield {
            "feature":
                feature,

            "level":
                "present",

            "encoding":
                "native_binary",

            "ref":
                clean_binary(
                    ref
                ),

            "cmp":
                clean_binary(
                    cmp
                )
        }

        return

    labels = sorted(
        set(
            ref
            .dropna()
            .astype(str)
        )
        |
        set(
            cmp
            .dropna()
            .astype(str)
        )
    )

    for label in labels:

        ref_indicator = (
            ref
            .astype("string")
            .eq(label)
            .astype(float)
        )

        cmp_indicator = (
            cmp
            .astype("string")
            .eq(label)
            .astype(float)
        )

        ref_indicator[
            ref.isna()
        ] = np.nan

        cmp_indicator[
            cmp.isna()
        ] = np.nan

        yield {
            "feature":
                feature,

            "level":
                label,

            "encoding":
                "one_vs_rest",

            "ref":
                ref_indicator,

            "cmp":
                cmp_indicator
        }


def run_mcnemar_characterisation(
    pairs: pd.DataFrame,
    *,
    config:
        CharacterisationConfig
        = CharacterisationConfig(),

    feature_bases:
        Optional[
            Sequence[str]
        ] = None,

    fixed_module:
        Optional[str]
        = None,

    by_fixed_module:
        bool = False
) -> pd.DataFrame:

    data = pairs.copy()

    group_cols = list(
        config.group_cols
    )

    if feature_bases is None:

        feature_bases = (
            paired_feature_bases(
                data,
                prefixes=
                    config.metadata_prefixes
            )
        )

    if fixed_module is not None:

        ref_module = (
            f"{fixed_module}_ref"
        )

        cmp_module = (
            f"{fixed_module}_cmp"
        )

        data = data[
            data[
                ref_module
            ].notna()
            &
            data[
                cmp_module
            ].notna()
            &
            data[
                ref_module
            ]
            .astype(str)
            .eq(
                data[
                    cmp_module
                ]
                .astype(str)
            )
        ].copy()

        if by_fixed_module:

            data[
                "fixed_module"
            ] = (
                data[
                    ref_module
                ]
                .astype(str)
            )

            group_cols.append(
                "fixed_module"
            )

    rows = []

    for keys, group in (
        data.groupby(
            group_cols,
            dropna=False,
            sort=True
        )
    ):

        if not isinstance(
            keys,
            tuple
        ):
            keys = (
                keys,
            )

        group_metadata = dict(
            zip(
                group_cols,
                keys
            )
        )

        for feature in (
            feature_bases
        ):

            # A itself is fixed and therefore
            # not tested in A-fixed analysis.
            if (
                fixed_module
                is not None
                and feature
                == fixed_module
            ):
                continue

            ref_col = (
                f"{feature}_ref"
            )

            cmp_col = (
                f"{feature}_cmp"
            )

            if (
                ref_col
                not in group.columns
                or cmp_col
                not in group.columns
            ):
                continue

            for indicator in (
                feature_indicators(
                    group,
                    feature
                )
            ):

                stats = mcnemar_test(
                    indicator[
                        "ref"
                    ],

                    indicator[
                        "cmp"
                    ],

                    exact_if_discordant_below=
                        config
                        .exact_if_discordant_below
                )

                rows.append(
                    {
                        **group_metadata,

                        "feature":
                            indicator[
                                "feature"
                            ],

                        "level":
                            indicator[
                                "level"
                            ],

                        "encoding":
                            indicator[
                                "encoding"
                            ],

                        **stats
                    }
                )

    return pd.DataFrame(
        rows
    )


def overall_results(
    selected_pairs: pd.DataFrame,
    *,
    config:
        CharacterisationConfig
        = CharacterisationConfig()
) -> pd.DataFrame:

    return run_mcnemar_characterisation(
        selected_pairs,
        config=config
    )


def a_fixed_results(
    selected_pairs: pd.DataFrame,
    *,
    module_feature: str = "A_module",
    by_module: bool = True,
    config:
        CharacterisationConfig
        = CharacterisationConfig()
) -> pd.DataFrame:

    return run_mcnemar_characterisation(
        selected_pairs,
        config=config,
        fixed_module=
            module_feature,
        by_fixed_module=
            by_module
    )


def add_position_terciles(
    pairs: pd.DataFrame,
    *,
    relative_position_feature:
        str = "relative_position"
) -> pd.DataFrame:

    output = pairs.copy()

    for side in [
        "ref",
        "cmp"
    ]:

        source = (
            f"{relative_position_feature}"
            f"_{side}"
        )

        target = (
            f"position_tercile"
            f"_{side}"
        )

        position = (
            pd.to_numeric(
                output[
                    source
                ],
                errors="coerce"
            )
        )

        output[
            target
        ] = np.select(
            [
                position < 1 / 3,
                position < 2 / 3
            ],
            [
                "Early",
                "Middle"
            ],
            default="Late"
        )

        output.loc[
            position.isna(),
            target
        ] = np.nan

    return output


def endpoint_long(
    pairs: pd.DataFrame,
    feature: str,
    *,
    deduplicate: bool = False
) -> pd.DataFrame:

    frames = []

    for side, condition_col in [
        (
            "ref",
            "reference_condition"
        ),
        (
            "cmp",
            "comparison_condition"
        )
    ]:

        frame = pd.DataFrame(
            {
                "model":
                    pairs[
                        "model"
                    ],

                "cue_type":
                    pairs[
                        "cue_type"
                    ],

                "condition":
                    pairs[
                        condition_col
                    ],

                "span_id":
                    pairs[
                        f"span_id_{side}"
                    ],

                feature:
                    pairs[
                        f"{feature}_{side}"
                    ]
            }
        )

        if (
            "counterfactual_unit_id"
            in pairs.columns
        ):

            frame[
                "counterfactual_unit_id"
            ] = pairs[
                "counterfactual_unit_id"
            ]

        frames.append(
            frame
        )

    output = pd.concat(
        frames,
        ignore_index=True
    )

    if deduplicate:

        keys = [
            key
            for key in [
                "model",
                "cue_type",
                "counterfactual_unit_id",
                "condition",
                "span_id"
            ]
            if key
            in output.columns
        ]

        output = (
            output
            .drop_duplicates(
                keys
            )
        )

    return output


def enrichment_ratio(
    selected_pairs: pd.DataFrame,
    all_analyzable_pairs: pd.DataFrame,
    *,
    feature: str,
    group_cols: Sequence[str] = (
        "model",
        "cue_type"
    ),
    deduplicate_endpoints: bool = False
) -> pd.DataFrame:

    selected = endpoint_long(
        selected_pairs,
        feature,
        deduplicate=
            deduplicate_endpoints
    )

    baseline = endpoint_long(
        all_analyzable_pairs,
        feature,
        deduplicate=
            deduplicate_endpoints
    )

    rows = []

    for keys, selected_group in (
        selected.groupby(
            list(
                group_cols
            ),
            dropna=False,
            sort=True
        )
    ):

        if not isinstance(
            keys,
            tuple
        ):
            keys = (
                keys,
            )

        metadata = dict(
            zip(
                group_cols,
                keys
            )
        )

        baseline_mask = pd.Series(
            True,
            index=baseline.index
        )

        for column, value in (
            metadata.items()
        ):

            baseline_mask &= (
                baseline[
                    column
                ].eq(
                    value
                )
            )

        baseline_group = baseline[
            baseline_mask
        ]

        selected_valid = (
            selected_group[
                selected_group[
                    feature
                ].notna()
            ]
        )

        baseline_valid = (
            baseline_group[
                baseline_group[
                    feature
                ].notna()
            ]
        )

        levels = sorted(
            set(
                selected_valid[
                    feature
                ].astype(str)
            )
            |
            set(
                baseline_valid[
                    feature
                ].astype(str)
            )
        )

        for level in levels:

            selected_n = int(
                selected_valid[
                    feature
                ]
                .astype(str)
                .eq(level)
                .sum()
            )

            baseline_n = int(
                baseline_valid[
                    feature
                ]
                .astype(str)
                .eq(level)
                .sum()
            )

            selected_share = (
                selected_n
                / len(
                    selected_valid
                )
            )

            baseline_share = (
                baseline_n
                / len(
                    baseline_valid
                )
            )

            rows.append(
                {
                    **metadata,

                    "feature":
                        feature,

                    "level":
                        level,

                    "selected_n":
                        selected_n,

                    "selected_share":
                        selected_share,

                    "analyzable_n":
                        baseline_n,

                    "analyzable_share":
                        baseline_share,

                    "localisation_rate":
                        (
                            selected_n
                            / baseline_n
                            if baseline_n
                            else np.nan
                        ),

                    "enrichment_ratio":
                        (
                            selected_share
                            / baseline_share
                            if baseline_share
                            else np.nan
                        )
                }
            )

    return pd.DataFrame(
        rows
    )
