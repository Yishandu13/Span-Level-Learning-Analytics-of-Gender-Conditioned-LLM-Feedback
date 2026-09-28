#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Span segmentation and external meta-annotation utilities.

This module provides reusable tools for:

1. converting response-level feedback into non-overlapping local spans;
2. sentence -> meaningful-clause segmentation;
3. enforcing configurable lexical-token guardrails;
4. assigning stable span identifiers;
5. preserving response/experimental metadata;
6. attaching annotations from an EXTERNAL annotation function.

No study-specific annotation rule bank is included in this file.

The intended workflow is:

    response-level feedback
        -> frozen local spans
        -> external meta-annotation

The span segmentation should be frozen before embedding and alignment.
Annotation rules may subsequently be refined as long as span_id and
span_text remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence
import hashlib
import re

import numpy as np
import pandas as pd


# ============================================================
# Configuration
# ============================================================

@dataclass(frozen=True)
class SegmentationConfig:
    min_tokens: int = 3
    max_tokens: int = 30
    spacy_model: str = "en_core_web_sm"


# ============================================================
# NLP loading
# ============================================================

def load_nlp(model: str = "en_core_web_sm"):
    """
    Load spaCy.

    If the requested dependency model is unavailable, the function falls
    back to a blank English pipeline with a rule-based sentencizer.

    Returns
    -------
    nlp
        spaCy Language object.
    has_parser : bool
        Whether dependency parsing is available.
    """

    try:
        import spacy
    except ImportError as exc:
        raise ImportError(
            "spaCy is required. Install with: pip install spacy"
        ) from exc

    try:
        nlp = spacy.load(model, disable=["ner"])

        if (
            "parser" not in nlp.pipe_names
            and "sentencizer" not in nlp.pipe_names
        ):
            nlp.add_pipe("sentencizer")

        return nlp, "parser" in nlp.pipe_names

    except OSError:

        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")

        return nlp, False


# ============================================================
# Text utilities
# ============================================================

def normalise_text(text: Any) -> str:

    text = "" if text is None else str(text)

    text = (
        text
        .replace("\u00a0", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
    )

    text = (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
    )

    lines = [
        re.sub(r"[ \t]+", " ", line).strip()
        for line in text.split("\n")
    ]

    return re.sub(
        r"\n{3,}",
        "\n\n",
        "\n".join(lines)
    ).strip()


def lexical_token_count(text: str, nlp) -> int:

    doc = nlp.make_doc(str(text))

    return sum(
        1
        for token in doc
        if not token.is_space
        and not token.is_punct
    )


def has_predicate(
    text: str,
    nlp,
    has_parser: bool,
) -> bool:
    """
    Conservative check that a potential clause contains a predicate.
    """

    if has_parser:

        doc = nlp(text)

        return any(
            token.pos_ in {"VERB", "AUX"}
            for token in doc
        )

    # Conservative parser-free fallback
    return bool(
        re.search(
            r"\b("
            r"am|is|are|was|were|be|been|being|"
            r"have|has|had|do|does|did|"
            r"can|could|may|might|must|should|would|will|"
            r"\w+(?:ed|ing)"
            r")\b",
            text,
            flags=re.I,
        )
    )


# ============================================================
# Clause segmentation
# ============================================================

CLAUSE_BOUNDARIES = [

    ("semicolon", re.compile(r";")),

    ("colon", re.compile(
        r":\s+(?=[A-Z])"
    )),

    ("but", re.compile(
        r",?\s+\bbut\b\s+",
        re.I,
    )),

    ("however", re.compile(
        r";?\s*\bhowever\b[, ]+\s*",
        re.I,
    )),

    ("although", re.compile(
        r",?\s+\balthough\b\s+",
        re.I,
    )),

    ("while", re.compile(
        r",?\s+\bwhile\b\s+",
        re.I,
    )),

    ("because", re.compile(
        r",?\s+\bbecause\b\s+",
        re.I,
    )),

    ("so", re.compile(
        r",?\s+\bso\b\s+",
        re.I,
    )),

    ("and", re.compile(
        r",?\s+\band\b\s+",
        re.I,
    )),
]


def split_meaningful_clauses(
    sentence: str,
    nlp,
    has_parser: bool,
    min_tokens: int = 3,
) -> list[str]:
    """
    Conservatively split sentences into non-overlapping clause-like units.
    """

    segments = [sentence.strip()]

    changed = True

    while changed:

        changed = False
        new_segments = []

        for segment in segments:

            split_done = False

            for label, pattern in CLAUSE_BOUNDARIES:

                for match in pattern.finditer(segment):

                    left = (
                        segment[:match.start()]
                        .strip(" ,;:")
                    )

                    right = (
                        segment[match.end():]
                        .strip(" ,;:")
                    )

                    if not left or not right:
                        continue

                    left_n = lexical_token_count(
                        left,
                        nlp,
                    )

                    right_n = lexical_token_count(
                        right,
                        nlp,
                    )

                    if (
                        left_n < min_tokens
                        or right_n < min_tokens
                    ):
                        continue

                    # "and" is treated conservatively
                    if (
                        label == "and"
                        and min(left_n, right_n) < 4
                    ):
                        continue

                    if not (
                        has_predicate(
                            left,
                            nlp,
                            has_parser,
                        )
                        and
                        has_predicate(
                            right,
                            nlp,
                            has_parser,
                        )
                    ):
                        continue

                    new_segments.extend([
                        left,
                        right,
                    ])

                    split_done = True
                    changed = True

                    break

                if split_done:
                    break

            if not split_done:
                new_segments.append(segment)

        segments = new_segments

    return [
        segment.strip()
        for segment in segments
        if segment.strip()
    ]


# ============================================================
# Length guardrail
# ============================================================

def split_long_segment(
    text: str,
    nlp,
    min_tokens: int = 3,
    max_tokens: int = 30,
) -> list[str]:
    """
    Recursively split spans longer than max_tokens.

    All characters remain assigned to exactly one output span.
    """

    n = lexical_token_count(
        text,
        nlp,
    )

    if n <= max_tokens:
        return [text.strip()]

    # Prefer punctuation near the midpoint
    candidates = [
        match.start()
        for match in re.finditer(
            r"[;:,.!?]\s+",
            text,
        )
    ]

    midpoint = len(text) / 2

    candidates = sorted(
        candidates,
        key=lambda x: abs(x - midpoint),
    )

    for pos in candidates:

        left = text[:pos + 1].strip()
        right = text[pos + 1:].strip()

        if (
            lexical_token_count(left, nlp)
            >= min_tokens
            and
            lexical_token_count(right, nlp)
            >= min_tokens
        ):

            return (
                split_long_segment(
                    left,
                    nlp,
                    min_tokens,
                    max_tokens,
                )
                +
                split_long_segment(
                    right,
                    nlp,
                    min_tokens,
                    max_tokens,
                )
            )

    # Last-resort lexical midpoint
    doc = nlp.make_doc(text)

    lexical_indices = [
        i
        for i, token in enumerate(doc)
        if not token.is_space
        and not token.is_punct
    ]

    if len(lexical_indices) <= max_tokens:
        return [text.strip()]

    token_i = lexical_indices[
        len(lexical_indices) // 2
    ]

    cut = doc[token_i].idx

    left = text[:cut].strip()
    right = text[cut:].strip()

    if not left or not right:
        return [text.strip()]

    return (
        split_long_segment(
            left,
            nlp,
            min_tokens,
            max_tokens,
        )
        +
        split_long_segment(
            right,
            nlp,
            min_tokens,
            max_tokens,
        )
    )


def merge_short_segments(
    segments: Sequence[str],
    nlp,
    min_tokens: int = 3,
    max_tokens: int = 30,
) -> list[str]:

    segments = [
        x.strip()
        for x in segments
        if x.strip()
    ]

    output = []

    i = 0

    while i < len(segments):

        current = segments[i]

        if (
            lexical_token_count(
                current,
                nlp,
            )
            >= min_tokens
        ):
            output.append(current)
            i += 1
            continue

        # Merge forward if possible
        if i + 1 < len(segments):

            merged = (
                current.rstrip(" ,;:")
                + " "
                + segments[i + 1].lstrip()
            ).strip()

            if (
                lexical_token_count(
                    merged,
                    nlp,
                )
                <= max_tokens
            ):
                segments[i + 1] = merged
                i += 1
                continue

        # Otherwise merge backward
        if output:

            merged = (
                output[-1].rstrip()
                + " "
                + current.lstrip()
            ).strip()

            if (
                lexical_token_count(
                    merged,
                    nlp,
                )
                <= max_tokens
            ):
                output[-1] = merged
                i += 1
                continue

        output.append(current)
        i += 1

    return output


# ============================================================
# Public segmentation API
# ============================================================

def segment_text(
    text: str,
    nlp=None,
    config: SegmentationConfig | None = None,
) -> list[str]:

    config = config or SegmentationConfig()

    if nlp is None:

        nlp, has_parser = load_nlp(
            config.spacy_model
        )

    else:

        has_parser = (
            "parser"
            in getattr(
                nlp,
                "pipe_names",
                [],
            )
        )

    text = normalise_text(text)

    if not text:
        return []

    doc = nlp(text)

    spans = []

    for sentence in doc.sents:

        sentence_text = sentence.text.strip()

        clauses = split_meaningful_clauses(
            sentence_text,
            nlp,
            has_parser,
            config.min_tokens,
        )

        for clause in clauses:

            spans.extend(
                split_long_segment(
                    clause,
                    nlp,
                    config.min_tokens,
                    config.max_tokens,
                )
            )

    spans = merge_short_segments(
        spans,
        nlp,
        config.min_tokens,
        config.max_tokens,
    )

    return [
        span.strip()
        for span in spans
        if span.strip()
    ]


def stable_span_id(
    response_id: Any,
    span_index: int,
    span_text: str,
) -> str:

    value = (
        f"{response_id}\x1f"
        f"{span_index}\x1f"
        f"{span_text}"
    )

    digest = hashlib.sha1(
        value.encode("utf-8")
    ).hexdigest()[:10]

    return (
        f"{response_id}"
        f"__s{span_index:03d}"
        f"__{digest}"
    )


def segment_dataframe(
    responses: pd.DataFrame,
    *,
    response_id_col: str,
    text_col: str,
    passthrough_cols: Sequence[str] = (),
    min_tokens: int = 3,
    max_tokens: int = 30,
    spacy_model: str = "en_core_web_sm",
    sectioner: Callable[
        [str],
        Sequence[tuple[Any, str]]
    ] | None = None,
) -> pd.DataFrame:
    """
    Convert a response-level dataframe into a frozen span table.

    sectioner
    ---------
    Optional external function:

        text -> [(section_label, section_text), ...]

    This permits users to recover prompt-defined feedback modules without
    hard-coding a particular module structure into this library.
    """

    required = {
        response_id_col,
        text_col,
        *passthrough_cols,
    }

    missing = sorted(
        required
        - set(responses.columns)
    )

    if missing:
        raise KeyError(
            f"Missing columns: {missing}"
        )

    nlp, _ = load_nlp(
        spacy_model
    )

    config = SegmentationConfig(
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        spacy_model=spacy_model,
    )

    records = []

    for _, row in responses.iterrows():

        response_id = row[
            response_id_col
        ]

        raw_text = normalise_text(
            row[text_col]
        )

        if sectioner is None:
            sections = [(None, raw_text)]
        else:
            sections = sectioner(raw_text)

        response_spans = []

        for section_label, section_text in sections:

            local_spans = segment_text(
                section_text,
                nlp=nlp,
                config=config,
            )

            for span in local_spans:

                response_spans.append(
                    (
                        section_label,
                        span,
                    )
                )

        n_response_spans = len(
            response_spans
        )

        for zero_i, (
            section_label,
            span_text,
        ) in enumerate(response_spans):

            span_index = zero_i + 1

            record = {

                "response_id":
                    response_id,

                "span_id":
                    stable_span_id(
                        response_id,
                        span_index,
                        span_text,
                    ),

                "span_index":
                    span_index,

                "n_response_spans":
                    n_response_spans,

                "relative_position":
                    (
                        (
                            span_index
                            - 0.5
                        )
                        /
                        n_response_spans
                    )
                    if n_response_spans
                    else np.nan,

                "section_label":
                    section_label,

                "span_text":
                    span_text,

                "span_token_n":
                    lexical_token_count(
                        span_text,
                        nlp,
                    ),
            }

            for col in passthrough_cols:
                record[col] = row[col]

            records.append(record)

    output = pd.DataFrame(
        records
    )

    if (
        len(output)
        and
        output["span_id"]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Duplicate span IDs generated."
        )

    return output



# External meta-annotation

MetaAnnotator = Callable[
    [str, Mapping[str, Any]],
    Mapping[str, Any],
]


def apply_meta_annotation(
    spans: pd.DataFrame,
    annotator: MetaAnnotator,
    *,
    text_col: str = "span_text",
) -> pd.DataFrame:
    """
    Attach annotations generated by an external annotation function.

    The external annotator must have the form:

        annotator(span_text, row_metadata) -> dict

    The function deliberately does not contain an annotation rule bank.
    """

    if text_col not in spans:
        raise KeyError(
            f"{text_col!r} not found."
        )

    original_ids = (
        spans["span_id"]
        .astype(str)
        .copy()
        if "span_id" in spans
        else None
    )

    original_text = (
        spans[text_col]
        .astype(str)
        .copy()
    )

    annotations = []

    for _, row in spans.iterrows():

        annotation = annotator(
            str(row[text_col]),
            row.to_dict(),
        )

        annotations.append(
            dict(annotation)
        )

    annotation_df = pd.DataFrame(
        annotations,
        index=spans.index,
    )

    overlap = (
        set(annotation_df.columns)
        &
        set(spans.columns)
    )

    if overlap:
        raise ValueError(
            "Annotation output would "
            f"overwrite columns: {sorted(overlap)}"
        )

    output = pd.concat(
        [
            spans.copy(),
            annotation_df,
        ],
        axis=1,
    )

    # Defensive freezing checks
    if original_ids is not None:

        if not (
            output["span_id"]
            .astype(str)
            .equals(original_ids)
        ):
            raise RuntimeError(
                "Annotation modified span IDs."
            )

    if not (
        output[text_col]
        .astype(str)
        .equals(original_text)
    ):
        raise RuntimeError(
            "Annotation modified frozen span text."
        )

    return output
