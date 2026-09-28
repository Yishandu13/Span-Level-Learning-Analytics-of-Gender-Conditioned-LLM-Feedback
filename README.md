# Span-Level Counterfactual Localisation and Pedagogical Characterisation of LLM Feedback

A modular Python toolkit for localising and interpreting fine-grained semantic divergence in counterfactual LLM-generated feedback.

The repository implements a four-stage workflow that separates **where semantic divergence occurs** from **how that divergence is pedagogically and linguistically realised**.

The toolkit was developed for analysing counterfactual gender-cue differences in LLM-generated educational feedback, but the core methods are intentionally implemented in a task- and condition-agnostic form and can be adapted to other counterfactual comparisons.

span-localisation-toolkit/

```text
│
├── 01_span_segmentation_meta_annotation.py
├── 02_embedding_counterfactual_alignment.py
├── 03_loo_permutation_localisation.py
├── 04_statistical_pedagogical_characterisation.py
│
├── annotation_bank_example.txt
├── README.md
└── requirements.txt
'''
---

## Overview

Whole-response similarity measures can identify whether two LLM outputs differ, but they provide limited information about **where** that difference occurs or **what pedagogical function it takes**.

This repository therefore implements a span-level pipeline:

```text
feedback responses
        |
        v
[1] span segmentation + meta-annotation
        |
        v
[2] semantic embedding + counterfactual span alignment
        |
        v
[3] leave-one-out attribution + permutation localisation
        |
        v
[4] statistical + pedagogical characterisation
