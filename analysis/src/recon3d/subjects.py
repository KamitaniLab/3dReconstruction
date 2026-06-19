"""Subject-name helpers used by the public analysis code."""

from __future__ import annotations

from typing import Any

import pandas as pd

from recon3d.metadata import PUBLIC_SUBJECTS

# Reproducing the paper's Bayesian estimates requires fitting with this
# alphabetical order of subjects. Temporary A-labels are used only during model
# fitting so the public subject IDs can remain S1...S5 in inputs and outputs.
BAYESIAN_SUBJECT_ORDER = [PUBLIC_SUBJECTS[index] for index in (2, 3, 1, 4, 0)]


def validate_public_subject(subject: str) -> str:
    """Return a public subject name after validating it."""
    subject = str(subject)
    if subject not in PUBLIC_SUBJECTS:
        raise ValueError(
            f"Unknown subject {subject!r}. Use one of: {', '.join(PUBLIC_SUBJECTS)}."
        )
    return subject


def to_public_subject(subject: str) -> str:
    """Return a validated public subject name."""
    return validate_public_subject(subject)


def apply_bayesian_subject_aliases(
    df: pd.DataFrame,
    *,
    subject_order: list[str] | None = None,
) -> dict[str, str]:
    """Replace public subject IDs with A1... labels for Bayesian fitting."""
    subject_order = subject_order or BAYESIAN_SUBJECT_ORDER
    subjects = set(df["subject"].astype(str))
    unknown = sorted(subjects - set(subject_order))
    if unknown:
        raise ValueError(f"Unexpected subject labels for Bayesian LMM: {unknown}")

    ordered_subjects = [subject for subject in subject_order if subject in subjects]
    subject_to_alias = {subject: f"A{i + 1}" for i, subject in enumerate(ordered_subjects)}
    alias_to_subject = {alias: subject for subject, alias in subject_to_alias.items()}
    df["subject"] = pd.Categorical(
        [subject_to_alias[subject] for subject in df["subject"].astype(str)],
        categories=[subject_to_alias[subject] for subject in ordered_subjects],
        ordered=False,
    )
    return alias_to_subject


def restore_bayesian_subject_labels(result: dict[str, Any], alias_to_subject: dict[str, str]) -> None:
    """Restore public subject labels in fitted-model data and InferenceData coords."""
    for model in result.get("models", {}).values():
        data = getattr(model, "data", None)
        if data is not None and "subject" in data:
            _restore_subject_column(data, alias_to_subject)

    for idata in result.get("idata", {}).values():
        for group in idata.groups():
            dataset = getattr(idata, group)
            for coord in dataset.coords:
                values = dataset.coords[coord].values
                if values.dtype.kind not in "OUS":
                    continue
                restored = [alias_to_subject.get(str(value), str(value)) for value in values]
                if restored != [str(value) for value in values]:
                    dataset.coords[coord] = restored


def _restore_subject_column(df: pd.DataFrame, alias_to_subject: dict[str, str]) -> None:
    restored = [alias_to_subject.get(str(value), str(value)) for value in df["subject"].astype(str)]
    categories = [subject for subject in BAYESIAN_SUBJECT_ORDER if subject in restored]
    df["subject"] = pd.Categorical(restored, categories=categories, ordered=False)
