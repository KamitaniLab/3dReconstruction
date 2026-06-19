"""Posterior slope extraction for Bambi fixed-effect parameterizations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from recon3d.stats.bayesian_lmm import BayesianLMMHelper


@dataclass
class PosteriorSlopeExtractor:
    """Compose emtrends-style slope draws from Bambi fixed-effect posteriors."""

    idata: Any
    x_var: str

    def base_slope(self) -> np.ndarray:
        return self.draws_1d(self.x_var).copy()

    def roi_slope(self, roi: str, roi_levels: list[str]) -> np.ndarray:
        samples = self.base_slope()
        roi_ref = roi_levels[0]
        if roi == roi_ref:
            return samples
        non_ref_rois = [level for level in roi_levels if level != roi_ref]
        return samples + self.term_draws_with_level(
            [f"{self.x_var}:roi", f"roi:{self.x_var}"],
            level=roi,
            non_ref_levels=non_ref_rois,
        )

    def stimulus_slope(self, stimulus: str, stimulus_levels: list[str]) -> np.ndarray:
        samples = self.base_slope()
        stim_ref = stimulus_levels[0]
        if stimulus == stim_ref:
            return samples
        non_ref_stimuli = [level for level in stimulus_levels if level != stim_ref]
        return samples + self.term_draws_with_level(
            [f"{self.x_var}:stimulus", f"stimulus:{self.x_var}"],
            level=stimulus,
            non_ref_levels=non_ref_stimuli,
        )

    def roi_stimulus_slope(
        self,
        *,
        roi: str,
        stimulus: str,
        roi_levels: list[str],
        stimulus_levels: list[str],
    ) -> np.ndarray:
        samples = self.base_slope()
        roi_ref = roi_levels[0]
        stim_ref = stimulus_levels[0]
        non_ref_rois = [level for level in roi_levels if level != roi_ref]
        non_ref_stimuli = [level for level in stimulus_levels if level != stim_ref]
        if roi != roi_ref:
            samples = samples + self.term_draws_with_level(
                [f"{self.x_var}:roi", f"roi:{self.x_var}"],
                level=roi,
                non_ref_levels=non_ref_rois,
            )
        if stimulus != stim_ref:
            samples = samples + self.term_draws_with_level(
                [f"{self.x_var}:stimulus", f"stimulus:{self.x_var}"],
                level=stimulus,
                non_ref_levels=non_ref_stimuli,
            )
        if roi != roi_ref and stimulus != stim_ref:
            samples = samples + self.interaction_draws(
                [
                    f"{self.x_var}:roi:stimulus",
                    f"{self.x_var}:stimulus:roi",
                    f"roi:{self.x_var}:stimulus",
                    f"roi:stimulus:{self.x_var}",
                    f"stimulus:{self.x_var}:roi",
                    f"stimulus:roi:{self.x_var}",
                ],
                levels=(roi, stimulus),
                non_ref_a=non_ref_rois,
                non_ref_b=non_ref_stimuli,
            )
        return samples

    def draws_1d(self, var: str, level=None) -> np.ndarray:
        return BayesianLMMHelper._draws_1d(self.idata, var, level)

    def zero_like_base(self) -> np.ndarray:
        return np.zeros_like(self.draws_1d(self.x_var), dtype=float)

    def term_draws_with_level(
        self,
        term_candidates: list[str],
        *,
        level,
        non_ref_levels: list[str],
    ) -> np.ndarray:
        for term in term_candidates:
            if term not in self.idata.posterior.data_vars:
                continue
            try:
                return self.draws_1d(term, level)
            except Exception:
                try:
                    return self.draws_1d(term, non_ref_levels.index(level))
                except Exception:
                    continue
        return self.zero_like_base()

    def interaction_draws(
        self,
        term_candidates: list[str],
        *,
        levels,
        non_ref_a: list[str],
        non_ref_b: list[str],
    ) -> np.ndarray:
        a_level, b_level = levels
        for term in term_candidates:
            if term not in self.idata.posterior.data_vars:
                continue
            try:
                return self.draws_1d(term, levels)
            except Exception:
                pass
            draws = self.idata.posterior[term]
            extra_dims = [dim for dim in draws.dims if dim not in ("chain", "draw")]
            try:
                if len(extra_dims) == 2:
                    term_factors = [part for part in term.split(":") if part != self.x_var]
                    index_by_factor = {
                        "roi": non_ref_a.index(a_level),
                        "stimulus": non_ref_b.index(b_level),
                    }
                    if term_factors == ["stimulus", "roi"]:
                        index_by_factor = {
                            "stimulus": non_ref_b.index(b_level),
                            "roi": non_ref_a.index(a_level),
                        }
                    return draws.isel(
                        {
                            dim: index_by_factor[factor]
                            for dim, factor in zip(extra_dims, term_factors, strict=True)
                        }
                    ).values.reshape(-1)
                if len(extra_dims) == 1:
                    idx = non_ref_a.index(a_level) * len(non_ref_b) + non_ref_b.index(b_level)
                    return draws.isel({extra_dims[0]: idx}).values.reshape(-1)
            except Exception:
                continue
        return self.zero_like_base()
