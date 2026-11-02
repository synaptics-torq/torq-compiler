# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Behavior tests for the torq.testing.comparison pytest shell.

The numeric core is delegated to torq.lab.verification.compare; these tests pin the pass/fail
outcomes and the stdout metric lines that the gen_config accuracy pipeline
parses.
"""

import contextlib
import io

import numpy as np
import pytest

from torq.gen_config._utils import parse_diff_metrics
from torq.testing.comparison import compare_results


class _FakeRequest:
    """Minimal request stub: only `getfixturevalue("tmpdir")` is used."""

    def __init__(self, tmpdir):
        self._tmpdir = tmpdir

    def getfixturevalue(self, name):
        assert name == "tmpdir"
        return self._tmpdir


def _run(tmp_path, observed, expected, **cfg_overrides):
    from torq.lab.verification.compare import DEFAULT_COMPARISON_CONFIG

    cfg = dict(DEFAULT_COMPARISON_CONFIG)
    cfg.update(cfg_overrides)
    buf = io.StringIO()
    err = None
    with contextlib.redirect_stdout(buf):
        try:
            compare_results(_FakeRequest(tmp_path), observed, expected, comparison_config=cfg)
        except AssertionError as exc:
            err = exc
    return err, buf.getvalue()


def test_float_match_passes(tmp_path):
    a = [np.array([1.0, 2.0, 3.0], np.float32)]
    err, out = _run(tmp_path, a, [a[0].copy()])
    assert err is None
    assert "Max relative difference:" in out
    assert "Max absolute difference:" in out


def test_float_within_tolerance_passes(tmp_path):
    exp = [np.array([1.0, 2.0, 3.0], np.float32)]
    obs = [np.array([1.0001, 2.0001, 3.0001], np.float32)]
    err, _ = _run(tmp_path, obs, exp, fp_max_tol=0.02)
    assert err is None


def test_float_large_diff_fails(tmp_path):
    exp = [np.array([1.0, 2.0, 3.0], np.float32)]
    obs = [np.array([10.0, 20.0, 30.0], np.float32)]
    err, out = _run(tmp_path, obs, exp)
    assert isinstance(err, AssertionError)
    # gen_config parses these stdout lines
    metrics = parse_diff_metrics(out)
    assert "max_rel_diff" in metrics and "max_abs_diff" in metrics
    assert "out of" in out and "%]" in out


def test_all_zero_observed_fails(tmp_path):
    exp = [np.array([1.0, 2.0], np.float32)]
    obs = [np.zeros(2, np.float32)]
    err, _ = _run(tmp_path, obs, exp)
    assert isinstance(err, AssertionError)


def test_all_zero_allowed_when_expected_zero(tmp_path):
    exp = [np.zeros(2, np.float32)]
    obs = [np.zeros(2, np.float32)]
    err, _ = _run(tmp_path, obs, exp)
    assert err is None


def test_length_mismatch_fails(tmp_path):
    err, _ = _run(tmp_path, [np.zeros(2, np.float32)], [np.zeros(2, np.float32), np.zeros(2, np.float32)])
    assert isinstance(err, AssertionError)


def test_stdout_metrics_are_parseable(tmp_path):
    exp = [np.array([1.0, 2.0, 3.0, 4.0], np.float32)]
    obs = [np.array([1.0, 2.0, 3.0, 9.0], np.float32)]
    _, out = _run(tmp_path, obs, exp)
    metrics = parse_diff_metrics(out)
    assert metrics["total_elements"] == 4
    assert 0 <= metrics["num_differences"] <= 4
