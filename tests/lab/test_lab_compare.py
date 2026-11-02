# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for torq.lab.verification.compare."""

import numpy as np

from torq.lab.verification.compare import compare_outputs


def test_int_beyond_threshold_fails():
    expected = np.array([0, 0], np.int32)
    observed = np.array([0, 5], np.int32)
    result = compare_outputs([observed], [expected])
    assert not result.passed
    assert result.tensors[0].max_abs_diff == 5.0
    assert "int_tol" in result.reason


def test_int_thld_below_int_tol_uses_int_tol():
    # Regression: int_thld below int_tol must not reject within-int_tol diffs.
    # int_tol is the effective integer tolerance (matching origin/main); a
    # caller setting int_thld=0 while widening int_tol still passes.
    expected = np.array([0, 100], np.int32)
    cfg = {"int_tol": 3, "int_thld": 0}
    within = compare_outputs([np.array([0, 102], np.int32)], [expected], cfg)
    assert within.passed
    assert within.tensors[0].max_abs_diff == 2.0
    beyond = compare_outputs([np.array([0, 110], np.int32)], [expected], cfg)
    assert not beyond.passed


def test_float_beyond_tolerance_fails():
    expected = np.array([1.0], np.float32)
    observed = np.array([2.0], np.float32)
    result = compare_outputs([observed], [expected])
    assert not result.passed
    assert result.tensors[0].max_rel_diff > 0.3


def test_allowed_wrong_fraction():
    expected = np.array([1.0, 1.0, 1.0, 1.0], np.float32)
    observed = np.array([1.0, 1.0, 1.0, 2.0], np.float32)
    assert not compare_outputs([observed], [expected]).passed
    assert compare_outputs([observed], [expected], {"allowed_wrong": 0.25}).passed


def test_all_zero_guard():
    expected = np.array([1, 2], np.int32)
    observed = np.array([0, 0], np.int32)
    assert not compare_outputs([observed], [expected]).passed
    assert "all zero" in compare_outputs([observed], [expected]).reason
    # expected all-zero -> guard does not trip
    zeros = np.array([0, 0], np.int32)
    assert compare_outputs([zeros], [zeros]).passed


def test_length_mismatch():
    a = np.array([1.0], np.float32)
    result = compare_outputs([a], [a, a])
    assert not result.passed
    assert "number of outputs differ" in result.reason


def test_nan_positions_differ():
    expected = np.array([np.nan, 1.0], np.float32)
    observed = np.array([1.0, 1.0], np.float32)
    result = compare_outputs([observed], [expected])
    assert not result.passed
    assert "NaN" in result.reason


