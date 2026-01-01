import numpy as np

from persiste.core.transition_cache import (
    clear_transition_cache,
    compute_binary_transition_matrix_cached,
    get_binary_transition_matrix,
    get_cache_info,
)


def test_compute_binary_transition_matrix_cached_counts_hits_and_misses():
    clear_transition_cache()
    info = get_cache_info()
    assert info.hits == 0
    assert info.misses == 0

    params = (0.8, 1.5, 0.2)
    first = compute_binary_transition_matrix_cached(*params)
    assert isinstance(first, tuple)

    info = get_cache_info()
    assert info.misses == 1
    assert info.hits == 0

    second = compute_binary_transition_matrix_cached(*params)
    assert second == first

    info = get_cache_info()
    assert info.hits == 1


def test_get_binary_transition_matrix_matches_cached_version():
    clear_transition_cache()
    gain_rate, loss_rate, branch = 0.4, 0.9, 0.5

    cached = get_binary_transition_matrix(gain_rate, loss_rate, branch, use_cache=True)
    uncached = get_binary_transition_matrix(gain_rate, loss_rate, branch, use_cache=False)

    np.testing.assert_allclose(cached, uncached)


def test_clear_transition_cache_resets_stats():
    clear_transition_cache()
    compute_binary_transition_matrix_cached(0.3, 0.7, 0.1)
    info = get_cache_info()
    assert info.misses == 1

    clear_transition_cache()
    info_after = get_cache_info()
    assert info_after.misses == 0
    assert info_after.hits == 0
