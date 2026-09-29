"""scripts/analysis/b2d_run_noise.py: the collision-lottery formula and pooled run variance (S-100, S-101)."""
import math

from scripts.analysis.b2d_run_noise import poisson_ds, pooled_var


def test_no_collisions_no_noise():
    assert poisson_ds(100.0, 0.0) == (100.0, 0.0)


def test_mean_is_generating_function_of_the_penalty():
    # E[0.6^K] for K ~ Poisson(lam) is exp(-0.4 lam)
    mean, _ = poisson_ds(100.0, 1.0)
    assert math.isclose(mean, 100 * math.exp(-0.4))


def test_noise_peaks_near_lambda_1_4_at_about_29():
    lams = [i / 100 for i in range(1, 500)]
    sds = [poisson_ds(100.0, lam)[1] for lam in lams]
    best = lams[sds.index(max(sds))]
    assert abs(best - math.log(1.25) / 0.16) < 0.02
    assert 28.0 < max(sds) < 29.5


def test_monte_carlo_agrees():
    import random
    rng = random.Random(0)

    def poisson(lam):
        k, p, L = 0, 1.0, math.exp(-lam)
        while True:
            p *= rng.random()
            if p <= L:
                return k
            k += 1
    xs = [100 * 0.6 ** poisson(1.4) for _ in range(40000)]
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    em, esd = poisson_ds(100.0, 1.4)
    assert abs(m - em) < 0.5 and abs(sd - esd) < 0.5


def test_pooled_var_uses_only_repeated_routes():
    v, dof = pooled_var({"a": [100, 60], "b": [50], "c": [10, 10, 40]})
    assert dof == 1 + 2
    assert math.isclose(v, (800 + 600) / 3)
