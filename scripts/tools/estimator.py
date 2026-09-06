"""
Batch and sliding-window maximum-a-posteriori localisation, on the "Lost in the
Woods" problem: a wheeled robot with odometry, measuring range and bearing to
known landmarks.

This is the 2-D version of the estimator in the AER1513 Assignment 3 report
(which runs on SE(3) with a stereo camera and an IMU on the Starry Night
dataset).  The structure is identical — stack every pose into one state,
linearise, solve one sparse Gauss-Newton system — and so are the conclusions,
which is the point: the window size is the only knob, and batch estimation is
the special case where the window is the whole trajectory.

State      x_k = [x, y, theta]
Motion     x_k = f(x_{k-1}, u_k) + w_k,   u = [v, omega],  w ~ N(0, Q)
Measure    y_k^j = [range, bearing] to landmark j + n,     n ~ N(0, R)
"""
import numpy as np


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class Problem:
    """A simulated trajectory, a set of landmarks, and the noisy data."""

    def __init__(self, T=400, dt=0.1, n_land=17, seed=0,
                 sigma_v=0.05, sigma_w=0.02, sigma_r=0.08, sigma_b=0.03,
                 max_range=4.0, blackout=(150, 200)):
        rng = np.random.default_rng(seed)
        self.dt, self.T = dt, T
        self.Q = np.diag([sigma_v ** 2, sigma_v ** 2, sigma_w ** 2])
        self.R = np.diag([sigma_r ** 2, sigma_b ** 2])
        self.max_range = max_range

        # a wandering ground-truth trajectory
        self.u = np.zeros((T, 2))
        t = np.arange(T) * dt
        self.u[:, 0] = 0.6 + 0.25 * np.sin(0.35 * t)
        self.u[:, 1] = 0.9 * np.sin(0.21 * t) + 0.35 * np.sin(0.07 * t)
        self.x_true = np.zeros((T + 1, 3))
        for k in range(T):
            self.x_true[k + 1] = self.f(self.x_true[k], self.u[k])

        # landmarks scattered over the area the robot visits
        lo = self.x_true[:, :2].min(0) - 1.5
        hi = self.x_true[:, :2].max(0) + 1.5
        self.land = rng.uniform(lo, hi, size=(n_land, 2))

        # noisy odometry
        self.u_meas = self.u + rng.normal(0, [sigma_v, sigma_w], size=(T, 2))

        # range-bearing measurements, with a stretch where nothing is visible
        self.obs = []
        for k in range(T + 1):
            rows = []
            if not (blackout[0] <= k < blackout[1]):
                for j, lm in enumerate(self.land):
                    d = lm - self.x_true[k, :2]
                    r = np.hypot(*d)
                    if r < max_range:
                        b = wrap(np.arctan2(d[1], d[0]) - self.x_true[k, 2])
                        rows.append((j, r + rng.normal(0, sigma_r),
                                     wrap(b + rng.normal(0, sigma_b))))
            self.obs.append(rows)
        self.visible = np.array([len(o) for o in self.obs])

    def f(self, x, u):
        v, w = u
        return np.array([x[0] + self.dt * v * np.cos(x[2]),
                         x[1] + self.dt * v * np.sin(x[2]),
                         wrap(x[2] + self.dt * w)])

    def F(self, x, u):
        """Jacobian of the motion model with respect to the previous pose."""
        v = u[0]
        return np.array([[1, 0, -self.dt * v * np.sin(x[2])],
                         [0, 1, self.dt * v * np.cos(x[2])],
                         [0, 0, 1]])

    def h(self, x, j):
        d = self.land[j] - x[:2]
        r = np.hypot(*d)
        return np.array([r, wrap(np.arctan2(d[1], d[0]) - x[2])])

    def H(self, x, j):
        dx, dy = self.land[j] - x[:2]
        r2 = dx * dx + dy * dy
        r = np.sqrt(r2)
        return np.array([[-dx / r, -dy / r, 0.0],
                         [dy / r2, -dx / r2, -1.0]])

    def dead_reckon(self):
        x = np.zeros((self.T + 1, 3))
        for k in range(self.T):
            x[k + 1] = self.f(x[k], self.u_meas[k])
        return x


def gauss_newton(prob, k0, k1, x_init, prior_mean=None, prior_cov=None,
                 iters=8, tol=1e-6):
    """One MAP solve over poses k0..k1 inclusive.  Returns the poses and the
    marginal covariance of the last pose."""
    n = k1 - k0 + 1
    x = x_init.copy()
    Qi = np.linalg.inv(prob.Q)
    Ri = np.linalg.inv(prob.R)
    A = None
    for _ in range(iters):
        A = np.zeros((3 * n, 3 * n))
        b = np.zeros(3 * n)

        if prior_cov is not None:
            Pi = np.linalg.inv(prior_cov)
            e = prior_mean - x[0]
            e[2] = wrap(e[2])
            A[:3, :3] += Pi
            b[:3] += Pi @ e

        for i in range(1, n):
            k = k0 + i
            pred = prob.f(x[i - 1], prob.u_meas[k - 1])
            e = pred - x[i]
            e[2] = wrap(e[2])
            F = prob.F(x[i - 1], prob.u_meas[k - 1])
            a, c = slice(3 * (i - 1), 3 * i), slice(3 * i, 3 * (i + 1))
            A[a, a] += F.T @ Qi @ F
            A[a, c] += -F.T @ Qi
            A[c, a] += -Qi @ F
            A[c, c] += Qi
            b[a] += -F.T @ Qi @ e
            b[c] += Qi @ e

        for i in range(n):
            k = k0 + i
            s = slice(3 * i, 3 * (i + 1))
            for (j, r, bear) in prob.obs[k]:
                e = np.array([r, bear]) - prob.h(x[i], j)
                e[1] = wrap(e[1])
                H = prob.H(x[i], j)
                A[s, s] += H.T @ Ri @ H
                b[s] += H.T @ Ri @ e

        A += 1e-9 * np.eye(3 * n)
        d = np.linalg.solve(A, b)
        x += d.reshape(n, 3)
        x[:, 2] = wrap(x[:, 2])
        if np.max(np.abs(d)) < tol:
            break

    cov_last = np.linalg.inv(A)[-3:, -3:]
    return x, cov_last


def run(prob, window=None, iters=8):
    """`window` is how many steps back the window reaches, so the solve covers
    window + 1 poses.  window = 1 is two poses, which is an extended Kalman
    filter; window = None is full batch, the whole trajectory in one solve."""
    dr = prob.dead_reckon()
    T = prob.T
    if window is None:
        x, _ = gauss_newton(prob, 0, T, dr, iters=iters)
        return x, None

    est = np.zeros((T + 1, 3))
    cov = np.zeros((T + 1, 3, 3))
    est[0] = dr[0]
    cov[0] = np.eye(3) * 1e-4
    for k in range(1, T + 1):
        k0 = max(0, k - window)
        init = np.vstack([est[k0:k], prob.f(est[k - 1], prob.u_meas[k - 1])])
        xs, P = gauss_newton(prob, k0, k, init, est[k0], cov[k0], iters=iters)
        est[k0:k + 1] = xs
        cov[k] = P
    return est, cov


def errors(est, truth):
    e = est - truth
    e[:, 2] = wrap(e[:, 2])
    return e
