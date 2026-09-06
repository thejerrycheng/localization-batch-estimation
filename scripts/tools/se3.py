"""
SE(3) batch and sliding-window MAP estimation — the Assignment 3 problem.

A vehicle carries an IMU (which measures the body-frame twist) and a
stereo camera (which sees known 3-D landmarks).  Every pose from k1 to k2 is
stacked into one state and solved for at once:

    x = [ T_k1, ..., T_k2 ],     T_k in SE(3)

    motion       T_k = Xi_k T_{k-1},   Xi_k = exp(dt * varpi_k ^)
    measurement  y_k^j = (1/z) M T_cv T_k p^j          (stereo, 4 numbers)

    e_v,k = ln( Xi_k T_{k-1} T_k^-1 )^v                (6-vector)
    e_y,k^j = y_k^j - h(T_k, p^j)                      (4-vector)

Gauss-Newton with a left perturbation, T <- exp(eps^) T, so the Jacobians are
the usual circle-dot / adjoint expressions rather than anything Euler-angled.
"""
import numpy as np


def hat(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def curly_hat(xi):
    """The 6x6 adjoint-algebra operator, xi^curly."""
    rho, phi = xi[:3], xi[3:]
    out = np.zeros((6, 6))
    out[:3, :3] = hat(phi)
    out[:3, 3:] = hat(rho)
    out[3:, 3:] = hat(phi)
    return out


def exp_so3(phi):
    a = np.linalg.norm(phi)
    if a < 1e-9:
        return np.eye(3) + hat(phi)
    ax = phi / a
    K = hat(ax)
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


def left_jacobian(phi):
    a = np.linalg.norm(phi)
    if a < 1e-9:
        return np.eye(3) + 0.5 * hat(phi)
    ax = phi / a
    K = hat(ax)
    return (np.sin(a) / a * np.eye(3)
            + (1 - np.sin(a) / a) * np.outer(ax, ax)
            + (1 - np.cos(a)) / a * K)


def exp_se3(xi):
    rho, phi = xi[:3], xi[3:]
    T = np.eye(4)
    T[:3, :3] = exp_so3(phi)
    T[:3, 3] = left_jacobian(phi) @ rho
    return T


def log_so3(C):
    c = (np.trace(C) - 1) / 2
    a = np.arccos(np.clip(c, -1, 1))
    if a < 1e-9:
        return np.array([C[2, 1] - C[1, 2], C[0, 2] - C[2, 0], C[1, 0] - C[0, 1]]) / 2
    return a / (2 * np.sin(a)) * np.array([C[2, 1] - C[1, 2],
                                           C[0, 2] - C[2, 0],
                                           C[1, 0] - C[0, 1]])


def log_se3(T):
    phi = log_so3(T[:3, :3])
    rho = np.linalg.solve(left_jacobian(phi), T[:3, 3])
    return np.concatenate([rho, phi])


def adjoint(T):
    C, r = T[:3, :3], T[:3, 3]
    A = np.zeros((6, 6))
    A[:3, :3] = C
    A[:3, 3:] = hat(r) @ C
    A[3:, 3:] = C
    return A


def circle_dot(p):
    """d(T p)/d(eps) for a homogeneous point p = [x, y, z, 1]."""
    out = np.zeros((4, 6))
    out[:3, :3] = np.eye(3)
    out[:3, 3:] = -hat(p[:3])
    return out


class StereoRig:
    """The stereo model from the report: M is 4x4, baseline b."""

    def __init__(self, fu=387.8, fv=387.8, cu=257.9, cv=197.0, b=0.24):
        self.fu, self.fv, self.cu, self.cv, self.b = fu, fv, cu, cv, b
        self.M = np.array([[fu, 0, cu, 0],
                           [0, fv, cv, 0],
                           [fu, 0, cu, -fu * b],
                           [0, fv, cv, 0]])

    def project(self, pc):
        """Camera-frame point -> [ul, vl, ur, vr]."""
        z = pc[2]
        return np.array([self.fu * pc[0] / z + self.cu,
                         self.fv * pc[1] / z + self.cv,
                         self.fu * (pc[0] - self.b) / z + self.cu,
                         self.fv * pc[1] / z + self.cv])

    def dproject(self, pc):
        """d(project)/d(camera point), 4x3."""
        x, y, z = pc[:3]
        return np.array([[self.fu / z, 0, -self.fu * x / z ** 2],
                         [0, self.fv / z, -self.fv * y / z ** 2],
                         [self.fu / z, 0, -self.fu * (x - self.b) / z ** 2],
                         [0, self.fv / z, -self.fv * y / z ** 2]])


class SE3Problem:
    def __init__(self, T=200, dt=0.1, n_land=40, seed=0,
                 sigma_v=0.06, sigma_w=0.02, sigma_px=1.2,
                 max_range=12.0, fov_deg=70.0, blackout=None):
        rng = np.random.default_rng(seed)
        self.T, self.dt = T, dt
        self.rig = StereoRig()
        self.Q = np.diag([sigma_v ** 2] * 3 + [sigma_w ** 2] * 3)
        self.R = np.eye(4) * sigma_px ** 2
        self.max_range, self.cos_fov = max_range, np.cos(np.deg2rad(fov_deg) / 2)

        # camera looks along the body x-axis, mounted level
        self.T_cv = np.eye(4)
        self.T_cv[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])

        # a gentle 3-D corkscrew, so all six degrees of freedom move
        self.varpi = np.zeros((T, 6))
        t = np.arange(T) * dt
        self.varpi[:, 0] = 1.2
        self.varpi[:, 1] = 0.15 * np.sin(0.5 * t)
        self.varpi[:, 2] = 0.20 * np.sin(0.3 * t)
        self.varpi[:, 3] = 0.10 * np.sin(0.7 * t)
        self.varpi[:, 4] = 0.12 * np.sin(0.4 * t)
        self.varpi[:, 5] = 0.35 * np.sin(0.22 * t)

        self.T_true = [np.eye(4)]
        for k in range(T):
            self.T_true.append(exp_se3(dt * self.varpi[k]) @ self.T_true[k])

        # landmarks around the path, in the inertial frame
        pos = np.array([np.linalg.inv(Tk)[:3, 3] for Tk in self.T_true])
        lo, hi = pos.min(0) - 6, pos.max(0) + 6
        self.land = rng.uniform(lo, hi, size=(n_land, 3))

        self.varpi_meas = self.varpi + rng.normal(
            0, [sigma_v] * 3 + [sigma_w] * 3, size=(T, 6))

        self.obs = []
        for k in range(T + 1):
            rows = []
            dark = blackout and blackout[0] <= k < blackout[1]
            if not dark:
                Tk = self.T_true[k]
                for j, p in enumerate(self.land):
                    ph = np.append(p, 1.0)
                    pc = (self.T_cv @ Tk @ ph)[:3]
                    if pc[2] < 0.5 or np.linalg.norm(pc) > max_range:
                        continue
                    if pc[2] / np.linalg.norm(pc) < self.cos_fov:
                        continue
                    y = self.rig.project(pc) + rng.normal(0, sigma_px, 4)
                    rows.append((j, y))
            self.obs.append(rows)
        self.visible = np.array([len(o) for o in self.obs])

    def h(self, Tk, j):
        ph = np.append(self.land[j], 1.0)
        pc = (self.T_cv @ Tk @ ph)[:3]
        return self.rig.project(pc), pc

    def H(self, Tk, j):
        """d(measurement)/d(left perturbation of T_k), 4x6."""
        ph = np.append(self.land[j], 1.0)
        p_v = Tk @ ph
        pc = (self.T_cv @ p_v)[:3]
        return self.rig.dproject(pc) @ self.T_cv[:3, :] @ circle_dot(p_v)

    def dead_reckon(self):
        out = [np.eye(4)]
        for k in range(self.T):
            out.append(exp_se3(self.dt * self.varpi_meas[k]) @ out[k])
        return out


def gauss_newton(prob, k0, k1, T_init, prior_T=None, prior_cov=None, iters=6):
    n = k1 - k0 + 1
    Ts = [t.copy() for t in T_init]
    Qi = np.linalg.inv(prob.Q)
    Ri = np.linalg.inv(prob.R)
    A = None
    for _ in range(iters):
        N = 6 * n
        A = np.zeros((N, N))
        b = np.zeros(N)

        if prior_cov is not None:
            Pi = np.linalg.inv(prior_cov)
            e = log_se3(prior_T @ np.linalg.inv(Ts[0]))
            A[:6, :6] += Pi
            b[:6] += Pi @ e

        for i in range(1, n):
            k = k0 + i
            Xi = exp_se3(prob.dt * prob.varpi_meas[k - 1])
            e = log_se3(Xi @ Ts[i - 1] @ np.linalg.inv(Ts[i]))
            F = adjoint(Xi)                     # d e / d eps_{i-1}
            a, c = slice(6 * (i - 1), 6 * i), slice(6 * i, 6 * (i + 1))
            A[a, a] += F.T @ Qi @ F
            A[a, c] += -F.T @ Qi
            A[c, a] += -Qi @ F
            A[c, c] += Qi
            b[a] += -F.T @ Qi @ e
            b[c] += Qi @ e

        for i in range(n):
            k = k0 + i
            s = slice(6 * i, 6 * (i + 1))
            for (j, y) in prob.obs[k]:
                yhat, pc = prob.h(Ts[i], j)
                if pc[2] < 0.2:
                    continue
                e = y - yhat
                H = prob.H(Ts[i], j)
                A[s, s] += H.T @ Ri @ H
                b[s] += H.T @ Ri @ e

        A += 1e-8 * np.eye(6 * n)
        d = np.linalg.solve(A, b)
        for i in range(n):
            Ts[i] = exp_se3(d[6 * i:6 * (i + 1)]) @ Ts[i]
        if np.max(np.abs(d)) < 1e-8:
            break

    cov_last = np.linalg.inv(A)[-6:, -6:]
    return Ts, cov_last


def run(prob, window=None, iters=6):
    dr = prob.dead_reckon()
    T = prob.T
    if window is None:
        Ts, _ = gauss_newton(prob, 0, T, dr, iters=iters)
        return Ts, None
    est = [dr[0].copy()]
    cov = [np.eye(6) * 1e-6]
    for k in range(1, T + 1):
        k0 = max(0, k - window)
        init = [est[i].copy() for i in range(k0, k)]
        init.append(exp_se3(prob.dt * prob.varpi_meas[k - 1]) @ est[k - 1])
        Ts, P = gauss_newton(prob, k0, k, init, est[k0], cov[k0], iters=iters)
        for i in range(k0, k + 1):
            if i < len(est):
                est[i] = Ts[i - k0]
            else:
                est.append(Ts[i - k0])
        cov.append(P) if len(cov) == k else cov.__setitem__(k, P)
    return est, cov


def errors(est, truth):
    """Left-invariant error, split into translation [m] and rotation [rad]."""
    out = np.zeros((len(est), 6))
    for k, (E, G) in enumerate(zip(est, truth)):
        out[k] = log_se3(E @ np.linalg.inv(G))
    return out
