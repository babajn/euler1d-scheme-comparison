# Standalone module extracted from notebooks/euler1d_stage1_light.ipynb
# Numerical algorithm is unchanged from the notebook.
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Optional, Tuple
import time
import numpy as np

class NumericalFailure(RuntimeError):
    """An inadmissible trial state; the integrator retains the last accepted state."""
    def __init__(self, stage, U=None, gamma=1.4, detail=""):
        self.stage = stage
        self.min_rho = np.nan
        self.min_p = np.nan
        self.bad_index = None
        if U is not None:
            rho = np.asarray(U)[..., 0]
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                pressure = (gamma - 1.0) * (U[..., 2] - 0.5 * U[..., 1]**2 / rho)
            self.min_rho = float(np.min(rho))
            self.min_p = float(np.min(pressure))
            bad = (~np.all(np.isfinite(U), axis=-1)) | (~np.isfinite(pressure)) | (rho <= 0.0) | (pressure <= 0.0)
            indices = np.flatnonzero(bad)
            self.bad_index = int(indices[0]) if len(indices) else None
        message = f"{stage}: {detail} min(rho)={self.min_rho:.5e}, min(p)={self.min_p:.5e}"
        super().__init__(message.strip())


def prim_to_cons(W: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    """基本変数 W=(rho,u,p) を保存変数 U=(rho,rho*u,E) に変換する。"""
    W = np.asarray(W, dtype=float)
    rho, u, p = W[..., 0], W[..., 1], W[..., 2]
    if gamma <= 1.0 or np.any(rho <= 0.0) or np.any(p <= 0.0) or not np.all(np.isfinite(W)):
        raise ValueError("gamma > 1, finite rho/u/p, rho > 0 and p > 0 are required.")
    return np.stack((rho, rho * u, p / (gamma - 1.0) + 0.5 * rho * u**2), axis=-1)


def cons_to_prim(U: np.ndarray, gamma: float = 1.4, check: bool = True, stage: str = "state") -> np.ndarray:
    """保存変数から基本変数を求める。非物理的な値を切り上げる操作はしない。"""
    U = np.asarray(U, dtype=float)
    rho = U[..., 0]
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        u = U[..., 1] / rho
        p = (gamma - 1.0) * (U[..., 2] - 0.5 * U[..., 1] * u)
    W = np.stack((rho, u, p), axis=-1)
    if check and (not np.all(np.isfinite(W)) or np.any(rho <= 0.0) or np.any(p <= 0.0)):
        raise NumericalFailure(stage, U, gamma)
    return W


def physical_flux(U: np.ndarray, gamma: float = 1.4, stage: str = "flux state") -> np.ndarray:
    W = cons_to_prim(U, gamma, stage=stage)
    rho, u, p = W[..., 0], W[..., 1], W[..., 2]
    return np.stack((rho * u, rho * u**2 + p, u * (U[..., 2] + p)), axis=-1)


def max_signal_speed(U: np.ndarray, gamma: float = 1.4) -> float:
    W = cons_to_prim(U, gamma)
    return float(np.max(np.abs(W[..., 1]) + np.sqrt(gamma * W[..., 2] / W[..., 0])))

RIEMANN_SOLVERS = ("Roe", "LLF", "HLL", "HLLC")
ALL_SOLVERS = ("LF", "LW", "Roe", "LLF", "HLL", "HLLC")


def roe_averages(UL: np.ndarray, UR: np.ndarray, WL: np.ndarray, WR: np.ndarray, gamma: float):
    rootL, rootR = np.sqrt(WL[..., 0]), np.sqrt(WR[..., 0])
    u = (rootL * WL[..., 1] + rootR * WR[..., 1]) / (rootL + rootR)
    HL, HR = (UL[..., 2] + WL[..., 2]) / WL[..., 0], (UR[..., 2] + WR[..., 2]) / WR[..., 0]
    H = (rootL * HL + rootR * HR) / (rootL + rootR)
    a2 = (gamma - 1.0) * (H - 0.5 * u**2)
    if np.any(a2 <= 0.0) or not np.all(np.isfinite(a2)):
        raise NumericalFailure("Roe average", detail="Nonpositive or nonfinite sound-speed squared.")
    return rootL * rootR, u, H, np.sqrt(a2)


def signal_bounds(UL, UR, WL, WR, gamma: float, method: str = "einfeldt"):
    """HLL/HLLCに同じ外側波速を使う。einfeldtの場合のHLLをHLLEと表示する。"""
    uL, uR = WL[..., 1], WR[..., 1]
    aL, aR = np.sqrt(gamma * WL[..., 2] / WL[..., 0]), np.sqrt(gamma * WR[..., 2] / WR[..., 0])
    if method == "einfeldt":
        _, u, _, a = roe_averages(UL, UR, WL, WR, gamma)
        SL, SR = np.minimum(uL - aL, u - a), np.maximum(uR + aR, u + a)
    elif method == "davis":
        SL, SR = np.minimum(uL - aL, uR - aR), np.maximum(uL + aL, uR + aR)
    elif method == "symmetric":
        amax = np.maximum(np.abs(uL) + aL, np.abs(uR) + aR)
        SL, SR = -amax, amax
    else:
        raise ValueError("Wave speeds must be 'einfeldt', 'davis', or 'symmetric'.")
    return SL, SR


def entropy_abs(lam: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """音響波だけに用いるHarten型の二次正則化。delta=0なら補正なし。"""
    value = np.abs(lam).copy()
    mask = (delta > 0.0) & (value < delta)
    value[mask] = 0.5 * (lam[mask]**2 / delta[mask] + delta[mask])
    return value


def numerical_flux(UL, UR, solver: str, gamma: float = 1.4, dt: Optional[float] = None,
                   dx: Optional[float] = None, wave_speeds: str = "einfeldt", roe_entropy: float = 0.1):
    """各セル境界の数値流束を返す。LLFのalphaは局所的、LFのalphaはdx/dt。"""
    if solver not in ALL_SOLVERS:
        raise ValueError(f"Unknown solver: {solver}")
    UL, UR = np.asarray(UL, dtype=float), np.asarray(UR, dtype=float)
    if UL.ndim != 2 or UL.shape != UR.shape or UL.shape[1] != 3:
        raise ValueError("Interface states must both have shape (number_of_faces, 3).")
    WL, WR = cons_to_prim(UL, gamma, stage="left interface"), cons_to_prim(UR, gamma, stage="right interface")
    FL, FR = physical_flux(UL, gamma), physical_flux(UR, gamma)
    rhoL, uL, pL = WL[..., 0], WL[..., 1], WL[..., 2]
    rhoR, uR, pR = WR[..., 0], WR[..., 1], WR[..., 2]
    aL, aR = np.sqrt(gamma * pL / rhoL), np.sqrt(gamma * pR / rhoR)
    jump = UR - UL
    if solver in ("LF", "LW"):
        if dt is None or dx is None or dt <= 0.0 or dx <= 0.0:
            raise ValueError("LF and LW require positive dt and dx.")
        if solver == "LF":
            return 0.5 * (FL + FR) - 0.5 * (dx / dt) * jump
        Uhalf = 0.5 * (UL + UR) - 0.5 * (dt / dx) * (FR - FL)
        return physical_flux(Uhalf, gamma, stage="LW half-step predictor")
    if solver == "LLF":
        alpha = np.maximum(np.abs(uL) + aL, np.abs(uR) + aR)
        return 0.5 * (FL + FR) - 0.5 * alpha[..., None] * jump
    if solver == "Roe":
        rho, u, H, a = roe_averages(UL, UR, WL, WR, gamma)
        drho, du, dp = rhoR - rhoL, uR - uL, pR - pL
        beta1, beta3 = (dp - rho * a * du) / (2.0 * a**2), (dp + rho * a * du) / (2.0 * a**2)
        beta2 = drho - dp / a**2
        r1 = np.stack((np.ones_like(u), u - a, H - u * a), axis=-1)
        r2 = np.stack((np.ones_like(u), u, 0.5 * u**2), axis=-1)
        r3 = np.stack((np.ones_like(u), u + a, H + u * a), axis=-1)
        delta = roe_entropy * a
        d1, d2, d3 = entropy_abs(u - a, delta), np.abs(u), entropy_abs(u + a, delta)
        dissipation = (beta1 * d1)[..., None] * r1 + (beta2 * d2)[..., None] * r2 + (beta3 * d3)[..., None] * r3
        return 0.5 * (FL + FR - dissipation)
    SL, SR = signal_bounds(UL, UR, WL, WR, gamma, wave_speeds)
    if solver == "HLL":
        Fmiddle = (SR[..., None] * FL - SL[..., None] * FR + (SL * SR)[..., None] * jump) / (SR - SL)[..., None]
        return np.where((SL >= 0.0)[..., None], FL, np.where((SR <= 0.0)[..., None], FR, Fmiddle))
    # HLLC: 音響波の間に接触波 Sstar を復元する。不要な超音速側のstar状態は計算しない。
    result = np.where((SL >= 0.0)[..., None], FL, FR).copy()
    middle = (SL < 0.0) & (SR > 0.0)
    if not np.any(middle):
        return result
    sl, sr = SL[middle], SR[middle]
    rl, rr, vl, vr, pl, pr = rhoL[middle], rhoR[middle], uL[middle], uR[middle], pL[middle], pR[middle]
    numerator = pr - pl + rl * vl * (sl - vl) - rr * vr * (sr - vr)
    denominator = rl * (sl - vl) - rr * (sr - vr)
    if np.any(denominator == 0.0):
        raise NumericalFailure("HLLC contact speed", detail="Zero denominator.")
    sm = numerator / denominator
    if np.any(sl >= sm) or np.any(sm >= sr):
        raise NumericalFailure("HLLC wave ordering", detail="Expected SL < Sstar < SR.")
    rsl, rsr = rl * (sl - vl) / (sl - sm), rr * (sr - vr) / (sr - sm)
    esl = rsl * (UL[middle, 2] / rl + (sm - vl) * (sm + pl / (rl * (sl - vl))))
    esr = rsr * (UR[middle, 2] / rr + (sm - vr) * (sm + pr / (rr * (sr - vr))))
    Usl, Usr = np.column_stack((rsl, rsl * sm, esl)), np.column_stack((rsr, rsr * sm, esr))
    Fsl, Fsr = FL[middle] + sl[:, None] * (Usl - UL[middle]), FR[middle] + sr[:, None] * (Usr - UR[middle])
    result[middle] = np.where((sm >= 0.0)[:, None], Fsl, Fsr)
    return result

@dataclass(frozen=True)
class Problem:
    key: str
    title: str
    bounds: Tuple[float, float]
    final_time: float
    boundary: str
    left: Optional[Tuple[float, float, float]] = None
    right: Optional[Tuple[float, float, float]] = None
    x0: float = 0.5
    velocity: float = 1.0


PROBLEMS = {
    "entropy": Problem("entropy", "Smooth entropy-wave advection", (0.0, 1.0), 1.0, "periodic"),
    "contact": Problem("contact", "Moving contact discontinuities", (0.0, 1.0), 0.25, "periodic"),
    "contact_static": Problem("contact_static", "Stationary contact discontinuities", (0.0, 1.0), 0.25, "periodic", velocity=0.0),
    "sod": Problem("sod", "Sod shock tube", (0.0, 1.0), 0.2, "outflow", (1.0, 0.0, 1.0), (0.125, 0.0, 0.1)),
    "double_rarefaction": Problem("double_rarefaction", "Two strong rarefactions", (-0.5, 0.5), 0.15, "outflow", (1.0, -2.0, 0.4), (1.0, 2.0, 0.4), x0=0.0),
    "lax": Problem("lax", "Lax shock tube", (-0.5, 0.5), 0.14, "outflow", (0.445, 0.698, 3.528), (0.5, 0.0, 0.571), x0=0.0),
    "transonic": Problem("transonic", "Transonic rarefaction", (0.0, 1.0), 0.2, "outflow", (1.0, 0.75, 1.0), (0.125, 0.0, 0.1), x0=0.3),
    "shu_osher": Problem("shu_osher", "Shu-Osher shock / entropy-wave interaction", (-5.0, 5.0), 1.8, "inflow_outflow", (27.0 / 7.0, 20.0 * np.sqrt(1.4) / 9.0, 31.0 / 3.0), x0=-4.0),
}


def grid_for(problem: Problem, n: int):
    if not isinstance(n, (int, np.integer)) or n < 8:
        raise ValueError("n must be an integer >= 8.")
    edges = np.linspace(problem.bounds[0], problem.bounds[1], n + 1)
    return edges, 0.5 * (edges[:-1] + edges[1:]), float(edges[1] - edges[0])


def advected_cell_average(problem: Problem, edges: np.ndarray, t: float, gamma: float):
    """正弦波/矩形波の厳密なセル平均。密度だけでなく保存変数を平均する。"""
    dx, x = np.diff(edges), 0.5 * (edges[:-1] + edges[1:])
    v, length = problem.velocity, problem.bounds[1] - problem.bounds[0]
    if problem.key == "entropy":
        rho = 1.0 + 0.2 * np.sinc(dx / length) * np.sin(2.0 * np.pi * (x - v * t) / length)
    else:
        shift = (v * t) % length
        a, b = 0.25 + shift, 0.45 + shift
        overlap = np.zeros_like(x)
        for k in (-1, 0, 1):
            overlap += np.maximum(0.0, np.minimum(edges[1:], b + k * length) - np.maximum(edges[:-1], a + k * length))
        rho = 1.0 + overlap / dx
    W = np.column_stack((rho, np.full_like(rho, v), np.ones_like(rho)))
    return prim_to_cons(W, gamma)


def initial_cell_average(problem: Problem, edges: np.ndarray, gamma: float = 1.4) -> np.ndarray:
    if problem.key in ("entropy", "contact", "contact_static"):
        return advected_cell_average(problem, edges, 0.0, gamma)
    dx = np.diff(edges)
    frac = np.clip((problem.x0 - edges[:-1]) / dx, 0.0, 1.0)
    UL = prim_to_cons(np.array(problem.left), gamma)
    if problem.key != "shu_osher":
        UR = prim_to_cons(np.array(problem.right), gamma)
        return frac[:, None] * UL + (1.0 - frac)[:, None] * UR
    # 正弦波部分の密度を解析積分する。波と初期衝撃波が同じセルにあってもよい。
    lo, hi = np.maximum(edges[:-1], problem.x0), edges[1:]
    active = hi > lo
    integral_rho = np.zeros_like(dx)
    integral_rho[active] = (hi[active] - lo[active]) + 0.04 * (np.cos(5.0 * lo[active]) - np.cos(5.0 * hi[active]))
    U = frac[:, None] * UL
    U[:, 0] += integral_rho / dx
    U[:, 2] += (1.0 - frac) / (gamma - 1.0)
    return U

@lru_cache(maxsize=128)
def riemann_structure(left: tuple, right: tuple, gamma: float = 1.4):
    """理想気体の厳密リーマン解。真空を含まない問題を対象にpressureを二分法で解く。"""
    rL, uL, pL = left
    rR, uR, pR = right
    if min(rL, rR, pL, pR) <= 0.0 or gamma <= 1.0:
        raise ValueError("Exact solver requires positive input states and gamma > 1.")
    cL, cR = np.sqrt(gamma * pL / rL), np.sqrt(gamma * pR / rR)
    if uR - uL >= 2.0 * (cL + cR) / (gamma - 1.0):
        raise ValueError("This exact solver does not implement a physical vacuum.")

    def wave_function(p, rho, pk, ck):
        if p > pk:
            return (p - pk) * np.sqrt((2.0 / ((gamma + 1.0) * rho)) / (p + (gamma - 1.0) * pk / (gamma + 1.0)))
        return (2.0 * ck / (gamma - 1.0)) * ((p / pk)**((gamma - 1.0) / (2.0 * gamma)) - 1.0)

    def residual(p):
        return wave_function(p, rL, pL, cL) + wave_function(p, rR, pR, cR) + uR - uL

    lower, upper = 0.0, max(pL, pR)
    for _ in range(100):
        if residual(upper) >= 0.0:
            break
        upper *= 2.0
    else:
        raise RuntimeError("Cannot bracket the star pressure.")
    for _ in range(180):
        p = 0.5 * (lower + upper)
        if residual(p) > 0.0:
            upper = p
        else:
            lower = p
        if upper - lower <= 2.0e-13 * max(upper, 1.0e-300):
            break
    ps = 0.5 * (lower + upper)
    us = 0.5 * (uL + uR + wave_function(ps, rR, pR, cR) - wave_function(ps, rL, pL, cL))
    b = (gamma - 1.0) / (gamma + 1.0)
    rLs = rL * ((ps / pL + b) / (b * ps / pL + 1.0) if ps > pL else (ps / pL)**(1.0 / gamma))
    rRs = rR * ((ps / pR + b) / (b * ps / pR + 1.0) if ps > pR else (ps / pR)**(1.0 / gamma))
    if ps > pL:
        sl = uL - cL * np.sqrt((gamma + 1.0) * ps / (2.0 * gamma * pL) + (gamma - 1.0) / (2.0 * gamma))
        left_speeds = (sl, sl)
    else:
        left_speeds = (uL - cL, us - cL * (ps / pL)**((gamma - 1.0) / (2.0 * gamma)))
    if ps > pR:
        sr = uR + cR * np.sqrt((gamma + 1.0) * ps / (2.0 * gamma * pR) + (gamma - 1.0) / (2.0 * gamma))
        right_speeds = (sr, sr)
    else:
        right_speeds = (us + cR * (ps / pR)**((gamma - 1.0) / (2.0 * gamma)), uR + cR)
    return {"pstar": ps, "ustar": us, "rhoLstar": rLs, "rhoRstar": rRs, "left_speeds": left_speeds, "right_speeds": right_speeds, "residual": residual(ps)}


def riemann_primitive(x: np.ndarray, t: float, problem: Problem, gamma: float = 1.4):
    original_shape = np.asarray(x).shape
    x = np.asarray(x, dtype=float).ravel()
    left, right = np.array(problem.left), np.array(problem.right)
    if t <= 0.0:
        return np.where((x < problem.x0)[:, None], left, right).reshape(original_shape + (3,))
    data = riemann_structure(tuple(problem.left), tuple(problem.right), gamma)
    ps, us = data["pstar"], data["ustar"]
    rL, uL, pL = left
    rR, uR, pR = right
    cL, cR = np.sqrt(gamma * pL / rL), np.sqrt(gamma * pR / rR)
    xi = (x - problem.x0) / t
    is_left = xi <= us
    W = np.where(is_left[:, None], np.array((data["rhoLstar"], us, ps)), np.array((data["rhoRstar"], us, ps)))
    lh, lt = data["left_speeds"]
    rt, rh = data["right_speeds"]
    W[xi < lh] = left
    W[xi > rh] = right
    if ps <= pL:
        fan = is_left & (xi >= lh) & (xi <= lt)
        c = (2.0 / (gamma + 1.0)) * (cL + 0.5 * (gamma - 1.0) * (uL - xi[fan]))
        W[fan, 0] = rL * (c / cL)**(2.0 / (gamma - 1.0))
        W[fan, 1] = (2.0 / (gamma + 1.0)) * (cL + 0.5 * (gamma - 1.0) * uL + xi[fan])
        W[fan, 2] = pL * (c / cL)**(2.0 * gamma / (gamma - 1.0))
    if ps <= pR:
        fan = (~is_left) & (xi >= rt) & (xi <= rh)
        c = (2.0 / (gamma + 1.0)) * (cR - 0.5 * (gamma - 1.0) * (uR - xi[fan]))
        W[fan, 0] = rR * (c / cR)**(2.0 / (gamma - 1.0))
        W[fan, 1] = (2.0 / (gamma + 1.0)) * (-cR + 0.5 * (gamma - 1.0) * uR + xi[fan])
        W[fan, 2] = pR * (c / cR)**(2.0 * gamma / (gamma - 1.0))
    return W.reshape(original_shape + (3,))


def exact_pointwise(problem: Problem, x: np.ndarray, t: float, gamma: float = 1.4):
    x = np.asarray(x, dtype=float)
    if problem.key == "shu_osher":
        raise ValueError("No exact solution is supplied for Shu-Osher.")
    if problem.key in ("entropy", "contact", "contact_static"):
        coordinate = (x - problem.velocity * t) % 1.0
        rho = 1.0 + 0.2 * np.sin(2.0 * np.pi * coordinate) if problem.key == "entropy" else 1.0 + ((coordinate >= 0.25) & (coordinate < 0.45))
        return np.stack((rho, np.full_like(x, problem.velocity), np.ones_like(x)), axis=-1)
    return riemann_primitive(x, t, problem, gamma)


def exact_cell_average(problem: Problem, edges: np.ndarray, t: float, gamma: float = 1.4, nq: int = 12):
    """リーマン解は波面で区間を分割したGauss積分。保存変数のセル平均を返す。"""
    if problem.key in ("entropy", "contact", "contact_static"):
        return advected_cell_average(problem, edges, t, gamma)
    if problem.key == "shu_osher":
        raise ValueError("Use a numerical reference for Shu-Osher.")
    if t <= 0.0:
        return initial_cell_average(problem, edges, gamma)
    data = riemann_structure(tuple(problem.left), tuple(problem.right), gamma)
    speeds = np.array(data["left_speeds"] + (data["ustar"],) + data["right_speeds"])
    fronts = problem.x0 + t * speeds
    partitions = np.unique(np.r_[edges, fronts[(fronts > edges[0]) & (fronts < edges[-1])]])
    lo, hi = partitions[:-1], partitions[1:]
    nodes, weights = np.polynomial.legendre.leggauss(nq)
    points = 0.5 * (lo + hi)[:, None] + 0.5 * (hi - lo)[:, None] * nodes
    states = prim_to_cons(riemann_primitive(points, t, problem, gamma), gamma)
    integrals = 0.5 * (hi - lo)[:, None] * np.einsum("q,sqk->sk", weights, states)
    owners = np.searchsorted(edges, 0.5 * (lo + hi), side="right") - 1
    averaged = np.zeros((len(edges) - 1, 3))
    np.add.at(averaged, owners, integrals)
    return averaged / np.diff(edges)[:, None]

def minmod(a: np.ndarray, b: np.ndarray):
    same_sign = np.sign(a) == np.sign(b)
    return np.where(same_sign, np.sign(a) * np.minimum(np.abs(a), np.abs(b)), 0.0)


def ghost_cells(U: np.ndarray, problem: Problem, gamma: float):
    """左右2個ずつのゴーストセルを作る。境界条件は各RK段階で更新する。"""
    if problem.boundary == "periodic":
        return np.concatenate((U[-2:], U, U[:2]), axis=0)
    if problem.boundary == "outflow":
        return np.concatenate((np.repeat(U[:1], 2, axis=0), U, np.repeat(U[-1:], 2, axis=0)), axis=0)
    if problem.boundary == "inflow_outflow":
        UL = prim_to_cons(np.array(problem.left), gamma)
        return np.concatenate((np.repeat(UL[None, :], 2, axis=0), U, np.repeat(U[-1:], 2, axis=0)), axis=0)
    raise ValueError(f"Unknown boundary condition: {problem.boundary}")


def interface_states(U: np.ndarray, problem: Problem, gamma: float, order: int, limiter: str):
    Ug = ghost_cells(U, problem, gamma)
    if order == 1:
        return Ug[1:-2].copy(), Ug[2:-1].copy()
    Wg = cons_to_prim(Ug, gamma, stage="MUSCL cell state")
    dm, dp = Wg[1:-1] - Wg[:-2], Wg[2:] - Wg[1:-1]
    slopes = np.zeros_like(Wg)
    if limiter == "minmod":
        slopes[1:-1] = minmod(dm, dp)
    elif limiter == "mc":
        slopes[1:-1] = minmod(0.5 * (dm + dp), minmod(2.0 * dm, 2.0 * dp))
    elif limiter == "unlimited":
        slopes[1:-1] = 0.5 * (dm + dp)
    else:
        raise ValueError("limiter must be 'minmod', 'mc', or 'unlimited'.")
    WL, WR = Wg[1:-2] + 0.5 * slopes[1:-2], Wg[2:-1] - 0.5 * slopes[2:-1]
    for state in (WL, WR):
        if not np.all(np.isfinite(state)) or np.any(state[:, 0] <= 0.0) or np.any(state[:, 2] <= 0.0):
            raise NumericalFailure("MUSCL reconstruction", detail="A reconstructed rho or p is not positive.")
    return prim_to_cons(WL, gamma), prim_to_cons(WR, gamma)


def advance(U, problem: Problem, solver: str, dt: float, dx: float, gamma: float,
            order: int, limiter: str, wave_speeds: str, roe_entropy: float):
    """更新後の保存変数と、同じ数値流束で積算した領域への正味流入量を返す。"""
    UL, UR = interface_states(U, problem, gamma, order, limiter)
    F1 = numerical_flux(UL, UR, solver, gamma, dt, dx, wave_speeds, roe_entropy)
    U1 = U - (dt / dx) * (F1[1:] - F1[:-1])
    cons_to_prim(U1, gamma, stage="RK stage 1" if order == 2 else "full-step update")
    if order == 1:
        return U1, dt * (F1[0] - F1[-1])
    UL, UR = interface_states(U1, problem, gamma, order, limiter)
    F2 = numerical_flux(UL, UR, solver, gamma, dt, dx, wave_speeds, roe_entropy)
    Unew = 0.5 * U + 0.5 * (U1 - (dt / dx) * (F2[1:] - F2[:-1]))
    cons_to_prim(Unew, gamma, stage="RK stage 2")
    exchange = 0.5 * dt * ((F1[0] - F1[-1]) + (F2[0] - F2[-1]))
    return Unew, exchange

@dataclass
class RunResult:
    problem: Problem
    solver: str
    n: int
    order: int
    gamma: float
    cfl: float
    limiter: str
    wave_speeds: str
    roe_entropy: float
    edges: np.ndarray
    x: np.ndarray
    U0: np.ndarray
    U: np.ndarray
    t: float
    target_time: float
    steps: int
    elapsed: float
    status: str
    initial_total: np.ndarray
    boundary_exchange: np.ndarray
    balance: np.ndarray
    history: np.ndarray
    fail_time: float = np.nan
    fail_dt: float = np.nan
    fail_min_rho: float = np.nan
    fail_min_p: float = np.nan
    fail_reason: str = ""

    @property
    def label(self):
        name = "HLL (HLLE)" if self.solver == "HLL" and self.wave_speeds == "einfeldt" else self.solver
        if self.solver == "LW":
            return "LW (Richtmyer)"
        if self.order == 2:
            return name + " + MUSCL"
        return name


def simulate(problem: Problem, solver: str, n: int = 200, gamma: float = 1.4,
             cfl: float = 0.4, order: int = 1, limiter: str = "minmod",
             wave_speeds: str = "einfeldt", roe_entropy: float = 0.1,
             t_end: Optional[float] = None, max_steps: int = 200000, history_stride: int = 10, observer=None):
    """失敗した試行は棄却して停止する。dt再試行・floor・他ソルバーへの切替は行わない。"""
    if solver not in ALL_SOLVERS or order not in (1, 2):
        raise ValueError("Invalid solver or order.")
    if order == 2 and solver not in RIEMANN_SOLVERS:
        raise ValueError("MUSCL+SSP-RK2 is used only with Roe, LLF, HLL, and HLLC.")
    if not (0.0 < cfl <= 0.9) or gamma <= 1.0 or roe_entropy < 0.0 or history_stride < 1:
        raise ValueError("Require 0 < CFL <= 0.9, gamma > 1, entropy parameter >= 0 and history_stride >= 1.")
    target = problem.final_time if t_end is None else float(t_end)
    if target < 0.0:
        raise ValueError("t_end must be nonnegative.")
    edges, x, dx = grid_for(problem, n)
    U0 = initial_cell_average(problem, edges, gamma)
    U = U0.copy()
    initial_total = dx * np.sum(U0, axis=0)
    exchanged = np.zeros(3)
    W0 = cons_to_prim(U0, gamma)
    history = [[0.0, float(W0[:, 0].min()), float(W0[:, 2].min()), 0.0]]
    t, steps, status = 0.0, 0, "ok"
    failure = {}
    start = time.perf_counter()
    if observer is not None:
        observer(t, U)  # 表示用の記録だけを行う。Uを変更しない。
    while t < target:
        if steps >= max_steps:
            status = "max_steps"
            failure["fail_reason"] = "Maximum step count reached."
            break
        dt_cfl = cfl * dx / max_signal_speed(U, gamma)
        remaining = target - t
        # LFでは極端に小さい最終dtが過剰な拡散を生むため、最後の2ステップを均等にする。
        dt = dt_cfl if remaining > 2.0 * dt_cfl else remaining / max(1, int(np.ceil(remaining / dt_cfl)))
        if not np.isfinite(dt) or dt <= 0.0 or t + dt == t:
            status = "failed"
            failure = {"fail_reason": "The time step cannot advance floating-point time.", "fail_time": t, "fail_dt": dt}
            break
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                trial, net_inflow = advance(U, problem, solver, dt, dx, gamma, order, limiter, wave_speeds, roe_entropy)
        except NumericalFailure as error:
            status = "failed"
            failure = {"fail_time": t + dt, "fail_dt": dt, "fail_min_rho": error.min_rho, "fail_min_p": error.min_p, "fail_reason": str(error)}
            break
        except FloatingPointError as error:
            status = "failed"
            failure = {"fail_time": t + dt, "fail_dt": dt, "fail_reason": "Floating-point failure: " + str(error)}
            break
        U = trial
        exchanged += net_inflow
        t = min(target, t + dt)
        steps += 1
        if observer is not None:
            observer(t, U)
        if steps % history_stride == 0 or t == target:
            W = cons_to_prim(U, gamma)
            balance = dx * np.sum(U, axis=0) - initial_total - exchanged
            history.append([t, float(W[:, 0].min()), float(W[:, 2].min()), float(np.max(np.abs(balance)))])
    elapsed = time.perf_counter() - start
    balance = dx * np.sum(U, axis=0) - initial_total - exchanged
    if history[-1][0] != t:
        W = cons_to_prim(U, gamma)
        history.append([t, float(W[:, 0].min()), float(W[:, 2].min()), float(np.max(np.abs(balance)))])
    return RunResult(problem, solver, n, order, gamma, cfl, limiter, wave_speeds, roe_entropy, edges, x, U0, U, t, target, steps, elapsed, status, initial_total, exchanged, balance, np.asarray(history), **failure)

def project_reference(reference: RunResult, edges: np.ndarray) -> np.ndarray:
    """同じ一様格子領域で整数比の細格子の保存変数を粗格子へ体積平均する。"""
    n = len(edges) - 1
    if reference.status != "ok" or reference.n % n or not np.allclose(reference.edges[[0, -1]], edges[[0, -1]], rtol=0.0, atol=1.0e-13):
        raise ValueError("The completed reference must cover the same domain with an integer grid ratio.")
    return reference.U.reshape(n, reference.n // n, 3).mean(axis=1)


def metrics(result: RunResult, reference: Optional[RunResult] = None) -> dict:
    W = cons_to_prim(result.U, result.gamma)
    row = {"problem": result.problem.key, "solver": result.solver, "order": 2 if result.solver == "LW" else result.order,
           "n": result.n, "cfl": result.cfl, "limiter": result.limiter if result.order == 2 else "constant",
           "wave_speeds": result.wave_speeds, "roe_entropy": result.roe_entropy,
           "status": result.status, "t_reached": result.t, "t_target": result.target_time, "steps": result.steps,
           "seconds": result.elapsed, "rho_min": float(W[:, 0].min()), "p_min": float(W[:, 2].min()),
           "balance_max": float(np.max(np.abs(result.balance))), "mass_balance": result.balance[0],
           "momentum_balance": result.balance[1], "energy_balance": result.balance[2],
           "L1_rho": np.nan, "L1_u": np.nan, "L1_p": np.nan, "amp_ratio": np.nan, "phase_rad": np.nan,
           "rho_overshoot": np.nan, "rho_undershoot": np.nan, "mixed_length": np.nan,
           "failure_trial_end": result.fail_time, "failure_dt": result.fail_dt, "failure_rho_min": result.fail_min_rho,
           "failure_p_min": result.fail_min_p, "failure_reason": result.fail_reason}
    if result.status != "ok":
        return row
    Uexact = None
    if result.problem.key != "shu_osher":
        Uexact = exact_cell_average(result.problem, result.edges, result.t, result.gamma)
    elif reference is not None:
        if abs(reference.t - result.t) > 1.0e-12 or reference.gamma != result.gamma:
            raise ValueError("Reference and numerical solutions must have the same time and gamma.")
        Uexact = project_reference(reference, result.edges)
    if Uexact is not None:
        Wexact = cons_to_prim(Uexact, result.gamma)
        error = np.mean(np.abs(W - Wexact), axis=0)
        row.update(dict(zip(("L1_rho", "L1_u", "L1_p"), error)))
    if result.problem.key == "entropy":
        phase = np.exp(-2j * np.pi * result.x)
        coefficient = np.sum((W[:, 0] - W[:, 0].mean()) * phase)
        exact_coefficient = np.sum((Wexact[:, 0] - Wexact[:, 0].mean()) * phase)
        ratio = coefficient / exact_coefficient
        row["amp_ratio"], row["phase_rad"] = float(abs(ratio)), float(np.angle(ratio))
    if result.problem.key in ("contact", "contact_static"):
        row["rho_overshoot"] = float(max(0.0, W[:, 0].max() - 2.0))
        row["rho_undershoot"] = float(max(0.0, 1.0 - W[:, 0].min()))
        row["mixed_length"] = float(np.sum((W[:, 0] > 1.1) & (W[:, 0] < 1.9)) * (result.edges[1] - result.edges[0]))
    return row


def shock_location_error(result: RunResult):
    """右向き衝撃波の近傍で最大密度勾配の位置を比較する簡易指標。"""
    if result.status != "ok" or result.problem.right is None:
        return np.nan
    data = riemann_structure(tuple(result.problem.left), tuple(result.problem.right), result.gamma)
    if data["pstar"] <= result.problem.right[2]:
        return np.nan
    expected = result.problem.x0 + result.t * data["right_speeds"][0]
    rho = result.U[:, 0]
    face_x = 0.5 * (result.x[:-1] + result.x[1:])
    dx = result.edges[1] - result.edges[0]
    window = np.abs(face_x - expected) < max(0.06 * (result.edges[-1] - result.edges[0]), 4.0 * dx)
    indices = np.flatnonzero(window)
    if not len(indices):
        return np.nan
    index = indices[np.argmax(np.abs(np.diff(rho)[indices]))]
    return float(face_x[index] - expected)

def run_unit_tests(gamma: float = 1.4):
    """数値流束、厳密解、境界の保存則を小さい計算で検証する。失敗時はassertで止まる。"""
    tests = []
    rng = np.random.default_rng(7341)
    W = np.column_stack((np.exp(rng.uniform(-1.0, 1.0, 31)), rng.uniform(-0.8, 0.8, 31), np.exp(rng.uniform(-1.0, 1.0, 31))))
    U = prim_to_cons(W, gamma)
    np.testing.assert_allclose(cons_to_prim(U, gamma), W, rtol=1.0e-13, atol=1.0e-13)
    tests.append("primitive/conserved round trip")
    for solver in ALL_SOLVERS:
        F = numerical_flux(U, U, solver, gamma, dt=0.01, dx=0.1)
        np.testing.assert_allclose(F, physical_flux(U, gamma), rtol=2.0e-13, atol=2.0e-13)
    tests.append("F*(U,U) = F(U), all six solvers")
    WL, WR = W[:15], W[15:30]
    UL, UR = prim_to_cons(WL, gamma), prim_to_cons(WR, gamma)
    Fhll = numerical_flux(UL, UR, "HLL", gamma, wave_speeds="symmetric")
    Fllf = numerical_flux(UL, UR, "LLF", gamma)
    np.testing.assert_allclose(Fhll, Fllf, rtol=2.0e-13, atol=2.0e-13)
    tests.append("symmetric-speed HLL equals LLF")
    for solver in ("Roe", "HLLC", "LW"):
        run = simulate(PROBLEMS["contact_static"], solver, n=40, gamma=gamma, t_end=0.02)
        assert run.status == "ok"
        np.testing.assert_allclose(run.U, run.U0, atol=3.0e-13, rtol=3.0e-13)
    tests.append("stationary contact: Roe, HLLC, and Richtmyer LW")
    for solver in ALL_SOLVERS:
        run = simulate(PROBLEMS["entropy"], solver, n=40, gamma=gamma, t_end=0.03)
        assert run.status == "ok"
        assert np.max(np.abs(run.balance)) < 1.0e-11
    tests.append("periodic conservation, all six solvers")
    for solver in RIEMANN_SOLVERS:
        run = simulate(PROBLEMS["entropy"], solver, n=40, gamma=gamma, order=2, t_end=0.03)
        assert run.status == "ok"
        assert np.max(np.abs(run.balance)) < 1.0e-11
    tests.append("MUSCL/SSP-RK2 periodic conservation")
    if abs(gamma - 1.4) < 1.0e-14:
        problem = PROBLEMS["sod"]
        data = riemann_structure(tuple(problem.left), tuple(problem.right), gamma)
        np.testing.assert_allclose((data["pstar"], data["ustar"]), (0.303130178050647, 0.927452620048950), rtol=2.0e-11)
        tests.append("Sod star pressure and velocity")
    for key in ("sod", "double_rarefaction", "lax", "transonic"):
        problem = PROBLEMS[key]
        edges, _, _ = grid_for(problem, 50)
        ref12 = exact_cell_average(problem, edges, problem.final_time, gamma, nq=12)
        ref24 = exact_cell_average(problem, edges, problem.final_time, gamma, nq=24)
        np.testing.assert_allclose(ref12, ref24, rtol=2.0e-10, atol=2.0e-12)
        cons_to_prim(ref12, gamma)
    tests.append("exact-solution quadrature and positive reference states")
    # 流入・流出がある場合に、生の総量変化ではなく境界流束を引いた残差を見る。
    run = simulate(PROBLEMS["shu_osher"], "HLLC", n=40, gamma=gamma, order=2, t_end=0.03)
    assert run.status == "ok" and np.max(np.abs(run.balance)) < 1.0e-10
    assert np.max(np.abs(run.boundary_exchange)) > 1.0e-6
    tests.append("nonperiodic boundary-flux budget")
    return tests