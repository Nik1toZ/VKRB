from __future__ import annotations
import time
import inspect
import numpy as np
import torch
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


_EPS_ZERO = 1e-10


def thomas(a, b, c, d):

    n = len(d)
    cp = np.empty(n - 1)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n - 1):
        m = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / m
    dp[n - 1] = ((d[n - 1] - a[n - 2] * dp[n - 2])
                 / (b[n - 1] - a[n - 2] * cp[n - 2]))
    x = np.empty(n)
    x[n - 1] = dp[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


def errors(u_pred, u_exact):

    u_pred = np.asarray(u_pred).ravel()
    u_exact = np.asarray(u_exact).ravel()
    valid = np.isfinite(u_pred) & np.isfinite(u_exact)
    u_pred = u_pred[valid]
    u_exact = u_exact[valid]
    diff = u_pred - u_exact
    abs_max = float(np.max(np.abs(diff))) if diff.size else 0.0
    abs_l2 = float(np.linalg.norm(diff))
    norm_exact = float(np.linalg.norm(u_exact))
    rel_defined = norm_exact >= _EPS_ZERO
    rel_l2 = abs_l2 / norm_exact if rel_defined else abs_max
    return {
        'rel_l2': rel_l2, 'abs_max': abs_max, 'abs_l2': abs_l2,
        'norm_exact': norm_exact, 'rel_defined': rel_defined,
    }


def err_summary(u_pred, u_exact, indent='  '):

    e = errors(u_pred, u_exact)
    if e['rel_defined']:
        return (f"{indent}Rel L2: {e['rel_l2']:.3e}   "
                f"Max |Δ|: {e['abs_max']:.3e}")
    return (f"{indent}Max |Δ|: {e['abs_max']:.3e}   "
            f"(Rel L2 не определена)")


class Boundary:


    def __init__(self, kind, alpha, beta, gamma, order=2):
        assert kind in ('dirichlet', 'neumann', 'robin')
        self.kind = kind
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = gamma
        self.order = order

    def __repr__(self):
        return (f'Boundary({self.kind}, α={self.alpha}, β={self.beta}, '
                f'order={self.order})')


def boundary_from_pinn_cond(cond, side_var, *, order=2):

    where = cond.get('where', {})

    if 't' in where:
        return None
    ctype = cond.get('type')

    if ctype is None:
        if 'deriv_var' in cond:
            ctype = 'neumann'
        else:
            ctype = 'dirichlet'
    val = cond['value_numpy']
    if not callable(val):
        const = float(val)
        gamma = lambda *args, _c=const: np.full_like(np.asarray(args[0],
                                                                dtype=float),
                                                    _c) if len(args) > 0                                       else _c
    else:
        gamma = val

    if ctype == 'dirichlet':
        return Boundary('dirichlet', alpha=1.0, beta=0.0, gamma=gamma,
                        order=order)
    elif ctype == 'neumann':
        return Boundary('neumann', alpha=0.0, beta=1.0, gamma=gamma,
                        order=order)
    elif ctype == 'robin':
        return Boundary('robin', alpha=cond['alpha'], beta=cond['beta'],
                        gamma=gamma, order=order)
    else:
        raise ValueError(f'Неизвестный тип ГУ: {ctype}')


def _build_interior_stencil_1d(nx, h, a, b, c, use_upwind=False, b_sign_for_upwind=None):

    a_arr = np.zeros(nx - 1)
    b_arr = np.zeros(nx)
    c_arr = np.zeros(nx - 1)

    coef_xx = a / h**2

    if not use_upwind:
        coef_x_lo = -b / (2 * h)
        coef_x_hi = +b / (2 * h)
    else:

        if b >= 0:
            coef_x_lo = -b / h
            coef_x_hi = 0.0
        else:
            coef_x_lo = 0.0
            coef_x_hi = +b / h


    for i in range(1, nx - 1):
        a_arr[i - 1] = coef_xx + coef_x_lo
        c_arr[i] = coef_xx + coef_x_hi
        if not use_upwind:
            b_arr[i] = -2.0 * coef_xx + c
        else:
            if b >= 0:
                b_arr[i] = -2.0 * coef_xx + b / h + c
            else:
                b_arr[i] = -2.0 * coef_xx - b / h + c
    return a_arr, b_arr, c_arr


def _apply_boundary_to_row(a_arr, b_arr, c_arr, d_val, side, bnd, h, t,
                           u_inner_extra=None):

    alpha, beta = bnd.alpha, bnd.beta
    gamma_val = bnd.gamma(t) if callable(bnd.gamma) else bnd.gamma
    if isinstance(gamma_val, np.ndarray):
        gamma_val = float(gamma_val.item()) if gamma_val.size == 1                    else float(gamma_val.ravel()[0])

    if bnd.kind == 'dirichlet':
        if side == 'left':
            return (None, 1.0, 0.0, gamma_val, None)
        else:
            return (0.0, 1.0, None, gamma_val, None)


    if bnd.order == 1:


        if side == 'left':


            return (None, alpha - beta / h, beta / h, gamma_val, None)
        else:


            return (-beta / h, alpha + beta / h, None, gamma_val, None)
    else:


        if side == 'left':
            return (None,
                    alpha - 3.0 * beta / (2 * h),
                    4.0 * beta / (2 * h),
                    gamma_val,
                    ('left_extra', -beta / (2 * h)))
        else:
            return (-4.0 * beta / (2 * h),
                    alpha + 3.0 * beta / (2 * h),
                    None,
                    gamma_val,
                    ('right_extra', beta / (2 * h)))


def solve_parabolic_1d(
    *,
    domain,
    a, b=0.0, c=0.0,
    source=None,
    initial,
    boundaries,
    nx=101, nt=2001,
    scheme='crank_nicolson',
    bc_order=2,
    use_upwind=None,
    verbose=False,
):

    assert scheme in ('explicit', 'implicit', 'crank_nicolson')
    x_lo, x_hi = domain['x']
    t_lo, t_hi = domain['t']
    x = np.linspace(x_lo, x_hi, nx)
    t = np.linspace(t_lo, t_hi, nt)
    h = x[1] - x[0]
    tau = t[1] - t[0]


    if use_upwind is None:
        use_upwind = (abs(b) * h / (2.0 * max(a, 1e-12))) > 1.0
    if verbose:
        print(f'  МКР-параболика: nx={nx}, nt={nt}, h={h:.4g}, '
              f'τ={tau:.4g}, схема={scheme}, upwind={use_upwind}')


    U = np.zeros((nt, nx))
    U[0, :] = initial(x)


    bL, bR = boundaries
    if bL.kind == 'dirichlet':
        gv = bL.gamma(t[0])
        U[0, 0] = float(gv) if np.ndim(gv) == 0 else float(np.asarray(gv).ravel()[0])
    if bR.kind == 'dirichlet':
        gv = bR.gamma(t[0])
        U[0, -1] = float(gv) if np.ndim(gv) == 0 else float(np.asarray(gv).ravel()[0])

    if scheme == 'explicit':
        return _solve_parabolic_1d_explicit(
            x, t, U, a, b, c, source, boundaries, h, tau, use_upwind, bc_order)
    elif scheme == 'implicit':
        return _solve_parabolic_1d_theta(
            x, t, U, a, b, c, source, boundaries, h, tau,
            use_upwind, bc_order, theta=1.0)
    else:
        return _solve_parabolic_1d_theta(
            x, t, U, a, b, c, source, boundaries, h, tau,
            use_upwind, bc_order, theta=0.5)


def _solve_parabolic_1d_explicit(x, t, U, a, b, c, source, boundaries,
                                 h, tau, use_upwind, bc_order):
    nx, nt = len(x), len(t)

    r_diff = a * tau / h**2
    if r_diff > 0.5 + 1e-12:
        print(f'  ⚠ Явная схема: r_diff = {r_diff:.3f} > 0.5, '
              f'возможна неустойчивость')
    bL, bR = boundaries
    for k in range(nt - 1):
        u_old = U[k, :]
        u_new = u_old.copy()
        if not use_upwind:

            u_new[1:-1] = (u_old[1:-1]
                           + tau * (a * (u_old[2:] - 2*u_old[1:-1] + u_old[:-2]) / h**2
                                    + b * (u_old[2:] - u_old[:-2]) / (2*h)
                                    + c * u_old[1:-1]))
        else:

            if b >= 0:
                conv = b * (u_old[1:-1] - u_old[:-2]) / h
            else:
                conv = b * (u_old[2:] - u_old[1:-1]) / h
            u_new[1:-1] = (u_old[1:-1]
                           + tau * (a * (u_old[2:] - 2*u_old[1:-1] + u_old[:-2]) / h**2
                                    + conv + c * u_old[1:-1]))

        if source is not None:
            u_new[1:-1] = u_new[1:-1] + tau * source(x[1:-1], t[k])

        u_new = _apply_explicit_bc(u_new, boundaries, h, t[k+1])
        U[k+1, :] = u_new
    return x, t, U


def _apply_explicit_bc(u, boundaries, h, t_now):

    bL, bR = boundaries
    nx = len(u)

    if bL.kind == 'dirichlet':
        u[0] = float(bL.gamma(t_now))
    else:

        gam = float(bL.gamma(t_now))
        if bL.order == 1:

            denom = bL.alpha - bL.beta / h
            u[0] = (gam - bL.beta * u[1] / h) / denom                if abs(denom) > 1e-14 else u[1]
        else:


            denom = bL.alpha - 3.0 * bL.beta / (2*h)
            rhs = gam - bL.beta * (4*u[1] - u[2]) / (2*h)
            u[0] = rhs / denom if abs(denom) > 1e-14 else u[1]

    if bR.kind == 'dirichlet':
        u[-1] = float(bR.gamma(t_now))
    else:
        gam = float(bR.gamma(t_now))
        if bR.order == 1:
            denom = bR.alpha + bR.beta / h
            u[-1] = (gam + bR.beta * u[-2] / h) / denom                if abs(denom) > 1e-14 else u[-2]
        else:
            denom = bR.alpha + 3.0 * bR.beta / (2*h)
            rhs = gam + bR.beta * (4*u[-2] - u[-3]) / (2*h)
            u[-1] = rhs / denom if abs(denom) > 1e-14 else u[-2]
    return u


def _solve_parabolic_1d_theta(x, t, U, a, b, c, source, boundaries,
                              h, tau, use_upwind, bc_order, theta):

    nx, nt = len(x), len(t)

    a_int, b_int, c_int = _build_interior_stencil_1d(
        nx, h, a, b, c, use_upwind=use_upwind)


    def build_matrix(theta_, sign):

        coef = sign * theta_ * tau if sign == -1 else sign * (1 - theta_) * tau

        diag_main = np.ones(nx) + coef * b_int
        diag_lo = coef * a_int
        diag_hi = coef * c_int

        return diag_main, diag_lo, diag_hi

    bL, bR = boundaries

    extra_left = None
    extra_right = None

    for k in range(nt - 1):

        main_L, lo_L, hi_L = build_matrix(theta, -1)

        main_R, lo_R, hi_R = build_matrix(theta, +1)

        u_old = U[k, :]
        rhs = np.zeros(nx)
        rhs[1:-1] = (main_R[1:-1] * u_old[1:-1]
                     + lo_R[:-1] * u_old[:-2]
                     + hi_R[1:] * u_old[2:])
        if source is not None:
            t_eval = t[k] + theta * tau
            rhs[1:-1] += tau * source(x[1:-1], t_eval)


        bcL_t = t[k+1]
        a_val, b_val, c_val, d_val, extra = _apply_boundary_to_row(
            None, None, None, None, 'left', bL, h, bcL_t)
        main_L[0] = b_val
        hi_L[0] = c_val
        rhs[0] = d_val
        extra_left = extra


        bcR_t = t[k+1]
        a_val, b_val, c_val, d_val, extra = _apply_boundary_to_row(
            None, None, None, None, 'right', bR, h, bcR_t)
        lo_L[-1] = a_val
        main_L[-1] = b_val
        rhs[-1] = d_val
        extra_right = extra


        if extra_left is not None or extra_right is not None:

            mat = sp.diags([lo_L, main_L, hi_L], offsets=[-1, 0, 1],
                           format='lil')
            if extra_left is not None:

                mat[0, 2] = extra_left[1]
            if extra_right is not None:

                mat[nx - 1, nx - 3] = extra_right[1]
            U[k+1, :] = spla.spsolve(mat.tocsc(), rhs)
        else:

            U[k+1, :] = thomas(lo_L, main_L, hi_L, rhs)
    return x, t, U


def interpolate_fdm_to_grid(fdm_x, fdm_t, fdm_U, eval_x, eval_t):

    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((fdm_t, fdm_x), fdm_U,
                                     method='linear', bounds_error=False,
                                     fill_value=None)
    Xg, Tg = np.meshgrid(eval_x, eval_t, indexing='ij')
    pts = np.stack([Tg.ravel(), Xg.ravel()], axis=-1)
    return interp(pts).reshape(len(eval_x), len(eval_t))


def interpolate_fdm_2d_to_grid(fdm_x, fdm_y, fdm_U, eval_x, eval_y):

    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((fdm_x, fdm_y), fdm_U,
                                     method='linear', bounds_error=False,
                                     fill_value=None)
    Xg, Yg = np.meshgrid(eval_x, eval_y, indexing='ij')
    pts = np.stack([Xg.ravel(), Yg.ravel()], axis=-1)
    return interp(pts).reshape(len(eval_x), len(eval_y))


def interpolate_fdm_3d_to_grid(fdm_x, fdm_y, fdm_t, fdm_U,
                                eval_x, eval_y, eval_t):

    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((fdm_t, fdm_x, fdm_y), fdm_U,
                                     method='linear', bounds_error=False,
                                     fill_value=None)
    nx_e, ny_e, nt_e = len(eval_x), len(eval_y), len(eval_t)
    Xg, Yg, Tg = np.meshgrid(eval_x, eval_y, eval_t, indexing='ij')
    pts = np.stack([Tg.ravel(), Xg.ravel(), Yg.ravel()], axis=-1)
    return interp(pts).reshape(nx_e, ny_e, nt_e)


def solve_hyperbolic_1d(
    *,
    domain,
    a, d=0.0, b=0.0, c=0.0,
    source=None,
    initial_u,
    initial_ut,
    boundaries,
    nx=201, nt=4001,
    bc_order=2,
    second_ic_order=2,
    verbose=False,
):

    x_lo, x_hi = domain['x']
    t_lo, t_hi = domain['t']
    x = np.linspace(x_lo, x_hi, nx)
    t = np.linspace(t_lo, t_hi, nt)
    h = x[1] - x[0]
    tau = t[1] - t[0]


    cfl = a * tau / h
    if cfl > 1.0 + 1e-12:
        print(f'  ⚠ CFL = a·τ/h = {cfl:.3f} > 1: возможна неустойчивость')
    if verbose:
        print(f'  МКР-гиперболика: nx={nx}, nt={nt}, h={h:.4g}, τ={tau:.4g}, '
              f'CFL={cfl:.3f}, d={d}, b={b}, c={c}')

    U = np.zeros((nt, nx))
    U[0, :] = initial_u(x)


    if second_ic_order >= 2:


        u0 = U[0, :]
        u_xx0 = np.zeros_like(u0)
        u_xx0[1:-1] = (u0[2:] - 2*u0[1:-1] + u0[:-2]) / h**2
        u_xx0[0] = u_xx0[1]
        u_xx0[-1] = u_xx0[-2]
        u_x0 = np.zeros_like(u0)
        u_x0[1:-1] = (u0[2:] - u0[:-2]) / (2*h)
        u_x0[0] = (u0[1] - u0[0]) / h
        u_x0[-1] = (u0[-1] - u0[-2]) / h
        ut0 = initial_ut(x)
        f0 = source(x, 0.0) if source is not None else np.zeros_like(x)
        u_tt0 = a**2 * u_xx0 + b * u_x0 + c * u0 + f0 - d * ut0
        U[1, :] = u0 + tau * ut0 + 0.5 * tau**2 * u_tt0
    else:

        U[1, :] = U[0, :] + tau * initial_ut(x)


    bL, bR = boundaries
    U[0, :] = _apply_explicit_bc(U[0, :], boundaries, h, t[0])
    U[1, :] = _apply_explicit_bc(U[1, :], boundaries, h, t[1])


    coef_lhs = 1.0 + d * tau / 2.0
    coef_uprev = 1.0 - d * tau / 2.0
    for k in range(1, nt - 1):
        u_n = U[k, :]
        u_p = U[k-1, :]

        u_xx = (u_n[2:] - 2*u_n[1:-1] + u_n[:-2]) / h**2
        u_x = (u_n[2:] - u_n[:-2]) / (2*h)
        Lu = a**2 * u_xx + b * u_x + c * u_n[1:-1]
        f = source(x[1:-1], t[k]) if source is not None else 0.0
        u_new_inner = (2 * u_n[1:-1] - coef_uprev * u_p[1:-1]
                        + tau**2 * (Lu + f)) / coef_lhs
        u_new = np.empty(nx)
        u_new[1:-1] = u_new_inner

        u_new[0] = 0.0
        u_new[-1] = 0.0
        u_new = _apply_explicit_bc(u_new, boundaries, h, t[k+1])
        U[k+1, :] = u_new
    return x, t, U


def _idx(i, j, ny):
    return i * ny + j


def solve_elliptic_2d(
    *,
    domain,
    a=1.0, bx=0.0, by=0.0, c=0.0,
    source=None,
    boundaries,
    nx=51, ny=51,
    bc_order=2,
    verbose=False,
):

    x_lo, x_hi = domain['x']
    y_lo, y_hi = domain['y']
    x = np.linspace(x_lo, x_hi, nx)
    y = np.linspace(y_lo, y_hi, ny)
    hx = x[1] - x[0]
    hy = y[1] - y[0]

    if verbose:
        print(f'  МКР-эллиптика: nx={nx}, ny={ny}, hx={hx:.4g}, hy={hy:.4g}')

    N = nx * ny
    A = sp.lil_matrix((N, N))
    rhs = np.zeros(N)


    for i in range(1, nx - 1):
        for j in range(1, ny - 1):
            k = _idx(i, j, ny)
            A[k, _idx(i-1, j, ny)] += a / hx**2
            A[k, _idx(i+1, j, ny)] += a / hx**2
            A[k, _idx(i, j-1, ny)] += a / hy**2
            A[k, _idx(i, j+1, ny)] += a / hy**2
            A[k, k] += -2.0 * a / hx**2 - 2.0 * a / hy**2
            if bx != 0.0:
                A[k, _idx(i+1, j, ny)] += bx / (2 * hx)
                A[k, _idx(i-1, j, ny)] += -bx / (2 * hx)
            if by != 0.0:
                A[k, _idx(i, j+1, ny)] += by / (2 * hy)
                A[k, _idx(i, j-1, ny)] += -by / (2 * hy)
            A[k, k] += c
            if source is not None:
                rhs[k] = source(x[i], y[j])

    bL = boundaries['left']
    bR = boundaries['right']
    bB = boundaries['bottom']
    bT = boundaries['top']

    def _set_row_x(row_k, side, bnd, coord_along_val):
        gam = bnd.gamma(coord_along_val) if callable(bnd.gamma) else bnd.gamma
        if isinstance(gam, np.ndarray) and gam.size == 1:
            gam = float(gam.ravel()[0])
        if bnd.kind == 'dirichlet':
            A[row_k, :] = 0
            A[row_k, row_k] = 1.0
            rhs[row_k] = float(gam)
            return
        alpha, beta = bnd.alpha, bnd.beta
        if side == 'left':
            i = 0
            j = row_k - i * ny
            A[row_k, :] = 0
            if bnd.order == 1:
                A[row_k, _idx(0, j, ny)] = alpha - beta / hx
                A[row_k, _idx(1, j, ny)] = beta / hx
            else:
                A[row_k, _idx(0, j, ny)] = alpha - 3.0 * beta / (2*hx)
                A[row_k, _idx(1, j, ny)] = 4.0 * beta / (2*hx)
                A[row_k, _idx(2, j, ny)] = -beta / (2*hx)
            rhs[row_k] = float(gam)
        else:
            i = nx - 1
            j = row_k - i * ny
            A[row_k, :] = 0
            if bnd.order == 1:
                A[row_k, _idx(nx-2, j, ny)] = -beta / hx
                A[row_k, _idx(nx-1, j, ny)] = alpha + beta / hx
            else:
                A[row_k, _idx(nx-3, j, ny)] = beta / (2*hx)
                A[row_k, _idx(nx-2, j, ny)] = -4.0 * beta / (2*hx)
                A[row_k, _idx(nx-1, j, ny)] = alpha + 3.0 * beta / (2*hx)
            rhs[row_k] = float(gam)

    def _set_row_y(row_k, side, bnd, coord_along_val):
        gam = bnd.gamma(coord_along_val) if callable(bnd.gamma) else bnd.gamma
        if isinstance(gam, np.ndarray) and gam.size == 1:
            gam = float(gam.ravel()[0])
        if bnd.kind == 'dirichlet':
            A[row_k, :] = 0
            A[row_k, row_k] = 1.0
            rhs[row_k] = float(gam)
            return
        alpha, beta = bnd.alpha, bnd.beta
        if side == 'bottom':
            j = 0
            i = (row_k - j) // ny
            A[row_k, :] = 0
            if bnd.order == 1:
                A[row_k, _idx(i, 0, ny)] = alpha - beta / hy
                A[row_k, _idx(i, 1, ny)] = beta / hy
            else:
                A[row_k, _idx(i, 0, ny)] = alpha - 3.0 * beta / (2*hy)
                A[row_k, _idx(i, 1, ny)] = 4.0 * beta / (2*hy)
                A[row_k, _idx(i, 2, ny)] = -beta / (2*hy)
            rhs[row_k] = float(gam)
        else:
            j = ny - 1
            i = (row_k - j) // ny
            A[row_k, :] = 0
            if bnd.order == 1:
                A[row_k, _idx(i, ny-2, ny)] = -beta / hy
                A[row_k, _idx(i, ny-1, ny)] = alpha + beta / hy
            else:
                A[row_k, _idx(i, ny-3, ny)] = beta / (2*hy)
                A[row_k, _idx(i, ny-2, ny)] = -4.0 * beta / (2*hy)
                A[row_k, _idx(i, ny-1, ny)] = alpha + 3.0 * beta / (2*hy)
            rhs[row_k] = float(gam)

    for j in range(ny):
        _set_row_x(_idx(0, j, ny), 'left', bL, y[j])
        _set_row_x(_idx(nx-1, j, ny), 'right', bR, y[j])
    for i in range(1, nx - 1):
        _set_row_y(_idx(i, 0, ny), 'bottom', bB, x[i])
        _set_row_y(_idx(i, ny-1, ny), 'top', bT, x[i])

    A = A.tocsr()
    u_flat = spla.spsolve(A, rhs)
    U = u_flat.reshape(nx, ny)
    return x, y, U


def _build_1d_implicit_matrix_with_bc(n, h, alpha_coef, bnd_lo, bnd_hi,
                                       t_bc):


    main = np.full(n, 1.0 + 2.0 * alpha_coef / h**2)
    lo = np.full(n - 1, -alpha_coef / h**2)
    hi = np.full(n - 1, -alpha_coef / h**2)


    return main, lo, hi


def _solve_2d_adi_xstep(U_in, x, y, ax, ay, hx, hy, tau,
                        bnd_left, bnd_right, bnd_bottom, bnd_top,
                        t_old, t_half, source):

    nx, ny = len(x), len(y)
    U_half = np.zeros_like(U_in)
    alpha_x = ax * tau / 2.0


    if source is not None:
        Xg, Yg = np.meshgrid(x, y, indexing='ij')
        F = source(Xg, Yg, t_half)
    else:
        F = None


    for j in range(ny):


        main = np.full(nx, 1.0 + 2.0 * alpha_x / hx**2)
        lo = np.full(nx - 1, -alpha_x / hx**2)
        hi = np.full(nx - 1, -alpha_x / hx**2)
        rhs = np.zeros(nx)


        u_n = U_in[:, j]
        if 1 <= j <= ny - 2:
            u_yy = (U_in[:, j+1] - 2*U_in[:, j] + U_in[:, j-1]) / hy**2
        else:
            u_yy = np.zeros(nx)
        rhs_int = u_n + tau / 2.0 * ay * u_yy
        if F is not None:
            rhs_int = rhs_int + tau / 2.0 * F[:, j]
        rhs[:] = rhs_int


        def _apply_x_left(b):
            gam = b.gamma(y[j], t_half) if callable(b.gamma) else b.gamma
            if isinstance(gam, np.ndarray):
                gam = float(gam.item()) if gam.size == 1 else float(gam.ravel()[0])
            if b.kind == 'dirichlet':
                main[0] = 1.0
                hi[0] = 0.0
                rhs[0] = float(gam)
                return False
            alpha, beta = b.alpha, b.beta
            if b.order == 1:
                main[0] = alpha - beta / hx
                hi[0] = beta / hx
                rhs[0] = float(gam)
                return False
            else:
                main[0] = alpha - 3.0 * beta / (2*hx)
                hi[0] = 4.0 * beta / (2*hx)
                rhs[0] = float(gam)

                return ('left', -beta / (2*hx))

        def _apply_x_right(b):
            gam = b.gamma(y[j], t_half) if callable(b.gamma) else b.gamma
            if isinstance(gam, np.ndarray):
                gam = float(gam.item()) if gam.size == 1 else float(gam.ravel()[0])
            if b.kind == 'dirichlet':
                main[-1] = 1.0
                lo[-1] = 0.0
                rhs[-1] = float(gam)
                return False
            alpha, beta = b.alpha, b.beta
            if b.order == 1:
                lo[-1] = -beta / hx
                main[-1] = alpha + beta / hx
                rhs[-1] = float(gam)
                return False
            else:
                lo[-1] = -4.0 * beta / (2*hx)
                main[-1] = alpha + 3.0 * beta / (2*hx)
                rhs[-1] = float(gam)
                return ('right', beta / (2*hx))

        extra_l = _apply_x_left(bnd_left)
        extra_r = _apply_x_right(bnd_right)


        if extra_l or extra_r:
            mat = sp.diags([lo, main, hi], offsets=[-1, 0, 1], format='lil')
            if extra_l:
                mat[0, 2] = extra_l[1]
            if extra_r:
                mat[nx-1, nx-3] = extra_r[1]
            U_half[:, j] = spla.spsolve(mat.tocsc(), rhs)
        else:
            U_half[:, j] = thomas(lo, main, hi, rhs)

    return U_half


def _solve_2d_adi_ystep(U_in, x, y, ax, ay, hx, hy, tau,
                        bnd_left, bnd_right, bnd_bottom, bnd_top,
                        t_half, t_new, source):

    nx, ny = len(x), len(y)
    U_new = np.zeros_like(U_in)
    alpha_y = ay * tau / 2.0

    if source is not None:
        Xg, Yg = np.meshgrid(x, y, indexing='ij')
        F = source(Xg, Yg, t_half)
    else:
        F = None

    for i in range(nx):
        main = np.full(ny, 1.0 + 2.0 * alpha_y / hy**2)
        lo = np.full(ny - 1, -alpha_y / hy**2)
        hi = np.full(ny - 1, -alpha_y / hy**2)
        rhs = np.zeros(ny)

        u_half_row = U_in[i, :]
        if 1 <= i <= nx - 2:
            u_xx = (U_in[i+1, :] - 2*U_in[i, :] + U_in[i-1, :]) / hx**2
        else:
            u_xx = np.zeros(ny)
        rhs_int = u_half_row + tau / 2.0 * ax * u_xx
        if F is not None:
            rhs_int = rhs_int + tau / 2.0 * F[i, :]
        rhs[:] = rhs_int

        def _apply_y_bottom(b):
            gam = b.gamma(x[i], t_new) if callable(b.gamma) else b.gamma
            if isinstance(gam, np.ndarray):
                gam = float(gam.item()) if gam.size == 1 else float(gam.ravel()[0])
            if b.kind == 'dirichlet':
                main[0] = 1.0
                hi[0] = 0.0
                rhs[0] = float(gam)
                return False
            alpha, beta = b.alpha, b.beta
            if b.order == 1:
                main[0] = alpha - beta / hy
                hi[0] = beta / hy
                rhs[0] = float(gam)
                return False
            else:
                main[0] = alpha - 3.0 * beta / (2*hy)
                hi[0] = 4.0 * beta / (2*hy)
                rhs[0] = float(gam)
                return ('bottom', -beta / (2*hy))

        def _apply_y_top(b):
            gam = b.gamma(x[i], t_new) if callable(b.gamma) else b.gamma
            if isinstance(gam, np.ndarray):
                gam = float(gam.item()) if gam.size == 1 else float(gam.ravel()[0])
            if b.kind == 'dirichlet':
                main[-1] = 1.0
                lo[-1] = 0.0
                rhs[-1] = float(gam)
                return False
            alpha, beta = b.alpha, b.beta
            if b.order == 1:
                lo[-1] = -beta / hy
                main[-1] = alpha + beta / hy
                rhs[-1] = float(gam)
                return False
            else:
                lo[-1] = -4.0 * beta / (2*hy)
                main[-1] = alpha + 3.0 * beta / (2*hy)
                rhs[-1] = float(gam)
                return ('top', beta / (2*hy))

        extra_b = _apply_y_bottom(bnd_bottom)
        extra_t = _apply_y_top(bnd_top)

        if extra_b or extra_t:
            mat = sp.diags([lo, main, hi], offsets=[-1, 0, 1], format='lil')
            if extra_b:
                mat[0, 2] = extra_b[1]
            if extra_t:
                mat[ny-1, ny-3] = extra_t[1]
            U_new[i, :] = spla.spsolve(mat.tocsc(), rhs)
        else:
            U_new[i, :] = thomas(lo, main, hi, rhs)

    return U_new


def solve_parabolic_2d_adi(
    *,
    domain,
    ax=1.0, ay=1.0,
    source=None,
    initial,
    boundaries,
    nx=51, ny=51, nt=201,
    bc_order=2,
    verbose=False,
):

    x_lo, x_hi = domain['x']
    y_lo, y_hi = domain['y']
    t_lo, t_hi = domain['t']
    x = np.linspace(x_lo, x_hi, nx)
    y = np.linspace(y_lo, y_hi, ny)
    t = np.linspace(t_lo, t_hi, nt)
    hx = x[1] - x[0]
    hy = y[1] - y[0]
    tau = t[1] - t[0]

    if verbose:
        print(f'  МКР-параболика 2D (ADI): nx={nx}, ny={ny}, nt={nt}, '
              f'hx={hx:.3g}, hy={hy:.3g}, τ={tau:.3g}')


    U = np.zeros((nt, nx, ny))
    Xg, Yg = np.meshgrid(x, y, indexing='ij')
    U[0, :, :] = initial(Xg, Yg)


    bL = boundaries['left']
    bR = boundaries['right']
    bB = boundaries['bottom']
    bT = boundaries['top']

    if bL.kind == 'dirichlet':
        U[0, 0, :] = np.array([bL.gamma(yj, t[0]) for yj in y])
    if bR.kind == 'dirichlet':
        U[0, -1, :] = np.array([bR.gamma(yj, t[0]) for yj in y])
    if bB.kind == 'dirichlet':
        U[0, :, 0] = np.array([bB.gamma(xi, t[0]) for xi in x])
    if bT.kind == 'dirichlet':
        U[0, :, -1] = np.array([bT.gamma(xi, t[0]) for xi in x])

    for k in range(nt - 1):
        t_old = t[k]
        t_new = t[k+1]
        t_half = 0.5 * (t_old + t_new)

        U_half = _solve_2d_adi_xstep(
            U[k], x, y, ax, ay, hx, hy, tau,
            bL, bR, bB, bT, t_old, t_half, source)


        if bB.kind == 'dirichlet':
            U_half[:, 0] = np.array([bB.gamma(xi, t_half) for xi in x])
        if bT.kind == 'dirichlet':
            U_half[:, -1] = np.array([bT.gamma(xi, t_half) for xi in x])


        U_new = _solve_2d_adi_ystep(
            U_half, x, y, ax, ay, hx, hy, tau,
            bL, bR, bB, bT, t_half, t_new, source)

        if bL.kind == 'dirichlet':
            U_new[0, :] = np.array([bL.gamma(yj, t_new) for yj in y])
        if bR.kind == 'dirichlet':
            U_new[-1, :] = np.array([bR.gamma(yj, t_new) for yj in y])

        U[k+1] = U_new

    return x, y, t, U


def _filter_kwargs(cls, kwargs):

    sig = inspect.signature(cls.__init__)
    valid = set(sig.parameters.keys()) - {'self'}
    bad = [k for k in kwargs if k not in valid]
    if bad:
        print(f'    [info] игнорирую устаревшие kwargs PINNSolver: {bad}')
    return {k: v for k, v in kwargs.items() if k in valid}


def _build_pinn_conditions(task_pinn_conditions):

    out = []
    for c in task_pinn_conditions:
        c2 = {k: v for k, v in c.items()
              if k not in ('value_numpy',)}
        if 'value_torch' in c2:
            c2['value'] = c2.pop('value_torch')
        out.append(c2)
    return out


def _make_boundary(spec):

    return Boundary(
        kind=spec['kind'],
        alpha=spec['alpha'],
        beta=spec['beta'],
        gamma=spec['gamma_numpy'],
        order=spec.get('order', 2),
    )


def run_fdm(task, *, kind, verbose=True):

    fdm_p = task['fdm']
    domain = task['domain']
    t0 = time.perf_counter()

    if kind == 'parabolic_1d':
        bL = _make_boundary(fdm_p['boundaries_spec'][0])
        bR = _make_boundary(fdm_p['boundaries_spec'][1])
        x, t, U = solve_parabolic_1d(
            domain=domain,
            a=fdm_p['a'], b=fdm_p.get('b', 0.0), c=fdm_p.get('c', 0.0),
            source=fdm_p.get('source_numpy'),
            initial=fdm_p['initial_numpy'],
            boundaries=[bL, bR],
            nx=fdm_p['nx'], nt=fdm_p['nt'],
            scheme=fdm_p.get('scheme', 'crank_nicolson'),
            verbose=verbose,
        )
        elapsed = time.perf_counter() - t0
        return {'kind': kind, 'x': x, 't': t, 'U': U, 'time': elapsed}

    elif kind == 'hyperbolic_1d':
        bL = _make_boundary(fdm_p['boundaries_spec'][0])
        bR = _make_boundary(fdm_p['boundaries_spec'][1])
        x, t, U = solve_hyperbolic_1d(
            domain=domain,
            a=fdm_p['a'], d=fdm_p.get('d', 0.0),
            b=fdm_p.get('b', 0.0), c=fdm_p.get('c', 0.0),
            source=fdm_p.get('source_numpy'),
            initial_u=fdm_p['initial_u_numpy'],
            initial_ut=fdm_p['initial_ut_numpy'],
            boundaries=[bL, bR],
            nx=fdm_p['nx'], nt=fdm_p['nt'],
            verbose=verbose,
        )
        elapsed = time.perf_counter() - t0
        return {'kind': kind, 'x': x, 't': t, 'U': U, 'time': elapsed}

    elif kind == 'elliptic_2d':
        bnds = {k: _make_boundary(v)
                for k, v in fdm_p['boundaries_spec'].items()}
        x, y, U = solve_elliptic_2d(
            domain=domain,
            a=fdm_p.get('a', 1.0),
            bx=fdm_p.get('bx', 0.0), by=fdm_p.get('by', 0.0),
            c=fdm_p.get('c', 0.0),
            source=fdm_p.get('source_numpy'),
            boundaries=bnds,
            nx=fdm_p['nx'], ny=fdm_p['ny'],
            verbose=verbose,
        )
        elapsed = time.perf_counter() - t0
        return {'kind': kind, 'x': x, 'y': y, 'U': U, 'time': elapsed}

    elif kind == 'parabolic_2d_adi':
        bnds = {k: _make_boundary(v)
                for k, v in fdm_p['boundaries_spec'].items()}
        x, y, t, U = solve_parabolic_2d_adi(
            domain=domain,
            ax=fdm_p['ax'], ay=fdm_p['ay'],
            source=fdm_p.get('source_numpy'),
            initial=fdm_p['initial_numpy'],
            boundaries=bnds,
            nx=fdm_p['nx'], ny=fdm_p['ny'], nt=fdm_p['nt'],
            verbose=verbose,
        )
        elapsed = time.perf_counter() - t0
        return {'kind': kind, 'x': x, 'y': y, 't': t, 'U': U,
                'time': elapsed}

    else:
        raise ValueError(f'Unknown kind: {kind}')


def run_pinn(task, *, PINNSolver_cls, verbose=True, device=None):

    pinn_p = task['pinn']
    raw_kwargs = dict(
        equation=pinn_p['equation_torch'],
        domain=task['domain'],
        conditions=_build_pinn_conditions(pinn_p['conditions']),
        device=device,
        **pinn_p['solver_kwargs'],
    )
    kwargs = _filter_kwargs(PINNSolver_cls, raw_kwargs)
    solver = PINNSolver_cls(**kwargs)

    t0 = time.perf_counter()
    solver.solve(verbose=verbose, **pinn_p['solve_kwargs'])
    elapsed = time.perf_counter() - t0
    return {'solver': solver, 'time': elapsed}


def evaluate_common(task, fdm_result, pinn_result, *, eval_n=200):

    kind = fdm_result['kind']
    domain = task['domain']
    exact_fn = task['exact']

    if kind == 'parabolic_1d' or kind == 'hyperbolic_1d':

        eval_x = np.linspace(*domain['x'], eval_n)
        eval_t = np.linspace(*domain['t'], eval_n)

        fdm_on_grid = interpolate_fdm_to_grid(
            fdm_result['x'], fdm_result['t'], fdm_result['U'],
            eval_x, eval_t)

        Xg, Tg = np.meshgrid(eval_x, eval_t, indexing='ij')
        pinn = pinn_result['solver']
        pinn_on_grid = pinn.predict(Xg.ravel(), Tg.ravel())
        if pinn_on_grid.ndim == 2 and pinn_on_grid.shape[1] == 1:
            pinn_on_grid = pinn_on_grid.ravel()
        pinn_on_grid = pinn_on_grid.reshape(eval_n, eval_n)

        exact_on_grid = exact_fn(Xg, Tg)
        return {
            'fdm_err': errors(fdm_on_grid, exact_on_grid),
            'pinn_err': errors(pinn_on_grid, exact_on_grid),
            'eval_grid': {'x': eval_x, 't': eval_t},
            'fdm_on_grid': fdm_on_grid,
            'pinn_on_grid': pinn_on_grid,
            'exact_on_grid': exact_on_grid,
        }

    elif kind == 'elliptic_2d':
        eval_x = np.linspace(*domain['x'], eval_n)
        eval_y = np.linspace(*domain['y'], eval_n)
        fdm_on_grid = interpolate_fdm_2d_to_grid(
            fdm_result['x'], fdm_result['y'], fdm_result['U'],
            eval_x, eval_y)
        Xg, Yg = np.meshgrid(eval_x, eval_y, indexing='ij')
        pinn = pinn_result['solver']
        pinn_on_grid = pinn.predict(Xg.ravel(), Yg.ravel())
        if pinn_on_grid.ndim == 2 and pinn_on_grid.shape[1] == 1:
            pinn_on_grid = pinn_on_grid.ravel()
        pinn_on_grid = pinn_on_grid.reshape(eval_n, eval_n)
        exact_on_grid = exact_fn(Xg, Yg)
        return {
            'fdm_err': errors(fdm_on_grid, exact_on_grid),
            'pinn_err': errors(pinn_on_grid, exact_on_grid),
            'eval_grid': {'x': eval_x, 'y': eval_y},
            'fdm_on_grid': fdm_on_grid,
            'pinn_on_grid': pinn_on_grid,
            'exact_on_grid': exact_on_grid,
        }

    elif kind == 'parabolic_2d_adi':

        eval_n_t = max(eval_n // 4, 25)
        eval_x = np.linspace(*domain['x'], eval_n)
        eval_y = np.linspace(*domain['y'], eval_n)
        eval_t = np.linspace(*domain['t'], eval_n_t)
        fdm_on_grid = interpolate_fdm_3d_to_grid(
            fdm_result['x'], fdm_result['y'], fdm_result['t'], fdm_result['U'],
            eval_x, eval_y, eval_t)
        Xg, Yg, Tg = np.meshgrid(eval_x, eval_y, eval_t, indexing='ij')
        pinn = pinn_result['solver']
        pinn_on_grid = pinn.predict(Xg.ravel(), Yg.ravel(), Tg.ravel())
        if pinn_on_grid.ndim == 2 and pinn_on_grid.shape[1] == 1:
            pinn_on_grid = pinn_on_grid.ravel()
        pinn_on_grid = pinn_on_grid.reshape(eval_n, eval_n, eval_n_t)
        exact_on_grid = exact_fn(Xg, Yg, Tg)
        return {
            'fdm_err': errors(fdm_on_grid, exact_on_grid),
            'pinn_err': errors(pinn_on_grid, exact_on_grid),
            'eval_grid': {'x': eval_x, 'y': eval_y, 't': eval_t},
            'fdm_on_grid': fdm_on_grid,
            'pinn_on_grid': pinn_on_grid,
            'exact_on_grid': exact_on_grid,
        }
    else:
        raise ValueError(f'Unknown kind: {kind}')


def format_row(name, fdm_err, fdm_t, pinn_err, pinn_t):
    def fmt(e, default='   —   '):
        if e is None:
            return default
        if e['rel_defined']:
            return f"{e['rel_l2']:.2e} / {e['abs_max']:.2e}"
        return f"  —    / {e['abs_max']:.2e}"
    return (f'{name:48s} | '
            f'{fmt(fdm_err):28s} | {fdm_t:7.2f}s | '
            f'{fmt(pinn_err):28s} | {pinn_t:7.2f}s')


def print_summary_table(rows):

    header = ('Задача                                           | '
              'МКР: rel L2 / max|Δ|         | время    | '
              'PINN: rel L2 / max|Δ|        | время   ')
    print(header)
    print('-' * len(header))
    for r in rows:
        print(format_row(r['name'],
                         r.get('fdm_err'), r.get('fdm_time', 0.0),
                         r.get('pinn_err'), r.get('pinn_time', 0.0)))


def run_task(task, *, kind, PINNSolver_cls, eval_n=200,
             skip_pinn=False, skip_fdm=False,
             verbose=True, device=None):

    print(f'\n{"="*78}')
    print(f'  {task["name"]}')
    print(f'  PDE: {task["pde_str"]}')
    print(f'{"="*78}')

    out = {'task': task, 'kind': kind}

    if not skip_fdm:
        print('--- МКР ---')
        fdm_res = run_fdm(task, kind=kind, verbose=verbose)
        out['fdm'] = fdm_res
        print(f'  МКР время: {fdm_res["time"]:.2f}s')
    else:
        out['fdm'] = None

    if not skip_pinn:
        print('--- PINN ---')
        pinn_res = run_pinn(task, PINNSolver_cls=PINNSolver_cls,
                            verbose=verbose, device=device)
        out['pinn'] = pinn_res
        print(f'  PINN время: {pinn_res["time"]:.2f}s')
    else:
        out['pinn'] = None

    if not skip_fdm and not skip_pinn:
        print('--- Оценка на общей сетке ---')
        ev = evaluate_common(task, out['fdm'], out['pinn'], eval_n=eval_n)
        out['eval'] = ev
        print(f'  МКР:  {err_summary(ev["fdm_on_grid"], ev["exact_on_grid"])}')
        print(f'  PINN: {err_summary(ev["pinn_on_grid"], ev["exact_on_grid"])}')

    return out


def plot_compare_2d(result, *, save_to=None, dpi=110):

    ev = result['eval']
    task = result['task']
    kind = result['kind']

    if kind in ('parabolic_1d', 'hyperbolic_1d'):
        v1, v2 = 'x', 't'
    elif kind == 'elliptic_2d':
        v1, v2 = 'x', 'y'
    else:
        raise ValueError(f'plot_compare_2d не поддерживает kind={kind}')

    A = ev['eval_grid'][v1]
    B = ev['eval_grid'][v2]
    Ag, Bg = np.meshgrid(A, B, indexing='ij')
    pinn = ev['pinn_on_grid']
    fdm = ev['fdm_on_grid']
    ex = ev['exact_on_grid']

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(task['name'], fontsize=13)

    panels = [
        ('PINN', pinn, 'viridis'),
        ('МКР', fdm, 'viridis'),
        ('аналитическое', ex, 'viridis'),
    ]
    for ax, (title, Z, cmap) in zip(axes[0], panels):
        cf = ax.contourf(Ag, Bg, Z, levels=30, cmap=cmap)
        ax.set_title(title)
        ax.set_xlabel(v1); ax.set_ylabel(v2)
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)

    err_panels = [
        ('|PINN - аналит|', np.abs(pinn - ex)),
        ('|МКР - аналит|', np.abs(fdm - ex)),
    ]
    for ax, (title, Z) in zip(axes[1, :2], err_panels):
        cf = ax.contourf(Ag, Bg, Z, levels=30, cmap='plasma')
        ax.set_title(title)
        ax.set_xlabel(v1); ax.set_ylabel(v2)
        plt.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)


    if 'pinn' in result and result['pinn'] is not None:
        ax_h = axes[1, 2]
        h = result['pinn']['solver'].history
        ax_h.semilogy(h['total'], label='total', alpha=0.85)
        ax_h.semilogy(h['residual'], label='residual', alpha=0.85)
        ax_h.semilogy(h['conditions'], label='BC/IC', alpha=0.85)
        ax_h.set_xlabel('iter'); ax_h.set_ylabel('loss')
        ax_h.set_title('история обучения PINN')
        ax_h.legend(fontsize=8)
        ax_h.grid(alpha=0.3)


    fdm_e = ev['fdm_err']
    pinn_e = ev['pinn_err']
    sub = (f"  МКР:  rel L2={fdm_e['rel_l2']:.2e}, max |Δ|={fdm_e['abs_max']:.2e}"
           f"   |   PINN: rel L2={pinn_e['rel_l2']:.2e}, max |Δ|={pinn_e['abs_max']:.2e}")
    fig.text(0.5, 0.92, sub, ha='center', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.91])
    if save_to:
        plt.savefig(save_to, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close('all')


def plot_compare_2d_t_snapshots(result, *, t_values=None,
                                  save_to=None, dpi=110):

    ev = result['eval']
    task = result['task']

    if t_values is None:
        t_grid = ev['eval_grid']['t']
        idx = np.linspace(0, len(t_grid)-1, 5, dtype=int)
        t_values = t_grid[idx]
    else:
        t_grid = ev['eval_grid']['t']
        idx = [int(np.argmin(np.abs(t_grid - tv))) for tv in t_values]

    x = ev['eval_grid']['x']
    y = ev['eval_grid']['y']
    Xg, Yg = np.meshgrid(x, y, indexing='ij')
    nt = len(t_values)

    fig, axes = plt.subplots(3, nt, figsize=(3.5*nt, 9))
    if nt == 1:
        axes = axes.reshape(3, 1)
    fig.suptitle(task['name'], fontsize=13)

    for col, (i_t, t_val) in enumerate(zip(idx, t_values)):
        pinn_slice = ev['pinn_on_grid'][:, :, i_t]
        fdm_slice = ev['fdm_on_grid'][:, :, i_t]
        ex_slice = ev['exact_on_grid'][:, :, i_t]

        cf0 = axes[0, col].contourf(Xg, Yg, pinn_slice, levels=20,
                                     cmap='viridis')
        axes[0, col].set_title(f'PINN @ t={t_val:.2f}')
        plt.colorbar(cf0, ax=axes[0, col], fraction=0.046, pad=0.04)

        cf1 = axes[1, col].contourf(Xg, Yg, fdm_slice, levels=20,
                                     cmap='viridis')
        axes[1, col].set_title(f'МКР @ t={t_val:.2f}')
        plt.colorbar(cf1, ax=axes[1, col], fraction=0.046, pad=0.04)


        err_max = np.maximum(np.abs(pinn_slice - ex_slice),
                              np.abs(fdm_slice - ex_slice))
        cf2 = axes[2, col].contourf(Xg, Yg, err_max, levels=20, cmap='plasma')
        axes[2, col].set_title(f'max |Δ| @ t={t_val:.2f}')
        plt.colorbar(cf2, ax=axes[2, col], fraction=0.046, pad=0.04)

        for r in range(3):
            axes[r, col].set_xlabel('x')
            axes[r, col].set_ylabel('y')

    fdm_e = ev['fdm_err']
    pinn_e = ev['pinn_err']
    sub = (f"  МКР:  rel L2={fdm_e['rel_l2']:.2e}, max |Δ|={fdm_e['abs_max']:.2e}"
           f"   |   PINN: rel L2={pinn_e['rel_l2']:.2e}, max |Δ|={pinn_e['abs_max']:.2e}")
    fig.text(0.5, 0.945, sub, ha='center', fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if save_to:
        plt.savefig(save_to, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close('all')


def plot_loss_history(result, save_to=None, dpi=110):

    if 'pinn' not in result or result['pinn'] is None:
        print('Нет PINN-результата для построения истории.')
        return
    h = result['pinn']['solver'].history
    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.semilogy(h['total'], label='total', alpha=0.85)
    ax.semilogy(h['residual'], label='residual', alpha=0.85)
    ax.semilogy(h['conditions'], label='BC/IC', alpha=0.85)
    ax.set_xlabel('iter'); ax.set_ylabel('loss')
    ax.set_title(f'История обучения PINN — {result["task"]["name"]}')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to, dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close('all')


_g1_a_t2 = 0.1
_g1_T_t2 = 1.0


def _g1_eq_t2_torch(u, c, D):
    return D(u, c['t']) - _g1_a_t2 * D(u, c['x'], 2)


def _g1_exact_t2(x, t):
    return x + np.exp(-np.pi**2 * _g1_a_t2 * t) * np.sin(np.pi * x)


LR1_TASK_2 = {
    'name': 'Параболическое 1D: неоднородные условия Дирихле',
    'pde_str': 'u_t = a u_xx,  u(0,t)=0, u(1,t)=1, u(x,0)=x+sin(πx)',
    'domain': {'x': (0.0, 1.0), 't': (0.0, _g1_T_t2)},
    'exact': _g1_exact_t2,
    'fdm': {
        'a': _g1_a_t2, 'b': 0.0, 'c': 0.0,
        'source_numpy': None,
        'initial_numpy': lambda x: x + np.sin(np.pi * x),
        'boundaries_spec': [
            {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
             'gamma_numpy': lambda t: 0.0},
            {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
             'gamma_numpy': lambda t: 1.0},
        ],
        'nx': 101, 'nt': 1001, 'scheme': 'crank_nicolson',
    },
    'pinn': {
        'equation_torch': _g1_eq_t2_torch,
        'conditions': [
            {'where': {'x': 0.0}, 'value_torch': 0.0,
             'value_numpy': lambda y_or_t=None: 0.0},
            {'where': {'x': 1.0}, 'value_torch': 1.0,
             'value_numpy': lambda y_or_t=None: 1.0},
            {'where': {'t': 0.0},
             'value_torch': lambda c: c['x'] + torch.sin(np.pi * c['x']),
             'value_numpy': lambda x: x + np.sin(np.pi * x)},
        ],
        'solver_kwargs': dict(
            hidden_size=64, num_hidden_layers=4, activation='tanh',
            loss_weighting='fixed', lambda_res=1.0, lambda_cond=10.0,
        ),
        'solve_kwargs': dict(
            n_epochs_adam=4000, n_collocation=1500, n_condition=200,
            lr=1e-3, use_lbfgs=True, lbfgs_max_iter=1500,
        ),
    },
}


_g1_a_t4 = 0.5
_g1_T_t4 = 2.0


def _g1_eq_t4_torch(u, c, D):
    return D(u, c['t']) - _g1_a_t4 * D(u, c['x'], 2)


def _g1_exact_t4(x, t):
    return np.exp(-_g1_a_t4 * t) * np.sin(x)


LR1_TASK_4 = {
    'name': 'Параболическое 1D: нестационарные условия Неймана',
    'pde_str': 'u_t = a u_xx, u_x(0,t)=exp(-at), u_x(π,t)=-exp(-at), u(x,0)=sin x',
    'domain': {'x': (0.0, np.pi), 't': (0.0, _g1_T_t4)},
    'exact': _g1_exact_t4,
    'fdm': {
        'a': _g1_a_t4, 'b': 0.0, 'c': 0.0,
        'source_numpy': None,
        'initial_numpy': lambda x: np.sin(x),
        'boundaries_spec': [
            {'kind': 'neumann', 'alpha': 0.0, 'beta': 1.0,
             'gamma_numpy': lambda t: np.exp(-_g1_a_t4 * t), 'order': 2},
            {'kind': 'neumann', 'alpha': 0.0, 'beta': 1.0,
             'gamma_numpy': lambda t: -np.exp(-_g1_a_t4 * t), 'order': 2},
        ],
        'nx': 101, 'nt': 2001, 'scheme': 'crank_nicolson',
    },
    'pinn': {
        'equation_torch': _g1_eq_t4_torch,
        'conditions': [
            {'type': 'neumann', 'deriv_var': 'x', 'where': {'x': 0.0},
             'value_torch': lambda c: torch.exp(-_g1_a_t4 * c['t']),
             'value_numpy': lambda t: np.exp(-_g1_a_t4 * np.asarray(t))},
            {'type': 'neumann', 'deriv_var': 'x', 'where': {'x': np.pi},
             'value_torch': lambda c: -torch.exp(-_g1_a_t4 * c['t']),
             'value_numpy': lambda t: -np.exp(-_g1_a_t4 * np.asarray(t))},
            {'where': {'t': 0.0},
             'value_torch': lambda c: torch.sin(c['x']),
             'value_numpy': lambda x: np.sin(x)},
        ],
        'solver_kwargs': dict(
            hidden_size=64, num_hidden_layers=4, activation='tanh',
            loss_weighting='fixed', lambda_res=1.0, lambda_cond=15.0,
        ),
        'solve_kwargs': dict(
            n_epochs_adam=5000, n_collocation=2000, n_condition=250,
            lr=1e-3, use_lbfgs=True, lbfgs_max_iter=1500,
        ),
    },
}


_g1_a_t9 = 0.5
_g1_b_t9 = 1.0
_g1_T_t9 = 2.0


def _g1_eq_t9_torch(u, c, D):
    return D(u, c['t']) - _g1_a_t9 * D(u, c['x'], 2) - _g1_b_t9 * D(u, c['x'])


def _g1_exact_t9(x, t):
    return np.exp(-_g1_a_t9 * t) * np.cos(x + _g1_b_t9 * t)


LR1_TASK_9 = {
    'name': 'Параболическое 1D: конвекция-диффузия, условия Робена',
    'pde_str': ('u_t = a u_xx + b u_x, '
                'u_x(0,t)-u(0,t)=-exp(-at)(cos bt+sin bt), '
                'u_x(π,t)-u(π,t)= exp(-at)(cos bt+sin bt), u(x,0)=cos x'),
    'domain': {'x': (0.0, np.pi), 't': (0.0, _g1_T_t9)},
    'exact': _g1_exact_t9,
    'fdm': {
        'a': _g1_a_t9, 'b': _g1_b_t9, 'c': 0.0,
        'source_numpy': None,
        'initial_numpy': lambda x: np.cos(x),
        'boundaries_spec': [
            {'kind': 'robin', 'alpha': -1.0, 'beta': 1.0,
             'gamma_numpy': lambda t: -np.exp(-_g1_a_t9*t)*(np.cos(_g1_b_t9*t)+np.sin(_g1_b_t9*t)),
             'order': 2},
            {'kind': 'robin', 'alpha': -1.0, 'beta': 1.0,
             'gamma_numpy': lambda t: np.exp(-_g1_a_t9*t)*(np.cos(_g1_b_t9*t)+np.sin(_g1_b_t9*t)),
             'order': 2},
        ],
        'nx': 151, 'nt': 2001, 'scheme': 'crank_nicolson',
    },
    'pinn': {
        'equation_torch': _g1_eq_t9_torch,
        'conditions': [
            {'type': 'robin', 'deriv_var': 'x', 'alpha': -1.0, 'beta': 1.0,
             'where': {'x': 0.0},
             'value_torch': lambda c: -torch.exp(-_g1_a_t9*c['t'])
                                      *(torch.cos(_g1_b_t9*c['t'])+torch.sin(_g1_b_t9*c['t'])),
             'value_numpy': lambda t: -np.exp(-_g1_a_t9*np.asarray(t))
                                      *(np.cos(_g1_b_t9*np.asarray(t))+np.sin(_g1_b_t9*np.asarray(t)))},
            {'type': 'robin', 'deriv_var': 'x', 'alpha': -1.0, 'beta': 1.0,
             'where': {'x': np.pi},
             'value_torch': lambda c: torch.exp(-_g1_a_t9*c['t'])
                                     *(torch.cos(_g1_b_t9*c['t'])+torch.sin(_g1_b_t9*c['t'])),
             'value_numpy': lambda t: np.exp(-_g1_a_t9*np.asarray(t))
                                     *(np.cos(_g1_b_t9*np.asarray(t))+np.sin(_g1_b_t9*np.asarray(t)))},
            {'where': {'t': 0.0},
             'value_torch': lambda c: torch.cos(c['x']),
             'value_numpy': lambda x: np.cos(x)},
        ],
        'solver_kwargs': dict(
            hidden_size=96, num_hidden_layers=5, activation='tanh',
            loss_weighting='grad_norm', lambda_res=1.0, lambda_cond=10.0,
        ),
        'solve_kwargs': dict(
            n_epochs_adam=7000, n_collocation=3000, n_condition=350,
            lr=1e-3, use_lbfgs=True, lbfgs_max_iter=1500,
        ),
    },
}


LR1_TASKS = [LR1_TASK_2, LR1_TASK_4, LR1_TASK_9]


_g2_a_t2 = 1.0
_g2_T_t2 = float(np.pi)


def _g2_eq_lr2_t2_torch(u, c, D):
    return D(u, c['t'], 2) - _g2_a_t2**2 * D(u, c['x'], 2)


def _g2_exact_lr2_t2(x, t):
    return np.sin(x - _g2_a_t2 * t) + np.cos(x + _g2_a_t2 * t)


LR2_TASK_2 = {
    'name': 'Гиперболическое 1D: условия Робена (α=-1, β=1)',
    'pde_str': 'u_tt = a² u_xx, u_x-u=0 на обоих концах',
    'domain': {'x': (0.0, np.pi), 't': (0.0, _g2_T_t2)},
    'exact': _g2_exact_lr2_t2,
    'fdm': {
        'a': _g2_a_t2, 'd': 0.0, 'b': 0.0, 'c': 0.0,
        'source_numpy': None,
        'initial_u_numpy': lambda x: np.sin(x) + np.cos(x),
        'initial_ut_numpy': lambda x: -_g2_a_t2 * (np.sin(x) + np.cos(x)),
        'boundaries_spec': [
            {'kind': 'robin', 'alpha': -1.0, 'beta': 1.0,
             'gamma_numpy': lambda t: 0.0, 'order': 2},
            {'kind': 'robin', 'alpha': -1.0, 'beta': 1.0,
             'gamma_numpy': lambda t: 0.0, 'order': 2},
        ],
        'nx': 201, 'nt': 4001,
    },
    'pinn': {
        'equation_torch': _g2_eq_lr2_t2_torch,
        'conditions': [
            {'type': 'robin', 'deriv_var': 'x', 'alpha': -1.0, 'beta': 1.0,
             'where': {'x': 0.0},
             'value_torch': 0.0, 'value_numpy': lambda t: 0.0},
            {'type': 'robin', 'deriv_var': 'x', 'alpha': -1.0, 'beta': 1.0,
             'where': {'x': np.pi},
             'value_torch': 0.0, 'value_numpy': lambda t: 0.0},
            {'where': {'t': 0.0},
             'value_torch': lambda c: torch.sin(c['x']) + torch.cos(c['x']),
             'value_numpy': lambda x: np.sin(x) + np.cos(x)},
            {'type': 'neumann', 'deriv_var': 't', 'where': {'t': 0.0},
             'value_torch': lambda c: -_g2_a_t2 * (torch.sin(c['x']) + torch.cos(c['x'])),
             'value_numpy': lambda x: -_g2_a_t2 * (np.sin(x) + np.cos(x))},
        ],
        'solver_kwargs': dict(
            hidden_size=96, num_hidden_layers=5, activation='sin',
            loss_weighting='grad_norm', lambda_res=1.0, lambda_cond=10.0,
        ),
        'solve_kwargs': dict(
            n_epochs_adam=8000, n_collocation=3000, n_condition=300,
            lr=1e-3, use_lbfgs=True, lbfgs_max_iter=2500,
        ),
    },
}


_g2_T_t3 = float(np.pi)


def _g2_eq_lr2_t3_torch(u, c, D):
    return D(u, c['t'], 2) - D(u, c['x'], 2) + 3 * u


def _g2_exact_lr2_t3(x, t):
    return np.cos(x) * np.sin(2 * t)


LR2_TASK_3 = {
    'name': 'Гиперболическое 1D: с реакционным членом, нестационарные Дирихле',
    'pde_str': 'u_tt = u_xx - 3u, u(0,t)=sin(2t), u(π,t)=-sin(2t)',
    'domain': {'x': (0.0, np.pi), 't': (0.0, _g2_T_t3)},
    'exact': _g2_exact_lr2_t3,
    'fdm': {
        'a': 1.0, 'd': 0.0, 'b': 0.0, 'c': -3.0,
        'source_numpy': None,
        'initial_u_numpy': lambda x: np.zeros_like(x),
        'initial_ut_numpy': lambda x: 2.0 * np.cos(x),
        'boundaries_spec': [
            {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
             'gamma_numpy': lambda t: float(np.sin(2 * t))},
            {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
             'gamma_numpy': lambda t: float(-np.sin(2 * t))},
        ],
        'nx': 201, 'nt': 4001,
    },
    'pinn': {
        'equation_torch': _g2_eq_lr2_t3_torch,
        'conditions': [
            {'where': {'x': 0.0},
             'value_torch': lambda c: torch.sin(2 * c['t']),
             'value_numpy': lambda t: np.sin(2 * np.asarray(t))},
            {'where': {'x': np.pi},
             'value_torch': lambda c: -torch.sin(2 * c['t']),
             'value_numpy': lambda t: -np.sin(2 * np.asarray(t))},
            {'where': {'t': 0.0},
             'value_torch': 0.0,
             'value_numpy': lambda x: np.zeros_like(np.asarray(x, dtype=float))},
            {'type': 'neumann', 'deriv_var': 't', 'where': {'t': 0.0},
             'value_torch': lambda c: 2.0 * torch.cos(c['x']),
             'value_numpy': lambda x: 2.0 * np.cos(x)},
        ],
        'solver_kwargs': dict(
            hidden_size=96, num_hidden_layers=5, activation='tanh',
            loss_weighting='grad_norm', lambda_res=1.0, lambda_cond=10.0,
        ),
        'solve_kwargs': dict(
            n_epochs_adam=8000, n_collocation=3000, n_condition=300,
            lr=1e-3, use_lbfgs=True, lbfgs_max_iter=2500,
        ),
    },
}


_g2_T_t8 = float(np.pi)


def _g2_eq_lr2_t8_torch(u, c, D):
    return (D(u, c['t'], 2) + 2.0 * D(u, c['t'])
            - D(u, c['x'], 2) - 2.0 * D(u, c['x']) + 3.0 * u)


def _g2_exact_lr2_t8(x, t):
    return np.exp(-t - x) * np.sin(x) * np.sin(2 * t)


LR2_TASK_8 = {
    'name': 'Гиперболическое 1D: с демпфированием и конвекцией',
    'pde_str': 'u_tt + 2u_t = u_xx + 2u_x − 3u, однор. Дирихле',
    'domain': {'x': (0.0, np.pi), 't': (0.0, _g2_T_t8)},
    'exact': _g2_exact_lr2_t8,
    'fdm': {
        'a': 1.0, 'd': 2.0, 'b': 2.0, 'c': -3.0,
        'source_numpy': None,
        'initial_u_numpy': lambda x: np.zeros_like(x),
        'initial_ut_numpy': lambda x: 2.0 * np.exp(-x) * np.sin(x),
        'boundaries_spec': [
            {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
             'gamma_numpy': lambda t: 0.0},
            {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
             'gamma_numpy': lambda t: 0.0},
        ],
        'nx': 201, 'nt': 4001,
    },
    'pinn': {
        'equation_torch': _g2_eq_lr2_t8_torch,
        'conditions': [
            {'where': {'x': 0.0}, 'value_torch': 0.0,
             'value_numpy': lambda t: 0.0},
            {'where': {'x': np.pi}, 'value_torch': 0.0,
             'value_numpy': lambda t: 0.0},
            {'where': {'t': 0.0}, 'value_torch': 0.0,
             'value_numpy': lambda x: np.zeros_like(np.asarray(x, dtype=float))},
            {'type': 'neumann', 'deriv_var': 't', 'where': {'t': 0.0},
             'value_torch': lambda c: 2.0 * torch.exp(-c['x']) * torch.sin(c['x']),
             'value_numpy': lambda x: 2.0 * np.exp(-x) * np.sin(x)},
        ],
        'solver_kwargs': dict(
            hidden_size=96, num_hidden_layers=5, activation='tanh',
            loss_weighting='grad_norm', lambda_res=1.0, lambda_cond=10.0,
        ),
        'solve_kwargs': dict(
            n_epochs_adam=8000, n_collocation=3000, n_condition=300,
            lr=1e-3, use_lbfgs=True, lbfgs_max_iter=2500,
        ),
    },
}


LR2_TASKS = [LR2_TASK_2, LR2_TASK_3, LR2_TASK_8]


def _g3_eq_lr3_t2_torch(u, c, D):
    return D(u, c['x'], 2) + D(u, c['y'], 2)


def _g3_exact_lr3_t2(x, y):
    return x**2 - y**2


LR3_TASK_2 = {
    'name': 'Эллиптическое 2D: смешанные условия Нейман+Дирихле',
    'pde_str': 'Δu = 0,  u_x(0,y)=0, u(1,y)=1-y², u_y(x,0)=0, u(x,1)=x²-1',
    'domain': {'x': (0.0, 1.0), 'y': (0.0, 1.0)},
    'exact': _g3_exact_lr3_t2,
    'fdm': {
        'a': 1.0, 'bx': 0.0, 'by': 0.0, 'c': 0.0,
        'source_numpy': None,
        'boundaries_spec': {
            'left':   {'kind': 'neumann', 'alpha': 0.0, 'beta': 1.0,
                       'gamma_numpy': lambda y: 0.0, 'order': 2},
            'right':  {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                       'gamma_numpy': lambda y: 1.0 - y**2},
            'bottom': {'kind': 'neumann', 'alpha': 0.0, 'beta': 1.0,
                       'gamma_numpy': lambda x: 0.0, 'order': 2},
            'top':    {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                       'gamma_numpy': lambda x: x**2 - 1.0},
        },
        'nx': 51, 'ny': 51,
    },
    'pinn': {
        'equation_torch': _g3_eq_lr3_t2_torch,
        'conditions': [
            {'type': 'neumann', 'deriv_var': 'x', 'where': {'x': 0.0},
             'value_torch': 0.0, 'value_numpy': lambda y: 0.0},
            {'where': {'x': 1.0},
             'value_torch': lambda c: 1.0 - c['y']**2,
             'value_numpy': lambda y: 1.0 - y**2},
            {'type': 'neumann', 'deriv_var': 'y', 'where': {'y': 0.0},
             'value_torch': 0.0, 'value_numpy': lambda x: 0.0},
            {'where': {'y': 1.0},
             'value_torch': lambda c: c['x']**2 - 1.0,
             'value_numpy': lambda x: x**2 - 1.0},
        ],
        'solver_kwargs': dict(
            hidden_size=64, num_hidden_layers=4, activation='tanh',
            loss_weighting='fixed', lambda_res=1.0, lambda_cond=15.0,
        ),
        'solve_kwargs': dict(
            n_epochs_adam=5000, n_collocation=2000, n_condition=250,
            lr=1e-3, use_lbfgs=True, lbfgs_max_iter=1500,
        ),
    },
}


def _g3_eq_lr3_t7_torch(u, c, D):
    return D(u, c['x'], 2) + D(u, c['y'], 2) + 2.0 * u


def _g3_exact_lr3_t7(x, y):
    return np.cos(x) * np.cos(y)


LR3_TASK_7 = {
    'name': 'Эллиптическое 2D: Δu+λu=0, условия Дирихле',
    'pde_str': 'Δu + 2u = 0, чисто Дирихле',
    'domain': {'x': (0.0, np.pi/2), 'y': (0.0, np.pi/2)},
    'exact': _g3_exact_lr3_t7,
    'fdm': {
        'a': 1.0, 'bx': 0.0, 'by': 0.0, 'c': 2.0,
        'source_numpy': None,
        'boundaries_spec': {
            'left':   {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                       'gamma_numpy': lambda y: np.cos(y)},
            'right':  {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                       'gamma_numpy': lambda y: 0.0},
            'bottom': {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                       'gamma_numpy': lambda x: np.cos(x)},
            'top':    {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                       'gamma_numpy': lambda x: 0.0},
        },
        'nx': 51, 'ny': 51,
    },
    'pinn': {
        'equation_torch': _g3_eq_lr3_t7_torch,
        'conditions': [
            {'where': {'x': 0.0},
             'value_torch': lambda c: torch.cos(c['y']),
             'value_numpy': lambda y: np.cos(y)},
            {'where': {'x': np.pi/2}, 'value_torch': 0.0,
             'value_numpy': lambda y: 0.0},
            {'where': {'y': 0.0},
             'value_torch': lambda c: torch.cos(c['x']),
             'value_numpy': lambda x: np.cos(x)},
            {'where': {'y': np.pi/2}, 'value_torch': 0.0,
             'value_numpy': lambda x: 0.0},
        ],
        'solver_kwargs': dict(
            hidden_size=80, num_hidden_layers=5, activation='tanh',
            loss_weighting='grad_norm', lambda_res=1.0, lambda_cond=10.0,
        ),
        'solve_kwargs': dict(
            n_epochs_adam=6000, n_collocation=2500, n_condition=300,
            lr=1e-3, use_lbfgs=True, lbfgs_max_iter=1800,
        ),
    },
}


def _g3_eq_lr3_t10_torch(u, c, D):
    return (D(u, c['x'], 2) + D(u, c['y'], 2)
            + 2.0 * D(u, c['x']) + 2.0 * D(u, c['y']) + 4.0 * u)


def _g3_exact_lr3_t10(x, y):
    return np.exp(-x - y) * np.cos(x) * np.cos(y)


LR3_TASK_10 = {
    'name': 'Эллиптическое 2D: конвекция-диффузия-реакция, условия Дирихле',
    'pde_str': 'Δu + 2u_x + 2u_y + 4u = 0',
    'domain': {'x': (0.0, np.pi/2), 'y': (0.0, np.pi/2)},
    'exact': _g3_exact_lr3_t10,
    'fdm': {
        'a': 1.0, 'bx': 2.0, 'by': 2.0, 'c': 4.0,
        'source_numpy': None,
        'boundaries_spec': {
            'left':   {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                       'gamma_numpy': lambda y: np.exp(-y) * np.cos(y)},
            'right':  {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                       'gamma_numpy': lambda y: 0.0},
            'bottom': {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                       'gamma_numpy': lambda x: np.exp(-x) * np.cos(x)},
            'top':    {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                       'gamma_numpy': lambda x: 0.0},
        },
        'nx': 81, 'ny': 81,
    },
    'pinn': {
        'equation_torch': _g3_eq_lr3_t10_torch,
        'conditions': [
            {'where': {'x': 0.0},
             'value_torch': lambda c: torch.exp(-c['y']) * torch.cos(c['y']),
             'value_numpy': lambda y: np.exp(-y) * np.cos(y)},
            {'where': {'x': np.pi/2}, 'value_torch': 0.0,
             'value_numpy': lambda y: 0.0},
            {'where': {'y': 0.0},
             'value_torch': lambda c: torch.exp(-c['x']) * torch.cos(c['x']),
             'value_numpy': lambda x: np.exp(-x) * np.cos(x)},
            {'where': {'y': np.pi/2}, 'value_torch': 0.0,
             'value_numpy': lambda x: 0.0},
        ],
        'solver_kwargs': dict(
            hidden_size=96, num_hidden_layers=5, activation='tanh',
            loss_weighting='softadapt', lambda_res=1.0, lambda_cond=10.0,
        ),
        'solve_kwargs': dict(
            n_epochs_adam=8000, n_collocation=3000, n_condition=350,
            lr=1e-3, use_lbfgs=True, lbfgs_max_iter=2500,
        ),
    },
}


LR3_TASKS = [LR3_TASK_2, LR3_TASK_7, LR3_TASK_10]


_g4_a_t1 = 0.1
_mu1_t1, _mu2_t1 = 1.0, 1.0
_g4_T_t1 = 2.0
_g4_lam_t1 = (_mu1_t1**2 + _mu2_t1**2) * _g4_a_t1


def _g4_eq_lr4_t1_torch(u, c, D):
    return D(u, c['t']) - _g4_a_t1 * (D(u, c['x'], 2) + D(u, c['y'], 2))


def _g4_exact_lr4_t1(x, y, t):
    return (np.cos(_mu1_t1 * x) * np.cos(_mu2_t1 * y)
            * np.exp(-_g4_lam_t1 * t))


LR4_TASK_1 = {
    'name': 'Параболическое 2D: изотропная диффузия, условия Дирихле',
    'pde_str': 'u_t = a·Δu, Дирихле на 4 сторонах',
    'domain': {'x': (0.0, np.pi), 'y': (0.0, np.pi), 't': (0.0, _g4_T_t1)},
    'exact': _g4_exact_lr4_t1,
    'fdm': {
        'ax': _g4_a_t1, 'ay': _g4_a_t1,
        'source_numpy': None,
        'initial_numpy': lambda X, Y: np.cos(_mu1_t1*X) * np.cos(_mu2_t1*Y),
        'boundaries_spec': {
            'left': {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                     'gamma_numpy': lambda y, t: np.cos(_mu2_t1*y) * np.exp(-_g4_lam_t1*t)},
            'right': {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                      'gamma_numpy': lambda y, t: ((-1.0)**_mu1_t1) * np.cos(_mu2_t1*y) * np.exp(-_g4_lam_t1*t)},
            'bottom': {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                       'gamma_numpy': lambda x, t: np.cos(_mu1_t1*x) * np.exp(-_g4_lam_t1*t)},
            'top': {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                    'gamma_numpy': lambda x, t: ((-1.0)**_mu2_t1) * np.cos(_mu1_t1*x) * np.exp(-_g4_lam_t1*t)},
        },
        'nx': 41, 'ny': 41, 'nt': 201,
    },
    'pinn': {
        'equation_torch': _g4_eq_lr4_t1_torch,
        'conditions': [
            {'where': {'x': 0.0},
             'value_torch': lambda c: torch.cos(_mu2_t1*c['y']) * torch.exp(-_g4_lam_t1*c['t']),
             'value_numpy': lambda y, t: np.cos(_mu2_t1*y) * np.exp(-_g4_lam_t1*t)},
            {'where': {'x': np.pi},
             'value_torch': lambda c: ((-1.0)**_mu1_t1) * torch.cos(_mu2_t1*c['y']) * torch.exp(-_g4_lam_t1*c['t']),
             'value_numpy': lambda y, t: ((-1.0)**_mu1_t1) * np.cos(_mu2_t1*y) * np.exp(-_g4_lam_t1*t)},
            {'where': {'y': 0.0},
             'value_torch': lambda c: torch.cos(_mu1_t1*c['x']) * torch.exp(-_g4_lam_t1*c['t']),
             'value_numpy': lambda x, t: np.cos(_mu1_t1*x) * np.exp(-_g4_lam_t1*t)},
            {'where': {'y': np.pi},
             'value_torch': lambda c: ((-1.0)**_mu2_t1) * torch.cos(_mu1_t1*c['x']) * torch.exp(-_g4_lam_t1*c['t']),
             'value_numpy': lambda x, t: ((-1.0)**_mu2_t1) * np.cos(_mu1_t1*x) * np.exp(-_g4_lam_t1*t)},
            {'where': {'t': 0.0},
             'value_torch': lambda c: torch.cos(_mu1_t1*c['x']) * torch.cos(_mu2_t1*c['y']),
             'value_numpy': lambda x, y: np.cos(_mu1_t1*x) * np.cos(_mu2_t1*y)},
        ],
        'solver_kwargs': dict(
            hidden_size=96, num_hidden_layers=5, activation='tanh',
            loss_weighting='grad_norm', lambda_res=1.0, lambda_cond=10.0,
        ),
        'solve_kwargs': dict(
            n_epochs_adam=8000, n_collocation=4000, n_condition=400,
            lr=1e-3, use_lbfgs=True, lbfgs_max_iter=2500,
        ),
    },
}


_g4_a_t5 = 0.1
_g4_T_t5 = 2.0


def _g4_eq_lr4_t5_torch(u, c, D):
    return D(u, c['t']) - _g4_a_t5 * (D(u, c['x'], 2) + D(u, c['y'], 2))


def _g4_exact_lr4_t5(x, y, t):
    return np.cos(2*x) * np.sinh(y) * np.exp(-3*_g4_a_t5*t)


LR4_TASK_5 = {
    'name': 'Параболическое 2D: Нейман снизу, Дирихле на остальных гранях',
    'pde_str': 'u_t = a·Δu, U=cos(2x)·sinh(y)·exp(-3at)',
    'domain': {'x': (0.0, np.pi/2), 'y': (0.0, np.log(2.0)),
               't': (0.0, _g4_T_t5)},
    'exact': _g4_exact_lr4_t5,
    'fdm': {
        'ax': _g4_a_t5, 'ay': _g4_a_t5,
        'source_numpy': None,
        'initial_numpy': lambda X, Y: np.cos(2*X) * np.sinh(Y),
        'boundaries_spec': {
            'left': {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                     'gamma_numpy': lambda y, t: np.sinh(y) * np.exp(-3*_g4_a_t5*t)},
            'right': {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                      'gamma_numpy': lambda y, t: -np.sinh(y) * np.exp(-3*_g4_a_t5*t)},
            'bottom': {'kind': 'neumann', 'alpha': 0.0, 'beta': 1.0,
                       'gamma_numpy': lambda x, t: np.cos(2*x) * np.exp(-3*_g4_a_t5*t),
                       'order': 2},
            'top': {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                    'gamma_numpy': lambda x, t: 0.75 * np.cos(2*x) * np.exp(-3*_g4_a_t5*t)},
        },
        'nx': 41, 'ny': 41, 'nt': 401,
    },
    'pinn': {
        'equation_torch': _g4_eq_lr4_t5_torch,
        'conditions': [
            {'where': {'x': 0.0},
             'value_torch': lambda c: torch.sinh(c['y']) * torch.exp(-3*_g4_a_t5*c['t']),
             'value_numpy': lambda y, t: np.sinh(y) * np.exp(-3*_g4_a_t5*t)},
            {'where': {'x': np.pi/2},
             'value_torch': lambda c: -torch.sinh(c['y']) * torch.exp(-3*_g4_a_t5*c['t']),
             'value_numpy': lambda y, t: -np.sinh(y) * np.exp(-3*_g4_a_t5*t)},
            {'type': 'neumann', 'deriv_var': 'y', 'where': {'y': 0.0},
             'value_torch': lambda c: torch.cos(2*c['x']) * torch.exp(-3*_g4_a_t5*c['t']),
             'value_numpy': lambda x, t: np.cos(2*x) * np.exp(-3*_g4_a_t5*t)},
            {'where': {'y': np.log(2.0)},
             'value_torch': lambda c: 0.75 * torch.cos(2*c['x']) * torch.exp(-3*_g4_a_t5*c['t']),
             'value_numpy': lambda x, t: 0.75 * np.cos(2*x) * np.exp(-3*_g4_a_t5*t)},
            {'where': {'t': 0.0},
             'value_torch': lambda c: torch.cos(2*c['x']) * torch.sinh(c['y']),
             'value_numpy': lambda x, y: np.cos(2*x) * np.sinh(y)},
        ],
        'solver_kwargs': dict(
            hidden_size=96, num_hidden_layers=5, activation='tanh',
            loss_weighting='grad_norm', lambda_res=1.0, lambda_cond=10.0,
        ),
        'solve_kwargs': dict(
            n_epochs_adam=8000, n_collocation=4000, n_condition=400,
            lr=1e-3, use_lbfgs=True, lbfgs_max_iter=2500,
        ),
    },
}


_g4_a_t9 = 1.0
_g4_b_t9 = 1.0
_g4_mu_t9 = 1.0
_g4_T_t9 = float(np.pi)


def _g4_eq_lr4_t9_torch(u, c, D):
    src = (torch.sin(c['x']) * torch.sin(c['y'])
           * (_g4_mu_t9 * torch.cos(_g4_mu_t9 * c['t'])
              + (_g4_a_t9 + _g4_b_t9) * torch.sin(_g4_mu_t9 * c['t'])))
    return D(u, c['t']) - _g4_a_t9 * D(u, c['x'], 2) - _g4_b_t9 * D(u, c['y'], 2) - src


def _g4_exact_lr4_t9(x, y, t):
    return np.sin(x) * np.sin(y) * np.sin(_g4_mu_t9 * t)


def _g4_src_lr4_t9_numpy(X, Y, t):
    return (np.sin(X) * np.sin(Y)
            * (_g4_mu_t9 * np.cos(_g4_mu_t9*t)
               + (_g4_a_t9 + _g4_b_t9) * np.sin(_g4_mu_t9*t)))


LR4_TASK_9 = {
    'name': 'Параболическое 2D: анизотропная диффузия с источником',
    'pde_str': 'u_t = a u_xx + b u_yy + sin x · sin y · (μ cos μt + (a+b) sin μt)',
    'domain': {'x': (0.0, np.pi/2), 'y': (0.0, np.pi), 't': (0.0, _g4_T_t9)},
    'exact': _g4_exact_lr4_t9,
    'fdm': {
        'ax': _g4_a_t9, 'ay': _g4_b_t9,
        'source_numpy': _g4_src_lr4_t9_numpy,
        'initial_numpy': lambda X, Y: np.zeros_like(X),
        'boundaries_spec': {
            'left': {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                     'gamma_numpy': lambda y, t: 0.0},
            'right': {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                      'gamma_numpy': lambda y, t: np.sin(y) * np.sin(_g4_mu_t9*t)},
            'bottom': {'kind': 'dirichlet', 'alpha': 1.0, 'beta': 0.0,
                       'gamma_numpy': lambda x, t: 0.0},
            'top': {'kind': 'neumann', 'alpha': 0.0, 'beta': 1.0,
                    'gamma_numpy': lambda x, t: -np.sin(x) * np.sin(_g4_mu_t9*t),
                    'order': 2},
        },
        'nx': 41, 'ny': 61, 'nt': 401,
    },
    'pinn': {
        'equation_torch': _g4_eq_lr4_t9_torch,
        'conditions': [
            {'where': {'x': 0.0}, 'value_torch': 0.0,
             'value_numpy': lambda y, t: 0.0},
            {'where': {'x': np.pi/2},
             'value_torch': lambda c: torch.sin(c['y']) * torch.sin(_g4_mu_t9*c['t']),
             'value_numpy': lambda y, t: np.sin(y) * np.sin(_g4_mu_t9*t)},
            {'where': {'y': 0.0}, 'value_torch': 0.0,
             'value_numpy': lambda x, t: 0.0},
            {'type': 'neumann', 'deriv_var': 'y', 'where': {'y': np.pi},
             'value_torch': lambda c: -torch.sin(c['x']) * torch.sin(_g4_mu_t9*c['t']),
             'value_numpy': lambda x, t: -np.sin(x) * np.sin(_g4_mu_t9*t)},
            {'where': {'t': 0.0}, 'value_torch': 0.0,
             'value_numpy': lambda x, y: 0.0},
        ],
        'solver_kwargs': dict(
            hidden_size=96, num_hidden_layers=5, activation='tanh',
            loss_weighting='grad_norm', lambda_res=1.0, lambda_cond=10.0,
        ),
        'solve_kwargs': dict(
            n_epochs_adam=10000, n_collocation=4000, n_condition=400,
            lr=1e-3, use_lbfgs=True, lbfgs_max_iter=2500,
        ),
    },
}


LR4_TASKS = [LR4_TASK_1, LR4_TASK_5, LR4_TASK_9]
