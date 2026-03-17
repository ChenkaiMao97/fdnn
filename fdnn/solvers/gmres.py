import torch
from tqdm import tqdm
from fdnn.utils.plot_field3d import plot_3slices

def MAE(a, b):
    return torch.mean(torch.abs(a-b))/torch.mean(torch.abs(b))

def c2r(x):
    bs, sx, sy, sz, _ = x.shape
    return torch.view_as_real(x).reshape(bs, sx, sy, sz, 6)

def r2c(x):
    bs, sx, sy, sz, _ = x.shape
    return torch.view_as_complex(x.reshape(bs, sx, sy, sz, 3, 2))

class mygmres():
    def __init__(self):
        self.myop = None

    def matvec(self, x):
        raise NotImplementedError

    def dot(self, x, y):
        raise NotImplementedError

    def zeros_like(self, x):
        raise NotImplementedError

    def scale(self, x, a):
        raise NotImplementedError

    def axby(self, a, x, b, y):
        raise NotImplementedError

    def vecnorm(self, x):
        return torch.sqrt(self.dot(x, x))

    @torch.no_grad()
    def solve(self, b, tol=1e-6, max_iter=100, start_iter=0, relres_history=None, init_b_norm=None, verbose=False, return_xr_history=False, plot_iters=None, complex_type=torch.complex128):
        print("return_xr_history: ", return_xr_history, "plot_iters: ", plot_iters)
        assert torch.is_complex(b), "b must be complex"
        b = b.to(complex_type)

        beta = self.vecnorm(b)
        if init_b_norm is None:
            init_b_norm = beta

        V = []
        Z = []
        V.append(self.scale(b, 1/beta))
        H = torch.zeros((max_iter + 1, max_iter), dtype=complex_type)

        # Arnoldi process
        relres_history = relres_history if relres_history is not None else [1.0]

        x_history = []
        r_history = []
        if return_xr_history:
            assert plot_iters is not None, "plot_iters must be provided if return_xr_history is True"

        pbar = tqdm(range(max_iter), total=max_iter, desc="GMRES", leave=False)
        for j in pbar:
            z = self.M(V[j].to(torch.complex64)).to(complex_type)
            Z.append(z)
            w = self.myop(z)
            for i in range(j + 1):
                H[i, j] = self.dot(V[i], w) # very important that the conjugate transpose is on V[i]
                w = self.axby(1, w, -H[i, j], V[i])
                assert torch.is_complex(w), "w must be complex"

            H[j + 1, j] = self.vecnorm(w)
            V.append(self.scale(w, 1/H[j + 1, j]))

            num_iter = j + 1
            # Solve the least squares problem
            e1 = torch.zeros(num_iter + 1, dtype=complex_type)
            e1[0] = beta
            result = torch.linalg.lstsq(H[:num_iter + 1, :num_iter], e1, rcond=None)
            y = result.solution
            # compute residual norm using y: ||H*y - e1||
            residual_norm = self.vecnorm(H[:num_iter + 1, :num_iter]@y - e1)
            relres_norm = torch.abs(residual_norm/init_b_norm)
            relres_history.append(relres_norm.item())

            # Check for convergence
            if verbose:
                pbar.set_description(f"GMRES: Iteration {num_iter + start_iter}, Res norm: {torch.abs(residual_norm):.2e}, rel-Res norm: {relres_norm:.2e}")

            if relres_norm < tol:
                break

            if return_xr_history and j in plot_iters:
                x = self.zeros_like(b).to(complex_type)
                for i in range(num_iter):
                    x = self.axby(1, x, y[i], Z[i])
                assert x.dtype == complex_type, f"x must be {complex_type}, but got {x.dtype}"

                # Compute the residual
                r = self.axby(1, b, -1, self.myop(x))
                assert torch.is_complex(r) and r.dtype == complex_type, f"r must be type {complex_type}, but got {r.dtype}"
                x_history.append(x)
                r_history.append(r)

        x = self.zeros_like(b).to(complex_type)
        for i in range(j+1):
            x = self.axby(1, x, y[i], Z[i])
        assert x.dtype == complex_type, f"x must be {complex_type}, but got {x.dtype}"

        return x, relres_history, j+1 + start_iter, x_history, r_history

class mygmrestorch(mygmres):
    def __init__(
        self,
        model,
        myop,
        tol=1e-8,
        max_iter=3,
        complex_type=torch.complex128
    ):
        super().__init__()
        self.model = model
        self.myop = myop
        self.M = None
        self.tol = tol
        self.max_iter = max_iter
        self.complex_type = complex_type

    def setup_eps(self, eps, freq):
        self.model.setup(eps, freq)
        self.M = lambda src: r2c(self.model(c2r(src), freq))

    def dot(self, x, y):
        prod = torch.sum(torch.conj(x) * y)
        return prod

    def zeros_like(self, x):
        return torch.zeros_like(x)

    def scale(self, x, a):
        return a * x

    def axby(self, a, x, b, y):
        return a * x + b * y

    def vecnorm(self, x):
        _norm = torch.sqrt(self.dot(x, x))
        return _norm

    @torch.no_grad()
    def solve(self, b, verbose=False, return_xr_history=False, plot_iters=None, init_x=None):
        assert torch.is_complex(b), "b must be complex"

        with torch.no_grad():
            init_b_norm = self.vecnorm(b)
            if init_x is not None:
                b = b - self.myop(init_x)
                sol = init_x
            else:
                sol = self.zeros_like(b)

            x, relres_history, num_iter, x_history, r_history = super().solve(
                b, tol=self.tol, max_iter=self.max_iter, verbose=verbose,
                return_xr_history=return_xr_history, plot_iters=plot_iters,
                complex_type=self.complex_type)

            sol = self.axby(1, sol, 1, x)

        return sol, relres_history, x_history, r_history

    def solve_with_restart(self, b, tol, max_iter, restart, verbose, init_x=None, return_xr_history=False, plot_iters=None):
        assert torch.is_complex(b), "b must be complex"
        print("init_x: ", init_x, "1: ", return_xr_history, plot_iters)

        with torch.no_grad():
            init_b_norm = self.vecnorm(b)
            if init_x is not None:
                b = b - self.myop(init_x)
                sol = init_x
            else:
                sol = self.zeros_like(b)
            num_iter = 0
            relres_history = [1.0]
            x_history, r_history = [], []
            while num_iter < max_iter:
                if plot_iters is not None:
                    this_plot_iters = [i - num_iter for i in plot_iters if i - num_iter >= 0]
                    this_return_xr_history = len(this_plot_iters) > 0
                else:
                    this_plot_iters = None
                    this_return_xr_history = False
                e, relres_history, num_iter, this_x_history, this_r_history = super().solve(b, tol, restart, start_iter=num_iter, relres_history=relres_history, init_b_norm=init_b_norm, verbose=verbose, return_xr_history=this_return_xr_history, plot_iters=this_plot_iters)
                sol = self.axby(1, sol, 1, e)
                b = b - self.myop(e)

                if this_return_xr_history:
                    x_history += this_x_history
                    r_history += this_r_history
                if relres_history[-1] < tol:
                    break

        if return_xr_history:
            return sol, relres_history, x_history, r_history
        else:
            return sol, relres_history
