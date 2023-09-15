from typing import *

import numpy as cnp
import jax
import jax.numpy as np

from chronostrain.inference.algs.vi.base.util import log_mm_exp
from chronostrain.model.io import TimeSeriesReads
from chronostrain.model.generative import AbundanceGaussianPrior
from chronostrain.inference.algs.base import AbstractModelSolver

from chronostrain.logging import create_logger
from chronostrain.database import StrainDatabase
from chronostrain.util.math import log_spspmm_exp_sparsey

logger = create_logger(__name__)



# noinspection PyPep8Naming
class AutogradMAPSolver(AbstractModelSolver):
    """
    MAP estimation via Expectation-Maximization.
    """
    def __init__(self,
                 generative_model: AbundanceGaussianPrior,
                 data: TimeSeriesReads,
                 db: StrainDatabase,
                 dtype):
        """
        Instantiates an EMSolver instance.

        :param generative_model: The underlying generative model with prior parameters.
        :param data: the observed read_frags, a time-indexed list of read collections.
        :param db: The StrainDatabase instance.
        :param lr: the learning rate (default: 1e-3)
        """
        super().__init__(generative_model, data, db)
        self.dtype = dtype

    def precompute_data_lls(self):
        data_lls = []
        log_y_inits = []
        for t, m in enumerate(self.data_likelihoods.matrices):
            data_ll_t = log_spspmm_exp_sparsey(
                self.model.fragment_frequencies_sparse.T,  # (F,S) transposed -> (S,F)
                m  # (F,R)
            )  # (S,R)

            # ========= Locate and filter out reads with no good alignments.
            nz = ~np.equal(np.sum(~np.isinf(data_ll_t), axis=0), 0)
            n_good_indices = np.sum(nz)
            if n_good_indices < data_ll_t.shape[1]:
                logger.debug("(t = {}) Found {} of {} reads without good alignments.".format(
                    t, data_ll_t.shape[1] - n_good_indices, data_ll_t.shape[1]
                ))
                data_ll_t = data_ll_t[:, nz]
            data_lls.append(data_ll_t)

            # Pick initializations.
            log_y_init_t = jax.nn.log_softmax(jax.nn.logsumexp(data_ll_t, axis=-1))
            log_y_init_t = np.where(np.isinf(log_y_init_t), -10000, log_y_init_t)
            if t == 0:
                log_y_inits.append(log_y_init_t - log_y_init_t.mean())
            else:
                log_y_inits.append(log_y_init_t)
        return data_lls, np.stack(log_y_inits, axis=0)

    def solve(self,
              iters: int = 20,
              num_epochs: int = 500,
              loss_tol: float = 1e-5,
              stepsize: float = 1e-3) -> np.ndarray:
        posterior_ll, x_init = self.compile_posterior()
        x_init = np.zeros(x_init.shape)

        logger.info("Running autograd MAP solver.")

        # ================ MAIN OPTIMIZER =================
        # import jax.scipy.optimize as jsp_opt
        # res = jsp_opt.minimize(
        #     lambda x: -self.posterior_ll(np.reshape(x, (self.model.num_times(), self.model.num_strains()))),
        #     method='BFGS',
        #     x0=np.zeros((self.model.num_times() * self.model.num_strains()), dtype=self.dtype),
        #     tol=1e-5
        # )
        # logger.debug("Optimizer result:")
        # logger.debug(f'Success: {res.success}, Status: {res.status}, Fun: {res.fun}, Iters: {res.nit}')
        # return np.reshape(res.x, (self.model.num_times(), self.model.num_strains()))

        return self.optimize(posterior_ll, x_init, num_epochs, iters, loss_tol)

    def compile_posterior(self):
        data_lls, initialization = self.precompute_data_lls()

        n_data = np.array([
            len(self.data.time_slices[t_idx])
            for t_idx in range(self.model.num_times())
        ], dtype=self.dtype)  # length T
        log_total_marker_lens = np.array([
            np.log(sum(len(m) for m in strain.markers))
            for strain in self.model.bacteria_pop.strains
        ], dtype=self.dtype)  # length S: total marker nucleotide length of each strain
        log_total_marker_lens = np.expand_dims(
            log_total_marker_lens, axis=[0, 1]
        )

        @jax.jit
        def posterior_ll(x: np.ndarray):
            # ll = self.model.log_likelihood_x(x=x).squeeze(0)

            n_times, n_strains = x.shape
            # eps = 1e-5
            # ll_first = -0.5 * n_strains * np.log(np.square(
            #     x[0] - self.model.mu
            # ).sum() + eps)
            # ll_rest = -0.5 * (n_times - 1) * n_strains * np.log(np.square(
            #     np.expand_dims(self.model.dt_sqrt_inverse, axis=[1, 2]) * np.diff(x, n=1, axis=0)
            # ).sum() + (n_times - 1) * eps)
            # ll = ll_first + ll_rest

            ll = 0.
            log_y = jax.nn.log_softmax(x, axis=-1)
            for t in range(n_times):
                log_y_t = jax.lax.dynamic_slice_in_dim(log_y, start_index=t, slice_size=1, axis=0)  # (1 x S)
                data_ll_t = data_lls[t]
                ll_data_t = log_mm_exp(
                    log_y_t,  # (1,S)
                    data_ll_t  # (S,R)
                ).sum()  # (1,R) summed up
                ll = ll + ll_data_t

            correction = -n_data * jax.scipy.special.logsumexp(log_y + log_total_marker_lens, axis=-1)
            ll = ll + np.sum(correction)
            return ll
        return posterior_ll, initialization

    def optimize(self,
                 posterior_ll: Callable,
                 x_init: np.ndarray,
                 num_epochs: int,
                 iters: int,
                 loss_tol: float):
        from chronostrain.util.benchmarking import RuntimeEstimator
        time_est = RuntimeEstimator(total_iters=num_epochs, horizon=10)

        logger.info("Starting Posterior optimization.")
        optim = CustomOptimizer(posterior_ll, maxiter=iters * num_epochs, minimize_objective=False)
        epoch_ll_prev = -cnp.inf
        optim.initialize(x_init)
        for epoch in range(1, num_epochs + 1, 1):
            # =========== Store ELBO values for reporting.
            time_est.stopwatch_click()
            for it in range(1, iters + 1, 1):
                # ========== Perform optimization for each iteration.
                optim.update()

            # ===========  End of epoch
            secs_elapsed = time_est.stopwatch_click()
            time_est.increment(secs_elapsed)
            ll_value = posterior_ll(optim.params)
            logger.info(
                "Epoch {epoch} | time left: {t:.2f} min. | Last log-p = {ll_value:.2f} | LR = {lr}".format(
                    epoch=epoch,
                    t=time_est.time_left() / 60000,
                    ll_value=ll_value,
                    lr=optim.current_learning_rate()
                )
            )

            optim.lr_scheduler.step(posterior_ll(optim.params))
            if loss_tol is not None:
                if cnp.abs(ll_value - epoch_ll_prev) < loss_tol * cnp.abs(epoch_ll_prev):
                    logger.info("Stopping criteria (Log-p rel. diff < {}) met after {} epochs.".format(loss_tol, epoch))
                    break
                epoch_ll_prev = ll_value

        # ========== End of optimization
        logger.info("Finished.")
        return optim.params


class CustomOptimizer:
    def __init__(self, objective_fn, maxiter, minimize_objective: bool = True):
        from chronostrain.util.optimization import ReduceLROnPlateauLast
        import jaxopt
        import optax
        self.objective_fn = objective_fn
        self.lr_scheduler = ReduceLROnPlateauLast(
            initial_lr=1e-3,
            min_lr=1e-6,
            mode='max',
            factor=0.25,
            patience=10,
            threshold=1e-4,
            threshold_mode='rel'
        )

        if not minimize_objective:  # want to maximize
            def obj(x):
                return -objective_fn(x)
        else:
            def obj(x):
                return objective_fn(x)

        print("USING ADAM")
        self.solver = jaxopt.OptaxSolver(
            fun=obj,
            opt=optax.adam(
                learning_rate=self.lr_scheduler.get_optax_scheduler()
            ),
            value_and_grad=False,
            maxiter=maxiter,
            jit=True,
            tol=1e-5
        )
        # self.solver = jaxopt.GradientDescent(
        #     fun=obj,
        #     maxiter=maxiter,
        #     jit=True,
        #     tol=1e-4,
        #     stepsize=lambda t: 1 / np.sqrt(t),
        # )
        self.params = None
        self.state = None

    def initialize(self, initial_params):
        self.params = initial_params
        self.state = self.solver.init_state(initial_params)

    def current_learning_rate(self) -> float:
        return self.lr_scheduler.get_current_lr()

    def update(self):
        if self.params is None or self.state is None:
            raise RuntimeError("Loss optimizer must be initialized before running.")
        self.params, self.state = self.solver.update(self.params, self.state)

# import optax
# class AdamCustom:
#     def __init__(
#             self,
#             objective_fn,
#             minimize_objective: bool = True,
#     ):
#         self.objective_fn = objective_fn
#         if not minimize_objective:  # want to maximize
#             def obj(x):
#                 return -objective_fn(x)
#         else:
#             def obj(x):
#                 return objective_fn(x)
#         self.obj_with_grad = jax.value_and_grad(obj)
#
#         self.scheduler = lr_scheduler
#         self.optim = optax.adam(
#             learning_rate=lr_scheduler.get_optax_scheduler(),
#             **hyperparameters
#         )
#         self.params = None
#         self.state = None
#         self.grad_sign = 1 if minimize_objective else -1
#
#     def initialize(self, initial_params: Union[Dict[str, np.ndarray], np.ndarray]):
#         self.params = initial_params
#         self.state = self.optim.init(initial_params)
#
#     def current_learning_rate(self) -> float:
#         return self.scheduler.get_current_lr()
#
#     def update(self, grad: Union[Dict[str, np.ndarray], np.ndarray]):
#         if self.params is None or self.state is None:
#             raise RuntimeError("Loss optimizer must be initialized before running.")
#
#         if isinstance(grad, Dict):
#             for k, w in grad.items():
#                 grad[k] = self.grad_sign * grad[k]
#         elif isinstance(grad, np.ndarray):
#             grad = self.grad_sign * grad
#         else:
#             raise RuntimeError("Unknown grad type `{}`.".format(type(grad)))
#         updates, new_opt_state = self.optim.update(grad, self.state, self.params)
#         self.params = optax.apply_updates(self.params, updates)
#         self.state = new_opt_state
