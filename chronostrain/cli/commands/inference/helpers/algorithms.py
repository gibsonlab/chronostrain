from logging import Logger

import jax.numpy as np
from chronostrain.algs import *
from chronostrain.database import StrainDatabase
from chronostrain.model import *
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.optimization import *
from chronostrain.config import cfg
from .model import create_model


def perform_advi(
        db: StrainDatabase,
        population: Population,
        fragments: FragmentSpace,
        reads: TimeSeriesReads,
        with_zeros: bool,
        initialize_with_map: bool,
        prune_strains: bool,
        num_epochs: int,
        lr_decay_factor: float,
        lr_patience: int,
        iters: int,
        learning_rate: float,
        num_samples: int,
        logger: Logger,
        accumulate_gradients: bool,
        prior_p: float = 0.5,
        min_lr: float = 1e-6,
        loss_tol: float = 1e-5,
        read_batch_size: int = 5000,
        correlation_type: str = "strain",
        save_elbo_history: bool = False,
        save_training_history: bool = False
):
    # ==== Run the solver.
    time_points = [time_slice.time_point for time_slice in reads]
    model = create_model(
        population=population,
        source_reads=reads,
        fragments=fragments,
        time_points=time_points,
        disable_quality=not cfg.model_cfg.use_quality_scores,
        db=db,
        logger=logger
    )

    if initialize_with_map:
        raise NotImplementedError("initialize_with_map option is not implemented.")
        # logger.debug("Running MAP solver, to initialize VI optimization.")
        # map_solver = AutogradMAPSolver(
        #     generative_model=model,
        #     data=reads,
        #     db=db,
        #     dtype=cfg.engine_cfg.dtype
        # )
        # initial_bias = map_solver.solve(iters=5, num_epochs=1000, loss_tol=1e-7, stepsize=1e-5)
        # np.save("map_soln.npy", initial_bias)
        # exit(1)
        # initial_bias, _, _ = em_solver.solve(iters=1000, thresh=1e-5, gradient_clip=1e3, print_debug_every=1)
    else:
        initial_bias = None

    lr_scheduler = ReduceLROnPlateauLast(
        initial_lr=learning_rate,
        mode='max',
        min_lr=0.1 * min_lr,
        factor=lr_decay_factor,
        patience=lr_patience,
        threshold=loss_tol * 1e1,
        threshold_mode='rel',
        eps=0.01 * min_lr
    )
    optimizer = Adam(
        lr_scheduler=lr_scheduler,
        minimize_objective=False
    )

    if with_zeros:
        from chronostrain.model.zeros import PopulationGlobalZeros
        zero_model = PopulationGlobalZeros(model.bacteria_pop.num_strains(), prior_p=prior_p)
        solver = ADVIGaussianZerosSolver(
            model=model,
            zero_model=zero_model,
            data=reads,
            db=db,
            optimizer=optimizer,
            correlation_type=correlation_type,
            accumulate_gradients=accumulate_gradients,
            read_batch_size=read_batch_size,
            dtype=cfg.engine_cfg.dtype,
            initial_gaussian_bias=initial_bias,
            prune_strains=prune_strains
        )
    else:
        solver = ADVIGaussianSolver(
            model=model,
            data=reads,
            optimizer=optimizer,
            correlation_type=correlation_type,
            accumulate_gradients=accumulate_gradients,
            db=db,
            read_batch_size=read_batch_size,
            dtype=cfg.engine_cfg.dtype,
            initial_gaussian_bias=initial_bias,
            prune_strains=prune_strains
        )

    callbacks = []
    uppers = [[] for _ in range(model.num_strains())]
    lowers = [[] for _ in range(model.num_strains())]
    medians = [[] for _ in range(model.num_strains())]
    elbo_history = []

    if save_training_history:
        logger.warning("Currently, save_training_history has been disabled to make callbacks more efficient.")
    #     def anim_callback(x_samples, uppers_buf, lowers_buf, medians_buf):
    #         # Plot VI posterior.
    #         abund_samples = model.latent_conversion(x_samples)
    #
    #         for s_idx in range(model.num_strains()):
    #             traj_samples = abund_samples[:, :, s_idx]  # (T x N)
    #             upper_quantile = np.quantile(traj_samples, q=0.975, axis=1)
    #             lower_quantile = np.quantile(traj_samples, q=0.025, axis=1)
    #             median = np.quantile(traj_samples, q=0.5, axis=1)
    #             uppers_buf[s_idx].append(upper_quantile)
    #             lowers_buf[s_idx].append(lower_quantile)
    #             medians_buf[s_idx].append(median)
    #
    #     callbacks.append(lambda epoch, elbo: anim_callback(x_samples, uppers, lowers, medians))

    if save_elbo_history:
        def elbo_callback(elbo, elbo_buf):
            elbo_buf.append(elbo)
        callbacks.append(lambda epoch, elbo: elbo_callback(elbo, elbo_history))

    import time
    start_time = time.time()
    solver.solve(
        iters=iters,
        num_epochs=num_epochs,
        num_samples=num_samples,
        min_lr=min_lr,
        loss_tol=loss_tol,
        callbacks=callbacks
    )
    end_time = time.time()
    logger.debug("Finished inference in {} sec.".format(
        (end_time - start_time)
    ))

    posterior = solver.posterior
    return solver, posterior, elbo_history, (uppers, lowers, medians)
