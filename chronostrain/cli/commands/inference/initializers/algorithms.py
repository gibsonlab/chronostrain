from logging import Logger

import jax.numpy as np
from chronostrain.algs import *
from chronostrain.database import StrainDatabase
from chronostrain.model import *
from chronostrain.model.io import TimeSeriesReads
from chronostrain.config import cfg
from .model import create_model


def perform_advi(
        db: StrainDatabase,
        population: Population,
        fragments: FragmentSpace,
        reads: TimeSeriesReads,
        with_zeros: bool,
        num_epochs: int,
        lr_decay_factor: float,
        lr_patience: int,
        iters: int,
        learning_rate: float,
        num_samples: int,
        logger: Logger,
        min_lr: float = 1e-6,
        loss_tol: float = 1e-5,
        read_batch_size: int = 5000,
        correlation_type: str = "strain",
        save_elbo_history: bool = False,
        save_training_history: bool = False
):
    read_types = {
        src.read_type
        for reads_t in reads.time_slices
        for src in reads_t.sources
    }

    # ==== Run the solver.
    time_points = [time_slice.time_point for time_slice in reads]
    model = create_model(
        population=population,
        read_types=read_types,
        mean=np.zeros(population.num_strains(), dtype=cfg.engine_cfg.dtype),
        fragments=fragments,
        time_points=time_points,
        disable_quality=not cfg.model_cfg.use_quality_scores,
        db=db,
        logger=logger
    )

    if with_zeros:
        from chronostrain.util.optimization import Adam, ReduceLROnPlateauLast
        from chronostrain.model.zeros import PopulationGlobalZeros
        zero_model = PopulationGlobalZeros(model.bacteria_pop.num_strains())
        lr_scheduler = ReduceLROnPlateauLast(
            initial_lr=learning_rate,
            mode='max',
            min_lr=min_lr,
            factor=lr_decay_factor,
            patience=lr_patience,
            threshold=1e-4,
            threshold_mode='rel'
        )
        optimizer = Adam(
            lr_scheduler=lr_scheduler,
            minimize_objective=False,
            eps=1e-4
        )
        import time
        print("TODO implement an option for elbo_mode argument.")
        time.sleep(3)

        solver = ADVIGaussianZerosSolver(
            model=model,
            zero_model=zero_model,
            data=reads,
            db=db,
            optimizer=optimizer,
            correlation_type=correlation_type,
            elbo_mode="default",
            read_batch_size=read_batch_size,
            dtype=cfg.engine_cfg.dtype
        )
    else:
        from chronostrain.util.optimization import ConstantLearningRate, SGD, Adam, ReduceLROnPlateauLast, DistributedShampoo
        lr_scheduler = ReduceLROnPlateauLast(
            initial_lr=learning_rate,
            mode='max',
            min_lr=min_lr,
            factor=lr_decay_factor,
            patience=lr_patience,
            threshold=1e-4,
            threshold_mode='rel'
        )
        optimizer = Adam(
            lr_scheduler=lr_scheduler,
            minimize_objective=False,
            eps=1e-4
        )
        solver = ADVIGaussianSolver(
            model=model,
            data=reads,
            optimizer=optimizer,
            correlation_type=correlation_type,
            db=db,
            read_batch_size=read_batch_size,
            dtype=cfg.engine_cfg.dtype
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
        lr_decay_factor=lr_decay_factor,
        lr_patience=lr_patience,
        callbacks=callbacks
    )
    end_time = time.time()
    logger.debug("Finished inference in {} sec.".format(
        (end_time - start_time)
    ))

    posterior = solver.posterior
    return solver, posterior, elbo_history, (uppers, lowers, medians)
