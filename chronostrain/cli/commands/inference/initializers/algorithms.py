import time
from logging import Logger

import numpy as np
import torch

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
        num_epochs: int,
        lr_decay_factor: float,
        lr_patience: int,
        iters: int,
        learning_rate: float,
        num_samples: int,
        logger: Logger,
        min_lr: float = 1e-4,
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
    if correlation_type == 'dirichlet':
        model = create_model(
            population=population,
            read_types=read_types,
            mean=torch.zeros(population.num_strains() - 1, device=cfg.torch_cfg.device),
            fragments=fragments,
            time_points=time_points,
            disable_quality=not cfg.model_cfg.use_quality_scores,
            db=db,
            logger=logger
        )
        solver = ADVIDirichletSolver(
            model=model,
            data=reads,
            db=db,
            read_batch_size=read_batch_size
        )
    else:
        model = create_model(
            population=population,
            read_types=read_types,
            mean=torch.zeros(population.num_strains(), device=cfg.torch_cfg.device),
            fragments=fragments,
            time_points=time_points,
            disable_quality=not cfg.model_cfg.use_quality_scores,
            db=db,
            logger=logger
        )
        solver = ADVIGaussianSolver(
            model=model,
            data=reads,
            correlation_type=correlation_type,
            db=db,
            read_batch_size=read_batch_size
        )

    callbacks = []
    uppers = [[] for _ in range(model.num_strains())]
    lowers = [[] for _ in range(model.num_strains())]
    medians = [[] for _ in range(model.num_strains())]
    elbo_history = []

    if save_training_history:
        from chronostrain.util.math.activations import sparsemax
        def anim_callback(x_samples, uppers_buf, lowers_buf, medians_buf):
            # Plot VI posterior.
            y_samples = model.latent_conversion(x_samples)
            abund_samples = y_samples.cpu().detach().numpy()

            for s_idx in range(model.num_strains()):
                traj_samples = abund_samples[:, :, s_idx]  # (T x N)
                upper_quantile = np.quantile(traj_samples, q=0.975, axis=1)
                lower_quantile = np.quantile(traj_samples, q=0.025, axis=1)
                median = np.quantile(traj_samples, q=0.5, axis=1)
                uppers_buf[s_idx].append(upper_quantile)
                lowers_buf[s_idx].append(lower_quantile)
                medians_buf[s_idx].append(median)

        callbacks.append(lambda epoch, x_samples, elbo: anim_callback(x_samples, uppers, lowers, medians))

    if save_elbo_history:
        def elbo_callback(elbo, elbo_buf):
            elbo_buf.append(elbo)
        callbacks.append(lambda epoch, x_samples, elbo: elbo_callback(elbo, elbo_history))

    start_time = time.time()
    solver.solve(
        optimizer_class=torch.optim.Adam,
        optimizer_args={'lr': learning_rate, 'betas': (0.9, 0.999), 'eps': 1e-7, 'weight_decay': 0.0},
        iters=iters,
        num_epochs=num_epochs,
        num_samples=num_samples,
        min_lr=min_lr,
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
