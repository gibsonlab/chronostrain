from logging import Logger
from typing import *

import jax.numpy as np
from chronostrain.algs import *
from chronostrain.database import StrainDatabase
from chronostrain.model import *
from chronostrain.model.io import TimeSeriesReads
from chronostrain.util.optimization import *
from chronostrain.config import cfg
from .model import create_model


def bias_EM_initializer(population: Population, time_points: List[float]) -> np.ndarray:
    raise NotImplementedError()


def bias_semisynthetic_debug_init(population: Population, time_points: List[float]) -> np.ndarray:
    import numpy as cnp
    target_accs = ['NZ_CP043213.1', 'NZ_CP060963.1', 'NZ_CP027140.1', 'NZ_CP030240.1', 'NZ_CP065611.1', 'NZ_CP024257.1']
    ground_truth = cnp.array([
        [0.26143791, 0.2745098, 0.09803922, 0.29411765, 0.0130719, 0.05882353],
        [0.31460674, 0.34831461, 0.13483146, 0.12359551, 0.03370787, 0.04494382],
        [0.25906736, 0.50086356, 0.07772021, 0.06908463, 0.08635579, 0.00690846],
        [0.15217391, 0.54347826, 0.06521739, 0.03913043, 0.19565217, 0.00434783],
        [0.06423983, 0.51391863, 0.04496788, 0.01284797, 0.25695931, 0.10706638],
        [0.0129199, 0.29715762, 0.01550388, 0.00258398, 0.51679587, 0.15503876]
    ])
    print("using ground truth {}".format(ground_truth))

    ground_truth_indices = {acc: i for i, acc in enumerate(target_accs)}
    target = cnp.zeros(shape=(len(time_points), population.num_strains()), dtype=float)

    targets_left = set(target_accs)
    for i, s in enumerate(population.strains):
        if s.id in ground_truth_indices:
            ground_truth_idx = ground_truth_indices[s.id]
            target[:, i] = ground_truth[:, ground_truth_idx]
            print("Found target: {} -> {}".format(s.id, target[:, i]))
            targets_left.remove(s.id)
    if len(targets_left) > 0:
        print("**** Strains not observed for initialization: {}".format(targets_left))

    eps = 1e-10
    target = target + eps  # padding to prepare for log
    target = target / target.sum(axis=-1, keepdims=True)  # renormalize
    target = cnp.log(target)  # log-space conversion
    return np.array(target, dtype=cfg.engine_cfg.dtype)


def bias_uniform_initializer(population: Population, time_points: List[float]) -> np.ndarray:
    return np.zeros(
        shape=(len(time_points), population.num_strains()),
        dtype=cfg.engine_cfg.dtype
    )



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
        bias_initializer = bias_EM_initializer
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
        # logger.info("DEBUGGING using ground truth initializer.")
        bias_initializer = bias_uniform_initializer

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
        if prior_p != 0.5:
            logger.info(f"Initialized model inclusion prior with p={prior_p}")
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
            bias_initializer=bias_initializer,
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
            bias_initializer=bias_initializer,
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
