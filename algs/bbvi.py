"""
  bbvi.py
  Black-box Variational Inference
  Author: Younhun Kim, with some code from https://github.com/jamesvuc/BBVI/blob/master/bbvi.py
"""

from abc import ABCMeta, abstractmethod

import datetime as dt

import numpy as np

from autograd import grad
from autograd.misc.optimizers import adam
from autograd.core import getval
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm as norm

# TODO: consider re-implementing in pytorch (and enable GPU support.)


"""
  Abstract class due to https://github.com/jamesvuc/BBVI/blob/master/bbvi.py
"""
class BaseBBVIModel(metaclass=ABCMeta):
    """
    An abstract base class providing the structure for a general Bayesian
    inference problem to be solved using black box variational inference.
    We provide a number of ELBO graient approximations, with ease of experimentation
    being a primary goal.

    To use this framework, one must derive their own class (i.e. model), and implement
    the user-specified mehtods indicated below.

    The mechanics follow those of
    https://github.com/HIPS/autograd/blob/master/examples/black_box_svi.py
    """

    def __init__(self):
        self._init_var_params = None
        self._var_params = None
        self.N_SAMPLES = None

    """
    =======User-specified methods=====
    These methods must be implemented when the model is derived from this base class.
    The user-specified signatures should match those below,.
    """

    # Variational approx

    @abstractmethod
    def log_var_approx(self, z, params):
        """
        Computes the log variational approximation of z to the posterior log_prob
        using variational parameters params. Should be vectorized over z.
        """
        pass

    @abstractmethod
    def sample_var_approx(self, params, n_samples=1000):
        """
        Returns samples from the variational approximation with parameters params.
        """
        pass

    # Joint Distribution
    @abstractmethod
    def log_prob(self, z):
        """
        Computes the log-posterior of latent variables z.
        """
        pass

    def callback(self, *args):
        """
        Optional method called once per optimization step.
        """
        pass

    """
    =======-Generic VI methods=======
    """

    """------Stochastic Search-------"""

    def _objfunc(self, params, t):
        """
        Implements an unadjusted stochastic-search BBVI gradient estimate according
        to https://arxiv.org/abs/1401.0118.
        """
        samps = self.sample_var_approx(getval(params), n_samples=self.N_SAMPLES)

        return np.mean(
            self.log_var_approx(samps, params) * (self.log_prob(samps) - self.log_var_approx(samps, getval(params))))

    def _objfuncCV(self, params, t):
        """
        Experimental: Implements a version of above with an estimated control variate.
        """
        raise NotImplementedError("TODO Fix the control variate scaling!")

        samps = self.sample_var_approx(getval(params), n_samples=self.N_SAMPLES)

        a_hat = np.mean(self.log_prob(samps) - self.log_var_approx(samps, getval(params)))  # TODO fix

        return np.mean(self.log_var_approx(samps, params) * (
                    self.log_prob(samps) - self.log_var_approx(samps, getval(params)) - a_hat))

    """-----Reparameterization Trick--------"""

    def _estimate_ELBO(self, params, t):
        """
        Implements the ELBO estimate from http://www.cs.toronto.edu/~duvenaud/papers/blackbox.pdf
        which in turn implements the reparamerization trick from https://arxiv.org/abs/1506.02557
        """
        samps = self.sample_var_approx(params, n_samples=self.N_SAMPLES)

        # estimates -E[log p(z)-log q(z)]
        return -np.mean(self.log_prob(samps) - self.log_var_approx(samps, params),
                        axis=0)  # this one appears to be correct

    def _estimate_ELBO_noscore(self, params, t):
        """
        Implements the ELBO estimate from
        https://papers.nips.cc/paper/7268-sticking-the-landing-simple-lower-variance-gradient-estimators-for-variational-inference.pdf
        which can reduce variance in certain cases.
        """
        samps = self.sample_var_approx(params, n_samples=self.N_SAMPLES)

        # eliminates the score function
        return -np.mean(self.log_prob(samps) - self.log_var_approx(samps, getval(params)),
                        axis=0)  # this one appears to be correct

    """-----Optimization------"""

    def run_VI(self, init_params, num_samples=50, step_size=0.01, num_iters=2000, method='stochsearch'):
        methods = ['stochsearch', 'reparam', 'noscore']
        if method not in methods:
            raise KeyError('Allowable VI methods are', methods)

        self.N_SAMPLES = num_samples

        # select the gradient type
        if method == 'stochsearch':
            # not CV
            _tmp_gradient = grad(self._objfunc)
        # CV
        # _tmp_gradient=grad(self._objfuncCV)

        elif method == 'reparam':
            _tmp_gradient = grad(self._estimate_ELBO)

        elif method == 'noscore':
            _tmp_gradient = grad(self._estimate_ELBO_noscore)

        else:
            raise Exception("Allowable ELBO estimates are", methods)

        # set the initial parameters
        self._init_var_params = init_params

        # start the clock
        s = dt.datetime.now()

        # run the VI
        self._var_params = adam(_tmp_gradient, self._init_var_params,
                                step_size=step_size,
                                num_iters=num_iters,
                                callback=self.callback
                                )

        return self._var_params


# ========= START EXAMPLE
class TestModel1(BaseBBVIModel):
    def __init__(self, D=2):
        self.dim=D
        plt.show(block=False)
        self.fig, self.ax=plt.subplots(2)
        self.elbo_hist=[]
        super().__init__(self)

    # specify the variational approximator
    def unpack_params(self, params):
        # print('params shape',params.shape)
        return params[:, 0], params[:, 1]

    def log_var_approx(self, z, params):
        mu, log_sigma=self.unpack_params(params)
        sigma=np.diag(np.exp(2*log_sigma))+1e-6
        return mvn.logpdf(z, mu, sigma)

    def sample_var_approx(self, params, n_samples=2000):
        mu, log_sigma=self.unpack_params(params)
        return npr.randn(n_samples, mu.shape[0])*np.exp(log_sigma)+mu

    # specify the distribution to be approximated
    def log_prob(self, z):
        mu, log_sigma = z[:, 0], z[:, 1]#this is a vectorized extraction of mu,sigma
        sigma_density = norm.logpdf(log_sigma, 0, 1.35)
        mu_density = norm.logpdf(mu, 0, np.exp(log_sigma))

        return sigma_density + mu_density

    def plot_isocontours(self, ax, func, xlimits=[-2, 2], ylimits=[-4, 2], numticks=101):
        x = np.linspace(*xlimits, num=numticks)
        y = np.linspace(*ylimits, num=numticks)
        X, Y = np.meshgrid(x, y)
        zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
        Z = zs.reshape(X.shape)
        # plt.contour(X, Y, Z)
        ax.contour(X, Y, Z)
        ax.set_yticks([])
        ax.set_xticks([])


    def callback(self, *args):
        self.elbo_hist.append(self._estimate_ELBO(args[0], 0))
        if args[1]%50==0:
            print(args[1])
            curr_params=args[0]
            for a in self.ax:
                a.cla()
            self.plot_isocontours(self.ax[0], lambda z:np.exp(self.log_prob(z)))
            self.plot_isocontours(self.ax[0], lambda z:np.exp(self.log_var_approx(z, curr_params)))
            self.ax[1].plot(self.elbo_hist)
            self.ax[1].set_title('elbo estimate='+str(round(self.elbo_hist[-1],4)))
            plt.pause(1.0/30.0)

            plt.draw()
 # ========= END EXAMPLE


class BBVIImplementation(BaseBBVIModel):
    def log_var_approx(self, z, params):
        pass

    def sample_var_approx(self, params, n_samples=1000):
        pass

    def log_prob(self, z):
        pass

    def run_VI(self, init_params, num_samples=50, step_size=0.01, num_iters=2000, how='blackwell-rao'):
        pass