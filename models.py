import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tinygp 
import inspect
import numpy as np

@tinygp.helpers.dataclass
class Multiband(tinygp.kernels.quasisep.Wrapper):
    """
    Implements the multiband GP model described here: 
    https://tinygp.readthedocs.io/en/latest/tutorials/quasisep-custom.html#multivariate-quasiseparable-kernels
    """
    
    amplitudes: jnp.ndarray

    def coord_to_sortable(self, X):
        return X[0]

    def observation_model(self, X):
        return self.amplitudes[X[1]] * self.kernel.observation_model(X[0])

def gauss_likelihood(resids, log_diag):

    return -(
        jnp.sum(jnp.square(resids) / (2 * jnp.exp(log_diag))) 
        + len(resids) * log_diag)

class multiband_no_gp():

    def __init__(self, t, nbands, mean, constant_params=[]):

        self.t = t
        self.n = len(t)
        self.nbands = nbands
        self.mean = mean
        self.constant_params = constant_params

        self.log_diags = ["log_diag[{}]".format(i) for i in range(self.nbands)]
        self.mean_args = inspect.getfullargspec(self.mean).args[1:]
        in_axes = [0] * len(self.mean_args)
        for h in self.constant_params:
            if h in self.mean_args:
                in_axes[self.mean_args.index(h)] = None

        unpack_key = (np.array(in_axes) == 0) * (self.nbands - 1) + 1
        self.mean_param_names = np.concatenate([
            [
                arg + '[{0}]'.format(i) if key > 1 else arg for i in range(key)
            ] for arg, key in zip(self.mean_args, unpack_key)
        ])

        self.vmapped_mean = jax.vmap(lambda *p: self.mean(self.t, *p), in_axes=in_axes)
        self.slices = []
        idx = 0
        for key in unpack_key:
            if key == 1:
                self.slices.append(idx)
                idx += 1
            else:
                self.slices.append(slice(idx, idx + key))
                idx += key

        self.param_names = [
            "noise:" + p for p in self.log_diags
        ] + [
            "mean:" + p for p in self.mean_param_names
        ]

    def mean_builder(self):
        """
        Returns: A function that returns a 2d array 
            of size (nbands, len(t)) representing the 
            mean for each band when passed a jax array of 
            parameters in the order specified by self.mean_param_names
        """

        def mean(p):
            param_list = [None] * len(self.mean_args)
            for i, key in enumerate(self.slices):
                param_list[i] = p[key]
            return self.vmapped_mean(*param_list)

        return mean

    def get_logp(self, y):
        """
        y: A jax array of size (nbands, len(t)) containing 
            a set of observations of the flux in each band 
            observed at times t. 

        Returns: A function that returns the log probability 
            of the model given the data, which takes as input 
            a jax array of parameters in the order 
            specified by self.param_names
        """

        build_mean = self.mean_builder()

        @jax.jit
        def logp(p):

            log_diags = p[:self.nbands]
            mean_params = p[self.nbands:]
            mean = build_mean(mean_params)
            lp = 0.0
            for yi, ld, m in zip(y, log_diags, mean):
                lp += gauss_likelihood(yi - m[0], ld)

            return lp

        return logp
            
        

class multiband_independent():
    """
    A multiband GP model where the noise realization is independent between bands. 

    t: A jax array representing the time coordinate
    nbands: the number of bands 
    mean: a function that called as 
        mean(t, *params) which returns a 1d jax array representing 
        the deterministic component of the flux in a single band. 
        This function is broadcasted across the nbands bands. 
    constant_params: (optional) A list of names of parameters of the mean 
        function that should be held constant across the bands. 
        These names should match the input parameters of the mean function. 
    """

    def __init__(self, t, nbands, term, mean, constant_params=[]):

        self.t = t
        self.n = len(t)
        self.nbands = nbands
        self.term = term
        self.mean = mean
        self.constant_params = constant_params

        kernel_params = list(inspect.signature(term).parameters)
        self.nkp = len(kernel_params)

        in_axes_gp = [0] * len(kernel_params)
        for h in self.constant_params:
            if h in kernel_params:
                in_axes_gp[kernel_params.index(h)] = None

        unpack_key_gp = (np.array(in_axes_gp) == 0) * (self.nbands - 1) + 1
        self.kernel_params = list(np.concatenate(
            [
                [
                    kp if kp in self.constant_params else kp + '[{0}]'.format(i) for i in range(key)
                ] for kp, key in zip(kernel_params, unpack_key_gp)
            ]
        ))
        log_diags = ["log_diag[{}]".format(i) for i in range(self.nbands)]
        self.gp_param_names = self.kernel_params + log_diags
        self.ngp_params = len(self.gp_param_names)

        self.map_gp_params = jax.vmap(lambda *p: jnp.array(p), in_axes=in_axes_gp)
        self.slices_gp = []
        idx = 0
        for key in unpack_key_gp:
            if key == 1:
                self.slices_gp.append(idx)
                idx += 1
            else:
                self.slices_gp.append(slice(idx, idx + key))
                idx += key

        self.mean_args = inspect.getfullargspec(self.mean).args[1:]
        in_axes = [0] * len(self.mean_args)
        for h in self.constant_params:
            if h in self.mean_args:
                in_axes[self.mean_args.index(h)] = None

        unpack_key = (np.array(in_axes) == 0) * (self.nbands - 1) + 1
        self.mean_param_names = np.concatenate([
            [
                arg + '[{0}]'.format(i) if key > 1 else arg for i in range(key)
            ] for arg, key in zip(self.mean_args, unpack_key)
        ])

        self.vmapped_mean = jax.vmap(lambda *p: self.mean(self.t, *p), in_axes=in_axes)
        self.slices = []
        idx = 0
        for key in unpack_key:
            if key == 1:
                self.slices.append(idx)
                idx += 1
            else:
                self.slices.append(slice(idx, idx + key))
                idx += key

        self.param_names = [
            "noise:" + p for p in self.gp_param_names
        ] + [
            "mean:" + p for p in self.mean_param_names
        ]

    def gp_builder(self):
        """
        Returns: A function that returns a list of 
            GPs, one per band, when passed a jax array 
            of parameters in the order specified by 
            self.gp_param_names.
        """

        def gps(p):

            gp_params = [None] * self.nkp
            for i, key in enumerate(self.slices_gp):
                gp_params[i] = p[key]
            gp_params = self.map_gp_params(*gp_params)
            kernels = [self.term(*gpp) for gpp in gp_params]
            log_diags = p[len(self.kernel_params):]
            gps = [
                tinygp.GaussianProcess(
                    kernel, 
                    self.t, 
                    diag=jnp.exp(ld)
                ) for kernel, ld in zip(kernels, log_diags)
            ]
            return gps
                
        return gp_arr

    def mean_builder(self):
        """
        Returns: A function that returns a 2d array 
            of size (nbands, len(t)) representing the 
            mean for each band when passed a jax array of 
            parameters in the order specified by self.mean_param_names
        """

        def mean(p):
            param_list = [None] * len(self.mean_args)
            for i, key in enumerate(self.slices):
                param_list[i] = p[key]
            return self.vmapped_mean(*param_list)

        return mean

    def get_logp(self, y):
        """
        y: A jax array of size (nbands, len(t)) containing 
            a set of observations of the flux in each band 
            observed at times t. 

        Returns: A function that returns the log probability 
            of the model given the data, which takes as input 
            a jax array of parameters in the order 
            specified by self.param_names
        """

        build_mean = self.mean_builder()
        build_gps = self.gp_builder()

        @jax.jit
        def logp(p):

            gp_params = p[:self.ngp_params]
            mean_params = p[self.ngp_params:]
            gps = build_gps(gp_params)
            mean = build_mean(mean_params)
            lp = 0.0
            for yi, gp, m in zip(y, gps, mean):
                lp += gp.log_probability(yi - m[0])
            return lp
            
        return logp


class multiband():
    """
    A multiband GP model where the noise realization is correlated across bands. 

    t: A jax array representing the time coordinate
    nbands: the number of bands 
    mean: a function that called as 
        mean(t, *params) which returns a 1d jax array representing 
        the deterministic component of the flux in a single band. 
        This function is broadcasted across the nbands bands. 
    constant_params: (optional) A list of names of parameters of the mean 
        function that should be held constant across the bands. 
        These names should match the input parameters of the mean function. 
    """

    def __init__(self, t, nbands, term, mean, constant_params=[]):

        self.t = t
        self.n = len(t)
        self.nbands = nbands
        self.term = term
        self.mean = mean
        self.constant_params = constant_params

        kernel_params = list(inspect.signature(term).parameters)
        self.nkp = len(kernel_params)
        
        scale_params = ["a[{0}]".format(i) for i in range(self.nbands - 1)]
        log_diags = ["log_diag[{}]".format(i) for i in range(self.nbands)]
        self.gp_param_names = kernel_params + scale_params + log_diags
        self.ngp_params = len(self.gp_param_names)

        self.mean_args = inspect.getfullargspec(self.mean).args[1:]
        in_axes = [0] * len(self.mean_args)
        for h in self.constant_params:
            if h in self.mean_args:
                in_axes[self.mean_args.index(h)] = None

        unpack_key = (np.array(in_axes) == 0) * (self.nbands - 1) + 1
        self.mean_param_names = np.concatenate([
            [
                arg + '[{0}]'.format(i) if key > 1 else arg for i in range(key)
            ] for arg, key in zip(self.mean_args, unpack_key)
        ])

        self.vmapped_mean = jax.vmap(lambda *p: self.mean(self.t, *p), in_axes=in_axes)
        self.slices = []
        idx = 0
        for key in unpack_key:
            if key == 1:
                self.slices.append(idx)
                idx += 1
            else:
                self.slices.append(slice(idx, idx + key))
                idx += key

        band_id = jnp.reshape(
            jnp.vstack(
                jnp.array([[jnp.ones(len(t), dtype=jnp.int32) * i] for i in range(nbands)])
            ).T
            , nbands * len(t)
        )
        x = jnp.sort(jnp.hstack([t] * nbands))
        self.X = (x, band_id)

        self.param_names = [
            "noise:" + p for p in self.gp_param_names
        ] + [
            "mean:" + p for p in self.mean_param_names
        ]
        
    def gp_builder(self):
        """
        Returns: A function that returns a multiband GP 
            when passed a jax array of parameters 
            in the order specified by self.gp_param_names.
        """

        def gp(p):

            idx = 0
            kernel = self.term(*p[:self.nkp])
            idx += self.nkp
            multiband_kernel = Multiband(
                kernel=kernel, 
                amplitudes=jnp.concatenate([jnp.array([1.0]), p[idx:idx + self.nbands - 1]])
            )
            idx += self.nbands - 1
            log_diags = jnp.tile(p[idx:], self.n)
            return tinygp.GaussianProcess(multiband_kernel, self.X, diag=jnp.exp(log_diags))

        return gp

    def mean_builder(self):
        """
        Returns: A function that returns a 2d array 
            of size (nbands, len(t)) representing the 
            mean for each band when passed a jax array of 
            parameters in the order specified by self.mean_param_names
        """

        def mean(p):
            param_list = [None] * len(self.mean_args)
            for i, key in enumerate(self.slices):
                param_list[i] = p[key]
            return self.vmapped_mean(*param_list)

        return mean

    def get_logp(self, y):
        """
        y: A jax array of size (nbands, len(t)) containing 
            a set of observations of the flux in each band 
            observed at times t. 

        Returns: A function that returns the log probability 
            of the model given the data, which takes as input 
            a jax array of parameters in the order 
            specified by self.param_names
        """

        build_mean = self.mean_builder()
        build_gp = self.gp_builder()

        Y = jnp.hstack(y.T)

        @jax.jit 
        def logp(p):

            gp_params = p[:self.ngp_params]
            mean_params = p[self.ngp_params:]
            gp = build_gp(gp_params)
            mean = build_mean(mean_params).T.flatten()
            return gp.log_probability(Y - mean)

        return logp

    def get_sampler(self):
        """
        Returns a function that returns samples from the model 
            when passed a jax array of parameters in the order specified 
            by model.param_names. The sample function also takes an 
            optional parameter, seed(=12345), which seeds jax's random 
            number generator for reproducibility. 
        """

        build_mean = self.mean_builder()
        build_gp = self.gp_builder()

        def sample(p, seed=12345):

            gp_params = p[:self.ngp_params]
            mean_params = p[self.ngp_params:]
            gp = build_gp(gp_params)
            mean = build_mean(mean_params).T.flatten()
            return gp.sample(jax.random.PRNGKey(seed)) + mean

        return sample

    def get_conditioner(self):
        """
        Returns a function that conditions the model on 
            a set of observations. The conditioning function 
            is called as condition(p, y, return_var=False) where
            p is a jax array of parameters in the order specified 
            by model.param_names and y is an nbands x len(t) array 
            of observations. If return_var=True the function returns 
            the variance of the conditioned GP in addition to the mean 
            of the conditioned GP. 
        """

        build_mean = self.mean_builder()
        build_gp = self.gp_builder()

        def condition(p, y, return_var=False):

            Y = jnp.hstack(y.T)
            gp_params = p[:self.ngp_params]
            mean_params = p[self.ngp_params:]
            gp = build_gp(gp_params)
            mean = build_mean(mean_params).T.flatten()
            cond_gp = gp.condition(Y - mean, self.X).gp

            if return_var:
                return cond_gp.loc + mean, cond_gp.variance + mean
            else:
                return cond_gp.loc + mean

        return condition

#def gp_generator_auxiliaries(t, nbands, naux, return_param_names=False):
#
#    pass