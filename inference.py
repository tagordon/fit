from datetime import date
import numpy as np
import blackjax
import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm

def window_adapt(logp, p, n, seed=12345):
    """
    Wrapper for blackjax.window_adaptation.

    logp: A function that takes a 1d jax array of parameters 
        and returns the log probability of a model.
    p: A set of parameters to initialize the adaptation. These should 
        be close to the best-fit parameters, but do not need to be 
        pre-optimized. 
    n: The number of iterations for the adaptation scheme. Recommend at least
        200, probably 500+. 

    Returns: The adaptation state and parameters. 
    """

    rng_key = jax.random.PRNGKey(seed)
    adapt = blackjax.window_adaptation(blackjax.nuts, logp, is_mass_matrix_diagonal=False)
    rng_key, sample_key = jax.random.split(rng_key)
    (state, parameters), info = adapt.run(sample_key, p, n)
    return state, parameters

def inference_loop(rng_key, kernel, p, n, progress=True):
    """
    Runs an inference loop. 

    rng_key: A seed for jax's random number generator 
    kernel: A blackjax mcmc kernel (recommend blackjax.nuts)
    p: A set of parameters to initialize the sampler. These should 
        generally be determined by first running an adaptation scheme. 

    Returns: A blackjax mcmc state object.
    """

    if progress:
        @scan_tqdm(n)
        @jax.jit
        def one_step(state, key_tup):
            sample_num, rng_key = key_tup
            state, _ = kernel(rng_key, state)
            return state, state
    else:
        @jax.jit
        def one_step(state, key_tup):
            sample_num, rng_key = key_tup
            state, _ = kernel(rng_key, state)
            return state, state

    keys = jax.random.split(rng_key, n)
    _, states = jax.lax.scan(one_step, p, (np.arange(n), keys))

    return states

# same but with no progress bar option. there's probably a better way to do this. 
def inference_loop_for_multichains(rng_key, kernel, p, n):
    
    @jax.jit
    def one_step(state, key_tup):
        sample_num, rng_key = key_tup
        state, _ = kernel(rng_key, state)
        return state, state

    keys = jax.random.split(rng_key, n)
    _, states = jax.lax.scan(one_step, p, (np.arange(n), keys))

    return states

inference_loop_multiple_chains = jax.pmap(
    inference_loop_for_multichains, 
    in_axes=(0, None, 0, None), 
    static_broadcasted_argnums=(1, 3)
)

def run_inference(logp, p, n, n_adapt=200, ncpu=1, adapt_params=None, infer_seed=12345, adapt_seed=12345, progress=True):
    """
    Wrapper for inference_loop which runs the window adaptation scheme (unless adapt_params is supplied) 
    and runs inference using the blackjax nuts sampler. 

    logp: A function that takes a 1d jax array of parameters and returns 
        the log probability for the model 
    p: A set of parameters to initialized the adaptation, or if adapt_params is supplied, then 
        p should be the final state from the adaptation scheme. 
    n: the number of iterations for the nuts sampler 
    n_adapt=200: (optional) the number of iterations for the window adaptation scheme 
    ncpu=1: (optional) the number of cpus available. If ncpus > 1, n iterations will be run for ncpu chains. 
    adapt_params=None: (optional) the result of a previously run adaptation scheme.
    infer_seed=12345: (optional) a seed for the nuts inference.
    adapt_seed=12345: (optional) a seed for the adaptation scheme if adapt_params is not supplied 
    progress=True: (optional) show a progress bar for the inference loop. 

    Returns: a blackjax mcmc state object. 
    """

    if adapt_params is None:
        
        state, adapt_params = window_adapt(logp, p, n_adapt, seed=adapt_seed)
        p = state.position
        
    nuts = blackjax.nuts(logp, **adapt_params)
    rng_key = jax.random.PRNGKey(infer_seed)
    if ncpu == 1:
        
        return inference_loop(rng_key, nuts.step, nuts.init(p), n, progress=progress)
        
    else:
        
        p = jnp.tile(p, (ncpu, 1))
        initial_states = jax.vmap(nuts.init, in_axes=(0))(p)
        sample_keys = jax.random.split(rng_key, ncpu)
        return inference_loop_multiple_chains(sample_keys, nuts.step, initial_states, n)