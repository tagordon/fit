import jax
import jax.numpy as jnp
from jaxoplanet import orbits, light_curves

def transit_with_trend(t, f0, m, p, q1, q2, r2, d, t0, b):

    tranmod = transit(t, p, q1, q2, r2, d, t0, b) * f0
    trend = f0 + m * (t - jnp.min(t))
    return trend + tranmod

def transit(t, p, q1, q2, r2, d, t0, b):

    u1 = 2 * jnp.sqrt(q1) * q2
    u2 = jnp.sqrt(q1) * (1 - 2 * q2)

    x2 = jnp.square(1 + jnp.sqrt(r2)) - jnp.square(b)
    speed = 2 * jnp.sqrt(jnp.maximum(0, x2)) / d

    speed = jnp.array([speed])
    p = jnp.array([p])
    d = jnp.array([d])
    t0 = jnp.array([t0])
    b = jnp.array([b])
    r2 = jnp.array([r2])

    orbit = orbits.TransitOrbit(
        speed=speed, 
        period=p, 
        duration=d, 
        time_transit=t0, 
        impact_param=b,
        radius=jnp.sqrt(r2)
    )
    
    return light_curves.QuadLightCurve(u1, u2).light_curve(orbit, t)