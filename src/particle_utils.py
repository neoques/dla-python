import numpy as np


def generate_new_particles(x, gen_radius, rows):
    new_particles = np.random.normal(0, 1, x[rows].shape)
    x[rows] = gen_radius * new_particles / np.linalg.norm(new_particles, axis=1, keepdims=True)
    return x


def regenerate_extreme_particles(x, gen_radius):
    kill_radius = gen_radius + 10
    dead_particle_rows = np.linalg.norm(x, axis=1) > kill_radius
    if any(dead_particle_rows):
        return generate_new_particles(x, gen_radius, dead_particle_rows)
    else:
        return x


def regenerate_fixed_particle(x, dead_inds, gen_radius):
    lengths = np.linalg.norm(x[dead_inds], axis=1)
    gen_radius = max(gen_radius, max(lengths) + 1)
    return generate_new_particles(x, gen_radius, np.where(dead_inds)), gen_radius


def init_moving_particles(live_particles, gen_radius, d):
    moving_particles = np.random.normal(0, 1, (live_particles, d)).astype('float32')
    moving_particles *= gen_radius / np.linalg.norm(moving_particles, axis=1, keepdims=True)
    return moving_particles
