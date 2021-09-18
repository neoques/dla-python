import faiss
import numpy as np
import matplotlib.pyplot as plt

import src.particle_utils as particle_utils

if __name__ == "__main__":
    np.random.seed(1234)
    max_iterations = 10000
    d = 2

    # particle radius
    p_radius = 2
    gen_radius = 2

    # initialize the count of particles
    fixed_particles = np.zeros((1, d), dtype='float32')
    live_particles = 2
    moving_particles = particle_utils.init_moving_particles(live_particles, gen_radius, d)

    index = faiss.IndexFlatL2(d)  # the other index
    index.add(fixed_particles)
    for i in range(max_iterations):
        if len(fixed_particles) > 10 * len(moving_particles):
            live_particles = len(moving_particles) * 2
            moving_particles = particle_utils.init_moving_particles(live_particles, gen_radius, d)
            print(f"{live_particles}, on iteration {i}")

        D, I = index.search(moving_particles, 1)

        fixing_indices = D[:, 0] < p_radius ** 2
        if any(fixing_indices):
            fixing_particles = moving_particles[fixing_indices]
            index.add(fixing_particles)
            fixed_particles = np.append(fixed_particles, fixing_particles, axis=0)
            moving_particles, gen_radius = particle_utils.regenerate_fixed_particle(moving_particles, fixing_indices, gen_radius)
        moving_particles += np.random.normal(0, 1, (live_particles, d)).astype('float32')
        moving_particles = particle_utils.regenerate_extreme_particles(moving_particles, gen_radius)

    plt.scatter(fixed_particles[:, 0], fixed_particles[:, 1])
    plt.title(f"Particle Count {len(fixed_particles)} GenRadius: {gen_radius}")
    plt.show()
