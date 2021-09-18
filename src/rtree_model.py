import rtree
import numpy as np
import matplotlib.pyplot as plt
import time


def point_to_bounderies(x):
    return np.concatenate(2 * [x, ])


def generate_point(gen_radius):
    new_point = np.random.normal(0, 1, 2)
    new_point /= np.linalg.norm(new_point)
    return new_point * np.sqrt(gen_radius)


if __name__ == "__main__":
    # params
    time_start = time.time()
    np.random.seed(1234)
    max_iterations = 1000000
    d = 2
    gen_radius2 = 2.5 ** 2
    particle_join_radius = 1

    first_particle = np.zeros(d, dtype='float32')
    index = rtree.index.Index()
    particle_count = 0
    index.add(particle_count, point_to_bounderies(first_particle))

    curr_particle = generate_point(gen_radius2)
    points = [first_particle, ]
    parent_inds = []
    for i in range(max_iterations):
        closest = next(index.nearest(point_to_bounderies(curr_particle), 1))
        curr_particle += np.random.normal(0, 1, 2)
        dist = points[closest] - curr_particle

        if np.dot(dist, dist) < particle_join_radius:
            parent_inds.append(closest)
            particle_count += 1
            points.append(curr_particle)
            index.add(particle_count, point_to_bounderies(curr_particle))
            if gen_radius2 < np.dot(curr_particle, curr_particle):
                gen_radius2 = np.dot(curr_particle, curr_particle)
            curr_particle = generate_point(gen_radius2)
            continue

        if np.dot(curr_particle, curr_particle) > gen_radius2 + 2:
            curr_particle = generate_point(gen_radius2)
    all_points = np.asarray(points)
    time_end = time.time()

    print(len(all_points))
    print(f"Time End {time_end - time_start}")

    plt.scatter(all_points[:, 0], all_points[:, 1])
    plt.show()