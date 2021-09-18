import svgwrite.extensions
import faiss
import numpy as np
import matplotlib.pyplot as plt
import svgwrite
import networkx as nx

import src.particle_utils as particle_utils


if __name__ == "__main__":
    page_size = (11 * 96, 17 * 96)
    max_iterations = 1000
    max_particles = 1000
    index_training_node_count = 100000
    grid_size = 2000
    np.random.seed(1234)
    center_bias = 0
    d = 2

    # particle radius
    p_radius = 2
    gen_radius = 2

    # Construct the index
    nlist = int(10 * np.sqrt(index_training_node_count))
    quantizer = faiss.IndexFlatL2(d)  # the other index
    index = faiss.IndexIVFFlat(quantizer, d, int(nlist))
    initial_vecs = (np.random.uniform(-1*grid_size, grid_size, (123318, 2))).astype('float32')
    index.train(initial_vecs)
    index.nprobe = 2

    # initialize the count of particles
    fixed_particles = np.zeros((1, d), dtype='float32')
    live_particles = 1
    moving_particles = particle_utils.init_moving_particles(live_particles, gen_radius, d)
    index.add(fixed_particles)

    # begin adding vectors to the index.
    a = 1
    particle_count = 1
    parent_indices = []
    i = 0
    last_particle_count = 1
    last_iteration = 0
    while particle_count < max_particles and i < max_iterations:
        i += 1
        # Increase the number of particles as the bounding circle gets larger
        if a*np.sqrt(particle_count)-5 > len(moving_particles):
            live_particles = int(np.sqrt(particle_count) * a)
            moving_particles = particle_utils.init_moving_particles(live_particles, gen_radius, d)
            print(f"Live: {live_particles:4}, Total: {particle_count:6}, on iteration {i:6} particles gained/iterations {(live_particles-last_particle_count)/(i-last_iteration)}")
            last_particle_count = live_particles
            last_iteration = i

        D, I = index.search(moving_particles, 1)
        fixing_indices = D[:, 0] < p_radius ** 2
        parent_indices.extend(I[fixing_indices])
        if any(fixing_indices):
            particle_count += sum(fixing_indices)
            fixing_particles = moving_particles[fixing_indices]
            index.add(fixing_particles)
            moving_particles, gen_radius = particle_utils.regenerate_fixed_particle(moving_particles, fixing_indices, gen_radius)
        moving_particles += np.random.normal(0, 1, (live_particles, d)).astype('float32')
        moving_particles -= moving_particles * center_bias/np.linalg.norm(moving_particles, axis=1, keepdims=True)
        moving_particles = particle_utils.regenerate_extreme_particles(moving_particles, gen_radius)

    # Reconstruct the points in the order they were added.
    index.make_direct_map()
    fixed_particles = index.reconstruct_n(0, int(particle_count)) + np.asarray(page_size)/2
    parent_indices = np.concatenate(parent_indices)
    parents = fixed_particles[parent_indices]

    # Build a graph
    G = nx.graph.Graph()
    for ind in range(len(fixed_particles)):
        G.add_node(ind)
        if ind > 0:
            G.add_edge(parent_indices[ind-1], ind)

    # Iterate over the edges of the graph
    edges = list(nx.algorithms.traversal.edgedfs.edge_dfs(G, source=0))
    grouped_edges = []
    for a_edge in edges:
        if len(grouped_edges) == 0 or grouped_edges[-1][-1] != a_edge[0]:
            grouped_edges.append(list(a_edge))
        else:
            grouped_edges[-1].append(a_edge[-1])

    # Group the nodes together
    group_strs = []
    paths = []
    fig, ax = plt.subplots()

    # Write the path
    dwg = svgwrite.Drawing("../outputs/out.svg", size=page_size)
    inkscape = svgwrite.extensions.Inkscape(dwg)
    layer = inkscape.layer()
    dwg.add(layer)
    for a_group in grouped_edges:
        curr_pnts = fixed_particles[a_group].astype('int')
        layer.add(svgwrite.shapes.Polyline(curr_pnts.tolist(),
                                           stroke="black",
                                           fill='none'))
    dwg.save()
