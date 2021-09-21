# dla-python
Implements several methods of simulating diffusion limited aggregation, with the intent of finding the fastest method.

Python generally is slow, but can rely on low-level c libraries to achieve performant results. The goal of this project was to find a quick way of performing diffusion limited aggregation.
The main computation in this project is detecting nearby points. The main challenge is dynamically updating the index used to perform the nearest neighbor detection.
To this end, I tried 2 methods. First I used facebook's faiss to index the plane over which the nearest neighbors would be computed. Then that index was used to apply the detection.
Additionally I used a python interface with the C library boost that implements a R tree (the search tree with the nicest dynamic update properties) to perform the same task.

The end result suggests that 
  faiss can simulate 250k points in 5 minutes~
  R-tree can simulate 25k points in 5 minutes~
