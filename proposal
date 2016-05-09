# Proposal
### Title
Parallel Cloud Simulation Based on Cellular Automata

### Summary
The purpose of this project is to create a dynamic, real-time cloud simulation, as described in [this paper](http://evasion.imag.fr/~Antoine.Bouthors/research/dea/sig00_cloud.pdf). I will parallelize the simulation using either Halide or CUDA, as well as parallelize the process used to render the clouds.

### Background
There are two parts to this project. The first part involves the cloud simulation based on cellular automata, which takes place in a 3-dimensional voxel grid. When a cell needs to be updated, it needs access to the 6 cells directly adjacent to it in the x, y, and z directions, as well as some additional cells adjacent to those.

The second part of this project is rendering the clouds. As of now, I don't quite understand the rendering method used in the paper, but there are alternative ways to render volumetric objects, like marching cubes or raycasting.

### Challenges
In this project, both the simulation part and the rendering part have opportunites for parallelism. I still do not know how I am going to render the clouds, so that is something that I have to figure out.

##### Workload
Because the simulation is based on cellular automata and the rules for updating cells involve boolean operations, it shouldn't be too difficult to parallelize the computation. However, the state of each voxel depends on the states of a somewhat irregular arrangement of voxels around it: two above, two in each of the four "horizontal" directions, and one below. Figuring out a good way to distribute the work will be a challenge. In addition, it is likely that there will be "inactive" regions during the simulation, and pruning these regions from the simulation may increase performance as well.

##### Constraints
3D voxel grids take up a large amount of memory. Regardless of how I decide to render the clouds, it will most likely involve using the GPU, so being bandwidth bound will probably be an issue.

### Resources
For this assignment, I will be using my personal laptop, which has an [Intel i7-4720HQ processor](http://ark.intel.com/products/78934/Intel-Core-i7-4720HQ-Processor-6M-Cache-up-to-3_60-GHz) and a [GTX 960M graphics card](http://www.geforce.com/hardware/notebook-gpus/geforce-gtx-960m/specifications).

I do not have any starter code, but I have the [above paper](http://evasion.imag.fr/~Antoine.Bouthors/research/dea/sig00_cloud.pdf), as well as [some slides on the same topic](http://www.cse.chalmers.se/edu/year/2011/course/TDA361/Advanced%20Computer%20Graphics/Seminar4-Clouds.pdf) - though the slides do not mention any rendering details. The paper also cites multiple other papers, but I don't have access to many of them at this time.

### Goals and Deliverables

##### What I Plan to Achieve
At minimum, I plan on having a functional cloud simulation that runs in real-time. I have a few metrics for benchmarking the performance of my simulation:
- The simulation in the paper takes 0.3 seconds per timestep on a voxel grid of size 256x128x20, and 0.5 seconds per timestep on a voxel grid of size 256x256x20. This simulation was run on an Intergraph TDZ 2000 GX1
(Pentium III 500MHz Dual).
- The simulation in the slides runs at 25 FPS on a core i7, though the slides do not specify how large the grid is.

I think having a real-time simulation is feasible because the simulation in the slides does not seem to be parallelized.

I also want some way of visualizing my simulation, so at minimum, I will produce a visualization similar to the one in the [slides](http://www.cse.chalmers.se/edu/year/2011/course/TDA361/Advanced%20Computer%20Graphics/Seminar4-Clouds.pdf) (see slide 31).

##### Stretch Goals that I Hope to Achieve
- Implement and optimize a more realistic way of rendering clouds, using ray casting or the method described in the paper, with realtime shadows and light shafts.
- Have an interactive demo similar to the one in this [video](https://www.youtube.com/watch?v=XLXU1o4F6pE) (based off of the paper).

### Platform Choice
I will be using my personal laptop for intial development and performance testing. Because the simulation is similar to image convolution, I think Halide is something worth looking into. The slides also mention using CUDA to perform the simulation on the GPU, which may or may not be better. For rendering, I will be using OpenGL - though CUDA may be a better choice if I end up doing the simulation on the GPU.

### Schedule
##### Week of 04/04/2016:
- Implement serial version of the cloud simulation.
- Implement basic visualization.
- Start thinking about how to parallelize the computation.

##### Week of 04/11/2016:
- Implement and test the performance of the parallel version of simulation.
- Write up project checkpoint.

##### Week of 04/18/2016:
- Keep working on improving the simulation if needed.
- Start looking into additional methods for rendering and potential ways to parallelize those methods.
- Prepare for exam on 04/25/2016.

##### Week of 04/25/2016
- Implement additional rendering methods.
- Test the performance of these rendering methods.

##### Week of 05/02/2016
- Final debugging and testing.
- Finish up writeup.
