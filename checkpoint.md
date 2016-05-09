# 15-418-Final-Project
# Checkpoint

### Work Completed
Setting up Visual Studio to build the projects was more complex than I originally thought, so I am a bit behind on schedule. However, I am finished with the cellular automata, and I am currently in the process of optomizing the code. I decided to use CUDA because if I wanted to render anything, I would use the GPU in some way, and using CUDA would prevent me from having to transfer large 3D voxel grids from the CPU to the GPU.

I have also started taking a few measurements on roughly how much room for improvement there is: my naive implementation takes around 16.1 ms per iteration to calculate the cellular automata, with an additional 1.34 ms per iteration for generating random numbers. The program is definitely bandwidth bound right now, when I comment out the memory accesses in the kernel, the 16.1 ms decreases to 3.1 ms.

### Goals and Deliverables
My original goal of having a realtime simulation was probably not high enough, as the naive implementation can perform the computation in almost realtime already (though this doesn't take into account the time needed for rendering). At this point, I want to get the 16.1 ms down to 3.1 ms as much as I can. I suspect that using shared memory can significantly increase the performance of the code, because it would greatly reduce the amount of global memory accesses that happen.

As for the rendering component, although it is nice to have, it is not as interesting from a parallelism perspective. However, having a rendering component would make it a lot easier to check the correctness of my implementation. At the parallelism competition, ideally, I want to have a basic rendering component, but if that doesn't work, I will have a graph comparing the performances of my different implementations.

### Concerns
The main concern that I have right now is that I still do not understand the rendering process described in the paper. I have not yet set up OpenGL with my project either, and that might also take a large amount of time.

For optimization, in additional to using shared memory, there are a few other ways that I can reduce the number of global memory accesses from the kernel, and I just need to implement those methods.

### Revised Schedule
##### Week of 04/18/2016
- Finish implementing shared memory (First Half)
- Study for exam (Second Half)

##### Week of 04/25/2016
- Implement additional ways of reducing global memory accesses and produce graphs (First Half)
- Set up OpenGL and possibly get started on rendering component (Second Half)

##### Week of 05/02/2016
- Implement a basic rendering component (First Half)
- Work on writeup (Second Half)
