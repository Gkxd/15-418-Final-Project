#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper for CUDA Error handling

#include <cuda_runtime.h>

#include <device_launch_parameters.h>

#include <curand.h>

#include <cuda_profiler_api.h>

#include <vector_types.h>

#include <GL/glew.h>
#include <GL/freeglut.h>

#include <cuda_gl_interop.h>
#include <helper_cuda_gl.h>

#include "cutil_math.h"

#define BLOCKDIM_X 8
#define BLOCKDIM_Y 8
#define BLOCKDIM_Z 8

#define WIDTH 640
#define HEIGHT 480

__device__ char hum(char c) {
    return c & 1;
}

__device__ char act(char c) {
    return (c >> 1) & 1;
}

__device__ char cld(char c) {
    return (c >> 2) & 1;
}

__device__ char hacToCell(char h, char a, char c) {
    return (h & 1) | ((a << 1) & 2) | ((c << 2) & 4);
}

__device__ int xyzToIdx(int x, int y, int z, int dimX, int dimZ) {
    return x + z * dimX + y * dimX * dimZ;
}

__device__ int xyzToIdx2(int x, int y, int z, dim3 d) {
    return x + y * d.x + z * d.x * d.y;
}

__device__ int xyToImgIdx(int x, int y, dim3 d) {
    return x + y * d.x;
}

__device__ float sampleClouds(char* cell_buf, float3 position, dim3 cloud_dim) {

    if (position.x < -128 || position.x > 128 || position.y < -10 || position.y > 10 || position.z < -128 || position.z > 128) {
        return 0;
    }

    int idx_x = clamp((int)(position.x + 128), 0, 255);
    int idx_y = clamp((int)(position.z + 128), 0, 255);
    int idx_z = clamp((int)(position.y + 10), 0, 19);

    //return 0.05f;

    char cell = cell_buf[xyzToIdx2(idx_x, idx_y, idx_z, cloud_dim)];
    if (cld(cell)) {
        return 1;
    }

    return 0;
}

// Assumes that d.x and d.y are multiples of 32
// Position of clouds is hard coded
// Basic ray marching method
__global__ void renderClouds(char* cell_buf, float* img_buf, dim3 img_dim, dim3 cloud_dim) {
    int x = 32 * blockIdx.x + threadIdx.x; // Pixel x coordinate
    int y = 32 * blockIdx.y + threadIdx.y; // Pixel y coordinate

    int idx = xyToImgIdx(x, y, img_dim); // (0, 0) is lower left corner

    float f = 0;// blockIdx.y / 30.f + blockIdx.x / 40.f;

    //float aspect = 0.75f; // dim.y / dim.x
    //float nearPlane = 1.f;

    float3 up = make_float3(0.8660254f, 0, -0.5f);// make_float3(-0.3830223f, 0.6427875f, 0.663414f);
    float3 right = make_float3(0.3535534f, 0.7071067f, 0.6123724f);// make_float3(0.8660254f, 0, 0.5f);
    float3 forward = make_float3(0.3535534f, -0.7071068f, 0.6123724f);// make_float3(-0.3213938f, -0.7660445f, 0.5566703f);

    float3 pos = make_float3(0, 20, 0) + right * 20 * ((x - 0.5f * img_dim.x) / img_dim.x) + up * 15 * ((y - 0.5f * img_dim.y) / img_dim.y);

    for (int dist = 0; dist < 30;) {
        float sample = sampleClouds(cell_buf, pos, cloud_dim);

        if (sample == 0) {
            dist += 1;
            pos += forward;
        }
        else {
            dist += 0.1f;
            pos += forward * 0.1f;
        }

        f += sample * 0.05f;
        if (f >= 1) break;
    }

    img_buf[idx] = f;
}

__global__ void updateCellNaive(char *src_buf, char *dst_buf, float* hum_buf, float* act_buf, float* cld_buf, float p_hum, float p_act, float p_cld, dim3 d) {
    int x = BLOCKDIM_X * blockIdx.x + threadIdx.x; // Horizontal "rows"
    int y = BLOCKDIM_Y * blockIdx.y + threadIdx.y; // Horizontal "columns"
    int z = BLOCKDIM_Z * blockIdx.z + threadIdx.z; // Vertical direction
    if (x < d.x && y < d.y && z < d.z) {
        int idx = xyzToIdx2(x, y, z, d);

        char cell = src_buf[idx];

        char h = hum(cell);
        char a = act(cell);
        char c = cld(cell);

        char new_h = (h & ~a) | (hum_buf[idx] < p_hum);
        char new_c = (c | a) & (cld_buf[idx] > p_cld);
        char new_a = ~a & h;

        char f = 0;

        for (int i = -2; i <= 2; i++) {
            if (f == 1) break;
            if (i == 0) continue;

            if (x + i >= 0 && x + i < d.x) {
                char adjacent = src_buf[xyzToIdx2(x + i, y, z, d)];
                char a2 = act(adjacent);
                f |= a2;
            }
            if (y + i >= 0 && y + i < d.y) {
                char adjacent = src_buf[xyzToIdx2(x, y + i, z, d)];
                char a2 = act(adjacent);
                f |= a2;
            }

            if (i < 2 && z + i >= 0 && z + i < d.z) {
                char adjacent = src_buf[xyzToIdx2(x, y, z + i, d)];
                char a2 = act(adjacent);
                f |= a2;
            }
        }

        new_a &= f;
        new_a |= (act_buf[idx] < p_act);

        dst_buf[idx] = hacToCell(new_h, new_a, new_c);
    }
}

// A better attempt at using shared memory than updateCell_SharedMem
// Assumes 8 x 8 x 8 blocksize
// This is still slower than the naive version without using shared memory...
// Looking into it a bit, http://supercomputingblog.com/cuda/cuda-memory-and-cache-architecture/
// According to that page, a typical GPU has 1 L1 cache per core that is 16-48kb
// A kernel can have at most 1024 threads. Since each cell requires information from 10 cells (including itself) to update,
// a (very generous) upper bound for the total amount of memory that is accessed per block is 10kb.

// This algorithm also accesses other buffers in global memory, but that should be irrelevant because those accesses occur
// either before or after all of the accesses to the cell buffer, and they won't be evicting any cache lines that are still in use.

// 10kb amount fits in most L1 caches, so the benefit of using shared memory might be outweighed by the overhead it requires.

// Okay, the above might be true if the Nvidia 960M was based on the Fermi architecture, but it's actually based on the more recent
// Maxwell architecture instead. So the comments above probably don't apply here.
// I think the actual reason that this might be happening is that this kernel is compute bound and not memory bound.
// For a 8x8x8 block size, we need to load 1584 bytes into shared memory, which is around 1.5kb
// However, to compute all the indices and offsets correctly, I require a lot more operations than the naive version, and the cost
// of performing these operations is not negligible
__global__ void updateCellNaive_SharedMem(char *src_buf, char *dst_buf, float* hum_buf, float* act_buf, float* cld_buf, float p_hum, float p_act, float p_cld, dim3 d) {
    __shared__ char sharedMem[1584]; // 12 * 12 * 11 elements need to be loaded

    int x = threadIdx.x;
    int y = threadIdx.y;
    int z = threadIdx.z;

    // 3 multiplies; Total: 3 multiplies
    int blockStartX = BLOCKDIM_X * blockIdx.x;
    int blockStartY = BLOCKDIM_Y * blockIdx.y;
    int blockStartZ = BLOCKDIM_Z * blockIdx.z;

    int threadNum = xyzToIdx2(x, y, z, dim3(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z)); // 0 to 511

    for (int iter = 0; iter < 4; iter++) {
        int iterThreadNum = threadNum + iter * 512; // load the iter-th group of 512 elements

        if (iter < 3 || iterThreadNum < 1584) {
            int cellX = blockStartX + iterThreadNum % 12 - 2;
            int cellY = blockStartY + (iterThreadNum / 12) % 12 - 2;
            int cellZ = blockStartZ + iterThreadNum / 144 - 2;

            if (cellX >= 0 && cellX < d.x && cellY >= 0 && cellY < d.y && cellZ >= 0 && cellZ < d.z) {
                int cellIndex = xyzToIdx2(cellX, cellY, cellZ, d);
                sharedMem[iterThreadNum] = src_buf[cellIndex];
            }
            else {
                sharedMem[iterThreadNum] = 0;
            }

            sharedMem[iterThreadNum] = src_buf[iterThreadNum];
        }
    }

    __syncthreads();

    // Do the simulation

    int global_x = BLOCKDIM_X * blockIdx.x + threadIdx.x; // Horizontal "rows"
    int global_y = BLOCKDIM_Y * blockIdx.y + threadIdx.y; // Horizontal "columns"
    int global_z = BLOCKDIM_Z * blockIdx.z + threadIdx.z; // Vertical direction


    if (global_x < d.x && global_y < d.y && global_z < d.z) {
        dim3 shareDim = dim3(12, 12, 11);

        int idx = xyzToIdx2(x + 2, y + 2, z + 2, shareDim); // index into shared memory
        int global_idx = xyzToIdx2(global_x, global_y, global_z, d); // index into global buffers

        char cell = sharedMem[idx];

        char h = hum(cell);
        char a = act(cell);
        char c = cld(cell);

        char new_h = (h & ~a) | (hum_buf[global_idx] < p_hum);
        char new_c = (c | a) & (cld_buf[global_idx] > p_cld);
        char new_a = ~a & h;

        char f = 0;

        for (int i = 0; i <= 4; i++) {
            if (f == 1) break;
            if (i == 2) continue;

            if (x + i >= 0 && x + i < d.x) {
                char adjacent = sharedMem[xyzToIdx2(x + i, y, z, shareDim)];
                char a2 = act(adjacent);
                f |= a2;
            }
            if (y + i >= 0 && y + i < d.y) {
                char adjacent = sharedMem[xyzToIdx2(x, y + i, z, shareDim)];
                char a2 = act(adjacent);
                f |= a2;
            }

            if (i < 4 && z + i >= 0 && z + i < d.z) {
                char adjacent = sharedMem[xyzToIdx2(x, y, z + i, shareDim)];
                char a2 = act(adjacent);
                f |= a2;
            }
        }

        new_a &= f;
        new_a |= (act_buf[global_idx] < p_act);

        dst_buf[global_idx] = hacToCell(new_h, new_a, new_c);
    }
}

__device__ float sampleCloudsCompact(char* cld_buf, float3 position, dim3 cloud_dim) {
    if (position.x < -128 || position.x > 128 || position.y < -10 || position.y > 10 || position.z < -128 || position.z > 128) {
        return 0;
    }

    int idx_x = clamp((int)(position.x + 128), 0, 255) / 2;
    int idx_y = clamp((int)(position.z + 128), 0, 255) / 2;
    int idx_z = clamp((int)(position.y + 10), 0, 19) / 2;

    char mask_x = 0x55 << ((int)(position.x + 128) % 2);
    char mask_y = 0x33 << (((int)(position.z + 128) % 2) << 1);
    char mask_z = 0xF << (((int)(position.y + 10) % 2) << 2);

    char cell = cld_buf[xyzToIdx2(idx_x, idx_y, idx_z, cloud_dim)];

    return !!(mask_x & mask_y & mask_z & cell);
}

__global__ void renderCloudsCompact(char* cld_buf, float* img_buf, dim3 img_dim, dim3 cloud_dim) {
    int x = 32 * blockIdx.x + threadIdx.x; // Pixel x coordinate
    int y = 32 * blockIdx.y + threadIdx.y; // Pixel y coordinate

    int idx = xyToImgIdx(x, y, img_dim); // (0, 0) is lower left corner

    float f = 0;

    float3 up = make_float3(0.8660254f, 0, -0.5f);
    float3 right = make_float3(0.3535534f, 0.7071067f, 0.6123724f);
    float3 forward = make_float3(0.3535534f, -0.7071068f, 0.6123724f);

    float3 pos = make_float3(0, 20, 0) + right * 20 * ((x - 0.5f * img_dim.x) / img_dim.x) + up * 15 * ((y - 0.5f * img_dim.y) / img_dim.y);

    for (int dist = 0; dist < 30;) {
        float sample = sampleCloudsCompact(cld_buf, pos, cloud_dim);

        if (sample == 0) {
            dist += 1;
            pos += forward;
        }
        else {
            dist += 0.1f;
            pos += forward * 0.1f;
        }

        f += sample * 0.05f;
        if (f >= 1) break;
    }

    img_buf[idx] = f;
}

// Uses separate hum, act, and cld buffers, and stores 8 bits of information in each byte
__global__ void updateCellCompact811(
    char* src_hum, char* src_act, char* src_cld,
    char* dst_hum, char* dst_act, char* dst_cld,
    float* hum_rng, float* act_rng, float* cld_rng,
    float p_hum, float p_act, float p_cld, dim3 d) { // d is size of buffer in bytes (!= dimensions of cloud buffer)

    int x = BLOCKDIM_X * blockIdx.x + threadIdx.x; // Horizontal "rows"
    int y = BLOCKDIM_Y * blockIdx.y + threadIdx.y; // Horizontal "columns"
    int z = BLOCKDIM_Z * blockIdx.z + threadIdx.z; // Vertical direction

    if (x < d.x && y < d.y && z < d.z) {
        int idx = xyzToIdx2(x, y, z, d);

        char h = src_hum[idx];
        char a = src_act[idx];
        char c = src_cld[idx];

        char new_h = (h & ~a) | (hum_rng[idx] < p_hum ? 0x55 : 0); // Add humidity to every other cell
        char new_c = (c | a) & (cld_rng[idx] > p_cld);
        char new_a = ~a & h;

        char f = 0;

        if (x - 1 >= 0) {
            char left = src_act[xyzToIdx2(x - 1, y, z, d)];
            char temp = ((left << 6) & 0xC0); // 0xC0 == 11000000
            temp |= (left << 7) & 0x80; // 0xC0 = 10000000

            f |= temp;
        }
        f |= (a >> 1) & 0x7F; // 0x7F = 01111111
        f |= a >> 2 & 0x3F; //0x3F = 00111111

        if (x + 1 < d.x) {
            char right = src_act[xyzToIdx2(x + 1, y, z, d)];
            char temp = (right >> 6) & 0x3; // 0x3 = 00000011
            temp |= (right >> 7) & 0x1; // 0x1 = 00000001

            f |= temp;
        }
        f |= (a << 1) & 0xFE; // 0xFE = 11111110
        f |= a >> 2 & 0xFC; //0xFC = 11111100

        if (y - 2 >= 0) {
            f |= src_act[xyzToIdx2(x, y - 2, z, d)];
        }
        if (y - 1 >= 0) {
            f |= src_act[xyzToIdx2(x, y - 1, z, d)];
        }
        if (y + 1 < d.y) {
            f |= src_act[xyzToIdx2(x, y + 1, z, d)];
        }
        if (y + 2 < d.y) {
            f |= src_act[xyzToIdx2(x, y + 2, z, d)];
        }

        if (z - 2 >= 0) {
            f |= src_act[xyzToIdx2(x, y, z - 2, d)];
        }
        if (z - 1 >= 0) {
            f |= src_act[xyzToIdx2(x, y, z - 1, d)];
        }
        if (z + 1 < d.z) {
            f |= src_act[xyzToIdx2(x, y, z + 1, d)];
        }

        new_a &= f;
        new_a |= (act_rng[idx] < p_act);

        dst_hum[idx] = new_h;
        dst_act[idx] = new_a;
        dst_cld[idx] = new_c;
    }
}

// Same as updateCellCompact811, except each byte stores information of a 2x2x2 block instead of a 8x1x1 block of clouds
__global__ void updateCellCompact222(
    char* src_hum, char* src_act, char* src_cld,
    char* dst_hum, char* dst_act, char* dst_cld,
    float* hum_rng, float* act_rng, float* cld_rng,
    float p_hum, float p_act, float p_cld, dim3 d) { // d is size of buffer in bytes (!= dimensions of cloud buffer)

    int x = BLOCKDIM_X * blockIdx.x + threadIdx.x; // Horizontal "rows"
    int y = BLOCKDIM_Y * blockIdx.y + threadIdx.y; // Horizontal "columns"
    int z = BLOCKDIM_Z * blockIdx.z + threadIdx.z; // Vertical direction

    if (x < d.x && y < d.y && z < d.z) {
        int idx = xyzToIdx2(x, y, z, d);

        char h = src_hum[idx];
        char a = src_act[idx];
        char c = src_cld[idx];

        char new_h = (h & ~a) | (hum_rng[idx] < p_hum ? 0xF : 0); // Add humidity to lower layer only
        char new_c = (c | a) & (cld_rng[idx] > p_cld);
        char new_a = ~a & h;

        char f = 0;

        if (x - 1 >= 0) {
            char left = src_act[xyzToIdx2(x - 1, y, z, d)];
            f |= left;
            f |= (left >> 1) & 0x55; // 0x55 == 01010101
        }
        f |= (a & 0x55) << 1;

        if (x + 1 < d.x) {
            char right = src_act[xyzToIdx2(x + 1, y, z, d)];
            f |= right;
            f |= (right & 0x55) << 1;
        }
        f |= (a >> 1) & 0x55;

        if (y - 1 >= 0) {
            char front = src_act[xyzToIdx2(x, y - 1, z, d)];
            f |= front;
            f |= (front >> 2) & 0x33; // 0x33 == 00110011
        }
        f |= (a & 0x33) << 2;

        if (y + 1 < d.y) {
            char back = src_act[xyzToIdx2(x, y + 1, z, d)];
            f |= back;
            f |= (back & 0x33) << 2;
        }
        f |= (a >> 2) & 0x33;

        if (z - 1 > 0) {
            char lower = src_act[xyzToIdx2(x, y, z - 1, d)];
            f |= lower;
            f |= (lower >> 4) & 0xF; // 0xF == 00001111
        }
        f |= (a & 0xF) << 4;

        if (z + 1 < d.z) {
            char upper = src_act[xyzToIdx2(x, y, z + 1, d)];
            f |= (upper & 0xF) << 4;
        }
        f |= (a >> 4) & 0xF;

        new_a &= f;
        new_a |= (act_rng[idx] < p_act);

        dst_hum[idx] = new_h;
        dst_act[idx] = new_a;
        dst_cld[idx] = new_c;
    }
}



// First version of the kernel. Pretty much the same as updateCellNaive(), except with y and z switched. The performace the two are the same.
__global__ void updateCell(char *src_buf, char *dst_buf, float* hum_buf, float* act_buf, float* cld_buf, float p_hum, float p_act, float p_cld, int dimX, int dimY, int dimZ) {
    int x = BLOCKDIM_X * blockIdx.x + threadIdx.x; // Horizontal "rows"
    int y = BLOCKDIM_Y * blockIdx.y + threadIdx.y; // Vertical direction
    int z = BLOCKDIM_Z * blockIdx.z + threadIdx.z; // Horizontal "columns"

    if (x < dimX && y < dimY && z < dimZ) {
        int idx = xyzToIdx(x, y, z, dimX, dimZ);

        char cell = src_buf[idx];

        char h = hum(cell);
        char a = act(cell);
        char c = cld(cell);

        char new_h = h & ~a | (hum_buf[idx] < p_hum);
        char new_c = c | a & (cld_buf[idx] > p_cld);
        char new_a = ~a & h;

        char f = 0;

        for (int i = -2; i <= 2; i++) {
            if (f == 1) break;
            if (i == 0) continue;

            if (x + i >= 0 && x + i < dimX) {
                char adjacent = src_buf[xyzToIdx(x + i, y, z, dimX, dimZ)];
                char a2 = act(adjacent);
                f |= a2;
            }

            if (z + i >= 0 && z + i < dimZ) {
                char adjacent = src_buf[xyzToIdx(x, y, z + i, dimX, dimZ)];
                char a2 = act(adjacent);
                f |= a2;
            }

            if (i < 2 && y + i >= 0 && y + i < dimY) {
                char adjacent = src_buf[xyzToIdx(x, y + i, z, dimX, dimZ)];
                char a2 = act(adjacent);
                f |= a2;
            }
        }
        new_a &= f;
        new_a |= (act_buf[idx] < p_act);

        dst_buf[idx] = hacToCell(new_h, new_a, new_c);
    }
}

// Total amount of shared memory per block: 49152 bytes
// Kernel assumes that block size y is 1 (Most of this function is commented out because I changed the block size later)
// This is a bad usage of shared memory, because the __syncthreads() causes the latency of copying cells to shared memory to be the latency of the slowest thread
// In this case, I used a 32x1x32 block size, resulting in a need to load 36x4x36 elements into shared memory. However, because work is not distributed evenly,
// there are a few threads that have to load 16 elements, causing the latency of the entire copy operation to be the same. This is slower than the naive method,
// since the naive method only needs to load 10 elements per thread to update values.
__global__ void updateCell_SharedMem(char *src_buf, char *dst_buf, float* hum_buf, float* act_buf, float* cld_buf, float p_hum, float p_act, float p_cld, int dimX, int dimY, int dimZ) {
    __shared__ char sharedMem[(32 + 4)*(32 + 4)*(1 + 3)]; // 36 * 36 * 4 = 5184 bytes
    /*
    int blockStartX = 32 * blockIdx.x;
    int blockStartY = 1 * blockIdx.y;
    int blockStartZ = 32 * blockIdx.z;

    int startX = blockStartX - 2;
    int startY = blockStartY - 2;
    int startZ = blockStartZ - 2;
    for (int i = 0; i < 4; i++) {
    int y = startY + i;

    if (y >= 0 && y < dimY) {
    int x1 = startX + threadIdx.x;
    int z1 = startZ + threadIdx.z;

    if (x1 >= 0 && z1 >= 0 && x1 < dimX && z1 < dimZ) {
    int idx1 = xyzToIdx(x1, y, z1, dimX, dimZ);
    int idx2 = xyzToIdx(threadIdx.x, i, threadIdx.z, 32 + 4, 32 + 4);

    sharedMem[idx2] = src_buf[idx1];
    }

    int x2 = x1 + 32;
    int z2 = z1 + 32;

    if (threadIdx.x < 4 && threadIdx.z < 4) {
    if (x2 >= 0 && z2 >= 0 && x2 <= dimX && z2 <= dimZ) {
    int idx1 = xyzToIdx(x2, y, z2, dimX, dimZ);
    int idx2 = xyzToIdx(threadIdx.x + 32, i, threadIdx.z + 32, 32 + 4, 32 + 4);

    sharedMem[idx2] = src_buf[idx1];
    }
    }

    if (threadIdx.z < 4) {
    if (x1 >= 0 && z2 >= 0 && x1 <= dimX && z2 <= dimZ) {
    int idx1 = xyzToIdx(x1, y, z2, dimX, dimZ);
    int idx2 = xyzToIdx(threadIdx.x, i, threadIdx.z + 32, 32 + 4, 32 + 4);

    sharedMem[idx2] = src_buf[idx1];
    }
    }

    if (threadIdx.x < 4) {
    if (x2 >= 0 && z1 >= 0 && x2 <= dimX && z2 <= dimZ) {
    int idx1 = xyzToIdx(x2, y, z1, dimX, dimZ);
    int idx2 = xyzToIdx(threadIdx.x + 32, i, threadIdx.z, 32 + 4, 32 + 4);

    sharedMem[idx2] = src_buf[idx1];
    }
    }
    }
    }

    __syncthreads();

    // Update using shared memory
    /*
    int x = threadIdx.x; // Horizontal "rows"
    int z = threadIdx.z; // Horizontal "columns"

    if (x < dimX && z < dimZ) {
    int idx = xyzToIdx(x + 2, 2, z + 2, 32 + 4, 32 + 4); // Index into shared memory
    int globalIdx = xyzToIdx(blockStartX + x, blockStartY, blockStartZ + z, dimX, dimZ); // Index into global buffers

    char cell = sharedMem[idx];

    char h = hum(cell);
    char a = act(cell);
    char c = cld(cell);

    h &= !a & (cld_buf[globalIdx] > p_cld);
    c |= a | (hum_buf[globalIdx] < p_hum);
    a = !a & h;
    char f = 0;

    for (int i = -2; i <= 2; i++) {
    if (i == 0) continue;

    if (blockStartX + x + i >= 0 && blockStartX + x + i < dimX) {
    char adjacent = sharedMem[xyzToIdx(x + i + 2, 2, z + 2, 32 + 4, 32 + 4)];
    char a2 = act(adjacent);
    f |= a2;
    }

    if (blockStartZ + z + i >= 0 && blockStartZ + z + i < dimZ) {
    char adjacent = sharedMem[xyzToIdx(x + 2, 2, z + i + 2, 32 + 4, 32 + 4)];
    char a2 = act(adjacent);
    f |= a2;
    }

    if (i < 2 && blockStartY + i > 0 && blockStartY + i <= dimY) {
    char adjacent = sharedMem[xyzToIdx(x + 2, i + 2, z + 2, 32 + 4, 32 + 4)];
    char a2 = act(adjacent);
    f |= a2;
    }
    }
    a &= f;
    a |= (act_buf[globalIdx] < p_act);

    dst_buf[globalIdx] = hacToCell(h, a, c);
    }*/
}


///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////

// runs simulation without rendering
void runSimulation(int simulationType, int dimX, int dimY, int dimZ, int iters);

// initializes renderer (used for checking correctness)
void init(int dimX, int dimY, int dimZ, int simulationType, bool checkRender, int iters, int argc, char **argv);

static struct {
    float host_imageBuffer[WIDTH][HEIGHT];

    char* dev_bufferA;
    char* dev_bufferB;

    char* dev_cldBufA;
    char* dev_humBufA;
    char* dev_actBufA;

    char* dev_cldBufB;
    char* dev_humBufB;
    char* dev_actBufB;

    float* dev_cldRngBuf;
    float* dev_humRngBuf;
    float* dev_actRngBuf;

    float* dev_imageBuffer;

    curandGenerator_t rng;

    dim3 dim;
    dim3 img_dim;
    dim3 compact_dim;

    int iter;

    bool compact;

    //Timing
    StopWatchInterface *timer;
    float totalKernelTime;
} cloudSimulation;

int main(int argc, char **argv) {

    ///*
    int num_args = argc;

    if (num_args == 6) {
        int s = atoi(argv[1]);
        int x = atoi(argv[2]);
        int y = atoi(argv[3]);
        int z = atoi(argv[4]);
        int i = atoi(argv[5]);
        printf("%d, %d, %d, %d, %d\n", s, x, y, z, i);
        runSimulation(s, x, y, z, i);
    }
    else if (num_args == 5) {
        int s = atoi(argv[1]);
        int x = atoi(argv[2]);
        int y = atoi(argv[3]);
        int z = atoi(argv[4]);
        printf("%d, %d, %d, %d\n", s, x, y, z);
        runSimulation(s, x, y, z, 30);
    }
    else {
        runSimulation(0, 256, 256, 20, 30);
    }
    //*/

    //std::getchar();

    //init(256, 256, 20, true, false, 500, argc, argv);
    //init(512, 512, 64, false, false, 2, argc, argv);
}

static inline int updiv(int n, int d) {
    return (n + d - 1) / d;
}

void renderPicture() {
    dim3 threadsPerBlockRender(32, 32, 1);
    dim3 numBlocksRender(updiv(WIDTH, 32), updiv(HEIGHT, 32), 1);

    sdkResetTimer(&cloudSimulation.timer);

    float kernelTime = 0;

    // Update Cloud Simulation
    if (cloudSimulation.compact) {
        dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
        dim3 numBlocks(updiv(cloudSimulation.dim.x / 2, BLOCKDIM_X), updiv(cloudSimulation.dim.y / 2, BLOCKDIM_Y), updiv(cloudSimulation.dim.z / 2, BLOCKDIM_Z));

        size_t size = cloudSimulation.dim.x * cloudSimulation.dim.y * cloudSimulation.dim.z / 8;
        curandGenerateUniform(cloudSimulation.rng, cloudSimulation.dev_humRngBuf, size);
        curandGenerateUniform(cloudSimulation.rng, cloudSimulation.dev_actRngBuf, size);
        curandGenerateUniform(cloudSimulation.rng, cloudSimulation.dev_cldRngBuf, size);

        checkCudaErrors(cudaDeviceSynchronize());

        if (cloudSimulation.iter % 2 == 0) {

            sdkStartTimer(&cloudSimulation.timer);

            updateCellCompact222 << <numBlocks, threadsPerBlock >> >(
                cloudSimulation.dev_humBufA, cloudSimulation.dev_actBufA, cloudSimulation.dev_cldBufA,
                cloudSimulation.dev_humBufB, cloudSimulation.dev_actBufB, cloudSimulation.dev_cldBufB,
                cloudSimulation.dev_humRngBuf, cloudSimulation.dev_actRngBuf, cloudSimulation.dev_cldRngBuf,
                0.001f, 0.001f, 0.001f, cloudSimulation.compact_dim);

            checkCudaErrors(cudaDeviceSynchronize());

            sdkStopTimer(&cloudSimulation.timer);
            kernelTime += sdkGetTimerValue(&cloudSimulation.timer);


            renderCloudsCompact << <numBlocksRender, threadsPerBlockRender >> >
                (cloudSimulation.dev_cldBufB, cloudSimulation.dev_imageBuffer, cloudSimulation.img_dim, cloudSimulation.compact_dim);
        }
        else {
            sdkStartTimer(&cloudSimulation.timer);

            updateCellCompact222 << <numBlocks, threadsPerBlock >> >(
                cloudSimulation.dev_humBufB, cloudSimulation.dev_actBufB, cloudSimulation.dev_cldBufB,
                cloudSimulation.dev_humBufA, cloudSimulation.dev_actBufA, cloudSimulation.dev_cldBufA,
                cloudSimulation.dev_humRngBuf, cloudSimulation.dev_actRngBuf, cloudSimulation.dev_cldRngBuf,
                0.001f, 0.001f, 0.001f, cloudSimulation.compact_dim);


            checkCudaErrors(cudaDeviceSynchronize());

            sdkStopTimer(&cloudSimulation.timer);
            kernelTime += sdkGetTimerValue(&cloudSimulation.timer);

            renderCloudsCompact << <numBlocksRender, threadsPerBlockRender >> >
                (cloudSimulation.dev_cldBufA, cloudSimulation.dev_imageBuffer, cloudSimulation.img_dim, cloudSimulation.compact_dim);
        }

        checkCudaErrors(cudaDeviceSynchronize());
    }
    else {
        dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
        dim3 numBlocks(updiv(cloudSimulation.dim.x, BLOCKDIM_X), updiv(cloudSimulation.dim.y, BLOCKDIM_Y), updiv(cloudSimulation.dim.z, BLOCKDIM_Z));

        size_t size = cloudSimulation.dim.x * cloudSimulation.dim.y * cloudSimulation.dim.z;
        curandGenerateUniform(cloudSimulation.rng, cloudSimulation.dev_humRngBuf, size);
        curandGenerateUniform(cloudSimulation.rng, cloudSimulation.dev_actRngBuf, size);
        curandGenerateUniform(cloudSimulation.rng, cloudSimulation.dev_cldRngBuf, size);

        checkCudaErrors(cudaDeviceSynchronize());

        if (cloudSimulation.iter % 2 == 0) {

            sdkStartTimer(&cloudSimulation.timer);

            updateCellNaive << <numBlocks, threadsPerBlock >> >(
                cloudSimulation.dev_bufferA, cloudSimulation.dev_bufferB,
                cloudSimulation.dev_humRngBuf, cloudSimulation.dev_actRngBuf, cloudSimulation.dev_cldRngBuf,
                0.001f, 0.001f, 0.001f, cloudSimulation.dim);

            checkCudaErrors(cudaDeviceSynchronize());

            sdkStopTimer(&cloudSimulation.timer);
            kernelTime += sdkGetTimerValue(&cloudSimulation.timer);

            // Render Clouds
            renderClouds << <numBlocksRender, threadsPerBlockRender >> >(cloudSimulation.dev_bufferB, cloudSimulation.dev_imageBuffer, cloudSimulation.img_dim, cloudSimulation.dim);

            checkCudaErrors(cudaDeviceSynchronize());
        }
        else {

            sdkStartTimer(&cloudSimulation.timer);

            updateCellNaive << <numBlocks, threadsPerBlock >> >(
                cloudSimulation.dev_bufferB, cloudSimulation.dev_bufferA,
                cloudSimulation.dev_humRngBuf, cloudSimulation.dev_actRngBuf, cloudSimulation.dev_cldRngBuf,
                0.001f, 0.001f, 0.001f, cloudSimulation.dim);

            checkCudaErrors(cudaDeviceSynchronize());

            sdkStopTimer(&cloudSimulation.timer);
            kernelTime += sdkGetTimerValue(&cloudSimulation.timer);

            renderClouds << <numBlocksRender, threadsPerBlockRender >> >(cloudSimulation.dev_bufferA, cloudSimulation.dev_imageBuffer, cloudSimulation.img_dim, cloudSimulation.dim);

            checkCudaErrors(cudaDeviceSynchronize());

        }
    }

    checkCudaErrors(cudaMemcpy(cloudSimulation.host_imageBuffer, cloudSimulation.dev_imageBuffer, WIDTH*HEIGHT*sizeof(float), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());

    printf("Iteration %d complete\n", cloudSimulation.iter);

    cloudSimulation.iter++;

    cloudSimulation.totalKernelTime += kernelTime;
    printf("Time spent in cloud simulation: %fms\t (Avg: %fms)\n", kernelTime, cloudSimulation.totalKernelTime / cloudSimulation.iter);
}

void handleKeyPress(unsigned char key, int x, int y) {

}

void handleDisplay() {
    renderPicture();

    glDisable(GL_DEPTH_TEST);
    glClearColor(1.f, 0.0f, 1.f, 1.f);
    glClear(GL_COLOR_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.f, WIDTH, 0.f, HEIGHT, -1.f, 1.f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glRasterPos2i(0, 0);
    glDrawPixels(WIDTH, HEIGHT, GL_LUMINANCE, GL_FLOAT, cloudSimulation.host_imageBuffer);

    glutSwapBuffers();
    glutPostRedisplay();
}

void init(int dimX, int dimY, int dimZ, bool compact, bool renderCorrectness, int iters, int argc, char **argv) {

    cloudSimulation.dim = dim3(dimX, dimY, dimZ); // Dimension of simulation in cloud cells
    cloudSimulation.compact_dim = dim3(dimX / 2, dimY / 2, dimZ / 2);
    cloudSimulation.img_dim = dim3(WIDTH, HEIGHT);
    cloudSimulation.compact = compact;

    if (renderCorrectness) {
        // Init GLUT
        glutInit(&argc, argv);
        glutInitWindowSize(WIDTH, HEIGHT);

        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
        glutCreateWindow("CMU 15-418 Final Project - Cloud Simulation");
        glutDisplayFunc(handleDisplay);
        glutKeyboardFunc(handleKeyPress);
    }

    // Init CUDA
    checkCudaErrors(cudaSetDevice(0));

    if (compact) {
        checkCudaErrors(cudaMalloc((void**)&cloudSimulation.dev_cldBufA, dimX * dimY * dimZ * sizeof(char) / 8));
        checkCudaErrors(cudaMalloc((void**)&cloudSimulation.dev_cldBufB, dimX * dimY * dimZ * sizeof(char) / 8));

        checkCudaErrors(cudaMalloc((void**)&cloudSimulation.dev_actBufA, dimX * dimY * dimZ * sizeof(char) / 8));
        checkCudaErrors(cudaMalloc((void**)&cloudSimulation.dev_actBufB, dimX * dimY * dimZ * sizeof(char) / 8));

        checkCudaErrors(cudaMalloc((void**)&cloudSimulation.dev_humBufA, dimX * dimY * dimZ * sizeof(char) / 8));
        checkCudaErrors(cudaMalloc((void**)&cloudSimulation.dev_humBufB, dimX * dimY * dimZ * sizeof(char) / 8));


        checkCudaErrors(cudaMalloc((void**)&cloudSimulation.dev_humRngBuf, dimX * dimY * dimZ * sizeof(float) / 8));
        checkCudaErrors(cudaMalloc((void**)&cloudSimulation.dev_actRngBuf, dimX * dimY * dimZ * sizeof(float) / 8));
        checkCudaErrors(cudaMalloc((void**)&cloudSimulation.dev_cldRngBuf, dimX * dimY * dimZ * sizeof(float) / 8));
    }
    else {
        checkCudaErrors(cudaMalloc((void**)&cloudSimulation.dev_bufferA, dimX * dimY * dimZ * sizeof(char)));
        checkCudaErrors(cudaMalloc((void**)&cloudSimulation.dev_bufferB, dimX * dimY * dimZ * sizeof(char)));

        checkCudaErrors(cudaMalloc((void**)&cloudSimulation.dev_humRngBuf, dimX * dimY * dimZ * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&cloudSimulation.dev_actRngBuf, dimX * dimY * dimZ * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&cloudSimulation.dev_cldRngBuf, dimX * dimY * dimZ * sizeof(float)));
    }

    if (renderCorrectness) {
        checkCudaErrors(cudaMalloc((void**)&cloudSimulation.dev_imageBuffer, WIDTH * HEIGHT * sizeof(float)));
    }


    curandCreateGenerator(&cloudSimulation.rng, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(cloudSimulation.rng, 12345);

    sdkCreateTimer(&cloudSimulation.timer);
    cloudSimulation.totalKernelTime = 0;

    if (renderCorrectness) {
        glutMainLoop();
    }
    else {
        double kernelTime = 0;

        if (compact) {
            dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
            dim3 numBlocks(updiv(cloudSimulation.dim.x / 2, BLOCKDIM_X), updiv(cloudSimulation.dim.y / 2, BLOCKDIM_Y), updiv(cloudSimulation.dim.z / 2, BLOCKDIM_Z));

            size_t size = cloudSimulation.dim.x * cloudSimulation.dim.y * cloudSimulation.dim.z / 8;

            for (int i = 0; i < iters; i++) {
                curandGenerateUniform(cloudSimulation.rng, cloudSimulation.dev_humRngBuf, size);
                curandGenerateUniform(cloudSimulation.rng, cloudSimulation.dev_actRngBuf, size);
                curandGenerateUniform(cloudSimulation.rng, cloudSimulation.dev_cldRngBuf, size);

                checkCudaErrors(cudaDeviceSynchronize());

                sdkResetTimer(&cloudSimulation.timer);

                if (i % 2 == 0) {
                    sdkStartTimer(&cloudSimulation.timer);

                    updateCellCompact222 << <numBlocks, threadsPerBlock >> >(
                        cloudSimulation.dev_humBufA, cloudSimulation.dev_actBufA, cloudSimulation.dev_cldBufA,
                        cloudSimulation.dev_humBufB, cloudSimulation.dev_actBufB, cloudSimulation.dev_cldBufB,
                        cloudSimulation.dev_humRngBuf, cloudSimulation.dev_actRngBuf, cloudSimulation.dev_cldRngBuf,
                        0.001f, 0.001f, 0.001f, cloudSimulation.compact_dim);

                    checkCudaErrors(cudaGetLastError());
                    checkCudaErrors(cudaDeviceSynchronize());

                    sdkStopTimer(&cloudSimulation.timer);
                }
                else {
                    sdkStartTimer(&cloudSimulation.timer);

                    updateCellCompact222 << <numBlocks, threadsPerBlock >> >(
                        cloudSimulation.dev_humBufB, cloudSimulation.dev_actBufB, cloudSimulation.dev_cldBufB,
                        cloudSimulation.dev_humBufA, cloudSimulation.dev_actBufA, cloudSimulation.dev_cldBufA,
                        cloudSimulation.dev_humRngBuf, cloudSimulation.dev_actRngBuf, cloudSimulation.dev_cldRngBuf,
                        0.001f, 0.001f, 0.001f, cloudSimulation.compact_dim);

                    checkCudaErrors(cudaGetLastError());
                    checkCudaErrors(cudaDeviceSynchronize());

                    sdkStopTimer(&cloudSimulation.timer);
                }

                cloudSimulation.totalKernelTime += sdkGetTimerValue(&cloudSimulation.timer);
                printf("Iteration %d complete in %fms\n", i, sdkGetTimerValue(&cloudSimulation.timer));
            }
        }
        else {
            dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
            dim3 numBlocks(updiv(cloudSimulation.dim.x, BLOCKDIM_X), updiv(cloudSimulation.dim.y, BLOCKDIM_Y), updiv(cloudSimulation.dim.z, BLOCKDIM_Z));

            size_t size = cloudSimulation.dim.x * cloudSimulation.dim.y * cloudSimulation.dim.z;

            for (int i = 0; i < iters; i++) {
                sdkResetTimer(&cloudSimulation.timer);

                curandGenerateUniform(cloudSimulation.rng, cloudSimulation.dev_humRngBuf, size);
                curandGenerateUniform(cloudSimulation.rng, cloudSimulation.dev_actRngBuf, size);
                curandGenerateUniform(cloudSimulation.rng, cloudSimulation.dev_cldRngBuf, size);

                checkCudaErrors(cudaDeviceSynchronize());

                if (i % 2 == 0) {
                    sdkStartTimer(&cloudSimulation.timer);

                    updateCellNaive << <numBlocks, threadsPerBlock >> >(
                        cloudSimulation.dev_bufferA, cloudSimulation.dev_bufferB,
                        cloudSimulation.dev_humRngBuf, cloudSimulation.dev_actRngBuf, cloudSimulation.dev_cldRngBuf,
                        0.001f, 0.001f, 0.001f, cloudSimulation.dim);

                    checkCudaErrors(cudaGetLastError());
                    checkCudaErrors(cudaDeviceSynchronize());

                    sdkStopTimer(&cloudSimulation.timer);
                }
                else {
                    sdkStartTimer(&cloudSimulation.timer);

                    updateCellNaive << <numBlocks, threadsPerBlock >> >(
                        cloudSimulation.dev_bufferB, cloudSimulation.dev_bufferA,
                        cloudSimulation.dev_humRngBuf, cloudSimulation.dev_actRngBuf, cloudSimulation.dev_cldRngBuf,
                        0.001f, 0.001f, 0.001f, cloudSimulation.dim);

                    checkCudaErrors(cudaGetLastError());
                    checkCudaErrors(cudaDeviceSynchronize());

                    sdkStopTimer(&cloudSimulation.timer);
                }

                cloudSimulation.totalKernelTime += sdkGetTimerValue(&cloudSimulation.timer);
                printf("Iteration %d complete in %fms\n", i, sdkGetTimerValue(&cloudSimulation.timer));
            }
        }

        printf("%d iterations complete. Total time: %fms\n", iters, cloudSimulation.totalKernelTime);
        cudaDeviceReset();
        cudaProfilerStop();
    }
}


void runSimulation(int simulationType, int dimX, int dimY, int dimZ, int iters) {
    // Choose a GPU to run on.
    checkCudaErrors(cudaSetDevice(0));

    // Timer
    StopWatchInterface *timer;
    sdkCreateTimer(&timer);

    float totalKernelTime = 0;

    // Random number generator
    curandGenerator_t rng;

    curandCreateGenerator(&rng, CURAND_RNG_PSEUDO_MTGP32);
    curandSetPseudoRandomGeneratorSeed(rng, 12345);

    float* dev_cldRngBuf = 0;
    float* dev_humRngBuf = 0;
    float* dev_actRngBuf = 0;

    if (simulationType == 0) { // Naive
        char* dev_bufferA = 0;
        char* dev_bufferB = 0;

        checkCudaErrors(cudaMalloc((void**)&dev_bufferA, dimX * dimY * dimZ * sizeof(char)));
        checkCudaErrors(cudaMalloc((void**)&dev_bufferB, dimX * dimY * dimZ * sizeof(char)));

        checkCudaErrors(cudaMalloc((void**)&dev_humRngBuf, dimX * dimY * dimZ * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&dev_actRngBuf, dimX * dimY * dimZ * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&dev_cldRngBuf, dimX * dimY * dimZ * sizeof(float)));

        dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
        dim3 numBlocks(updiv(dimX, BLOCKDIM_X), updiv(dimY, BLOCKDIM_Y), updiv(dimZ, BLOCKDIM_Z));
        dim3 dim = dim3(dimX, dimY, dimZ);

        size_t size = dimX * dimY * dimZ;

        for (int i = 0; i < iters; i++) {
            sdkResetTimer(&timer);

            curandGenerateUniform(rng, dev_humRngBuf, size);
            curandGenerateUniform(rng, dev_actRngBuf, size);
            curandGenerateUniform(rng, dev_cldRngBuf, size);

            checkCudaErrors(cudaDeviceSynchronize());

            if (i % 2 == 0) {
                sdkStartTimer(&timer);

                updateCellNaive << <numBlocks, threadsPerBlock >> >(
                    dev_bufferA, dev_bufferB,
                    dev_humRngBuf, dev_actRngBuf, dev_cldRngBuf,
                    0.001f, 0.001f, 0.001f, dim);

                checkCudaErrors(cudaGetLastError());
                checkCudaErrors(cudaDeviceSynchronize());

                sdkStopTimer(&timer);
            }
            else {
                sdkStartTimer(&timer);

                updateCellNaive << <numBlocks, threadsPerBlock >> >(
                    dev_bufferB, dev_bufferA,
                    dev_humRngBuf, dev_actRngBuf, dev_cldRngBuf,
                    0.001f, 0.001f, 0.001f, dim);

                checkCudaErrors(cudaGetLastError());
                checkCudaErrors(cudaDeviceSynchronize());

                sdkStopTimer(&timer);
            }

            totalKernelTime += sdkGetTimerValue(&timer);
            printf("Iteration %d complete in %fms\n", i, sdkGetTimerValue(&timer));
        }
    }
    else if (simulationType == 1) { // Naive with shared memory
        char* dev_bufferA = 0;
        char* dev_bufferB = 0;

        checkCudaErrors(cudaMalloc((void**)&dev_bufferA, dimX * dimY * dimZ * sizeof(char)));
        checkCudaErrors(cudaMalloc((void**)&dev_bufferB, dimX * dimY * dimZ * sizeof(char)));

        checkCudaErrors(cudaMalloc((void**)&dev_humRngBuf, dimX * dimY * dimZ * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&dev_actRngBuf, dimX * dimY * dimZ * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&dev_cldRngBuf, dimX * dimY * dimZ * sizeof(float)));

        dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
        dim3 numBlocks(updiv(dimX, BLOCKDIM_X), updiv(dimY, BLOCKDIM_Y), updiv(dimZ, BLOCKDIM_Z));
        dim3 dim = dim3(dimX, dimY, dimZ);

        size_t size = dimX * dimY * dimZ;

        for (int i = 0; i < iters; i++) {
            sdkResetTimer(&timer);

            curandGenerateUniform(rng, dev_humRngBuf, size);
            curandGenerateUniform(rng, dev_actRngBuf, size);
            curandGenerateUniform(rng, dev_cldRngBuf, size);

            checkCudaErrors(cudaDeviceSynchronize());

            if (i % 2 == 0) {
                sdkStartTimer(&timer);

                updateCellNaive_SharedMem << <numBlocks, threadsPerBlock >> >(
                    dev_bufferA, dev_bufferB,
                    dev_humRngBuf, dev_actRngBuf, dev_cldRngBuf,
                    0.001f, 0.001f, 0.001f, dim);

                checkCudaErrors(cudaGetLastError());
                checkCudaErrors(cudaDeviceSynchronize());

                sdkStopTimer(&timer);
            }
            else {
                sdkStartTimer(&timer);

                updateCellNaive_SharedMem << <numBlocks, threadsPerBlock >> >(
                    dev_bufferB, dev_bufferA,
                    dev_humRngBuf, dev_actRngBuf, dev_cldRngBuf,
                    0.001f, 0.001f, 0.001f, dim);

                checkCudaErrors(cudaGetLastError());
                checkCudaErrors(cudaDeviceSynchronize());

                sdkStopTimer(&timer);
            }

            totalKernelTime += sdkGetTimerValue(&timer);
            printf("Iteration %d complete in %fms\n", i, sdkGetTimerValue(&timer));
        }

    }
    else if (simulationType == 2) { // Compact 8x1x1
        char* dev_cldBufA = 0;
        char* dev_humBufA = 0;
        char* dev_actBufA = 0;

        char* dev_cldBufB = 0;
        char* dev_humBufB = 0;
        char* dev_actBufB = 0;

        checkCudaErrors(cudaMalloc((void**)&dev_cldBufA, dimX * dimY * dimZ * sizeof(char) / 8));
        checkCudaErrors(cudaMalloc((void**)&dev_cldBufB, dimX * dimY * dimZ * sizeof(char) / 8));

        checkCudaErrors(cudaMalloc((void**)&dev_actBufA, dimX * dimY * dimZ * sizeof(char) / 8));
        checkCudaErrors(cudaMalloc((void**)&dev_actBufB, dimX * dimY * dimZ * sizeof(char) / 8));

        checkCudaErrors(cudaMalloc((void**)&dev_humBufA, dimX * dimY * dimZ * sizeof(char) / 8));
        checkCudaErrors(cudaMalloc((void**)&dev_humBufB, dimX * dimY * dimZ * sizeof(char) / 8));


        checkCudaErrors(cudaMalloc((void**)&dev_humRngBuf, dimX * dimY * dimZ * sizeof(float) / 8));
        checkCudaErrors(cudaMalloc((void**)&dev_actRngBuf, dimX * dimY * dimZ * sizeof(float) / 8));
        checkCudaErrors(cudaMalloc((void**)&dev_cldRngBuf, dimX * dimY * dimZ * sizeof(float) / 8));


        dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
        dim3 numBlocks(updiv(dimX / 8, BLOCKDIM_X), updiv(dimY, BLOCKDIM_Y), updiv(dimZ, BLOCKDIM_Z));
        dim3 dim = dim3(dimX / 8, dimY, dimZ);

        size_t size = dimX * dimY * dimZ / 8;

        for (int i = 0; i < iters; i++) {
            curandGenerateUniform(rng, dev_humRngBuf, size);
            curandGenerateUniform(rng, dev_actRngBuf, size);
            curandGenerateUniform(rng, dev_cldRngBuf, size);

            checkCudaErrors(cudaDeviceSynchronize());

            sdkResetTimer(&timer);

            if (i % 2 == 0) {
                sdkStartTimer(&timer);

                updateCellCompact811 << <numBlocks, threadsPerBlock >> >(
                    dev_humBufA, dev_actBufA, dev_cldBufA,
                    dev_humBufB, dev_actBufB, dev_cldBufB,
                    dev_humRngBuf, dev_actRngBuf, dev_cldRngBuf,
                    0.001f, 0.001f, 0.001f, dim);

                checkCudaErrors(cudaGetLastError());
                checkCudaErrors(cudaDeviceSynchronize());

                sdkStopTimer(&timer);
            }
            else {
                sdkStartTimer(&timer);

                updateCellCompact811 << <numBlocks, threadsPerBlock >> >(
                    dev_humBufB, dev_actBufB, dev_cldBufB,
                    dev_humBufA, dev_actBufA, dev_cldBufA,
                    dev_humRngBuf, dev_actRngBuf, dev_cldRngBuf,
                    0.001f, 0.001f, 0.001f, dim);

                checkCudaErrors(cudaGetLastError());
                checkCudaErrors(cudaDeviceSynchronize());

                sdkStopTimer(&timer);
            }

            totalKernelTime += sdkGetTimerValue(&timer);
            printf("Iteration %d complete in %fms\n", i, sdkGetTimerValue(&timer));
        }
    }
    else if (simulationType == 3) { // Compact 2x2x2
        char* dev_cldBufA = 0;
        char* dev_humBufA = 0;
        char* dev_actBufA = 0;

        char* dev_cldBufB = 0;
        char* dev_humBufB = 0;
        char* dev_actBufB = 0;

        checkCudaErrors(cudaMalloc((void**)&dev_cldBufA, dimX * dimY * dimZ * sizeof(char) / 8));
        checkCudaErrors(cudaMalloc((void**)&dev_cldBufB, dimX * dimY * dimZ * sizeof(char) / 8));

        checkCudaErrors(cudaMalloc((void**)&dev_actBufA, dimX * dimY * dimZ * sizeof(char) / 8));
        checkCudaErrors(cudaMalloc((void**)&dev_actBufB, dimX * dimY * dimZ * sizeof(char) / 8));

        checkCudaErrors(cudaMalloc((void**)&dev_humBufA, dimX * dimY * dimZ * sizeof(char) / 8));
        checkCudaErrors(cudaMalloc((void**)&dev_humBufB, dimX * dimY * dimZ * sizeof(char) / 8));


        checkCudaErrors(cudaMalloc((void**)&dev_humRngBuf, dimX * dimY * dimZ * sizeof(float) / 8));
        checkCudaErrors(cudaMalloc((void**)&dev_actRngBuf, dimX * dimY * dimZ * sizeof(float) / 8));
        checkCudaErrors(cudaMalloc((void**)&dev_cldRngBuf, dimX * dimY * dimZ * sizeof(float) / 8));


        dim3 threadsPerBlock(BLOCKDIM_X, BLOCKDIM_Y, BLOCKDIM_Z);
        dim3 numBlocks(updiv(dimX / 2, BLOCKDIM_X), updiv(dimY / 2, BLOCKDIM_Y), updiv(dimZ / 2, BLOCKDIM_Z));
        dim3 dim = dim3(dimX / 2, dimY / 2, dimZ / 2);

        size_t size = dimX * dimY * dimZ / 8;

        for (int i = 0; i < iters; i++) {
            curandGenerateUniform(rng, dev_humRngBuf, size);
            curandGenerateUniform(rng, dev_actRngBuf, size);
            curandGenerateUniform(rng, dev_cldRngBuf, size);

            checkCudaErrors(cudaDeviceSynchronize());

            sdkResetTimer(&timer);

            if (i % 2 == 0) {
                sdkStartTimer(&timer);

                updateCellCompact222 << <numBlocks, threadsPerBlock >> >(
                    dev_humBufA, dev_actBufA, dev_cldBufA,
                    dev_humBufB, dev_actBufB, dev_cldBufB,
                    dev_humRngBuf, dev_actRngBuf, dev_cldRngBuf,
                    0.001f, 0.001f, 0.001f, dim);

                checkCudaErrors(cudaGetLastError());
                checkCudaErrors(cudaDeviceSynchronize());

                sdkStopTimer(&timer);
            }
            else {
                sdkStartTimer(&timer);

                updateCellCompact222 << <numBlocks, threadsPerBlock >> >(
                    dev_humBufB, dev_actBufB, dev_cldBufB,
                    dev_humBufA, dev_actBufA, dev_cldBufA,
                    dev_humRngBuf, dev_actRngBuf, dev_cldRngBuf,
                    0.001f, 0.001f, 0.001f, dim);

                checkCudaErrors(cudaGetLastError());
                checkCudaErrors(cudaDeviceSynchronize());

                sdkStopTimer(&timer);
            }

            totalKernelTime += sdkGetTimerValue(&timer);
            printf("Iteration %d complete in %fms\n", i, sdkGetTimerValue(&timer));
        }
    }

    printf("%d iterations complete. Total time: %fms\n", iters, totalKernelTime);
    printf("Average: %fms\n", totalKernelTime / iters);
    cudaDeviceReset();
}