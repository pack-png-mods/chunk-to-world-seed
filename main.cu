//"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin\nvcc.exe"  -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64" -o main main.cu -O3

// IDE indexing
#ifdef __JETBRAINS_IDE__
#define __host__
#define __device__
#define __shared__
#define __constant__
#define __global__
#define __CUDACC__
#include <__clang_cuda_builtin_vars.h>
#include <__clang_cuda_cmath.h>
#include <__clang_cuda_complex_builtins.h>
#include <__clang_cuda_intrinsics.h>
#include <__clang_cuda_math_forward_declares.h>
#include <device_functions.h>
#endif

#include <iostream>
#include <math.h>
#include <cmath>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>

//Used for sleep function
#include <chrono>
#include <thread>

#define CHECK_GPU_ERR(code) gpuAssert((code), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
        exit(code);
    }
}

#ifndef CHUNK_X
#define CHUNK_X 3
#endif
#define CHUNK_Z -3
#define OUTPUT_SEED_ARRAY_SIZE (1ULL << 20)
#define WORKER_COUNT (1ULL << 16)
#define MAXCHAR 1000

#define MASK48 ((1ULL << 48) - 1ULL)
#define MASK32 ((1ULL << 32) - 1ULL)
#define MASK16 ((1ULL << 16) - 1ULL)

#define M1 25214903917ULL
#define ADDEND1 11ULL

#define M2 205749139540585ULL
#define ADDEND2 277363943098ULL

#define M4 55986898099985ULL
#define ADDEND4 49720483695876ULL

inline __host__ __device__ int64_t nextLong(uint64_t* seed) {
    *seed = (*seed * M1 + ADDEND1) & MASK48;
    int32_t u = *seed >> 16;
    *seed = (*seed * M1 + ADDEND1) & MASK48;
    return ((uint64_t)u << 32) + (int32_t)(*seed >> 16);
}

inline __device__ void addSeed(uint64_t seed, uint64_t* seeds, uint64_t* seedCounter)
{
    // unsigned long long* cast is required for CUDA 9 :thonkgpu:
    uint64_t id = atomicAdd((unsigned long long*) seedCounter, 1ULL);
    seeds[id] = seed;
}

inline __host__ __device__ uint64_t makeMask(int32_t bits) {
    return (1ULL << bits) - 1;
}

inline __host__ __device__ int countTrailingZeroes(uint64_t v) {
    int c;

    v = (v ^ (v - 1)) >> 1;

    for(c = 0; v != 0; c++)  {
        v >>= 1;
    }

    return c;
}

inline __host__ __device__ uint64_t modInverse(uint64_t x) {
    uint64_t inv = 0;
    uint64_t b = 1;
    for (int32_t i = 0; i < 16; i++) {
        inv |= (1ULL << i) * (b & 1);
        b = (b - x * (b & 1)) >> 1;
    }
    return inv;
}

inline __host__ __device__ uint64_t getChunkSeed(uint64_t worldSeed) {
    uint64_t seed = (worldSeed ^ M1) & MASK48;
    int64_t a = nextLong(&seed) / 2 * 2 + 1;
    int64_t b = nextLong(&seed) / 2 * 2 + 1;
    return (uint64_t)(((CHUNK_X * a + CHUNK_Z * b) ^ worldSeed) & MASK48);
}

inline __host__ __device__ uint64_t getPartialAddend(uint64_t partialSeed, int32_t bits) {
    uint64_t mask = makeMask(bits);
    return ((uint64_t)CHUNK_X) * (((int32_t)(((M2 * ((partialSeed ^ M1) & mask) + ADDEND2) & MASK48) >> 16)) / 2 * 2 + 1) +
           ((uint64_t)CHUNK_Z) * (((int32_t)(((M4 * ((partialSeed ^ M1) & mask) + ADDEND4) & MASK48) >> 16)) / 2 * 2 + 1);
}

inline __device__ void addWorldSeed(uint64_t firstAddend, int32_t multTrailingZeroes, uint64_t firstMultInv,
                                    uint64_t c, uint64_t chunkSeed, uint64_t* seeds, uint64_t* seedCounter) {
    if(countTrailingZeroes(firstAddend) < multTrailingZeroes)
        return;
    uint64_t bottom32BitsChunkseed = chunkSeed & MASK32;

    uint64_t b = (((firstMultInv * firstAddend) >> multTrailingZeroes) ^ (M1 >> 16)) & makeMask(16 - multTrailingZeroes);
    if (multTrailingZeroes != 0) {
        uint64_t smallMask = makeMask(multTrailingZeroes);
        uint64_t smallMultInverse = smallMask & firstMultInv;
        uint64_t target = (((b ^ (bottom32BitsChunkseed >> 16)) & smallMask) -
                                (getPartialAddend((b << 16) + c, 32 - multTrailingZeroes) >> 16)) & smallMask;
        b += (((target * smallMultInverse) ^ (M1 >> (32 - multTrailingZeroes))) & smallMask) << (16 - multTrailingZeroes);
    }
    uint64_t bottom32BitsSeed = (b << 16) + c;
    uint64_t target2 = (bottom32BitsSeed ^ bottom32BitsChunkseed) >> 16;
    uint64_t secondAddend = (getPartialAddend(bottom32BitsSeed, 32) >> 16);
    secondAddend &= MASK16;
    uint64_t topBits = ((((firstMultInv * (target2 - secondAddend)) >> multTrailingZeroes) ^ (M1 >> 32)) & makeMask(16 - multTrailingZeroes));

    for (; topBits < (1ULL << 16); topBits += (1ULL << (16 - multTrailingZeroes))) {
        if (getChunkSeed((topBits << 32) + bottom32BitsSeed) == chunkSeed) {
            addSeed((topBits << 32) + bottom32BitsSeed, seeds, seedCounter);
        }
    }

}

__global__ void crack(uint64_t seedInputCount, uint64_t* seedInputArray, uint64_t* seedOutputCounter, uint64_t* seedOutputArray,
                        int32_t multTrailingZeroes, uint64_t firstMultInv, int32_t xCount, int32_t zCount, int32_t totalCount) {
    uint64_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id > seedInputCount)
        return;

    uint64_t chunkSeed = seedInputArray[global_id];
    int32_t x = CHUNK_X;
    int32_t z = CHUNK_Z;

#if CHUNK_X == 0 && CHUNK_Z == 0
    addSeed(chunkSeed, seedOutputArray, seedOutputCounter);
#else
    uint64_t f = chunkSeed & MASK16;
    uint64_t c = xCount == zCount ? chunkSeed & ((1ULL << (xCount + 1)) - 1) :
                                    chunkSeed & ((1ULL << (totalCount + 1)) - 1) ^ (1 << totalCount);
    for (; c < (1ULL << 16); c += (1ULL << (totalCount + 1))) {
        uint64_t target = (c ^ f) & MASK16;
        uint64_t magic = (uint64_t)(x * ((M2 * ((c ^ M1) & MASK16) + ADDEND2) >> 16)) +
                         (uint64_t)(z * ((M4 * ((c ^ M1) & MASK16) + ADDEND4) >> 16));
        addWorldSeed(target - (magic & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
#if CHUNK_X != 0
        addWorldSeed(target - ((magic + x) & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
#endif
#if CHUNK_Z != 0 && CHUNK_X != CHUNK_Z
        addWorldSeed(target - ((magic + z) & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
#endif
#if CHUNK_X != 0 && CHUNK_Z != 0 && CHUNK_X + CHUNK_Z != 0
        addWorldSeed(target - ((magic + x + z) & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
#endif
#if CHUNK_X != 0 && CHUNK_X != CHUNK_Z
        addWorldSeed(target - ((magic + 2 * x) & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
#endif
#if CHUNK_Z != 0 && CHUNK_X != CHUNK_Z
        addWorldSeed(target - ((magic + 2 * z) & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
#endif
#if CHUNK_X != 0 && CHUNK_Z != 0 && CHUNK_X + CHUNK_Z != 0 && CHUNK_X * 2 + CHUNK_Z != 0
        addWorldSeed(target - ((magic + 2 * x + z) & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
#endif
#if CHUNK_X != 0 && CHUNK_Z != 0 && CHUNK_X != CHUNK_Z && CHUNK_X + CHUNK_Z != 0 && CHUNK_X + CHUNK_Z * 2 != 0
        addWorldSeed(target - ((magic + x + 2 * z) & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
#endif
#if CHUNK_X != 0 && CHUNK_Z != 0 && CHUNK_X + CHUNK_Z != 0
        addWorldSeed(target - ((magic + 2 * x + 2 * z) & MASK16), multTrailingZeroes, firstMultInv, c, chunkSeed, seedOutputArray, seedOutputCounter);
#endif
    }
#endif // !(CHUNK_X == 0 && CHUNK_Z == 0)
}

#ifndef OUTPUT_FILE
#define OUTPUT_FILE "WorldSeeds.txt"
#endif

#ifndef INPUT_FILE
#define INPUT_FILE "SEEDS.txt"
#endif

#undef int
int main() {
    #define int uint32_t
        setbuf(stdout, NULL);
        FILE *fp;
        FILE *fp_out;
        char str[MAXCHAR];
        fp = fopen("SEEDS.txt", "r");
        uint64_t totalInputSeeds = 0;
        if (!fp) {
            printf("Could not open file\n");
            return 1;
        }
        printf("Counting input size...\n");
        while (fgets(str, MAXCHAR, fp))
            totalInputSeeds++;
        fclose(fp);
        fp = fopen("SEEDS.txt", "r");
        if (!fp) {
            printf("Could not open file\n");
            return 1;
        }
        fp_out = fopen("WorldSeeds.txt", "w");

        uint64_t* buffer = (uint64_t*)malloc(WORKER_COUNT * sizeof(uint64_t));

        int inputSeedCount = WORKER_COUNT;


        uint64_t* inputSeeds;
        CHECK_GPU_ERR(cudaMallocManaged(&inputSeeds, sizeof(*inputSeeds) * (inputSeedCount)));

        uint64_t* outputSeedCount;
        CHECK_GPU_ERR(cudaMallocManaged(&outputSeedCount, sizeof(*outputSeedCount)));

        uint64_t* outputSeeds;
        CHECK_GPU_ERR(cudaMallocManaged(&outputSeeds, sizeof(*outputSeeds) * OUTPUT_SEED_ARRAY_SIZE));

        //Inital copy
        for(uint64_t i = 0; i < WORKER_COUNT; i++)
        {
            if(fgets(str, MAXCHAR, fp) != NULL)
            {
                sscanf(str, "%lu", &inputSeeds[i]);
            }
        }

        printf("Beginning converting %lu seeds\n", totalInputSeeds);
        int count = 0; // Counter used for end bit
        int64_t numSearched = 0;
        int64_t totalSeeds = 0;
        clock_t lastIteration = clock();
        clock_t startTime = clock();

        uint64_t firstMultiplier = (M2 * CHUNK_X + M4 * CHUNK_Z) & MASK16;
        int32_t multTrailingZeroes = countTrailingZeroes(firstMultiplier);
        uint64_t firstMultInv = modInverse(firstMultiplier >> multTrailingZeroes);

        int32_t xCount = countTrailingZeroes(CHUNK_X);
        int32_t zCount = countTrailingZeroes(CHUNK_Z);
        int32_t totalCount = countTrailingZeroes(CHUNK_X | CHUNK_Z);

        while (true) {
            crack<<<WORKER_COUNT >> 9, 1 << 9>>>(inputSeedCount, inputSeeds,
                                                outputSeedCount, outputSeeds,
                                                multTrailingZeroes, firstMultInv,
                                                xCount, zCount, totalCount);

            bool doneFlag = false;
            count = 0;
            for(uint64_t i = 0; i < WORKER_COUNT; i++) {

                if(fgets(str, MAXCHAR, fp) != NULL) {
                    sscanf(str, "%lu", &buffer[i]);
                    count++;
                } else {
                    doneFlag = true;
                }
            }
            CHECK_GPU_ERR(cudaPeekAtLastError());
            CHECK_GPU_ERR(cudaDeviceSynchronize());
            for(uint64_t i = 0; i < WORKER_COUNT; i++) {
                inputSeeds[i] = buffer[i];
            }

            double iterationTime = (double)(clock() - lastIteration) / CLOCKS_PER_SEC;
            double timeElapsed = (double)(clock() - startTime) / CLOCKS_PER_SEC;
            lastIteration = clock();
            numSearched += WORKER_COUNT;
            double speed = WORKER_COUNT / iterationTime / 1000.0;
            double progress = (double) numSearched / (double) totalInputSeeds * 100.0;
            double estimatedTime = (double) (totalInputSeeds - numSearched) / (double) WORKER_COUNT * iterationTime;
            char suffix = 's';
            if (estimatedTime >= 3600) {
                suffix = 'h';
                estimatedTime /= 3600;
            } else if (estimatedTime >= 60) {
                suffix = 'm';
                estimatedTime /= 60;
            }
            if (progress >= 100) {
                estimatedTime = 0;
                suffix = 's';
            }
            totalSeeds += *outputSeedCount;

            printf("Searched: %ld seeds. Found %ld matches. Uptime: %.1fs. Speed: %.2fk seeds/s. Completion: %.3f%%. ETA: %.1f%c.\n", numSearched, totalSeeds, timeElapsed, speed, progress, estimatedTime, suffix);

            for (int i = 0; i < *outputSeedCount; i++) {
                fprintf(fp_out, "%lu\n", outputSeeds[i]);
            }
            fflush(fp_out);

            *outputSeedCount = 0;
            if (doneFlag) {
                printf("DONE\n");
                break;
            }
        }

        crack<<<WORKER_COUNT >> 9, 1 << 9>>>(count, inputSeeds, outputSeedCount,
                                            outputSeeds, multTrailingZeroes,
                                            firstMultInv, xCount, zCount,
                                            totalCount);
        CHECK_GPU_ERR(cudaDeviceSynchronize());
        for (int i = 0; i < *outputSeedCount; i++) {
            fprintf(fp_out, "%lu\n", outputSeeds[i]);
        }
        fflush(fp_out);
        fclose(fp);
        fclose(fp_out);
    }
