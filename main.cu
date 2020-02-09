//"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin\nvcc.exe"  -ccbin "C:\Program Files (x86)\Microsoft Visual Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64" -o main main.cu -O3
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

#define int int32_t
inline __host__ __device__ uint64_t makeSecondAddend(int x, uint64_t k, int z);
inline __device__ void addWorldSeed(uint64_t firstAddend, int multTrailingZeroes, uint64_t firstMultInv, uint64_t c, uint64_t e, int x, int z, uint64_t populationSeed, uint64_t* seeds, uint64_t* seedCounter);
inline __host__ __device__ int countTrailingZeroes(uint64_t v);
inline __host__ __device__ uint64_t nextLong(uint64_t* seed);
inline __host__ __device__ uint64_t getPopulationSeed(uint64_t worldSeed, int x, int z);

#define CHUNK_X 3
#define CHUNK_Z -3
#define OUTPUT_SEED_ARRAY_SIZE (1ULL << 20)
#define WORKER_COUNT (1ULL << 16)
#define MAXCHAR 1000

#define MASK_48 ((1ULL << 48) - 1)
#define MASK_32 ((1ULL << 32) - 1)
#define MASK_16 ((1ULL << 16) - 1)

#define JAVA_LCG_MULTIPLIER 25214903917ULL
#define JAVA_LCG_ADDEND 11ULL

#define SKIP_2_MULTIPLIER 205749139540585ULL
#define SKIP_2_ADDEND 277363943098ULL

#define SKIP_4_MULTIPLIER 55986898099985ULL
#define SKIP_4_ADDEND 49720483695876ULL

inline __device__ void addSeed(uint64_t seed, uint64_t* seeds, uint64_t* seedCounter)
{
    uint64_t id = atomicAdd(seedCounter, 1);
    seeds[id] = seed;
}

__global__ void crack(uint64_t seedInputCount, uint64_t* seedInputArray, uint64_t* seedOutputCounter, uint64_t* seedOutputArray, 
    int multTrailingZeroes, uint64_t firstMultInv) {
    uint64_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_id > seedInputCount)
        return;
    
    uint64_t seed = seedInputArray[global_id];
    
    seed = seed ^ JAVA_LCG_MULTIPLIER; // xor with const ASSUMPTION THAT original seeds arnt xored
    
    int x = CHUNK_X;
    int z = CHUNK_Z;
    
    uint64_t c = 0ULL;
    uint64_t e = seed & MASK_32; 
    uint64_t f = seed & MASK_16;
    
    if ((x ^ z) & 1) { 
        c = (seed & 1) ^ 1;
    } else {
        c = (seed & 1);
    }
    
    for(; c < (1ULL << 16); c += 2) {
        uint64_t target = (c ^ f) & MASK_16;
        uint64_t magic = x * ((SKIP_2_MULTIPLIER * ((c ^ JAVA_LCG_MULTIPLIER) & MASK_16) + SKIP_2_ADDEND) >> 16) 
                       + z * ((SKIP_4_MULTIPLIER * ((c ^ JAVA_LCG_MULTIPLIER) & MASK_16) + SKIP_4_ADDEND) >> 16);
        
        addWorldSeed(target - (magic & MASK_16), multTrailingZeroes, firstMultInv, c, e, x, z, seed, seedOutputArray, seedOutputCounter); 
        
#if CHUNK_X != 0
        addWorldSeed(target - ((magic + x) & MASK_16), multTrailingZeroes, firstMultInv, c, e, x, z, seed, seedOutputArray, seedOutputCounter);
#endif
#if CHUNK_Z != 0        
        addWorldSeed(target - ((magic + z) & MASK_16), multTrailingZeroes, firstMultInv, c, e, x, z, seed, seedOutputArray, seedOutputCounter); 
#endif
#if CHUNK_X + CHUNK_Z != 0 && CHUNK_X != 0 && CHUNK_Z != 0
        addWorldSeed(target - ((magic + x + z) & MASK_16), multTrailingZeroes, firstMultInv, c, e, x, z, seed, seedOutputArray, seedOutputCounter);
#endif
    }
}

inline __host__ __device__ uint64_t makeSecondAddend(int x, uint64_t k, int z) {
    return ((x * ((((SKIP_2_MULTIPLIER * ((k ^ JAVA_LCG_MULTIPLIER) & MASK_32) + SKIP_2_ADDEND) & MASK_48) >> 16) | 1ULL) +
             z * ((((SKIP_4_MULTIPLIER * ((k ^ JAVA_LCG_MULTIPLIER) & MASK_32) + SKIP_4_ADDEND) & MASK_48) >> 16) | 1ULL)) >> 16) & MASK_16;
}

inline __device__ void addWorldSeed(uint64_t firstAddend, int multTrailingZeroes, uint64_t firstMultInv, uint64_t c, uint64_t e, int x, int z, uint64_t populationSeed, uint64_t* seeds, uint64_t* seedCounter) {
    if(countTrailingZeroes(firstAddend) < multTrailingZeroes)
        return;

    uint64_t n = 1ULL << (16 - multTrailingZeroes);
    uint64_t b = ((((firstMultInv * firstAddend) >> multTrailingZeroes) ^ (JAVA_LCG_MULTIPLIER >> 16)) & (n - 1));
    uint64_t max = 1ULL << 16;
    
    for(; b < max; b += n) {
        uint64_t k = (b << 16) + c;
        uint64_t target2 = (k ^ e) >> 16;
        uint64_t secondAddend = makeSecondAddend(x, k, z);

        if (countTrailingZeroes(target2 - secondAddend) >= multTrailingZeroes) { 
            uint64_t a = ((((firstMultInv * (target2 - secondAddend)) >> multTrailingZeroes) ^ (JAVA_LCG_MULTIPLIER >> 32)) & (n - 1));
            for(; a < max; a += n) { 
                if((getPopulationSeed((a << 32) + k, x, z) & MASK_48) == populationSeed) {
                    addSeed((a << 32) + k, seeds, seedCounter);
                }
            }
        }
    }
    
}

inline __host__ __device__ uint64_t getPopulationSeed(uint64_t worldSeed, int x, int z) {
    uint64_t seed = (worldSeed ^ JAVA_LCG_MULTIPLIER) & MASK_48;
    uint64_t a = nextLong(&seed) | 1ULL;
    uint64_t b = nextLong(&seed) | 1ULL;
    return x * a + z * b ^ worldSeed;
}

inline __host__ __device__ int countTrailingZeroes(uint64_t v) {
    int c;  

    v = (v ^ (v - 1)) >> 1;  

    for(c = 0; v != 0; c++)  {
        v >>= 1;
    }

    return c;
}

inline __host__ __device__ uint64_t nextLong(uint64_t* seed) {
    *seed = (*seed * JAVA_LCG_MULTIPLIER + JAVA_LCG_ADDEND) & MASK_48;
    int u = *seed >> 16;
    *seed = (*seed * JAVA_LCG_MULTIPLIER + JAVA_LCG_ADDEND) & MASK_48;
    return ((uint64_t)u << 32) + (*seed >> 16);
}

uint64_t modInverse(uint64_t x) { 
    uint64_t inv = 0;
    uint64_t b = 1;

    for(int i = 0; i < 16; i++) {
        if((b & 1) == 1) {
            inv |= 1ULL << i;
            b = (b - x) >> 1;
        } else {
            b >>= 1;
        }
    }

    return inv;
}

#undef int
int main() {
    setbuf(stdout, NULL);
#define int uint64_t
    FILE *fp;
    FILE *fp_out;
    char str[MAXCHAR];
    fp = fopen("SEEDS.txt", "r");
    fp_out = fopen("WorldSeeds.txt", "w");
    if (fp == NULL) {
        printf("Could not open file");
        return 1;
    }
    
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
            sscanf(str, "%llu", &inputSeeds[i]);
        }
    }
    
    uint64_t firstMultiplier = (SKIP_2_MULTIPLIER * CHUNK_X + SKIP_4_MULTIPLIER * CHUNK_Z) & MASK_16;
    int32_t multTrailingZeroes = countTrailingZeroes(firstMultiplier); 
    uint64_t firstMultInv = modInverse(firstMultiplier >> multTrailingZeroes);
    
    printf("Beginning converting\n");
    int count = 0; // Counter used for end bit
    while (true) {
        clock_t startTime = clock();
        
        crack<<<WORKER_COUNT >> 9, 1 << 9>>>(inputSeedCount, inputSeeds, outputSeedCount, outputSeeds, 
            multTrailingZeroes, firstMultInv);
    
        bool doneFlag = false;
        count = 0;
        for(uint64_t i = 0; i < WORKER_COUNT; i++) {
            
            if(fgets(str, MAXCHAR, fp) != NULL) {
                sscanf(str, "%llu", &buffer[i]);
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
        
        double timeElapsed = (double)(clock() - startTime);
        timeElapsed /= CLOCKS_PER_SEC;
        
        printf("Seed count %llu, Time %.3fs, Speed: %.3fm/s\n", *outputSeedCount, timeElapsed, WORKER_COUNT / timeElapsed / 1000000);
    
        for (int i = 0; i < *outputSeedCount; i++) {
            fprintf(fp_out, "%llu\n", outputSeeds[i]);
            fflush(fp_out);
            outputSeeds[i] = 0;
        }
        
        *outputSeedCount = 0;
        if (doneFlag) {
            printf("DONE\n");
            break;
        }
    }
    
    crack<<<WORKER_COUNT >> 9, 1 << 9>>>(count, inputSeeds, outputSeedCount, outputSeeds, 
        multTrailingZeroes, firstMultInv);
    CHECK_GPU_ERR(cudaDeviceSynchronize());
    for (int i = 0; i < *outputSeedCount; i++) {
        fprintf(fp_out, "%llu\n", outputSeeds[i]);
        fflush(fp_out);
        outputSeeds[i] = 0;
    }
    fclose(fp);
    fclose(fp_out);
}
