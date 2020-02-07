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
inline __host__ __device__ uint64_t modInverse(uint64_t x, int mod);
inline __host__ __device__ int countTrailingZeroes(uint64_t v);
inline __host__ __device__ uint64_t nextLong(uint64_t* seed);
inline __host__ __device__ uint64_t getPopulationSeed(uint64_t worldSeed, int x, int z);









#define CHUNK_X 3
#define CHUNK_Z -3

















inline __device__ void addSeed(uint64_t seed,uint64_t* seeds,uint64_t* seedCounter)
{
	uint64_t Id=atomicAdd(seedCounter,1);
	seeds[Id]=seed;
}

__global__ void crack(uint64_t seedInputCount,uint64_t* seedInputArray,uint64_t* seedOutputCounter,uint64_t* seedOutputArray) {
    uint64_t global_id = blockIdx.x * blockDim.x + threadIdx.x;
	if((global_id)>seedInputCount)
		return;
	
    uint64_t seed = seedInputArray[global_id];
    
	seed = seed ^ 25214903917ULL;//xor with const ASSUMPTION THAT original seeds arnt xored
	
    int x = CHUNK_X;
    int z = CHUNK_Z;
    
    uint64_t c = 0ULL;
    uint64_t e = seed & (1ULL << 32) - 1; 
    uint64_t f = seed & ((1ULL << 16) - 1); 

    uint64_t firstMultiplier = (205749139540585ULL * x + 55986898099985ULL * z) & ((1ULL << 16) - 1);
    int multTrailingZeroes = countTrailingZeroes(firstMultiplier); 
    uint64_t firstMultInv = modInverse(firstMultiplier >> multTrailingZeroes, 16);
    
    if(((x ^ z) & 1)) { 
        c = (seed & 1) ^ 1;
    } else {
        c = (seed & 1);
    }
    
    for(; c < (1L << 16); c += 2) {
        uint64_t target = (c ^ f) & ((1ULL << 16) - 1);
        uint64_t magic = x * ((205749139540585ULL * ((c ^ 25214903917ULL) & ((1ULL << 16) - 1)) + 277363943098ULL) >> 16) + z * ((55986898099985ULL * ((c ^ 25214903917ULL) & ((1ULL << 16) - 1)) + 49720483695876ULL) >> 16);
        
		//printf("%llu %llu\n",target,magic);
		
        addWorldSeed(target - (magic & ((1ULL << 16) - 1)), multTrailingZeroes, firstMultInv, c, e, x, z, seed, seedOutputArray, seedOutputCounter); 
        
        addWorldSeed(target - ((magic + x) & ((1ULL << 16) - 1)), multTrailingZeroes, firstMultInv, c, e, x, z, seed, seedOutputArray, seedOutputCounter);
        addWorldSeed(target - ((magic + z) & ((1ULL << 16) - 1)), multTrailingZeroes, firstMultInv, c, e, x, z, seed, seedOutputArray, seedOutputCounter); 
        
        addWorldSeed(target - ((magic + x + z) & ((1ULL << 16) - 1)), multTrailingZeroes, firstMultInv, c, e, x, z, seed, seedOutputArray, seedOutputCounter); 
    }
}

inline __host__ __device__ uint64_t makeSecondAddend(int x, uint64_t k, int z) {
    return ((x*((((205749139540585ULL * ((k ^ 25214903917ULL) & (1ULL << 32) - 1) + 277363943098ULL) & (1ULL << 48) - 1) >> 16) | 1ULL) +
            z*((((55986898099985ULL * ((k ^ 25214903917ULL) & (1ULL << 32) - 1) + 49720483695876ULL) & (1ULL << 48) - 1) >> 16) | 1ULL)) >> 16) & ((1ULL << 16) - 1);
}

inline __device__ void addWorldSeed(uint64_t firstAddend, int multTrailingZeroes, uint64_t firstMultInv, uint64_t c, uint64_t e, int x, int z, uint64_t populationSeed, uint64_t* seeds, uint64_t* seedCounter) {
    if(countTrailingZeroes(firstAddend) < multTrailingZeroes)return;
 
    if(multTrailingZeroes > 16) {
        return;
    }

    uint64_t n = 1ULL << (16 - multTrailingZeroes);
    
    for(uint64_t b = ((((firstMultInv * firstAddend) >> multTrailingZeroes) ^ (25214903917ULL >> 16)) & ((1ULL << (16 - multTrailingZeroes)) - 1)); b < 0xFFFFULL; b += n) {
        uint64_t k = (b << 16) + c;
        uint64_t target2 = (k ^ e) >> 16;
        uint64_t secondAddend = makeSecondAddend(x, k, z);

        if (countTrailingZeroes(target2 - secondAddend) >= multTrailingZeroes) { 
            uint64_t a = ((((firstMultInv * (target2 - secondAddend)) >> multTrailingZeroes) ^ (25214903917ULL >> 32)) & ((1ULL << (16 - multTrailingZeroes)) - 1));

            for(; a < (1L << 16); a += (1L << (16 - multTrailingZeroes))) { 
                if(getPopulationSeed((a << 32) + k, x, z) == populationSeed) {
					addSeed((a << 32) + k,seeds,seedCounter);
					return;//KAP HAD IT LIKE THIS
                }
            }
        }
    }
    
}

inline __host__ __device__ uint64_t getPopulationSeed(uint64_t worldSeed, int x, int z) {
    uint64_t seed = worldSeed ^ 25214903917ULL;
    return (x * (nextLong(&seed) | 1ULL) + z * (nextLong(&seed) | 1ULL) ^ worldSeed)&((1ULL<<48)-1);
}

inline __host__ __device__ int countTrailingZeroes(uint64_t v) {
    int c;  

    v = (v ^ (v - 1)) >> 1;  

    for(c = 0; v != 0; c++)  {
        v >>= 1;
    }

    return c;
}

inline __host__ __device__ uint64_t modInverse(uint64_t x, int mod) { 
    uint64_t inv = 0;
    uint64_t b = 1;

    for(int i = 0; i < mod; i++) {
        if((b & 1) == 1) {
            inv |= 1ULL << i;
            b = (b - x) >> 1;
        } else {
            b >>= 1;
        }
    }

    return inv;
}

inline __host__ __device__ uint64_t nextLong(uint64_t* seed) {
    *seed = (*seed * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
    int u = *seed >> 16;
    *seed = (*seed * 0x5DEECE66DULL + 0xBULL) & ((1ULL << 48) - 1);
    return ((uint64_t)u << 32) + (*seed >> 16);
}






















#define OUTPUT_SEED_ARRAY_SIZE (1ULL<<20)





#define WORKER_COUNT (1ULL<<16)


#define MAXCHAR 1000

#undef int
int main() {
	setbuf(stdout, NULL);
#define int uint64_t
	FILE *fp;
    char str[MAXCHAR];
	fp = fopen("SEEDS.txt", "r");
	if (fp == NULL){
        printf("Could not open file");
        return 1;
    }
	
	uint64_t* buffer=(uint64_t*)malloc(WORKER_COUNT*sizeof(uint64_t));
	
	
	
	int inputSeedCount=WORKER_COUNT;
	
	
	uint64_t* inputSeeds;
	CHECK_GPU_ERR(cudaMallocManaged(&inputSeeds, sizeof(*inputSeeds)*(inputSeedCount)));


	
	
	uint64_t* outputSeedCount;
	CHECK_GPU_ERR(cudaMallocManaged(&outputSeedCount, sizeof(*outputSeedCount)));
	
	uint64_t* outputSeeds;
	CHECK_GPU_ERR(cudaMallocManaged(&outputSeeds, sizeof(*outputSeeds) * OUTPUT_SEED_ARRAY_SIZE));
	
	//Inital copy
	for(uint64_t i=0;i<WORKER_COUNT;i++)
	{
		if(fgets(str, MAXCHAR, fp) != NULL)
		{
			sscanf(str,"%llu",&inputSeeds[i]);
		}
	}
	
	
	printf("Begining cracking\n");
	int count=0;//Counter used for end bit
	while(true)
	{
		clock_t startTime = clock();
		
		
		crack<<<WORKER_COUNT>>9,1<<9>>>(inputSeedCount,inputSeeds,outputSeedCount,outputSeeds);
	
		bool doneFlag=false;
		count=0;
		for(uint64_t i=0;i<WORKER_COUNT;i++)
		{
			if(fgets(str, MAXCHAR, fp) != NULL)
			{
				sscanf(str,"%llu",&buffer[i]);
				count++;
			}
			else
			{
			
				doneFlag=true;
			}
		}
		CHECK_GPU_ERR(cudaPeekAtLastError());
		CHECK_GPU_ERR(cudaDeviceSynchronize());
		for(uint64_t i=0;i<WORKER_COUNT;i++)
		{
			inputSeeds[i]=buffer[i];
		}
		
		double timeElapsed = (double)(clock() - startTime);
		timeElapsed /= CLOCKS_PER_SEC;
		
		printf("Seed count %llu, Time %.1fs %i\n",*outputSeedCount,timeElapsed,doneFlag);
	
		for (int i = 0; i < *outputSeedCount; i++)
		{
			//PRINT TO FILE HERE
			outputSeeds[i] = 0;
		}
		*outputSeedCount=0;
		if(doneFlag)
		{
			printf("DONE\n");
			break;
		}

	}
	
	crack<<<WORKER_COUNT>>9,1<<9>>>(count,inputSeeds,outputSeedCount,outputSeeds);
	CHECK_GPU_ERR(cudaDeviceSynchronize());
	for (int i = 0; i < *outputSeedCount; i++)
	{
		
		
		//PRINT TO FILE HERE
		outputSeeds[i] = 0;
	}
    fclose(fp);
}


































