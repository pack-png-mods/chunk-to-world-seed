// MathDotSqrt

//"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin\nvcc.exe" -ccbin
//"C:\Program Files (x86)\Microsoft Visual
// Studio\2017\Community\VC\Tools\MSVC\14.16.27023\bin\Hostx64\x64" -o main
// main.cu -O3

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

// reduce array of zeros
#include <array>
#include <memory>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>

#include <assert.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <math.h>
#include <sstream>
#include <string>
#include <vector>

// Used for sleep function
#include <chrono>
#include <thread>

#define GPU_ASSERT(code) gpuAssert((code), __FILE__, __LINE__)
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s (code %d) %s %d\n", cudaGetErrorString(code), code, file, line);
    exit(code);
  }
}

/*FILE PATHS*/
constexpr const char *INPUT_FILE_PATH = "data/big_chunk_seeds.txt";
constexpr const char *OUTPUT_FILE_PATH = "data/WorldSeeds.txt";
/*FILE PATHS*/

/*CHUNK CONSTANTS*/
constexpr int32_t CHUNK_X = 3;
constexpr int32_t CHUNK_Z = -3;
constexpr uint64_t INVALID_SEED = 0;
/*CHUNK CONSTANTS*/

/*CUDA LAUNCH CONSTANTS*/
constexpr int32_t BLOCK_DIM_X = 128;
constexpr int32_t BLOCK_DIM_Y = 1;  //should be 1
constexpr int32_t BLOCK_DIM_Z = 1;  //should be 1

constexpr int32_t GRID_DIM_X = 4096;
constexpr int32_t GRID_DIM_Y = 1;   //should be 1
constexpr int32_t GRID_DIM_Z = 1;   //should be 1
/*CUDA LAUNCH CONSTANTS*/

/*MAGIC NUMBERS*/
constexpr uint64_t mod_inv(uint64_t x) {
  uint64_t inv = 0;
  uint64_t b = 1;
  for (int32_t i = 0; i < 16; i++) {
    inv |= (1ULL << i) * (b & 1);
    b = (b - x * (b & 1)) >> 1;
  }
  return inv;
}

constexpr int32_t count_trailing_zeros(uint64_t v) {
  int c = 0;
  v = (v ^ (v - 1)) >> 1;

  for (c = 0; v != 0; c++) {
    v >>= 1;
  }

  return c;
}

constexpr uint64_t MASK48 = ((1ULL << 48) - 1ULL);
constexpr uint64_t MASK32 = ((1ULL << 32) - 1ULL);
constexpr uint64_t MASK16 = ((1ULL << 16) - 1ULL);

constexpr uint64_t M1 = 25214903917ULL;
constexpr uint64_t ADDEND1 = 11ULL;

constexpr uint64_t M2 = 205749139540585ULL;
constexpr uint64_t ADDEND2 = 277363943098ULL;

constexpr uint64_t M4 = 55986898099985ULL;
constexpr uint64_t ADDEND4 = 49720483695876ULL;

constexpr auto FIRST_MULT = (M2 * (uint64_t)CHUNK_X + M4 * (uint64_t)CHUNK_Z) & MASK16;
constexpr auto MULT_TRAILING_ZEROS = count_trailing_zeros(FIRST_MULT);
constexpr auto FIRST_MULT_INV = (uint64_t)mod_inv(FIRST_MULT >> MULT_TRAILING_ZEROS);

constexpr auto X_COUNT = count_trailing_zeros((uint64_t)CHUNK_X);
constexpr auto Z_COUNT = count_trailing_zeros((uint64_t)CHUNK_Z);
constexpr auto TOTAL_COUNT = count_trailing_zeros(CHUNK_X | CHUNK_Z);

constexpr auto C_MAX = (1ULL << 16);
constexpr auto C_STRIDE = (1ULL << (TOTAL_COUNT + 1));
/*MAGIC NUMBERS*/

/*DETAILS*/
constexpr int32_t SEEDS_PER_LAUNCH = BLOCK_DIM_X * GRID_DIM_X;
constexpr int32_t WORLD_SEEDS_PER_CHUNK_SEED = 8;
constexpr size_t INPUT_SEED_ARRAY_SIZE = SEEDS_PER_LAUNCH;//SEEDS_PER_LAUNCH;
constexpr size_t OUTPUT_SEED_ARRAY_SIZE = SEEDS_PER_LAUNCH * WORLD_SEEDS_PER_CHUNK_SEED;//1 << 20;
constexpr int32_t MAX_LINE = 1000;
/*DETAILS*/

__host__ __device__
int64_t next_long(uint64_t *seed) {
  *seed = (*seed * M1 + ADDEND1) & MASK48;
  int32_t u = *seed >> 16;
  *seed = (*seed * M1 + ADDEND1) & MASK48;
  return ((uint64_t)u << 32) + (int32_t)(*seed >> 16);
}

__host__ __device__
uint64_t make_mask(int32_t bits) { return (1ULL << bits) - 1; }

__device__
int ctz(uint64_t v) {
  // return __popcll((v & (-v))-1);
  return __popcll(v ^ (v - 1)) - 1;
}


__device__
void clear_seed(uint64_t *bucket) {
  *bucket = INVALID_SEED;
}

__device__
void add_seed_cond(bool cond, uint64_t new_seed, uint64_t *bucket, uint32_t *index) {
    if(cond){
      bucket[*index] = new_seed;
      *index += 1;
    }

}

__host__ __device__
uint64_t get_chunk_seed(uint64_t worldSeed) {
  uint64_t seed = (worldSeed ^ M1) & MASK48;
  int64_t a = next_long(&seed) / 2 * 2 + 1;
  int64_t b = next_long(&seed) / 2 * 2 + 1;
  return (uint64_t)(((CHUNK_X * a + CHUNK_Z * b) ^ worldSeed) & MASK48);
}

__host__ __device__
uint64_t get_partial_addend(uint64_t partialSeed, int32_t bits) {
  uint64_t mask = make_mask(bits);
  /* clang-format off */
  return ((uint64_t)CHUNK_X) * (((int32_t)(((M2 * ((partialSeed ^ M1) & mask) + ADDEND2) & MASK48) >> 16)) / 2 * 2 + 1) +
         ((uint64_t)CHUNK_Z) * (((int32_t)(((M4 * ((partialSeed ^ M1) & mask) + ADDEND4) & MASK48) >> 16)) / 2 * 2 + 1);
  /* clang-format on */
}

__device__
void add_world_seed(uint64_t firstAddend, uint64_t c, uint64_t chunkSeed, uint64_t *bucket, uint32_t *index) {
  if(ctz(firstAddend) < MULT_TRAILING_ZEROS){
    return;
  }
  uint64_t bottom32BitsChunkseed = chunkSeed & MASK32;
  uint64_t b = (((FIRST_MULT_INV * firstAddend) >> MULT_TRAILING_ZEROS) ^ (M1 >> 16)) & make_mask(16 - MULT_TRAILING_ZEROS);
  if (MULT_TRAILING_ZEROS != 0) {
    uint64_t smallMask = make_mask(MULT_TRAILING_ZEROS);
    uint64_t smallMultInverse = smallMask & FIRST_MULT_INV;
    uint64_t target = (((b ^ (bottom32BitsChunkseed >> 16)) & smallMask) -
                       (get_partial_addend((b << 16) + c, 32 - MULT_TRAILING_ZEROS) >> 16)) &
                      smallMask;
    b += (((target * smallMultInverse) ^ (M1 >> (32 - MULT_TRAILING_ZEROS))) & smallMask) << (16 - MULT_TRAILING_ZEROS);
  }
  uint64_t bottom32BitsSeed = (b << 16) + c;
  uint64_t target2 = (bottom32BitsSeed ^ bottom32BitsChunkseed) >> 16;
  uint64_t secondAddend = (get_partial_addend(bottom32BitsSeed, 32) >> 16);
  secondAddend &= MASK16;
  uint64_t topBits = ((((FIRST_MULT_INV * (target2 - secondAddend)) >> MULT_TRAILING_ZEROS) ^ (M1 >> 32)) &
                      make_mask(16 - MULT_TRAILING_ZEROS));

  for (; topBits < (1ULL << 16); topBits += (1ULL << (16 - MULT_TRAILING_ZEROS))) {
    bool condition = get_chunk_seed((topBits << 32) + bottom32BitsSeed) == chunkSeed;
    uint64_t seed_candidate = (topBits << 32) + bottom32BitsSeed;
    add_seed_cond(condition, seed_candidate, bucket, index);
  }
  //__syncthreads();
}

__device__
void add_some_seeds(uint64_t chunk_seed, uint64_t c, uint64_t *bucket, uint32_t *index){
  constexpr auto x = (uint64_t)CHUNK_X;
  constexpr auto z = (uint64_t)CHUNK_Z;

  const auto f = chunk_seed & MASK16;
  const auto target = (c ^ f) & MASK16;
  uint64_t magic = (uint64_t)(x * ((M2 * ((c ^ M1) & MASK16) + ADDEND2) >> 16)) +
                   (uint64_t)(z * ((M4 * ((c ^ M1) & MASK16) + ADDEND4) >> 16));

  add_world_seed(target - (magic & MASK16), c, chunk_seed, bucket, index);
  //nvcc optimizes this branching conditional statically
  //no need for macros here
  if (CHUNK_X != 0) {
    add_world_seed(target - ((magic + x) & MASK16), c, chunk_seed, bucket, index);
  }
  if (CHUNK_Z != 0 && CHUNK_X != CHUNK_Z) {
    add_world_seed(target - ((magic + z) & MASK16), c, chunk_seed, bucket, index);
  }
  if (CHUNK_X != 0 && CHUNK_Z != 0 && CHUNK_X + CHUNK_Z != 0) {
    add_world_seed(target - ((magic + x + z) & MASK16), c, chunk_seed, bucket, index);
  }
  if (CHUNK_X != 0 && CHUNK_X != CHUNK_Z) {
    add_world_seed(target - ((magic + 2 * x) & MASK16), c, chunk_seed, bucket, index);
  }
  if (CHUNK_Z != 0 && CHUNK_X != CHUNK_Z) {
    add_world_seed(target - ((magic + 2 * z) & MASK16), c, chunk_seed, bucket, index);
  }
  if (CHUNK_X != 0 && CHUNK_Z != 0 && CHUNK_X + CHUNK_Z != 0 && CHUNK_X * 2 + CHUNK_Z != 0) {
    add_world_seed(target - ((magic + 2 * x + z) & MASK16), c, chunk_seed, bucket, index);
  }
  if (CHUNK_X != 0 && CHUNK_Z != 0 && CHUNK_X != CHUNK_Z && CHUNK_X + CHUNK_Z != 0 && CHUNK_X + CHUNK_Z * 2 != 0) {
    // is the x supposed to be multiplied
    add_world_seed(target - ((magic + x + 2 * z) & MASK16), c, chunk_seed, bucket, index);
  }
  if (CHUNK_X != 0 && CHUNK_Z != 0 && CHUNK_X + CHUNK_Z != 0) {
    add_world_seed(target - ((magic + 2 * x + 2 * z) & MASK16), c, chunk_seed, bucket, index);
  }
}

__global__
void crack(uint64_t input_seed_count, uint64_t *input_seed_array, uint64_t *output_seed_array) {
  const int32_t thread_id = blockIdx.x * BLOCK_DIM_X + threadIdx.x;

  const int32_t input_seed_index = thread_id;
  const int32_t output_seed_index = thread_id * WORLD_SEEDS_PER_CHUNK_SEED;

  if(input_seed_index >= input_seed_count){
    return;
  }
  uint64_t chunk_seed = input_seed_array[thread_id];
  uint32_t index_count = 0;

  const uint64_t start_c = X_COUNT == Z_COUNT ? chunk_seed & ((1ULL << (X_COUNT + 1)) - 1)
                                : chunk_seed & ((1ULL << (TOTAL_COUNT + 1)) - 1) ^ (1 << TOTAL_COUNT);
  for(uint64_t c = start_c; c < C_MAX; c += C_STRIDE){
    add_some_seeds(chunk_seed, c, output_seed_array + output_seed_index, &index_count);
  }
}

FILE *open_file(const char *path, const char *mode) {
  auto fp = fopen(path, mode);
  if (fp == nullptr) {
    printf("Error: could not open file %s with mode %s", path, mode);
    exit(1);
  }
  return fp;
}

int32_t count_file_length(FILE *file) {
  static char line[MAX_LINE];
  int32_t total = 0;
  while (fgets(line, MAX_LINE, file))
    total++;

  // seeks to beginning of file
  rewind(file);
  return total;
}

size_t file_to_buffer(FILE *source, uint64_t *dest, size_t N) {
  static char line[MAX_LINE];
  for (size_t i = 0; i < N; i++) {
    if (fgets(line, MAX_LINE, source) != nullptr) {
      sscanf(line, "%llu", &dest[i]); // THIS IS SUPPOSED TO BE LLU
      //printf("seed %llu | c %llu\n", dest[i], c);
    } else {
      return i;
    }
  }
  return N;
}

int32_t buffer_to_file(uint64_t *source, FILE *dest, size_t N) {
  int32_t count = 0;
  for (size_t i = 0; i < N; i++) {
    if(source[i] != INVALID_SEED){
      count++;
      fprintf(dest, "%llu\n", source[i]); // THIS IS SUPPOSED TO BE LLU
    }
  }
  //printf("COUNT %d\n", count);
  fflush(dest);
  return count;
}

int main() {
  using clock=std::chrono::high_resolution_clock;
  using h_duration=std::chrono::duration<double, std::ratio<60 * 60>>;
  using m_duration=std::chrono::duration<double, std::ratio<60>>;
  using s_duration=std::chrono::duration<double>;
  using ms_duration=std::chrono::duration<double, std::milli>;


  //my implementation doesnt work for special case of CHUNK_X == CHUNK_Z == 0
  assert(CHUNK_X != 0 || CHUNK_Z != 0);
  setbuf(stdout, NULL);
  std::cout << "Init...\n";

  const dim3 GRID_DIM(GRID_DIM_X, GRID_DIM_Y, GRID_DIM_Z);
  const dim3 BLOCK_DIM(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z);

  std::cout << "Opening files...\n";
  FILE *in = open_file(INPUT_FILE_PATH, "r");
  FILE *out = open_file(OUTPUT_FILE_PATH, "w");

  const int32_t total_input_seeds = count_file_length(in);
  uint64_t *input_seeds_cpu = (uint64_t *)malloc(sizeof(uint64_t) * INPUT_SEED_ARRAY_SIZE);
  uint64_t *output_seeds_cpu = (uint64_t *)calloc(OUTPUT_SEED_ARRAY_SIZE, sizeof(uint64_t));   //needs default zeros

  uint64_t *input_seeds_gpu = nullptr;
  uint64_t *output_seeds_gpu = nullptr;

  //not using managed memory because it is slow
  GPU_ASSERT(cudaMalloc(&input_seeds_gpu, sizeof(uint64_t) * INPUT_SEED_ARRAY_SIZE));
  GPU_ASSERT(cudaMalloc(&output_seeds_gpu, sizeof(uint64_t) * OUTPUT_SEED_ARRAY_SIZE));

  uint64_t file_input_count = file_to_buffer(in, input_seeds_cpu, INPUT_SEED_ARRAY_SIZE);
  GPU_ASSERT(cudaMemcpy(input_seeds_gpu, input_seeds_cpu, file_input_count * sizeof(uint64_t), cudaMemcpyHostToDevice));

  std::cout << "Total seeds: " << total_input_seeds << "\n";
  std::cout << "Launching kernel...\n";
  auto start_time = clock::now();
  auto prev_time = start_time;
  auto current_time = start_time;
  int32_t total_searched = 0;
  int32_t total_found = 0;
  while (file_input_count > 0) {
    crack<<<GRID_DIM, BLOCK_DIM>>>(file_input_count, input_seeds_gpu, output_seeds_gpu);
    auto num_seeds_found = buffer_to_file(output_seeds_cpu, out, OUTPUT_SEED_ARRAY_SIZE);
    auto prev_file_input_count = file_input_count;
    file_input_count = file_to_buffer(in, input_seeds_cpu, INPUT_SEED_ARRAY_SIZE);

    GPU_ASSERT(cudaPeekAtLastError());
    GPU_ASSERT(cudaDeviceSynchronize());
    //output_count = *output_seed_count;
    GPU_ASSERT(cudaMemcpy(input_seeds_gpu, input_seeds_cpu, file_input_count * sizeof(uint64_t), cudaMemcpyHostToDevice));
    GPU_ASSERT(cudaMemcpy(output_seeds_cpu, output_seeds_gpu, OUTPUT_SEED_ARRAY_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    GPU_ASSERT(cudaMemset(output_seeds_gpu, INVALID_SEED, OUTPUT_SEED_ARRAY_SIZE * sizeof(uint64_t)));


    current_time = clock::now();
    total_searched += prev_file_input_count;
    total_found += num_seeds_found;


    auto s_delta = s_duration(current_time - prev_time).count();
    auto k_seeds_per_second = prev_file_input_count / s_delta / 1000.0;
    auto completion = (double)total_searched / total_input_seeds * 100;
    auto e_time = (double) (total_input_seeds - total_searched) / INPUT_SEED_ARRAY_SIZE * s_delta;
    char suffix = 's';
    if(e_time >= 60 * 60){
      e_time /= 3600.0;
      suffix = 'h';
    }
    else if(e_time >= 60){
      e_time /= 60.0;
      suffix = 'm';
    }

    auto uptime = s_duration(current_time - start_time).count();
    //Searched Uptime
    printf("Searched: %d seeds | Found: %d seeds | Speed: %.2lfk seeds/s | Completion: %.3lf%% | ETA: %.1lf%c | Uptime: %.1lfs\n",
      total_searched, total_found, k_seeds_per_second, completion, e_time, suffix, uptime
    );
    // std::cout << "Searched: " << total_searched << " Found: " << total_found
    // << " Uptime: " << uptime << "s Seeds " << seeds_per_second << "seed/s \n";
    prev_time = current_time;
  }
  total_found += buffer_to_file(output_seeds_cpu, out, OUTPUT_SEED_ARRAY_SIZE);

  auto stop_time = clock::now();
  std::cout << "Total world seeds converted: " << total_found << " seeds\n";
  std::cout << "Total execution time: " << s_duration( stop_time - start_time).count() << "s\n";

  free(input_seeds_cpu);
  free(output_seeds_cpu);

  cudaFree(input_seeds_gpu);
  cudaFree(output_seeds_gpu);
  //cudaFree(output_seed_count);
  fclose(in);
  fflush(out);
  fclose(out);
}
