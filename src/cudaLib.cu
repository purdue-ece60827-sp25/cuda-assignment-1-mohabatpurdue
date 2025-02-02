
#include "cpuLib.h"
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if(i<size) y[i] += scale*x[i];
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	int size_bytes = vectorSize * sizeof(float);
	
	float scale = rand()%100;

	float *X, *Y, *C;
	X = (float *) malloc(size_bytes);
	Y = (float *) malloc(size_bytes);
	C = (float *) malloc(size_bytes);


	if (X == NULL || Y == NULL || C == NULL) {
		printf("Unable to malloc memory ... Exiting!");
		return -1;
	}

	vectorInit(X, vectorSize);
	vectorInit(X, vectorSize);

	float *X_d, *Y_d;

	cudaMalloc((void **) &X_d, size_bytes);
	cudaMemcpy(X_d, X, size_bytes, cudaMemcpyHostToDevice);
	cudaMalloc((void **) &Y_d, size_bytes);
	cudaMemcpy(Y_d, Y, size_bytes, cudaMemcpyHostToDevice);

	auto tStart = std::chrono::high_resolution_clock::now();
	//Run ceil(vectorSize/1024) blocks of 1024 threads each
	saxpy_gpu<<<ceil(vectorSize/1024.0), 1024>>>(X_d, Y_d, scale, vectorSize);

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	cudaMemcpy(C, Y_d, size_bytes, cudaMemcpyDeviceToHost);

	cudaFree(X_d);
	cudaFree(Y_d);
	
	int errorCount = verifyVector(X, Y, C, scale, vectorSize);
	
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	
	free(X);
	free(Y);
	free(C);

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= pSumSize) return;

	curandState_t rng;
	curand_init(clock64(), i, 0, &rng);
	uint64_t hitCount = 0;
	float x, y;
	for (uint64_t idx = 0; idx < sampleSize; ++idx) {
		x = curand_uniform(&rng);
		y = curand_uniform(&rng);
		
		if ( int(x * x + y * y) == 0 ) {
			++ hitCount;
		}
	}
	pSums[i] = hitCount;
}

__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i >= pSumSize/reduceSize) return;
	uint64_t running_sum = 0;
	for (uint64_t idx = 0; idx < reduceSize; ++idx) {
		running_sum += pSums[idx + reduceSize*i];
	}
	totals[i] = running_sum;

}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	auto tEnd= std::chrono::high_resolution_clock::now();

	std::cout << std::setprecision(10);
	std::cout << "Estimated Pi = " << approxPi << "\n";


	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;
	uint64_t *pSums_d;
	uint64_t pSumSize = generateThreadCount;
	uint64_t *totals_d;
	uint64_t totals_size_bytes = reduceThreadCount*sizeof(uint64_t);

	uint64_t *totals;
	totals = (uint64_t *)malloc(totals_size_bytes);

	cudaMalloc((void **)&pSums_d, pSumSize*sizeof(uint64_t));
	cudaMalloc((void **)&totals_d, totals_size_bytes);

	generatePoints<<<ceil(pSumSize/1024.0), 1024>>>(pSums_d, pSumSize, sampleSize);

	reduceCounts<<<reduceThreadCount, 1>>>(pSums_d, totals_d, pSumSize, reduceSize);

	cudaMemcpy(totals, totals_d, totals_size_bytes, cudaMemcpyDeviceToHost);
	uint64_t totals_sum = 0;
	for (uint64_t i = 0; i < reduceThreadCount; i++)
	{
		totals_sum += totals[i];
	}

	approxPi = ((double)totals_sum / sampleSize) / generateThreadCount;
	approxPi *= 4.0;

	return approxPi;
}
