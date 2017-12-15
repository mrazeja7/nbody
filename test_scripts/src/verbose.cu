/* nbody.cu */
/*
this is a version of the implementation that is supposed to be run on an NVIDIA GPU.
It's a rather straightforward design, with each body calculating its own force vector
from all other bodies in one thread. I think this is appropriate granularity-wise,
if I wanted to use a thread for every (i,j) pair I would potentially spawn
trillions of threads, which is not favorable.
*/
#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdio>
#include <string>
#include <algorithm>
#include <chrono>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdlib>
#include <random>
using namespace std;
using namespace std::chrono;
__device__ __constant__ float G = 6.674e-11;
#define BLOCKSIZE 512
float *xPos;
float *yPos;
float *xVel;
float *yVel;
float *mass;
int n;
int currentTime;
int32_t gifW, gifH, gifDelay;
float timeInSeconds;
float gflops;
int getCurrentTime()
{
	return currentTime;
}
int getCount()
{
	return n;
}

__global__ void updateKernel(float *xPos, float *yPos, float *xVel, float *yVel, float *mass, int count)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID >= count)
		return;
	int i = threadID;
	float G = 6.674e-11;

	float Fx = 0.0f; // 1
	float Fy = 0.0f; // 1
	for (unsigned int j = 0; j < count; j++)
	{
		if (j != i)
		{
			float dx = xPos[j] - xPos[i]; // 2
			float dy = yPos[j] - yPos[i]; // 2
			float r2 = dx*dx + dy*dy + 0.001f; // 5
			float invertedR2 = 1.0f / r2; // 2
			Fx += dx * invertedR2 * mass[j]; // 4
			Fy += dy * invertedR2 * mass[j]; // 4
		}
	}
	xVel[i] += G * Fx * mass[i]; // 4
	yVel[i] += G * Fy * mass[i]; // 4
	xPos[i] += xVel[i]; // 2
	yPos[i] += yVel[i]; // 2
}

void optimizedUpdate() // estimated FLOP counts in comments on each line
{
	cudaError_t cudaStatus;
	int nblocks = (n + BLOCKSIZE - 1) / BLOCKSIZE;
	cout << "attempting to start kernel,  " << nblocks << " " << BLOCKSIZE << endl;
	updateKernel << <nblocks, BLOCKSIZE >> >(xPos, yPos, xVel, yVel, mass, n);
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		cout << "update failed: " << cudaGetErrorString(cudaStatus) << endl;

	cout << "kernel should be running " << endl;
	cudaDeviceSynchronize();
	currentTime++;
}
void setGifProps(int w, int h, int d)
{
	gifW = w;
	gifH = h;
	gifDelay = d;
}

__global__ void initializeBodies(float *xPos, float *yPos, float *xVel, float *yVel, float *mass, int count, int w, int h, int seed)
{
	int threadID = blockIdx.x * blockDim.x + threadIdx.x;
	if (threadID >= count)
		return;
	curandState_t state;
	curand_init(seed, 0, 0, &state);

	if (threadID != count-1)
	{
		xPos[threadID] = curand_uniform(&state) * w - w / 2;
		yPos[threadID] = curand_uniform(&state) * h - h / 2;
		mass[threadID] = curand_uniform(&state) * 10000.0 + 10000.0;
		xVel[threadID] = yVel[threadID] = 0.0;
	}
	else
	{
		xPos[threadID] = 0.0;
		yPos[threadID] = 0.0;
		mass[threadID] = 100000.0;
		xVel[threadID] = yVel[threadID] = 0.0;
	}
}

void randomBodies_onDev_(int count) // initialize everything on the device using a different kernel
{
	cudaError_t cudaStatus;
	cudaMalloc((void**)&xPos, count * sizeof(float));
	cudaMalloc((void**)&yPos, count * sizeof(float));
	cudaMalloc((void**)&xVel, count * sizeof(float));
	cudaMalloc((void**)&yVel, count * sizeof(float));
	cudaMalloc((void**)&mass, count * sizeof(float));

	// 1 thread is enough
	int nblocks = (count + BLOCKSIZE - 1) / BLOCKSIZE;
	initializeBodies << <nblocks, BLOCKSIZE >> > (xPos, yPos, xVel, yVel, mass, count, gifW, gifH, unsigned(time(NULL)));
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		cout << "kernel failed: " << cudaGetErrorString(cudaStatus) << endl;

	cudaDeviceSynchronize();
	n = count;
}

void randomBodies_onDev(int count) // initialize everything on the device using a different kernel
{
	cout << "starting malloc" << endl;
	cudaError_t cudaStatus;
	float *dummy;
	cudaStatus = cudaMalloc((void**)&dummy, count * sizeof(float));
	cout << "dummy malloc: " << cudaGetErrorString(cudaStatus) << endl;
	cudaStatus = cudaMalloc((void**)&xPos, count * sizeof(float));
	cout << "malloc: " << cudaGetErrorString(cudaStatus) << endl;
	cudaStatus = cudaMalloc((void**)&yPos, count * sizeof(float));
	cout << "malloc: " << cudaGetErrorString(cudaStatus) << endl;
	cudaStatus = cudaMalloc((void**)&xVel, count * sizeof(float));
	cout << "malloc: " << cudaGetErrorString(cudaStatus) << endl;
	cudaStatus = cudaMalloc((void**)&yVel, count * sizeof(float));
	cout << "malloc: " << cudaGetErrorString(cudaStatus) << endl;
	cudaStatus = cudaMalloc((void**)&mass, count * sizeof(float));
	cout << "malloc: " << cudaGetErrorString(cudaStatus) << endl;

	// 1 thread is enough
	int nblocks = (count + BLOCKSIZE - 1) / BLOCKSIZE;
	initializeBodies << <nblocks, BLOCKSIZE >> > (xPos, yPos, xVel, yVel, mass, count, gifW, gifH, unsigned(time(NULL)));
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess)
		cout << "kernel failed: " << cudaGetErrorString(cudaStatus) << endl;

	cudaDeviceSynchronize();
	n = count;
}

void randomBodies(int count) // initialize on the host, then copy over to device
{
	cudaMalloc((void**)&xPos, count * sizeof(float));
	cudaMalloc((void**)&yPos, count * sizeof(float));
	cudaMalloc((void**)&xVel, count * sizeof(float));
	cudaMalloc((void**)&yVel, count * sizeof(float));
	cudaMalloc((void**)&mass, count * sizeof(float));

	default_random_engine generator;
	std::uniform_int_distribution<int> xpos(-gifW / 2, gifW / 2);
	std::uniform_int_distribution<int> ypos(-gifH / 2, gifH / 2);
	std::uniform_real_distribution<float> massgen(10000.0, 20000.0);

	float *xPos_h = new float[count];
	float *yPos_h = new float[count];
	float *xVel_h = new float[count];
	float *yVel_h = new float[count];
	float *mass_h = new float[count];

	for (int i = 0; i < count - 1; ++i)
	{
		xPos_h[i] = xpos(generator);
		yPos_h[i] = ypos(generator);
		mass_h[i] = massgen(generator);
		xVel_h[i] = yVel_h[i] = 0.0;
	}
	xPos_h[count - 1] = 0.0;
	yPos_h[count - 1] = 0.0;
	mass_h[count - 1] = 100000.0;
	xVel_h[count - 1] = yVel_h[count - 1] = 0.0;
	n = count;

	cudaMemcpy(xPos, xPos_h, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(yPos, yPos_h, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(xVel, xVel_h, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(yVel, yVel_h, count * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(mass, mass_h, count * sizeof(float), cudaMemcpyHostToDevice);

	delete[] xPos_h;
	delete[] yPos_h;
	delete[] xVel_h;
	delete[] yVel_h;
	delete[] mass_h;
}

void checkBodies()
{
	float *tmp = new float[100];
	cudaMemcpy(tmp, xVel, 100 * sizeof(float), cudaMemcpyDeviceToHost);

	for (size_t i = 0; i < 20; i++)
	{
		cout << tmp[i] << " ";
	}
	cout << endl;
}

void simulate_(int bodies, int iters)
{
	setGifProps(1024, 1024, 1);
	randomBodies_onDev(bodies);
	cout << "generated on dev" << endl;
	high_resolution_clock::time_point start = high_resolution_clock::now();
	for (int i = 0; i < iters; ++i)
		optimizedUpdate();

	float finish = duration_cast<duration<float>>(high_resolution_clock::now() - start).count();
	// (19*(n-1) + 14)*n*k total floating point operations == (19n-5)*n*k
	int appxFlops = (19 * getCount() - 5)*getCount()*iters;
	timeInSeconds = finish;
	gflops = 1e-9 * appxFlops / finish;
	cout << "sim done" << endl;
}

void simulate(int bodies, int iters)
{
	setGifProps(1024, 1024, 1);
	cout << "set gif" << endl;
	randomBodies_onDev(bodies);
	cout << "generated on dev" << endl;
	high_resolution_clock::time_point start = high_resolution_clock::now();
	for (int i = 0; i < iters; ++i)
		optimizedUpdate();
	cout << "optimized" << endl;

	float finish = duration_cast<duration<float>>(high_resolution_clock::now() - start).count();
	// (19*(n-1) + 14)*n*k total floating point operations == (19n-5)*n*k
	int appxFlops = (19 * getCount() - 5)*getCount()*iters;
	timeInSeconds = finish;
	gflops = 1e-9 * appxFlops / finish;
	cout << "sim done" << endl;
}

int main(int argc, char **argv)
{
	//cout << "Number of bodies: ";
	int b;
	cin >> b;
	//cout << "Number of iterations: ";
	int k;
	cin >> k;

	cout << "CUDA" << /*" threads\t"*/"\t";
	simulate(b, k);
	cout << getCount() << /*" bodies\t"*/"\t" << getCurrentTime() << /*" iterations\t"*/"\t" << timeInSeconds << /*" seconds\t"*/"\t" << gflops /*<< " GFlops/s." */ << endl;
	//cout << getCount() << " bodies\n" << getCurrentTime() << " iterations\n" << timeInSeconds << " seconds\n" << gflops << " GFlops/s." << endl;

	checkBodies();
	//system("PAUSE");
	return 0;
}