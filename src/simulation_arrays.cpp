/* simulation_arrays.cpp */
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <string>
#include <algorithm> 
#include <omp.h>
#include <chrono>
//#include "gif.h" // gif library for generating animations of the simulation
using namespace std;
using namespace std::chrono;

float G = 6.674e-11;
int nthreads;
float *xPos;
float *yPos;
float *xVel;
float *yVel;
float *mass;
uint64_t n;
uint64_t currentTime;
int32_t gifW,gifH,gifDelay;
float timeInSeconds;
uint64_t getCurrentTime()
{
	return currentTime;
}
uint64_t getCount()
{
	return n;
}
void optimizedUpdate() 
{
	#pragma omp parallel for schedule(dynamic) num_threads(nthreads)
	for (unsigned int i = 0; i < n; i++) 
	{  
		float Fx = 0.0f;
		float Fy = 0.0f;
		#pragma omp simd reduction(+:Fx,Fy)
		for (unsigned int j = 0; j < n; j++) 
		{
			if (j!=i)
			{
				const float dx = xPos[j] - xPos[i];
				const float dy = yPos[j] - yPos[i];
				const float r2 = dx*dx+dy*dy+0.001f;
				const float invertedR2 = 1.0f/r2;
				Fx += dx * invertedR2 * mass[j];
				Fy += dy * invertedR2 * mass[j];
			}
		}
		xVel[i] += G * Fx * mass[i];
		yVel[i] += G * Fy * mass[i];
		xPos[i] += xVel[i];
		yPos[i] += yVel[i];
	}
	currentTime++;
}
void setGifProps(int w, int h, int d)
{
	gifW = w;
	gifH = h;
	gifDelay = d;		
}
void randomBodies(uint64_t count)
{
	currentTime = 0;
	default_random_engine generator;
	std::uniform_int_distribution<int> xpos(-gifW/2,gifW/2);
	std::uniform_int_distribution<int> ypos(-gifH/2,gifH/2);
	std::uniform_real_distribution<float> massgen(10000.0,20000.0);

	xPos = new float[count];
	yPos = new float[count];
	xVel = new float[count];
	yVel = new float[count];
	mass = new float[count];

	for (uint64_t i = 0; i < count - 1; ++i)
	{
		xPos[i] = xpos(generator);
		yPos[i] = ypos(generator);
		mass[i] = massgen(generator);
		xVel[i] = yVel[i] = 0.0;
	}
	xPos[count-1] = 0.0;
	yPos[count-1] = 0.0;
	mass[count-1] = 100000.0;
	xVel[count-1] = yVel[count-1] = 0.0;
	n = count;
}

void simulate(int bodies,int iters)
{
	setGifProps(1024,1024,1);
	randomBodies(bodies);
	high_resolution_clock::time_point start = high_resolution_clock::now();
	for (int i = 0; i < iters; ++i)
		optimizedUpdate();

	float finish = duration_cast<duration<float>>(high_resolution_clock::now()-start).count();
	timeInSeconds = finish;
	
}

int main(int argc,char **argv)
{
	cout << "Number of bodies: ";
	int b;
	cin >> b;
	cout << "Number of iterations: ";	
	int k;
	cin >> k;
	cout << "Number of threads: ";
	cin >> nthreads;	
	
	cout << nthreads << /*" threads\t"*/"\t";
	simulate(b,k);
	cout << getCount() << /*" bodies\t"*/"\t" << getCurrentTime() << /*" iterations\t"*/"\t" << timeInSeconds << /*" seconds" << */endl;	
	cout << getCount() << " bodies\t" << getCurrentTime() << " iterations\t" << timeInSeconds << " seconds" << endl;	
	return 0;
}