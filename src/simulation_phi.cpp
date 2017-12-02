/* simulation_phi.cpp */
/* 
	This is an implementation of my project that is meant to be optimized for Intel's Xeon Phi coprocessor.
	This version of my project will be run on a Xeon Phi 7120X (Knight's Corner product line)
*/
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

#pragma offload_attribute(pop)
#pragma omp declare target

float G = 6.674e-11;
float *xPos __attribute__((aligned(256)));
float *yPos __attribute__((aligned(256)));
float *xVel __attribute__((aligned(256)));
float *yVel __attribute__((aligned(256)));
float *mass __attribute__((aligned(256)));
uint64_t n;
uint64_t currentTime;
int32_t gifW,gifH,gifDelay;
float timeInSeconds;
float gflops;
uint64_t getCurrentTime()
{
	return currentTime;
}
uint64_t getCount()
{
	return n;
}
void optimizedUpdate() // estimated FLOP counts in comments on each line
{
	#pragma omp parallel for schedule(dynamic)
	for (unsigned int i = 0; i < n; i++) 
	{  
		float Fx = 0.0f; // 1
		float Fy = 0.0f; // 1
		#pragma omp simd reduction(+:Fx,Fy)
		for (unsigned int j = 0; j < n; j++)
		{
			if (j!=i)
			{
				const float dx = xPos[j] - xPos[i]; // 2
				const float dy = yPos[j] - yPos[i]; // 2
				const float r2 = dx*dx+dy*dy+0.001f; // 5
				const float invertedR2 = 1.0f/r2; // 2
				Fx += dx * invertedR2 * mass[j]; // 4
				Fy += dy * invertedR2 * mass[j]; // 4
			}
		}
		xVel[i] += G * Fx * mass[i]; // 4
		yVel[i] += G * Fy * mass[i]; // 4
		xPos[i] += xVel[i]; // 2
		yPos[i] += yVel[i]; // 2
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
	// (19*(n-1) + 14)*n*k total floating point operations == (19n-5)*n*k
	uint64_t appxFlops = (19*getCount()-5)*getCount()*iters;
	timeInSeconds = finish;
	gflops = 1e-9 * appxFlops / finish;
}

int main(int argc,char **argv)
{
	cout << "Number of bodies: ";
	int b;
	cin >> b;
	cout << "Number of iterations: ";	
	int k;
	cin >> k;
	int threads = -1;
	cout << "Use Xeon Phi [y/n]? ";
	char c;
	cin >> c;
	if (c=='y')
	{
		#pragma offload target(mic: 0)
		{
			threads = omp_get_max_threads();
			cout << threads << /*" threads\t"*/"\t";// << endl;
			simulate(b,k);	
			cout << getCount() << /*" bodies\t"*/"\t" << getCurrentTime() << /*" iterations\t"*/"\t" << timeInSeconds << /*" seconds\t"*/"\t" << gflops /*<< " GFlops/s." */<< endl;	
		}
	}
	else
	{
		threads = omp_get_max_threads();
		cout << threads << /*" threads\t"*/"\t";// << endl;
		simulate(b,k);
		cout << getCount() << /*" bodies\t"*/"\t" << getCurrentTime() << /*" iterations\t"*/"\t" << timeInSeconds << /*" seconds\t"*/"\t" << gflops /*<< " GFlops/s." */<< endl;	
	}
	return 0;
}