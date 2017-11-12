/* simulation.cpp */
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdio>
#include <string>
#include <algorithm> 
#include <omp.h>
#include <chrono>
#include "gif.h" // gif library for generating animations of the simulation
using namespace std;
using namespace std::chrono;

/*  
	global gravitational constant as used in Newton's Law of universal gravitation
	and Einstein's General theory of relativity 
*/
double G=6.674e-11; 

double dist(double x,double y)
{
	return sqrt(x*x+y*y);
}

class Body
{
public:
	double xPos,yPos; // position coordinates
	double mass;
	double xVel,yVel; // velocity vector coordinates
	Body(double _x, double _y, double _w)
	{
		xPos = _x;
		yPos = _y;
		mass = _w;
		xVel = yVel = 0;
	}
	void updatePhysics(Body* x) // perform body-to-body influence evaluation
	{
		double xr = x->xPos - xPos;
		double yr = x->yPos - yPos;
		double r = dist(xr,yr);
		double xAccel,yAccel;
		xAccel = G * x->mass * xr / pow(r,2.0);
		yAccel = G * x->mass * yr / pow(r,2.0);
		xVel = xVel + xAccel;
		yVel = yVel + yAccel;		
		xPos = xPos + xVel;
		yPos = yPos + yVel;
	}
};

class Simulation
{
private:
	uint64_t n; // body count
	vector<Body*> bodies;
	uint64_t currentTime; // current time instant (starts at zero)
	// gif
	int32_t gifW,gifH,gifDelay; // gif dimensions
	GifWriter gWriter;
public:
	uint8_t *image; // pointer to an array that represents each frame of the gif

	uint64_t getCount()
	{
		return n;
	}

	uint64_t getCurrentTime()
	{
		return currentTime;
	}

	~Simulation()
	{
		for (uint64_t i = 0; i < n; ++i)
			delete bodies[i];
		GifEnd(&gWriter);
	}	

	void randomBodies(uint64_t count) // generate random bodies surrounding the area of the gif
	{
		currentTime = 0;
		default_random_engine generator;
		std::uniform_int_distribution<int> xpos(-gifW,gifW);
		std::uniform_int_distribution<int> ypos(-gifH,gifH);
		std::uniform_real_distribution<double> mass(10000.0,20000.0);
		std::uniform_real_distribution<double> vel(-5.0,5.0);

		for (uint64_t i = 0; i < count - 1; ++i)
			bodies.push_back(new Body(xpos(generator),ypos(generator),mass(generator)));

		bodies.push_back(new Body(0,0,100000000.0)); // a huge single mass in the middle (the Sun)
		n = bodies.size();
	}

	void printUniverse()
	{
		cout << endl << "Bodies at time instant " << currentTime << ":" << endl;
		for (uint64_t i = 0; i < n; ++i)
		{
			cout << "Position: " << bodies[i]->xPos << " " << bodies[i]->yPos << endl
			     << "Velocity: " << bodies[i]->xVel << " " << bodies[i]->yVel << endl;
		}
	}

	void updatePositions() // updates the positions and properties of all bodies using the brute-force O(n^2) algorithm
	{
		for (unsigned int i = 0; i < n; i++) 
		{  
			for (unsigned int j = 0; j < n; j++) 
			{
				if (i != j) 
					bodies[i]->updatePhysics(bodies[j]);
			}
		}
		currentTime++;
	}

	void advance(uint64_t time) // advances the time by a given number of frames
	{
		for (uint64_t i = 0; i < time; ++i)
			updatePositions();
	}	

	// gif handling section begins here
	void setGifProps(int w, int h, int d) // sets all gif properties and initializes the gif writer
	{
		gifW=w;
		gifH=h;
		gifDelay=d; // 100ths of a second
		GifBegin(&gWriter, "simulation.gif", gifW, gifH, gifDelay);
	}	
	void assembleFrame() // puts together a single frame of the gif based on all visible bodies
	{
		image = new uint8_t[4*gifW*gifH]();
		for (uint64_t i = 0; i < n; ++i)
		{	 
			// 0.5x zoom to fit more bodies on the frame	
			int x = bodies[i]->xPos/2 + gifW/2;
			int y = bodies[i]->yPos/2 + gifH/2;
			if (x>=gifW || y>=gifH || x<0 || y<0) 
				continue; // this body won't be displayed

			// shade the body depending on its mass
			int shade = (bodies[i]->mass/20000.0)*256;
			uint32_t gifIndex = 4 * (y * gifW + x); // linear array, each pixel is represented by 4 adjacent values (alpha, R, G, B)

			// different shades of green on a black background (I thought it looked the best and clearest)
			image[gifIndex]   = shade;
			image[gifIndex+1] = 255;
			image[gifIndex+2] = shade;
			image[gifIndex+3] = 0;
		}
		GifWriteFrame(&gWriter, image, gifW, gifH, gifDelay);
		delete[] image;
	}
	// end of gif handling section
	
};

int main(int argc,char **argv)
{
	Simulation sim;
	sim.setGifProps(1024,1024,0);
	sim.randomBodies(1000);
	
	high_resolution_clock::time_point start = high_resolution_clock::now();

	for (int i = 0; i < 1000; ++i)
	{
		sim.advance(1);
		//sim.assembleFrame();
		//if (i % 10 == 0) 
		//	cout << i << endl;
	}
	double finish = duration_cast<duration<double>>(high_resolution_clock::now()-start).count();
	cout << sim.getCount() << " bodies, " << sim.getCurrentTime() << " iterations, " << finish << " seconds." << endl;

	return 0;
}