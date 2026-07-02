//Optimized using shared memory and on chip memory 																																			
// nvcc microPlastics.cu -o microPlastics -lglut -lm -lGLU -lGL
//To stop hit "control c" in the window you launched it from.

// =====================================================================
//  SCALED UNIT SYSTEM (consistent, near-real-time-friendly)
//
//   Length  : 1 unit = 1 micrometer   (um)
//   Time    : 1 unit = 1 millisecond  (ms)
//   Mass    : 1 unit = 1 picogram     (pg)
//
//  Derived:
//   Velocity         : um/ms        ( = mm/s in SI)
//   Acceleration     : um/ms^2      ( = m/s^2 in SI — gravity g = 9.81!)
//   Force            : pg*um/ms^2   ( = femtoNewton, fN, = 1e-15 N)
//   Energy           : pg*um^2/ms^2 ( = zettaJoule,  zJ, = 1e-21 J)
//   Density          : pg/um^3      ( = g/cm^3)
//
//  Useful constants in these units:
//   kB              = 1.38e-2 pg*um^2/ms^2/K     (Boltzmann)
//   eta (water)     = 1.0e3    pg/(um*ms)         (dynamic viscosity) 
//   c (light)       = 3.0e11   um/ms              (speed of light) 
//   g (gravity)     = 9.81     um/ms^2
// =====================================================================


#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <GL/glut.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
using namespace std;

FILE* ffmpeg;

#define BOLD_ON  "\e[1m"
#define BOLD_OFF   "\e[m"

#define PI 3.141592654
#define BLOCK 256

#define K_BOLTZMANN 1.38e-2f	//boltzmann constant in pg*um^2/ms^2/K 

#include "./forceFunctions.h"
#include "./histogram.h"
#include "./rayTraceLaser.h"

FILE* MovieFile;
int* Buffer;
int MovieFlag; // 0 movie off, 1 movie on

// Globals to be read in from parameter file.
int NumberOfMicroPlastics;
double MaxMicroPlastics1DVelocity;
double DensityOfMicroPlasticMin;
double DensityOfMicroPlasticMax;
float DiameterOfMicroPlasticMin;
float DiameterOfMicroPlasticMax;

int NumberOfPolymerChains;
int PolymersChainLengthMin;
int PolymersChainLengthMax;

float PolymersConnectionLength;
double DensityOfPolymer;
float DiameterOfPolymer;

float BeakerRadius; //4900.0;
float FluidHeight; //118000.0;

float FluidDensity;
float Drag;

float TotalRunTime;
float Dt;
int DrawRate;
int PrintRate;

float PolymerRed;
float PolymerGreen;
float PolymerBlue;

float MicroPlasticRed;
float MicroPlasticGreen;
float MicroPlasticBlue;

// Other Globals
int Pause;
int ViewFlag; // 0 orthoganal, 1 fulstum
int TopView;
int NumberOfBodies;
int NumberOfPolymers;

//flags for container on or off and particle interactions on or off
int ContainerFlag;
int ParticleInteractionFlag;

float4 *BodyPosition, *BodyVelocity, *BodyForce;
RayData *RaysGPU;
float4  *LaserForceGPU;
float4 *BodyPositionGPU, *BodyVelocityGPU, *BodyForceGPU;
float3 *MicroPlasticColors; // to make them rainbow- take out if you want them back to normal 
int *PolymerChainLength;
int *PolymerConnectionA, *PolymerConnectionB;
int *PolymerConnectionAGPU, *PolymerConnectionBGPU;
curandState_t* DevStates;
dim3 Blocks, Grids;
int DrawTimer, PrintTimer;
float RunTime;
float4 CenterOfSimulation;
float4 DistanceFromCenter = make_float4(0.0, 0.0, 0.0, 0.0);
float4 AngleOfSimulation;
float ShakeItUpMag;
float LaserBeamRadius;
float LaserBeamCenterY;
int   LaserDebugMode = 0;        // 0 = normal, 1 = freeze on first hit
int   FirstHitRayIndex = -1;     // -1 = no hit yet
int   FirstHitParticle = -1;
float FirstHitX, FirstHitY, FirstHitZ;
float FirstHitFs, FirstHitFg;
float FirstHitFx, FirstHitFy, FirstHitFz;
float FirstHitAngleI, FirstHitAngleR;
float4 *DebugHitInfoGPU;  // per-ray debug data
float4 *DebugHitInfo;     // CPU side
float4 *DebugForceGPU;
float4 *DebugForce;


int DebugFlag;
int RadialConfinementViewingAids;
int StirFlag;
float StirAngularVelosity;
float Theta;
int LaserFlag;
int ShakeItUpFlag;
int GravityFlag;
int BrownianFlag;
int DragFlag;
float UsedDrag;

// Window globals
int Window;
int XWindowSize;
int YWindowSize;
double Near;
double Far;
double EyeX;
double EyeY;
double EyeZ;
double CenterX;
double CenterY;
double CenterZ;
double UpX;
double UpY;
double UpZ;



// Prototyping functions
void readSimulationParameters();
void setNumberOfBodies();
void allocateMemory();
__global__ void getForcesSetup(curandState_t*, float4 *, float4 *, float4 *, int *, int *, float, int, int, float, float, float, int, float, int, float );
void polymerShakeUp(float4 *, float4 *, float4 *, int *, int *, float, int, float, float, float, float);
void setInitailConditions();
void drawPicture();
void errorCheck(const char*);
__global__ void init_curand(unsigned int, curandState_t*);
__global__ void getForces(curandState_t*, float4 *, float4 *, float4 *, int *, int *, float, int, int, float, float, float, int, float, int, float, int, int, int, int, float, float4 *);
__global__ void moveBodies(float4 *, float4 *, float4 *, float, int);
void nBody();
float3 getLinearMomentumOfMicroplastics();
float3 getAngularMomentumOfMicroplastics();
float getKineticEnergyOfMicroplastics();
void terminalPrint();
void setup();

#include "./callBackFunctions.h"

void readSimulationParameters()
{
	ifstream data;
	string name;
	
	data.open("./simulationSetup");
	
	if(data.is_open() == 1)
	{
		getline(data,name,'=');
		data >> NumberOfMicroPlastics;
		
		getline(data,name,'=');
		data >> MaxMicroPlastics1DVelocity;
		
		getline(data,name,'=');
		data >> DensityOfMicroPlasticMin;

		getline(data,name,'=');
		data >> DensityOfMicroPlasticMax;
		
		getline(data,name,'=');
		data >> DiameterOfMicroPlasticMin;

		getline(data,name,'=');
		data >> DiameterOfMicroPlasticMax;
		
		getline(data,name,'=');
		data >> NumberOfPolymerChains;
		
		getline(data,name,'=');
		data >> PolymersChainLengthMin;

		getline(data,name,'=');
		data >> PolymersChainLengthMax;
		
		getline(data,name,'=');
		data >> PolymersConnectionLength;
		
		getline(data,name,'=');
		data >> DensityOfPolymer;
		
		getline(data,name,'=');
		data >> DiameterOfPolymer;
		
		getline(data,name,'=');
		data >> BeakerRadius;

		getline(data,name,'=');
		data >> FluidHeight;

		getline(data,name,'=');
		data >> FluidDensity;

		getline(data,name,'=');
		data >> Drag;
		
		getline(data,name,'=');
		data >> TotalRunTime;
		
		getline(data,name,'=');
		data >> Dt;
		
		getline(data,name,'=');
		data >> DrawRate;
		
		getline(data,name,'=');
		data >> PrintRate;

		getline(data,name,'=');
		data >> PolymerRed;

		getline(data,name,'=');
		data >> PolymerGreen;

		getline(data,name,'=');
		data >> PolymerBlue;

		getline(data,name,'=');
		data >> MicroPlasticRed;

		getline(data,name,'=');
		data >> MicroPlasticGreen;

		getline(data,name,'=');
		data >> MicroPlasticBlue;
		
	}
	else
	{
		printf("\nTSU Error could not open simulationSetup file\n");
		exit(0);
	}
	data.close();
	
	if(DebugFlag == 1)
	{
	//prinf all the parameters
		printf("\n\n Number of MicroPlastics = %d", NumberOfMicroPlastics);
		printf("\n DensityOfMicroPlasticMin = %f", DensityOfMicroPlasticMin);
		printf("\n DensityOfMicroPlasticMax = %f", DensityOfMicroPlasticMax);
		printf("\n DiameterOfMicroPlasticMin = %f", DiameterOfMicroPlasticMin);
		printf("\n DiameterOfMicroPlasticMax = %f", DiameterOfMicroPlasticMax);
		printf("\n NumberOfPolymerChains = %d", NumberOfPolymerChains);
		printf("\n PolymersChainLengthMin = %d", PolymersChainLengthMin);
		printf("\n PolymersChainLengthMax = %d", PolymersChainLengthMax);
		printf("\n PolymersConnectionLength = %f", PolymersConnectionLength);
		printf("\n DensityOfPolymer = %f", DensityOfPolymer);
		printf("\n DiameterOfPolymer = %f", DiameterOfPolymer);
		printf("\n BeakerRadius = %f", BeakerRadius);
		printf("\n FluidHeight = %f", FluidHeight);
		printf("\n FluidDensity = %f", FluidDensity);
		printf("\n Drag = %f", Drag);
		printf("\n TotalRunTime = %f", TotalRunTime);
		printf("\n Dt = %f", Dt);
		printf("\n DrawRate = %d", DrawRate);
		printf("\n PrintRate = %d", PrintRate);
		printf("\n PolymerRed = %f", PolymerRed);
		printf("\n PolymerGreen = %f", PolymerGreen);
		printf("\n PolymerBlue = %f", PolymerBlue);
		printf("\n MicroPlasticRed = %f", MicroPlasticRed);
		printf("\n MicroPlasticGreen = %f", MicroPlasticGreen);
		printf("\n MicroPlasticBlue = %f", MicroPlasticBlue);
	}
	printf("\n Parameter file has been read.\n");
}

void setNumberOfBodies()
{
	time_t t;
	
	PolymerChainLength = (int*)malloc(NumberOfPolymerChains*sizeof(int));
	
	srand((unsigned) time(&t));
	for(int i = 0; i < NumberOfPolymerChains; i++)
	{
		PolymerChainLength[i] = ((float)rand()/(float)RAND_MAX)*(PolymersChainLengthMax - PolymersChainLengthMin) + PolymersChainLengthMin;
		//printf("\n PolymerChainLength[%d] = %d", i, PolymerChainLength[i]);	
	}
	
	NumberOfPolymers = 0;
	for(int i = 0; i < NumberOfPolymerChains; i++)
	{
		NumberOfPolymers += PolymerChainLength[i];	
	}
	
	NumberOfBodies = NumberOfMicroPlastics + NumberOfPolymers;
	
	if(DebugFlag == 1)
	{
		printf("\n\n Number of Polymers = %d", NumberOfPolymers);
		printf("\n Number of MicroPlastics = %d", NumberOfMicroPlastics);
		printf("\n Total number of bodies = %d", NumberOfBodies);
	}
	
	printf("\n Number of bodies has been set.\n");
}

void allocateMemory()
{
	Blocks.x = BLOCK;
	Blocks.y = 1;
	Blocks.z = 1;
	
	Grids.x = (NumberOfBodies - 1)/Blocks.x + 1;
	Grids.y = 1;
	Grids.z = 1;
	
	BodyPosition = (float4*)malloc(NumberOfBodies*sizeof(float4));
	BodyVelocity = (float4*)malloc(NumberOfBodies*sizeof(float4));
	BodyForce    = (float4*)malloc(NumberOfBodies*sizeof(float4));
	
	PolymerConnectionA    = (int*)malloc(NumberOfPolymers*sizeof(int));
	PolymerConnectionB    = (int*)malloc(NumberOfPolymers*sizeof(int));
	
	cudaMalloc( (void**)&BodyPositionGPU, NumberOfBodies *sizeof(float4));
	errorCheck("cudaMalloc BodyPositionGPU");
	cudaMalloc( (void**)&BodyVelocityGPU, NumberOfBodies *sizeof(float4));
	errorCheck("cudaMalloc BodyDiameterOfBodyVelocityGPU");
	cudaMalloc( (void**)&BodyForceGPU, NumberOfBodies *sizeof(float4));
	errorCheck("cudaMalloc BodyForceGPU");
	
	cudaMalloc( (void**)&PolymerConnectionAGPU, NumberOfPolymers *sizeof(int));
	errorCheck("cudaMalloc BodyForceGPU");
	cudaMalloc( (void**)&PolymerConnectionBGPU, NumberOfPolymers *sizeof(int));
	errorCheck("cudaMalloc BodyForceGPU");
	
	cudaMalloc((void**)&DevStates, NumberOfBodies * sizeof(curandState_t));

	cudaMalloc((void**)&RaysGPU, NUM_RAYS * sizeof(RayData));
	errorCheck("cudaMalloc RaysGPU");
	cudaMalloc((void**)&LaserForceGPU, NumberOfBodies * sizeof(float4));
	errorCheck("cudaMalloc LaserForceGPU");

	/*cudaMalloc((void**)&DebugHitInfoGPU, NUM_RAYS * sizeof(float4));
	errorCheck("cudaMalloc DebugHitInfoGPU");
	DebugHitInfo = (float4*)malloc(NUM_RAYS * sizeof(float4));

	cudaMalloc((void**)&DebugForceGPU, NUM_RAYS * sizeof(float4));
	errorCheck("cudaMalloc DebugForceGPU");
	DebugForce = (float4*)malloc(NUM_RAYS * sizeof(float4));*/

	int debugSize = NUM_RAYS * DEBUG_HITS_PER_RAY;
	cudaMalloc((void**)&DebugHitInfoGPU, debugSize * sizeof(float4));
	errorCheck("cudaMalloc DebugHitInfoGPU");
	DebugHitInfo = (float4*)malloc(debugSize * sizeof(float4));
	cudaMalloc((void**)&DebugForceGPU, debugSize * sizeof(float4));
	errorCheck("cudaMalloc DebugForceGPU");
	DebugForce = (float4*)malloc(debugSize * sizeof(float4));

	
	printf("\n Memory has been allocated.\n");
}

/******************************************************************************
 This function does most of the nbody work when shaking up the polymers for setup.
*******************************************************************************/
__global__ void getForcesSetup(curandState_t* states, float4 *pos, float4 *vel, float4 *force, int *linkA, int *linkB, float length, int nPolymer, int nPlastics, float beakerRadius, float fluidHeight, float fluidDensity, int stirFlag, float theta, int shakeItUpFlag, float drag)
{
	int myId, yourId;
	float4 forceVector, forceVectorSum;
	float4 posMe, posYou;
	
	myId = threadIdx.x + blockDim.x*blockIdx.x;
    	if(myId < nPolymer)
    	{
		posMe.x = pos[myId].x;
		posMe.y = pos[myId].y;
		posMe.z = pos[myId].z;
		posMe.w = pos[myId].w;
		
		forceVectorSum.x = 0.0;
		forceVectorSum.y = 0.0;
		forceVectorSum.z = 0.0;
		
		for(yourId = 0; yourId < nPolymer; yourId++)
		{	
			if(yourId != myId) // Making sure you are not working on youself.
			{
				posYou.x = pos[yourId].x;
				posYou.y = pos[yourId].y;
				posYou.z = pos[yourId].z;
				posYou.w = pos[yourId].w;
			
				if(myId < nPolymer && yourId < nPolymer)
				{
					// Polymer-polymer force
					forceVector = getPolymerPolymerForce(posMe, posYou, linkA[myId], linkB[myId], yourId, length, myId);
				}
				
			    	forceVectorSum.x += forceVector.x;
			    	forceVectorSum.y += forceVector.y;
			    	forceVectorSum.z += forceVector.z;
		    	}
		}
		
		// This adds on the forces to keep the bodies in the container.
		forceVector = getContainerForces(posMe, beakerRadius, fluidHeight);
		forceVectorSum.x += forceVector.x;
		forceVectorSum.y += forceVector.y;
		forceVectorSum.z += forceVector.z;
		
		// This adds on the fliud drag force.
		forceVector = getDragForces(pos[myId], vel[myId]);
		forceVectorSum.x += forceVector.x;
		forceVectorSum.y += forceVector.y;
		forceVectorSum.z += forceVector.z;
		
		// Tranfering all the forces to my force function
		force[myId].x = forceVectorSum.x;
		force[myId].y = forceVectorSum.y;
		force[myId].z = forceVectorSum.z;
    	}
}

void polymerShakeUp(float4 *pos, float4 *vel, float4 *force, int *linkA, int *linkB, float length, int n, float drag, float dt, float beakerRadius, float fluidHeight)
{
	float magx, magy, magz, mag;
	float dragTemp = drag;
	
	mag = 100000.0;//before was 10
	float stopTime = 2.0;
	float time = 0;
	DrawTimer = 0;
	
	cudaMemcpy( PolymerConnectionAGPU, PolymerConnectionA, NumberOfPolymers*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( PolymerConnectionBGPU, PolymerConnectionB, NumberOfPolymers*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( BodyPositionGPU, BodyPosition, NumberOfBodies*sizeof(float4), cudaMemcpyHostToDevice );
	cudaMemcpy( BodyVelocityGPU, BodyVelocity, NumberOfBodies*sizeof(float4), cudaMemcpyHostToDevice );
	cudaMemcpy( BodyForceGPU, BodyForce, NumberOfBodies*sizeof(float4), cudaMemcpyHostToDevice );
		
	while(time < stopTime)
	{
		for(int i = 0; i < n; i++)
		{
			force[i].x = 0.0;
			force[i].y = 0.0;
			force[i].z = 0.0;
		}
		
		for(int i = 0; i < n; i++)
		{
			mag = 1000000.0; //before was 10
			if(time < 1.0)
			{
				dragTemp = 0.001;
				if(linkA[i] == -1) 
				{
					force[i].y -= 500.0;
					force[i].x -= 10.0;
				}
				if(linkB[i] == -1) 
				{
					force[i].y += 500.0;
					force[i].x += 10.0;
				}
				
				magx = mag*((float)rand()/RAND_MAX*2.0 - 1.0);
				magy = mag*((float)rand()/RAND_MAX*2.0 - 1.0);
				magz = mag*((float)rand()/RAND_MAX*2.0 - 1.0);
				
				force[i].x += magx;
				force[i].y += magy;
				force[i].z += magz;
			}
			else if(time < 1.5)
			{
				dragTemp = 0.01;
			}
			else
			{

				dragTemp = drag;

			}	

		}
		
		cudaMemcpy( BodyForceGPU, BodyForce, NumberOfBodies*sizeof(float4), cudaMemcpyHostToDevice );
		getForcesSetup<<<Grids, Blocks>>>(DevStates, BodyPositionGPU, BodyVelocityGPU, BodyForceGPU, PolymerConnectionAGPU, PolymerConnectionBGPU, PolymersConnectionLength, NumberOfPolymers, NumberOfMicroPlastics, BeakerRadius, FluidHeight, FluidDensity, StirFlag, Theta, ShakeItUpFlag, dragTemp);
		errorCheck("getForcesSetup");
		
		moveBodies<<<Grids, Blocks>>>(BodyPositionGPU, BodyVelocityGPU, BodyForceGPU, Dt, NumberOfBodies);
		errorCheck("moveBodies");
		cudaMemcpy( BodyForce, BodyForceGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost );
			
		time += dt;
	}
	
	// Taking the shaken up polymers back to the GPU so that the microplastics can be added on.
	cudaMemcpy( BodyPosition, BodyPositionGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost );
	cudaMemcpy( BodyVelocity, BodyVelocityGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost );
	cudaMemcpy( BodyForce, BodyForceGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost );

	
	printf("\n Polymers have been shoken up.\n");
}

void setInitailConditions()
{
	time_t t;
	srand((unsigned) time(&t));
	double density;
	double angle;
	double dx, dy, dz, d2, d;
	int k;
	int index;
	int test;
	double TotalPolymerLength;
	double spaceBetweenPolymerCenters;
	double startX,startY,startZ;


	// Zeroing out everything just for safety
	for(int i = 0; i < NumberOfBodies; i++)
	{
		BodyPosition[i].x = 0.0;
		BodyPosition[i].y = 0.0;
		BodyPosition[i].z = 0.0;
		BodyPosition[i].w = 0.0;
		
		BodyVelocity[i].x = 0.0;
		BodyVelocity[i].y = 0.0;
		BodyVelocity[i].z = 0.0;
		BodyVelocity[i].w = 0.0;
		
		BodyForce[i].x = 0.0;
		BodyForce[i].y = 0.0;
		BodyForce[i].z = 0.0;
		BodyForce[i].w = 0.0;
	}
	
	// Loading velocity, diameter, density, and mass of polymers
	for(int i = 0; i < NumberOfPolymers; i++)
	{
		// Setting diameter
		BodyPosition[i].w = DiameterOfPolymer;	
		
		// Setting density
		BodyVelocity[i].w = DensityOfPolymer;
		
		// Setting mass
		BodyForce[i].w = DensityOfPolymer*(4.0/3.0)*PI*(BodyPosition[i].w/2.0)*(BodyPosition[i].w/2.0)*(BodyPosition[i].w/2.0);
	}
	
	// Setting, diameter, density, and mass of microplastics
	for(int i = NumberOfPolymers; i < NumberOfBodies; i++)
	{
		// Setting diameter
		BodyPosition[i].w = ((double)rand()/(double)RAND_MAX)*(DiameterOfMicroPlasticMax - DiameterOfMicroPlasticMin) + DiameterOfMicroPlasticMin;
		
		// Setting density
		density = ((double)rand()/(double)RAND_MAX)*(DensityOfMicroPlasticMax - DensityOfMicroPlasticMin) + DensityOfMicroPlasticMin;
		BodyVelocity[i].w = density;
		
		// Setting mass
		BodyForce[i].w = density*(4.0/3.0)*PI*(BodyPosition[i].w/2.0)*(BodyPosition[i].w/2.0)*(BodyPosition[i].w/2.0);	
	}

		// Assign random colors to each microplastic - not necessary just makes it more fun to look at. You can take it out if you want them all to be the same color again.
	MicroPlasticColors = (float3*)malloc(NumberOfMicroPlastics * sizeof(float3));
	for(int i = 0; i < NumberOfMicroPlastics; i++)
	{
		MicroPlasticColors[i].x = (float)rand() / (float)RAND_MAX; // Red
		MicroPlasticColors[i].y = (float)rand() / (float)RAND_MAX; // Green
		MicroPlasticColors[i].z = (float)rand() / (float)RAND_MAX; // Blue
	}


	// Setting intial positions of polymers
	spaceBetweenPolymerCenters = PolymersConnectionLength+DiameterOfPolymer;
	k = 0;
	for(int i = 0; i < NumberOfPolymerChains; i++)
	{
		test = 0;
		while(test == 0)
		{
			angle = 2.0*PI*(double)rand()/(double)RAND_MAX;
			BodyPosition[k].x = ((double)rand()/(double)RAND_MAX)*BeakerRadius * cos(angle);
			BodyPosition[k].z = ((double)rand()/(double)RAND_MAX)*BeakerRadius * sin(angle);
			
			test = 1;
			index = 0;
			for(int j = 0; j < i; j++)
			{
				// Checking against the leading element of the polymer chain.
				dx = BodyPosition[k].x - BodyPosition[index].x;
				dz = BodyPosition[k].z - BodyPosition[index].z;
				d2  = dx*dx + dz*dz;
				d = sqrt(d2); 
				if(d < spaceBetweenPolymerCenters)
				{
					test = 0;
				}
				index += PolymerChainLength[j];
			}
		}
		
		TotalPolymerLength = spaceBetweenPolymerCenters * (double)PolymerChainLength[i];
		BodyPosition[k].y = ((double)rand()/(double)RAND_MAX)*(FluidHeight - TotalPolymerLength) +  TotalPolymerLength;
		
		startX = BodyPosition[k].x;
		startY = BodyPosition[k].y;
		startZ = BodyPosition[k].z;
		
		PolymerConnectionA[k] = -1;
		PolymerConnectionB[k] = -1;
		k++;
		
		for(int j = 1; j < PolymerChainLength[i]; j++)
		{
			PolymerConnectionB[k-1] = k;
			PolymerConnectionA[k] = k-1;
			PolymerConnectionB[k] = -1;
			BodyPosition[k].x = startX;
			BodyPosition[k].y = startY - j*spaceBetweenPolymerCenters;
			BodyPosition[k].z = startZ;
			k++;
		}
	}
	
	if(DebugFlag == 1)
	{
		// Printing our polymer chains for debuging.
		k = 0;
		for(int i = 0; i < NumberOfPolymerChains; i++)
		{
			printf("\n ******************* polymer chain %d **********************\n", i);
			for(int j = 0; j < PolymerChainLength[i]; j++)
			{
				printf("PolymerPosition[%d] = (%f, %f, %f) linkA = %d linkB = %d \n", k, BodyPosition[k].x, BodyPosition[k].y, BodyPosition[k].z, PolymerConnectionA[k], PolymerConnectionB[k]);
				printf("PolymerVelocity[%d] = (%f, %f, %f) \n", k, BodyVelocity[k].x, BodyVelocity[k].y, BodyVelocity[k].z);
				k++;
			}
		}
	}
	
	// Shaking the polymers out of their unnatural intial positions.
	polymerShakeUp(BodyPosition, BodyVelocity, BodyForce, PolymerConnectionA, PolymerConnectionB, PolymersConnectionLength, NumberOfPolymers, Drag, Dt, BeakerRadius, FluidHeight);
	
	// Setting intial positions and velocities of the micro plastics
	for(int i = NumberOfPolymers; i < NumberOfBodies; i++)
	{
		BodyVelocity[i].x = ((double)rand()/(double)RAND_MAX)*2.0*MaxMicroPlastics1DVelocity - MaxMicroPlastics1DVelocity;
		BodyVelocity[i].y = ((double)rand()/(double)RAND_MAX)*2.0*MaxMicroPlastics1DVelocity - MaxMicroPlastics1DVelocity;
		BodyVelocity[i].z = ((double)rand()/(double)RAND_MAX)*2.0*MaxMicroPlastics1DVelocity - MaxMicroPlastics1DVelocity;
		
		test = 0;
		while(test == 0)
		{
			angle = 2.0*PI*(double)rand()/(double)RAND_MAX;
			BodyPosition[i].x = ((double)rand()/(double)RAND_MAX)*(BeakerRadius * cos(angle));
			BodyPosition[i].y = ((double)rand()/(double)RAND_MAX)*(FluidHeight);
			BodyPosition[i].z = ((double)rand()/(double)RAND_MAX)*(BeakerRadius * sin(angle));
			
			test = 1;
			for(int j = 0; j < i; j++)
			{
				dx = BodyPosition[i].x - BodyPosition[j].x;
				dy = BodyPosition[i].y - BodyPosition[j].y;
				dz = BodyPosition[i].z - BodyPosition[j].z;
				d2  = dx*dx + dy*dy + dz*dz;
				d = sqrt(d2); 
				
				if(d < BodyPosition[i].w + BodyPosition[j].w)
				{
					test = 0;
				}
			}
		}
	}
	
	if(DebugFlag == 1)
	{
		// Printing micro plastics for debugging.
		printf("\n ****************************************** \n");
		for(int i = NumberOfPolymers; i < NumberOfBodies; i++)
		{
			printf(" MicrPlasticPosition[%d] = (%f, %f, %f) \n", i, BodyPosition[i].x, BodyPosition[i].y, BodyPosition[i].z);
		}
	}
		
	printf("\n Initial conditions have been set.\n");
}


void drawPicture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	
	// Drawing Polymers
	for(int i = 0; i < NumberOfPolymers; i++)
	{
		glColor3d(PolymerRed, PolymerGreen, PolymerBlue);
		glPushMatrix();
			glTranslatef(BodyPosition[i].x, BodyPosition[i].y, BodyPosition[i].z);
			glutSolidSphere(BodyPosition[i].w/2.0, 30, 30);
		glPopMatrix();
		
		// Drawing polymer connections.
		// Note: there is no need to draw both. If you draw just the one above or below all 
		// connections will be drawn.
		glLineWidth(3.0);
		glColor3d(1.0, 0.0, 0.0);
		glBegin(GL_LINES);
			if(PolymerConnectionA[i] != -1)
			{
				glVertex3f(BodyPosition[i].x, BodyPosition[i].y, BodyPosition[i].z);
				glVertex3f(BodyPosition[PolymerConnectionA[i]].x, BodyPosition[PolymerConnectionA[i]].y, BodyPosition[PolymerConnectionA[i]].z);;
			}
		glEnd();
		
		if(DebugFlag == 1)
		{
			if(PolymerConnectionA[i] == -1)
			{
				glColor3d(0.0, 0.0, 1.0);
				glPushMatrix();
					glTranslatef(BodyPosition[i].x, BodyPosition[i].y, BodyPosition[i].z);
					glutSolidSphere(2.0*BodyPosition[i].w/2.0, 30, 30);
				glPopMatrix();
			}
			if(PolymerConnectionB[i] == -1)
			{
				glColor3d(0.0, 1.0, 1.0);
				glPushMatrix();
					glTranslatef(BodyPosition[i].x, BodyPosition[i].y, BodyPosition[i].z);
					glutSolidSphere(2.0*BodyPosition[i].w/2.0, 30, 30);
				glPopMatrix();
			}
		}
	}
	
	/*
	// Drawing Microplastics- white
	for(int i = NumberOfPolymers; i < NumberOfBodies; i++)
	{
		glColor3d(MicroPlasticRed, MicroPlasticGreen, MicroPlasticBlue);
		glPushMatrix();
			glTranslatef(BodyPosition[i].x, BodyPosition[i].y, BodyPosition[i].z);
			glutSolidSphere(BodyPosition[i].w/2.0, 30, 30);
		glPopMatrix();
	}
	*/

	// Drawing Microplastics with random colors- just for fun. You can take out the random colors and make them all the same color again if you want.
	for(int i = NumberOfPolymers; i < NumberOfBodies; i++)
	{
		int colorIndex = i - NumberOfPolymers;
		glColor3d(MicroPlasticColors[colorIndex].x, MicroPlasticColors[colorIndex].y, MicroPlasticColors[colorIndex].z);
		glPushMatrix();
			glTranslatef(BodyPosition[i].x, BodyPosition[i].y, BodyPosition[i].z);
			glutSolidSphere(BodyPosition[i].w/2.0, 30, 30);
		glPopMatrix();
	}

	// Drawint a red sphere at the origin for reference.
	glColor3d(1.0, 0.0, 0.0);
	glPushMatrix();
		glTranslatef(0, 0, 0);
		glutSolidSphere(DiameterOfMicroPlasticMax, 30, 30);
	glPopMatrix();
	
	
	// Drawing the outline of the Beaker.
	if(RadialConfinementViewingAids == 1)
	{
		glLineWidth(1.0);
		float divitions = 60.0;
		float angle = 2.0*PI/divitions;
		
		// Drawing top ring.
		glColor3d(1.0,1.0,0.0);
		for(int i = 0; i < divitions; i++)
		{
			glBegin(GL_LINES);
				glVertex3f(sin(angle*i)*BeakerRadius, FluidHeight, cos(angle*i)*BeakerRadius);
				glVertex3f(sin(angle*(i+1))*BeakerRadius, FluidHeight, cos(angle*(i+1))*BeakerRadius);
			glEnd();
		}
		
		// Drawing lines down the sides of the beaker.
		glColor3d(0.0,1.0,0.0);
		for(int i = 0; i < divitions; i++)
		{
			glBegin(GL_LINES);
				glVertex3f(sin(angle*i)*BeakerRadius, 0.0, cos(angle*i)*BeakerRadius);
				glVertex3f(sin(angle*(i))*BeakerRadius, FluidHeight, cos(angle*(i))*BeakerRadius);
			glEnd();
		}
		
		// Drawing the bottom ring.
		glColor3d(0.0,0.0,1.0);
		for(int i = 0; i < divitions; i++)
		{
			glBegin(GL_LINES);
				glVertex3f(sin(angle*i)*BeakerRadius, 0.0, cos(angle*i)*BeakerRadius);
				glVertex3f(sin(angle*(i+1))*BeakerRadius, 0.0, cos(angle*(i+1))*BeakerRadius);
			glEnd();
		}
	}
	
	// Drawing the stirring.
	if(StirFlag == 1)
	{
		glLineWidth(6.0);
		glColor3d(0.0,1.0,1.0);
		glBegin(GL_LINES);
			glVertex3f(0.0, 0.0, 0.0);
			glVertex3f(BeakerRadius*cos(Theta), 0.0, BeakerRadius*sin(Theta));
		glEnd();
	}
	
	// Draw laser rays if laser is on
	if(LaserFlag == 1)
	{
		static RayData *raysHost = NULL;
		if(raysHost == NULL) raysHost = (RayData*)malloc(NUM_RAYS * sizeof(RayData));
		cudaError_t err = cudaMemcpy(raysHost, RaysGPU,
                             NUM_RAYS * sizeof(RayData),
                             cudaMemcpyDeviceToHost);

		if(err != cudaSuccess)
		{
			printf("Ray copy failed: %s\n", cudaGetErrorString(err));
		}

		glDisable(GL_LIGHTING);
		glLineWidth(1.0f);
		glColor3f(1.0f, 0.2f, 0.2f);
		glBegin(GL_LINES);
		for(int i = 0; i < NUM_RAYS; i += 200)
		{
			if(i == FirstHitRayIndex) continue;
			// Draw the ray from its origin (above the beaker) all the way down through the fluid.
			// This is purely visual — physics is unaffected.
			float t = FluidHeight * 4.0f;   // long enough to pass through the entire beaker
			glVertex3f(raysHost[i].origin.x, raysHost[i].origin.y, raysHost[i].origin.z);
			glVertex3f(raysHost[i].origin.x + raysHost[i].direction.x * t,
				 raysHost[i].origin.y + raysHost[i].direction.y * t, 
				 raysHost[i].origin.z + raysHost[i].direction.z * t);
		}
		glEnd();

		// === NEW MAGENTA HIT-RAY BLOCK GOES HERE ===
		if(FirstHitRayIndex >= 0 && FirstHitRayIndex < NUM_RAYS)
		{
			// Bright magenta thick line for the hit ray - very obvious [4]
			glLineWidth(8.0f);
			glColor3f(1.0f, 0.0f, 1.0f);  // bright magenta
			glBegin(GL_LINES);

			// Draw the ray going its full length through the scene
			float t = FluidHeight * 4.0f;   // long enough to pass through the entire beaker
			glVertex3f(raysHost[FirstHitRayIndex].origin.x,
					raysHost[FirstHitRayIndex].origin.y,
					raysHost[FirstHitRayIndex].origin.z);
			glVertex3f(raysHost[FirstHitRayIndex].origin.x + raysHost[FirstHitRayIndex].direction.x * t,
					raysHost[FirstHitRayIndex].origin.y + raysHost[FirstHitRayIndex].direction.y * t,
					raysHost[FirstHitRayIndex].origin.z + raysHost[FirstHitRayIndex].direction.z * t);
			glEnd();

			// Draw a bright yellow sphere at the first hit point to mark it
			glEnable(GL_LIGHTING);
			glColor3f(1.0f, 1.0f, 0.0f);  // bright yellow
			glPushMatrix();
				glTranslatef(FirstHitX, FirstHitY, FirstHitZ);
				glutSolidSphere(BodyPosition[FirstHitParticle].w * 0.3, 20, 20);
			glPopMatrix();
			glDisable(GL_LIGHTING);
		}
		// === END NEW BLOCK ===

		glEnable(GL_LIGHTING);
	}

	glutSwapBuffers();

	// Captures frames if you are making a movie.
	if(MovieFlag == 1)
	{
		glReadPixels(5, 5, XWindowSize, YWindowSize, GL_RGBA, GL_UNSIGNED_BYTE, Buffer);
		fwrite(Buffer, sizeof(int)*XWindowSize*YWindowSize, 1, MovieFile);
	}
}

void errorCheck(const char *message)
{
	cudaError_t  error;
	error = cudaGetLastError();

	if(error != cudaSuccess)
	{
		printf("\n CUDA ERROR: %s = %s\n", message, cudaGetErrorString(error));
		exit(0);
	}
}

/******************************************************************************
 This function initializes CUDA Rand making it so every thread can have its own set
 of random numbers.
*******************************************************************************/
__global__ void init_curand(unsigned int seed, curandState_t* states) 
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, idx, 0, &states[idx]);
}

/******************************************************************************
 This function controlls all the forces that act on the bodies.
*******************************************************************************/
__global__ void getForces(curandState_t* states, float4 *pos, float4 *vel, float4 *force, int *linkA, int *linkB, float length, int nPolymer, int nPlastics, float beakerRadius, float fluidHeight, float fluidDensity, int stirFlag, float theta, int shakeItUpFlag, float ShakeItUpMag, int dragFlag, int laserFlag, int gravityFlag, int brownianFlag, int containerFlag, int particleInteractionFlag, float dt, float4 *laserForce)
{
	int myId, yourId;
	int nBodies;
	float4 forceVector, forceVectorSum;
	float4 velocityVector;
	float4 posMe, posYou;
	float4 velMe;
	float densityMe, massMe;
	
	nBodies = nPolymer + nPlastics;
	myId = threadIdx.x + blockDim.x*blockIdx.x;
    	if(myId < nBodies)
    	{
		posMe.x = pos[myId].x;
		posMe.y = pos[myId].y;
		posMe.z = pos[myId].z;
		posMe.w = pos[myId].w;
		
		velMe.x = vel[myId].x;
		velMe.y = vel[myId].y;
		velMe.z = vel[myId].z;
		velMe.w = vel[myId].w;
		
		//diameterMe = posMe.w;
		densityMe = vel[myId].w;
		massMe = pos[myId].w;   // TEMP FIX (better than force[].w)
		
		forceVectorSum.x = 0.0;
		forceVectorSum.y = 0.0;
		forceVectorSum.z = 0.0;
		
	// ===WRAPPED WITH PARTICLE INTERACTION FLAG===
	if(particleInteractionFlag == 1)
	{
		for(yourId = 0; yourId < nBodies; yourId++)
		{
			posYou.x = pos[yourId].x;
			posYou.y = pos[yourId].y;
			posYou.z = pos[yourId].z;
			posYou.w = pos[yourId].w;
			
			if(yourId != myId) // Making sure you are not working on youself.
			{
				if(myId < nPolymer)
				{
					if(yourId < nPolymer)
					{
						// Polymer-polymer force
						forceVector = getPolymerPolymerForce(posMe, posYou, linkA[myId], linkB[myId], yourId, length, myId);
					}
					else
					{
						// Polymer-microPlastic force
						forceVector = getPolymerMicroPlasticForce(posMe, posYou);
					}
				}
				else
				{
					if(yourId < nPolymer)
					{
						// Polymer-microPlastic force
						forceVector = getPolymerMicroPlasticForce(posMe, posYou);
					}
					else
					{
						// microPlastic-microPlastic force
						forceVector = getMicroPlasticMicroPlasticForce(posMe, posYou);
					}
				}
				
			    	forceVectorSum.x += forceVector.x;
			    	forceVectorSum.y += forceVector.y;
			    	forceVectorSum.z += forceVector.z;
		    	}
		}
	}
	//==END OF PARTICLE INTERACTION WRAP===
		
		// This adds on a gravity pull based on density
		if(gravityFlag == 1)
		{
			forceVector = getGravityForces(densityMe, massMe, fluidDensity);
			forceVectorSum.x += forceVector.x;
			forceVectorSum.y += forceVector.y;
			forceVectorSum.z += forceVector.z;
		}
		
		//===WRAPPED WITH CONTAINER FLAG===
		if(containerFlag == 1)
		{
		// This adds on the forces to keep the bodies in the container.
		forceVector = getContainerForces(posMe, beakerRadius, fluidHeight);
		forceVectorSum.x += forceVector.x;
		forceVectorSum.y += forceVector.y;
		forceVectorSum.z += forceVector.z;
		}
		//===END OF CONTAINER FLAG WRAP===
		
		// This adds on the forces caused by stirring.
		if(stirFlag == 1)
		{
			forceVector = getStirringForces(states, myId, posMe, velMe, beakerRadius, fluidHeight, theta);
			forceVectorSum.x += forceVector.x;
			forceVectorSum.y += forceVector.y;
			forceVectorSum.z += forceVector.z;
		}
		
		if(brownianFlag == 1)
		{
			forceVector = brownian_motion(states, myId, posMe, dt);
			forceVectorSum.x += forceVector.x;
			forceVectorSum.y += forceVector.y;
			forceVectorSum.z += forceVector.z;
		}
		
		// This just adds random motion to the system.
		if(shakeItUpFlag == 1)
		{
			velocityVector = shakeItUp(states, myId, ShakeItUpMag);
			vel[myId].x += velocityVector.x;
			vel[myId].y += velocityVector.y;
			vel[myId].z += velocityVector.z;
		}
		
		/*// This is the force from fliud drag.
		if(dragFlag == 1)
		{
			forceVector = getDragForces(posMe, velMe);
			forceVectorSum.x += forceVector.x;
			forceVectorSum.y += forceVector.y;
			forceVectorSum.z += forceVector.z;
		}
			*/
		
		// This is the force from the laser.
		if(laserFlag == 1)
		{
			forceVector = getLaserForces(laserForce[myId]);
			forceVectorSum.x += forceVector.x;
			forceVectorSum.y += forceVector.y;
			forceVectorSum.z += forceVector.z;
		}
		
		// Tranfering all the forces to my force function
		force[myId].x = forceVectorSum.x;
		force[myId].y = forceVectorSum.y;
		force[myId].z = forceVectorSum.z;
    	}
}

/******************************************************************************
 This function moves the bodies.
*******************************************************************************/
__global__ void moveBodies(float4 *pos, float4 *vel, float4 *force, float dt, int n)
{
    int id = threadIdx.x + blockDim.x * blockIdx.x;

    if(id < n)
    {
        float mass = force[id].w;
		if(mass < 1e-20f)
		{
			vel[id].x = 0.0f;
			vel[id].y = 0.0f;
			vel[id].z = 0.0f;
			return;
		}
        // Radius and drag coefficient (Stokes)
        float R = pos[id].w * 0.5f;

        const float eta = 1.0e3f;
        const float drag_scale = 0.01f;

        float gamma = 6.0f * 3.141592654f * eta * R * drag_scale;

        // acceleration
        float ax = force[id].x / mass;
        float ay = force[id].y / mass;
        float az = force[id].z / mass;

        // semi-implicit stable update
        float invDenom = 1.0f / (1.0f + gamma * dt / mass);

        vel[id].x = (vel[id].x + ax * dt) * invDenom;
        vel[id].y = (vel[id].y + ay * dt) * invDenom;
        vel[id].z = (vel[id].z + az * dt) * invDenom;

        // NaN safety
        if(!isfinite(vel[id].x)) vel[id].x = 0.0f;
        if(!isfinite(vel[id].y)) vel[id].y = 0.0f;
        if(!isfinite(vel[id].z)) vel[id].z = 0.0f;

        // update position
        pos[id].x += vel[id].x * dt;
        pos[id].y += vel[id].y * dt;
        pos[id].z += vel[id].z * dt;

        // NaN safety
        if(!isfinite(pos[id].x)) pos[id].x = 0.0f;
        if(!isfinite(pos[id].y)) pos[id].y = 0.0f;
        if(!isfinite(pos[id].z)) pos[id].z = 0.0f;
    }
}
void nBody()
{
    if(Pause != 1)
    {
        if(LaserFlag == 1)
        {
			int debugSize = NUM_RAYS * DEBUG_HITS_PER_RAY;

            dim3 rayGrid((NUM_RAYS - 1)/BLOCK + 1, 1, 1);
            dim3 bodyGrid((NumberOfBodies - 1)/BLOCK + 1, 1, 1);

            zeroLaserForceKernel<<<bodyGrid, Blocks>>>(LaserForceGPU, NumberOfBodies);
            errorCheck("zeroLaserForceKernel");

            initRaysKernel<<<rayGrid, Blocks>>>(RaysGPU, DevStates, LaserBeamRadius, LaserBeamCenterY);
            errorCheck("initRaysKernel");

            // Zero the debug arrays so old hits don't linger
            cudaMemset(DebugHitInfoGPU, 0, debugSize * sizeof(float4));
            cudaMemset(DebugForceGPU, 0, debugSize * sizeof(float4));

            traceRaysKernel<<<rayGrid, Blocks>>>(RaysGPU, BodyPositionGPU, LaserForceGPU,
                                                  DebugHitInfoGPU, DebugForceGPU,
                                                  NumberOfPolymers, NumberOfBodies);
            errorCheck("traceRaysKernel");

            // If in debug mode, find first hit ray and freeze
            if(LaserDebugMode == 1 && FirstHitRayIndex < 0)
            {
                cudaMemcpy(DebugHitInfo, DebugHitInfoGPU, debugSize * sizeof(float4), cudaMemcpyDeviceToHost);
                cudaMemcpy(DebugForce, DebugForceGPU, debugSize * sizeof(float4), cudaMemcpyDeviceToHost);

                // Find the first ray that hit something and print ALL its bounces
int foundRay = -1;
for(int i = 0; i < NUM_RAYS; i++)
{
    int slot0 = i * DEBUG_HITS_PER_RAY;
    if(DebugForce[slot0].x != 0.0f || DebugForce[slot0].y != 0.0f || DebugForce[slot0].z != 0.0f)
    {
        foundRay = i;
        break;
    }
}

				if(foundRay >= 0)
				{
					FirstHitRayIndex = foundRay;
					cudaMemcpy(BodyPosition, BodyPositionGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost);

					printf("\n\033[0;36m");
					printf("==========================================================\n");
					printf("  RAY %d - TRACING ALL HITS (shadowing) [4]\n", foundRay);
					printf("==========================================================\n");
					printf("\033[0m");

					int hitCount = 0;
					for(int b = 0; b < DEBUG_HITS_PER_RAY; b++)
					{
						int slot = foundRay * DEBUG_HITS_PER_RAY + b;
						if(DebugForce[slot].x == 0.0f && DebugForce[slot].y == 0.0f && DebugForce[slot].z == 0.0f)
							break;

						hitCount++;
						int particleHit = (int)DebugHitInfo[slot].x;
						float thI = DebugHitInfo[slot].y;
						float thR = DebugHitInfo[slot].z;
						float Fs = DebugHitInfo[slot].w;
						float Fg = DebugForce[slot].w;

						printf("\n\033[0;33m  --- HIT #%d (bounce %d) ---\033[0m\n", hitCount, b);
						printf("  Hit particle index:  %d\n", particleHit);
						if(particleHit >= 0 && particleHit < NumberOfBodies)
						{
							printf("  Particle position:   (%.3f, %.3f, %.3f)\n",
								BodyPosition[particleHit].x,
								BodyPosition[particleHit].y,
								BodyPosition[particleHit].z);
						}
						printf("  Angle of incidence:  %.4f rad  (%.2f deg)\n", thI, thI*180.0f/PI);
						printf("  Angle of refraction: %.4f rad  (%.2f deg)\n", thR, thR*180.0f/PI);
						printf("  Fs (parallel to beam):      %.4e N\n", Fs);
						printf("  Fg (perpendicular to beam): %.4e N\n", Fg);
						printf("  Total force on particle:\n");
						printf("    Fx = %.4e\n", DebugForce[slot].x);
						printf("    Fy = %.4e  (parallel to beam)\n", DebugForce[slot].y);
						printf("    Fz = %.4e\n", DebugForce[slot].z);
					}

					// Save first hit point for drawing
					int slot0 = foundRay * DEBUG_HITS_PER_RAY;
					FirstHitParticle = (int)DebugHitInfo[slot0].x;
					if(FirstHitParticle >= 0 && FirstHitParticle < NumberOfBodies)
					{
						FirstHitX = BodyPosition[FirstHitParticle].x;
						FirstHitY = BodyPosition[FirstHitParticle].y + BodyPosition[FirstHitParticle].w / 2.0f;
						FirstHitZ = BodyPosition[FirstHitParticle].z;
					}

					printf("\n==========================================================\n");
					printf("  Total hits on this ray: %d\n", hitCount);
					printf("  Simulation frozen. Press 'r' to resume, 'L' to re-arm.\n\n");

					cudaDeviceSynchronize();
					Pause = 1;
					LaserDebugMode = 0;
				}
            }
        }

        getForces<<<Grids, Blocks>>>(DevStates, BodyPositionGPU, BodyVelocityGPU, BodyForceGPU,
                                      PolymerConnectionAGPU, PolymerConnectionBGPU,
                                      PolymersConnectionLength, NumberOfPolymers, NumberOfMicroPlastics,
                                      BeakerRadius, FluidHeight, FluidDensity, StirFlag, Theta,
                                      ShakeItUpFlag, ShakeItUpMag, DragFlag, LaserFlag,
                                      GravityFlag, BrownianFlag, ContainerFlag,
                                      ParticleInteractionFlag, Dt, LaserForceGPU);
        errorCheck("getForces");

        moveBodies<<<Grids, Blocks>>>(BodyPositionGPU, BodyVelocityGPU, BodyForceGPU, Dt, NumberOfBodies);
        errorCheck("moveBodies");

        DrawTimer++;
        if(DrawTimer == DrawRate)
        {
            cudaMemcpy(BodyPosition, BodyPositionGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost);
            cudaMemcpy(BodyVelocity, BodyVelocityGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost);
            cudaMemcpy(BodyForce, BodyForceGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost);
            drawPicture();

            updateMomentumHistograms(BodyVelocity, BodyForce, NumberOfPolymers, NumberOfBodies);
            glutSetWindow(HistogramWindow);
            glutPostRedisplay();
            glutSetWindow(Window);
            DrawTimer = 0;
        }

        PrintTimer++;
        if(PrintRate <= PrintTimer)
        {
            cudaMemcpy(BodyPosition, BodyPositionGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost);
            cudaMemcpy(BodyVelocity, BodyVelocityGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost);
            cudaMemcpy(BodyForce, BodyForceGPU, NumberOfBodies*sizeof(float4), cudaMemcpyDeviceToHost);
            terminalPrint();
            PrintTimer = 0;
        }

        RunTime += Dt;
        if(TotalRunTime < RunTime)
        {
            printf("\n\n Done\n");
            exit(0);
        }

        Theta += StirAngularVelosity*Dt;
        if(2.0*PI < Theta) Theta = 0.0;
    }
}

float3 getLinearMomentumOfMicroplastics()
{
	float3 linearMomentum;
	linearMomentum.x = 0.0;
	linearMomentum.y = 0.0;
	linearMomentum.z = 0.0;
		
	for(int i = NumberOfPolymers; i < NumberOfBodies; i++)
	{
		linearMomentum.x += BodyForce[i].w*BodyVelocity[i].x;
		linearMomentum.y += BodyForce[i].w*BodyVelocity[i].y;
		linearMomentum.z += BodyForce[i].w*BodyVelocity[i].z;
	}
	return(linearMomentum);
}

float3 getAngularMomentumOfMicroplastics()
{
	float3 angularMomentum;
	angularMomentum.x = 0.0;
	angularMomentum.y = 0.0;
	angularMomentum.z = 0.0;
		
	for(int i = NumberOfPolymers; i < NumberOfBodies; i++)
	{
		angularMomentum.x += BodyForce[i].w*(BodyPosition[i].y*BodyVelocity[i].z - BodyPosition[i].z*BodyVelocity[i].y);
		angularMomentum.y += BodyForce[i].w*(BodyPosition[i].x*BodyVelocity[i].z - BodyPosition[i].z*BodyVelocity[i].x);
		angularMomentum.z += BodyForce[i].w*(BodyPosition[i].x*BodyVelocity[i].y - BodyPosition[i].y*BodyVelocity[i].x);
	}
	return(angularMomentum);
}

float getKineticEnergyOfMicroplastics()
{
	float kineticEnergy = 0.0;
		
	for(int i = NumberOfPolymers; i < NumberOfBodies; i++)
	{
		kineticEnergy += 0.5*BodyForce[i].w*(BodyVelocity[i].x*BodyVelocity[i].x + BodyVelocity[i].y*BodyVelocity[i].y + BodyVelocity[i].z*BodyVelocity[i].z);
	}
	return(kineticEnergy);
}

void terminalPrint()
{
	float3 linearMomentum = getLinearMomentumOfMicroplastics();
	float3 angularMomentum = getAngularMomentumOfMicroplastics();
	double totalLinearMomentum = sqrt(linearMomentum.x*linearMomentum.x + linearMomentum.y*linearMomentum.y + linearMomentum.z*linearMomentum.z);
	double totalAngularMomentum = sqrt(angularMomentum.x*angularMomentum.x + angularMomentum.y*angularMomentum.y + angularMomentum.z*angularMomentum.z);
	float KineticEnergy = getKineticEnergyOfMicroplastics();
	system("clear");
	//printf("\033[0;34m"); // blue.
	//printf("\033[0;36m"); // cyan
	//printf("\033[0;33m"); // yellow
	//printf("\033[0;31m"); // red
	//printf("\033[0;32m"); // green
	printf("\033[0m"); // back to white.
	
	printf("\n");
	printf("\033[0;33m");
	printf("\n **************************** Simulation Stats ****************************");
	printf("\033[0m");
	
	printf("\n Total run time = %7.2f milliseconds", RunTime);
	printf("\n Total Linear Momentum of Microplatics  = %f", totalLinearMomentum);
	printf("\n Total Angular Momentum of Microplatics = %f", totalAngularMomentum);
	printf("\n Kinetic Energy of Microplatics = %f", KineticEnergy);
	
	printf("\033[0;33m");
	printf("\n **************************** Terminal Comands ****************************");
	printf("\033[0m");
	//printf("\n h: Help");
	//printf("\n c: Recenter View");
	printf("\n q: Quit");
	printf("\n m: Screenshot");
	//printf("\n k: Save Current Run");
	printf("\n");
	
	printf("\n Toggles");
	printf("\n r: Run/Pause            - ");
	if(Pause == 0) 
	{
		printf("\033[0;32m");
		printf(BOLD_ON "Simulation Running" BOLD_OFF);
	} 
	else
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Simulation Paused" BOLD_OFF);
	}
	printf("\n v: Orthogonal/Frustum   - ");
	if (ViewFlag == 0) 
	{
		printf("\033[0;36m"); // cyan
		printf(BOLD_ON "Orthogonal" BOLD_OFF); 
	}
	else 
	{
		printf("\033[0;36m"); // cyan
		printf(BOLD_ON "Frustrum" BOLD_OFF);
	}
	printf("\n t: Top/Side view        - ");
	if(TopView == 0) 
	{
		printf("\033[0;36m"); // cyan
		printf(BOLD_ON "Side View" BOLD_OFF);
	}
	else 
	{
		printf("\033[0;36m"); // cyan
		printf(BOLD_ON "Top View" BOLD_OFF);
	}
	printf("\n e: Viewing Aids         - ");
	if(RadialConfinementViewingAids == 0) 
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Viewing Aids Off" BOLD_OFF);
	}
	else 
	{
		printf("\033[0;32m");
		printf(BOLD_ON "Viewing Aids On" BOLD_OFF);
	}
	printf("\n M: Video On/Off         - ");
	if (MovieFlag == 0) 
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Video Recording Off" BOLD_OFF); 
	}
	else 
	{
		printf("\033[0;32m");
		printf(BOLD_ON "Video Recording On" BOLD_OFF);
	}
	printf("\n l: Laser On/Off         - ");
	if (LaserFlag == 0) 
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Laser is Off" BOLD_OFF); 
	}
	else 
	{
		printf("\033[0;32m");
		printf(BOLD_ON "Laser is On" BOLD_OFF);
	}

	printf("\n L: Laser Debug          - ");
	if (LaserDebugMode == 0) 
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Debug Off (press capital L to arm)" BOLD_OFF); 
	}
	else 
	{
		printf("\033[0;33m");
		printf(BOLD_ON "Debug ARMED - waiting for first hit..." BOLD_OFF);
	}

	printf("\n g: Gravity On/Off       - ");
	if (GravityFlag == 0) 
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Gravity is Off" BOLD_OFF); 
	}
	else 
	{
		printf("\033[0;32m");
		printf(BOLD_ON "Gravity is On" BOLD_OFF);
	}
	printf("\n b: Brownian On/Off      - ");
	if (BrownianFlag == 0) 
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Brownian is Off" BOLD_OFF); 
	}
	else 
	{
		printf("\033[0;32m");
		printf(BOLD_ON "Brownian is On" BOLD_OFF);
	}
	printf("\n D: Drag On/Off          - ");
	if (DragFlag == 0) 
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Drag is Off" BOLD_OFF); 
	}
	else 
	{
		printf("\033[0;32m");
		printf(BOLD_ON "Drag is On" BOLD_OFF);
	}
	printf("\n x: Stirring On/Off      - ");
	if(StirFlag == 0) 
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Stirring Off" BOLD_OFF);
	}
	else 
	{
		printf("\033[0;32m");
		
		printf(BOLD_ON "Stirring..." BOLD_OFF);
	}
	printf("\n c: Shake It Up On/Off   - ");
	if(ShakeItUpFlag == 0) 
	{
		printf("\033[0;31m");
		printf(BOLD_ON "Shake Off" BOLD_OFF);
	}
	else 
	{
		printf("\033[0;32m");
		printf(BOLD_ON "Shake On" BOLD_OFF);
	}
	
	if(TopView == 0) //side view
	{
		printf("\n");
		printf("\n Adjust views");
		printf("\n k/l: Rotate CW/CCW");
		printf("\n a/d: Translate Left/Right");
		printf("\n s/w: Translate Down/Up");
		printf("\n z/Z: Translate Out/In");
		printf("\n f:   Recenter");
		printf("\n");
		printf("\n ********************************************************************");
		printf("\033[0m");
		printf("\n");

	}
	else //top view controls
	{
		printf("\n");
		printf("\n Adjust views");
		printf("\n k/l: Rotate CW/CCW");
		printf("\n a/d: Translate Left/Right");
		printf("\n z/Z: Translate Down/Up");
		printf("\n w/s: Translate Out/In");
		printf("\n f:   Recenter");
		printf("\n");
		printf("\n ********************************************************************");
		printf("\033[0m");
		printf("\n");
	}
}

void setup()
{	
	readSimulationParameters();
	setNumberOfBodies();
	allocateMemory();

	// Initialize CURAND
	//unsigned int seed = static_cast<unsigned int>(time(0));
    int threads = 256;
	int blocks = (NumberOfBodies + threads - 1) / threads;

	init_curand<<<blocks, threads>>>(1234, DevStates);
	errorCheck("init_curand");

	setInitailConditions();
	cudaDeviceSynchronize();

	printf("\n=== UNIT SANITY CHECK ===\n");
	printf(" BeakerRadius  = %.3f um\n", BeakerRadius);
	printf(" FluidHeight   = %.3f um\n", FluidHeight);
	printf(" Dt            = %.6f ms\n", Dt);
	printf(" Particle dia  = %.3f um\n", BodyPosition[NumberOfPolymers].w);
	printf(" Particle mass = %.3e pg\n", BodyForce[NumberOfPolymers].w);
	printf("=========================\n");

	cudaMemcpy( BodyPositionGPU, BodyPosition, NumberOfBodies*sizeof(float4), cudaMemcpyHostToDevice );
	cudaMemcpy( BodyVelocityGPU, BodyVelocity, NumberOfBodies*sizeof(float4), cudaMemcpyHostToDevice );
	cudaMemcpy( BodyForceGPU, BodyForce, NumberOfBodies*sizeof(float4), cudaMemcpyHostToDevice );
	cudaMemcpy( PolymerConnectionAGPU, PolymerConnectionA, NumberOfPolymers*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( PolymerConnectionBGPU, PolymerConnectionB, NumberOfPolymers*sizeof(int), cudaMemcpyHostToDevice );
	
	cudaSetDevice(0); // Select GPU device 0
    	
    	errorCheck("cudaSetDevice");
	
	DrawTimer = 0;
	PrintTimer = 0;
	RunTime = 0.0;
	Pause = 1;
	MovieFlag = 0;
	ViewFlag = 1;
	TopView = 1;
	LaserFlag = 0;
	GravityFlag = 0;
	BrownianFlag = 0;
	RadialConfinementViewingAids = 1;
	StirFlag = 0;
	ShakeItUpFlag = 0;
	DebugFlag = 0;
	Theta = 0.0;

	LaserBeamRadius = BeakerRadius / 10.0f;     // wider entry circle for cone
	// Calculate the starting height of the rays to place the focal point at the beaker's vertical center.
	// The focal point is at y = FluidHeight / 2.0f.
	// The formula from initRaysKernel is: focusY = beamCenterY - beamRadius / sinThetaMax
	// So, beamCenterY = focusY + beamRadius / sinThetaMax
	// SIN_THETA_MAX is 0.8f, defined in rayTraceLaser.h [4]
	LaserBeamCenterY = FluidHeight * 2.0f;

	DragFlag = 0;      // Turn ON for proper Brownian motion physics
	UsedDrag = Drag;   // Set UsedDrag to the value read from simulationSetup
	ContainerFlag = 1; //0 container off, 1 container on 
	ParticleInteractionFlag = 1; // 0 particle interactions off, 1 particle interactions on.

	StirAngularVelosity = (2.0*PI)/(100.0); // This is 10 revolution per second in milliseconds
	
	CenterOfSimulation.x = 0.0;
	CenterOfSimulation.y = 0.0;
	CenterOfSimulation.z = 0.0;
	CenterOfSimulation.w = 0.0;
	
	AngleOfSimulation.x = 0.0;
	AngleOfSimulation.y = 1.0;
	AngleOfSimulation.z = 0.0;
	AngleOfSimulation.w = 0.0;
	
	terminalPrint();
}


int main(int argc, char** argv)
{
	setup();
	
	XWindowSize = 2000;
	YWindowSize = 2000; 
	Buffer = new int[XWindowSize*YWindowSize];

	// Clip plains
	Near = 0.2;
	Far = BeakerRadius*10.0;

	//Direction here your eye is located location
	EyeX = 0.0;
	EyeY = FluidHeight+FluidHeight/3.0;
	EyeZ = 1.0; //BeakerRadius+BeakerRadius/10.0;

	//Where you are looking
	CenterX = 0.0;
	CenterY = 0.0;
	CenterZ = 0.0;

	//Up vector for viewing
	UpX = 0.0;
	UpY = 1.0;
	UpZ = 0.0;
	
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(5,5);
	Window = glutCreateWindow("Laser Microplatics Capture");
	
	gluLookAt(EyeX, EyeY, EyeZ, CenterX, CenterY, CenterZ, UpX, UpY, UpZ);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, Near, Far);
	glMatrixMode(GL_MODELVIEW);
	glClearColor(0.0, 0.0, 0.0, 0.0);
	
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMouseFunc(mymouse);
	glutKeyboardFunc(KeyPressed);
	glutIdleFunc(idle);

	int numMicroplastics = NumberOfBodies - NumberOfPolymers;
	createHistogramWindow(argc, argv, numMicroplastics);

	glutMainLoop();
	
	return 0;
}

