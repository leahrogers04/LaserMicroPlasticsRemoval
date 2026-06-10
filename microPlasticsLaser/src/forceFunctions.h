__device__ float4 shakeItUp(curandState_t*, int, float);
__device__ float4 getPolymerPolymerForce(float4, float4, int, int, int, float, int);
__device__ float4 getPolymerMicroPlasticForce(float4, float4);
__device__ float4 getMicroPlasticMicroPlasticForce(float4, float4);
__device__ float4 getGravityForces(float, float, float);
__device__ float4 getDragForces(float, float4);
__device__ float4 getLaserForces(float4);
__device__ float4 getContainerForces(float4, float , float);
__device__ float4 getStirringForces(curandState_t*, int, float4, float4, float, float, float);
__device__ float4 brownian_motion(curandState_t*, int, float4, float);

/******************************************************************************
 This function just shakes the whole system up
*******************************************************************************/
__device__ float4 shakeItUp(curandState_t* states, int id, float shakeItUpMag)
{
	float mag = shakeItUpMag;
	float4 v;
	float randx = mag*(curand_uniform(&states[id])*2.0 - 1.0);
    	float randy = mag*(curand_uniform(&states[id])*2.0 - 1.0);
        float randz = mag*(curand_uniform(&states[id])*2.0 - 1.0);
        
        v.x = randx;
        v.y = randy;
        v.z = randz;
	
	return(v);
}

/******************************************************************************
 This is the Polymer to Polymer interaction function.
 Place any comments and papers you used to get parameters for this function here.
 
*******************************************************************************/                                 
__device__ float4 getPolymerPolymerForce(float4 p0, float4 p1, int linkA, int linkB, int yourId, float length, int myId)
{
    float4 f;
    float force;
    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    float dz = p1.z - p0.z;
    float r2 = dx*dx + dy*dy + dz*dz + 0.000001;
    float r = sqrt(r2);
    float penitration = (p0.w + p1.w)/2.0 - r;
    float k1 = 100.0;
    float k2 = 100.0;
    
    force  = 0.0;
    
    if(0.0 < penitration)
    {
    	// PolymerPolymer shell repulsion
    	force  += -k1*penitration*penitration;
    }
    else
    {
    	// PolymerPolymer atraction
    	force  += 0.0;
    }
    
    // Polymer chain connection forces.
    if(linkA != -1 && yourId == linkA)
    { 
    	force  += -k2*(length - r);
    }
    if(linkB != -1 && yourId == linkB)
    { 
    	force  += -k2*(length - r);
    }
    
    f.x = force*dx/r;
    f.y = force*dy/r;
    f.z = force*dz/r;
    
    return(f);
}

/******************************************************************************
 This is the Polymer to micro-plastic interaction function.
 Place any comments and papers you used to get parameters for this function here.
*******************************************************************************/
__device__ float4 getPolymerMicroPlasticForce(float4 p0, float4 p1)
{
    float4 f;
    float force;
    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    float dz = p1.z - p0.z;
    float r2 = dx*dx + dy*dy + dz*dz + 0.0001;
    float r = sqrt(r2);
    float G = 15000.0; //before was 100.0
    float penitration = (p0.w + p1.w)/2.0 - r;
    float k = 100.0;
    
    force  = 0.0;
 
    if(0.0 < penitration)
    {
    	// Polymer microPlastic shell repulsion.
    	force  += -k*penitration*penitration;
    }
    else
    {
    	// Polymer microPlastic actraction
    	force += G*(p0.w*p1.w)/r2;
    	//force += 0.0;
    	//printf("\n force = %f", force);
    }
    
    f.x = force*dx/r;
    f.y = force*dy/r;
    f.z = force*dz/r;
    return(f);
}

/******************************************************************************
 This is the micro-plasic to micro-plastic interaction function.
 Place any comments and papers you used to get parameters for this function here.
 Self-Assembled Plasmonic Nanoparticle Clusters: 
 https://www.science.org/doi/10.1126/science.1187949#editor-abstract
*******************************************************************************/
__device__ float4 getMicroPlasticMicroPlasticForce(float4 p0, float4 p1)
{
    float4 f;
    float force;
    float dx = p1.x - p0.x;
    float dy = p1.y - p0.y;
    float dz = p1.z - p0.z;
    float r2 = dx*dx + dy*dy + dz*dz + 0.0001;
    float r = sqrt(r2);
    float penitration = (p0.w + p1.w)/2.0 - r;
    float k = 100.0;
    
    force  = 0.0;
    
    if(0.0 < penitration)
    {
    	// MicroPlastic microPlastic shell repulsion.
    	// Using a second order because the surface of intersection is a circle.
    	// So if we are using a pressure type thought process here, force is pressure time area.
    	//force  += -k*penitration*penitration;
    	
    	// Or use a first order to make it simple.
    	force  += -k*penitration;
    	
    }
    else
    {
    	// MicroPlastic microPlastic actraction
    	force  += 0.0;
    }
    
    f.x = force*dx/r;
    f.y = force*dy/r;
    f.z = force*dz/r;
    
    return(f);
}

/******************************************************************************
 This is the gravity function add complexity at will.
*******************************************************************************/
__device__ float4 getGravityForces(float density, float mass, float fluidDensity)
{
	float4 f;
	float G = 9.81; // When you take meters/second^2 to micrometers/millisecond^2 everything cancels out so you get 9.81
	f.x = 0.0;
	f.z = 0.0;
	
	// May want to do something more accurate. I just made a linear function that  pulled stuff dowm if its density is
	// greater than the fluid pushed it up if its density is less than the fluid.
	f.y = -G*mass*(density - fluidDensity);
	
	return(f);
}

/******************************************************************************
 This is the fluid drag force function.
*******************************************************************************/
__device__ float4 getDragForces(float drag, float4 vel)
{
        float4 f;
	f.x = -drag*vel.x;
	f.y = -drag*vel.y;
	f.z = -drag*vel.z;
	
	return(f);
}

/******************************************************************************
 This is the laser force function.
*******************************************************************************/
__device__ float4 getLaserForces(float4 pos)
{
        float4 f;
	f.x = -pos.x;
	f.y = 5.0 - pos.y;
	f.z = -pos.z;
	
	return(f);
}

/******************************************************************************
 This function keeps the bodies in the container.
*******************************************************************************/
__device__ float4 getContainerForces(float4 posMe, float beakerRadius, float fluidHeight)
{
	float4 f;
	float force;
	float r2 = posMe.x*posMe.x + posMe.z*posMe.z;
	float r = sqrt(r2);
	float k = 100.0;
	
	f.x = 0.0;
	f.y = 0.0;
	f.z = 0.0;
	
	if(beakerRadius < r)
	{
		force = k*(beakerRadius - r);
		f.x = force*posMe.x/r;
		f.z = force*posMe.z/r;
	}
	
	if(fluidHeight < posMe.y)
	{
		force = k*(fluidHeight - posMe.y);
		f.y = force;
	}
	else if(posMe.y < 0.0)
	{
		force = -k*(posMe.y);
		f.y = force;
	}
	
	return(f);
}

/******************************************************************************
 This is the stirring function add complexity at will.
*******************************************************************************/
__device__ float4 getStirringForces(curandState_t* states, int id, float4 posMe, float4 velMe, float beakerRadius, float fluidHeight, float theta)
{
	float4 f;
	float angle;
	float magRand = 10000.0;
	float centerMag;
	//float temp;
	float magStir = 100000.0; //before was 200
	//float mag2 = 10.0;
	float r2 = posMe.x*posMe.x + posMe.z*posMe.z;
	float r = sqrt(r2);
	float range = PI/6.0;
	
	float randx = magRand*(curand_uniform(&states[id])*2.0 - 1.0);
        float randy = magRand*(curand_uniform(&states[id])*2.0 - 1.0);
        float randz = magRand*(curand_uniform(&states[id])*2.0 - 1.0);
	
	f.x = 0.0;
	f.y = 0.0;
	f.z = 0.0;
	
	if(0.0 < r)
	{
		// This gives a radial motion
		//f.x = mag1*(-posMe.z/r);
		//f.z = mag1*(posMe.x/r);
		
		// This gives a pulling down in the center and up on the sides.
		//f.y = mag2*(r*2.0/beakerRadius - 1.0);
		
		// This is suposed to move it in from the top and out on the bottom.
		//temp = 10.0*(1.0 - posMe.y/fluidHeight); //mag2*(-r*2.0/beakerRadius + 1.0);
		//f.x = temp*(posMe.x/r);
		//f.z = temp*(posMe.z/r);
		
		angle = atan(posMe.z/posMe.x);
		if(0.0 < posMe.x)
		{
			if(0.0 < posMe.z)
			{
				angle += 0.0;
			}
			else
			{
				angle += 2.0*PI;
			}
		}
		else
		{
			if(0.0 < posMe.z)
			{
				angle += PI;
			}
			else
			{
				angle += PI;
			}
		}
		
		if(0.0 < (angle - theta) && (angle - theta) < range || (2*PI - (theta - angle)) < range)
		{
			centerMag = -(r/beakerRadius - 1.0)*(r/beakerRadius - 1.0) + 1.0; // This makes it full in the middle and die off on the ends,
			f.x = randx + centerMag*magStir*(-posMe.z/r);
			f.y = randy;
			f.z = randz + centerMag*magStir*(posMe.x/r);
		}
	}
	
	return(f);
}

/*********************************************************************************
Brownian Motion using Langevin Equation and Wiener Process
 
[1] Simulation of a Brownian particle in an optical trap https://doi.org/10.1119/1.4772632:
 where W(t) is a Wiener process which is the white noise term that models the random collisions with the fluid molecules.

[2] ThorLabs user guide 

 The Langevin equation ([1], Eq. 1):
	m*x_ddot = -gamma*x_dot + sqrt(2*k_B*T*gamma)*W(t)

 The thermal noise amplitude sqrt(2*k_B*T*gamma) comes from the 
 fluctuation-dissipation theorem as shown in the Volpe paper.
 
Wiener process properties ([1], section 2 below Eq. 1):
  "W(t) is characterized by: mean <W(t)> = 0 for all t; 
      <W(t)^2> = 1 for each value t; and W(t1) and W(t2) are 
      independent of each other for t1 != t2"

Stokes frictional force ([2], Section 7.5, Eq. 22):
      F_R = 6*pi*eta*R*v
 This implies the drag coefficient gamma = 6*pi*eta*R
 
 The finite difference approach from [1] uses:
     x_i = x_{i-1} + sqrt(2*D*Dt) * w_i
 where w_i is N(0,1) and N(0,1) is a Gaussian random number with zero mean and unit variance. and D = k_B*T/gamma is the diffusion coefficient.

 Smaller particles move more ([2], Section 7.5):
     "The 1 um spheres can be more easily sent into motion by impact 
      with the water molecules than larger spheres"


 Key change from original: using curand_normal() (Gaussian) instead of 
 curand_uniform() (uniform) for proper Wiener process statistics [1].

*********************************************************************************/
__device__ float4 brownian_motion(curandState_t* states, int id, float4 pos, float dt)
{
	float4 f;

	// get particle radius from diameter stored in pos.w
	float radius = pos.w / 2.0f;

	// physical params
	float temperature = 298.0f; // room temperature in K
	float eta = 1.0e-3f; // effective viscosity of the water in Pa*s (kg/(m*s))

	 // Drag coefficient gamma = 6*pi*eta*R 
    // (derived from Stokes frictional force F_R = 6*pi*eta*R*v, 
    //  ThorLabs Section 7.5.2, Eq. 22)
    float gamma = 6.0f * PI * eta * radius;
    
    // Diffusion coefficient D = k_B*T/gamma
    // This relates to ThorLabs Eq. 23: m = 2*k_B*T/(3*pi*eta*R)
    float D = K_BOLTZMANN * temperature / gamma;
    
    // From Volpe & Volpe finite difference solution:
    // x_i = x_{i-1} + sqrt(2*D*Dt) * w_i
    // To convert to force: F = displacement * gamma / dt
    // This gives: F = sqrt(2*D*gamma^2/dt) * w_i = sqrt(2*k_B*T*gamma/dt) * w_i
    float thermalAmplitude = sqrtf(2.0f * D * gamma * gamma / dt);
    
    // Wiener process: Gaussian random numbers N(0,1)
    // From Volpe & Volpe: "w_i are random numbers with zero mean and unit variance"
    float dW_x = curand_normal(&states[id]);
    float dW_y = curand_normal(&states[id]);
    float dW_z = curand_normal(&states[id]);
    
    f.x = thermalAmplitude * dW_x;
    f.y = thermalAmplitude * dW_y;
    f.z = thermalAmplitude * dW_z;
    f.w = 0.0f;
    
    return f;
}



/******************************************************************************
 This is the Brownian Motion function.
 Place any comments and papers you used to get parameters for this function here.
 The below commented out function is one I ased to get Brownian Motion in another project.
*******************************************************************************/
/*
__device__ float4 brownian_motion(curandState_t* states, int id)
{
	float mag = 100.0;
	float4 f;
	float randx = mag*(curand_uniform(&states[id])*2.0 - 1.0);
        float randy = mag*(curand_uniform(&states[id])*2.0 - 1.0);
        float randz = mag*(curand_uniform(&states[id])*2.0 - 1.0);
        
        f.x = randx;
        f.y = randy;
        f.z = randz;
	
	return(f);
}
*/

// This is an example of brownian motion I used in another project.
/*
void brownian_motion(float3 *force)
{
	int i,under_normal_curve;
	float mag, angle1, angle2;
	float x,y,normal_hieght,temp;
	
	temp = 4.0*g_drag*DT;
	under_normal_curve = NO;
	
	while(under_normal_curve == NO)
	{
		x = 2.0*1.0*(float)rand()/RAND_MAX - 1.0;
		y = 1.0*(float)rand()/RAND_MAX;
		normal_hieght = 1.0*exp(-x*x/temp);
		if(y <= normal_hieght)
		{
			mag = x;
			under_normal_curve = YES;
		}
	}	
	
	for(i = 0; i < NUMBER_OF_BODIES; i++)
	{
		angle1 = PI*(float)rand()/RAND_MAX;
		angle2 = 2.0*PI*(float)rand()/RAND_MAX;
		force[i].x += mag*sinf(angle1)*cosf(angle2);
		force[i].y += mag*sinf(angle1)*sinf(angle2);
		force[i].z += mag*cosf(angle1);
	}
}
*/

