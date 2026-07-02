__device__ float4 shakeItUp(curandState_t*, int, float);
__device__ float4 getPolymerPolymerForce(float4, float4, int, int, int, float, int);
__device__ float4 getPolymerMicroPlasticForce(float4, float4);
__device__ float4 getMicroPlasticMicroPlasticForce(float4, float4);
__device__ float4 getGravityForces(float, float, float);
__device__ float4 getDragForces(float4, float4);
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
float4 f = {0.0f, 0.0f, 0.0f, 0.0f};

float dx = p1.x - p0.x;
float dy = p1.y - p0.y;
float dz = p1.z - p0.z;
float r2 = dx * dx + dy * dy + dz * dz + 1.0e-6f;
float r_orig = sqrtf(r2);

float fMag = 0.0f;

// --- LJ part (purely repulsive for polymers) ---
float sigma = (p0.w + p1.w) * 0.5f;
float r_cut_lj = 1.122462f * sigma; // WCA cutoff

if (r_orig < r_cut_lj)
{
// A "soft" clamp that prevents the force from becoming infinite
float r_safe = sqrtf(r_orig*r_orig + (0.1f*sigma)*(0.1f*sigma));
float epsilon = 500.0f; 
float sr = sigma / r_safe;
float sr2 = sr * sr;
float sr6 = sr2 * sr2 * sr2;
float sr12 = sr6 * sr6;
fMag += 24.0f * epsilon * (2.0f * sr12 - sr6) / r_safe;
}

// --- Chain spring (Hookean) for connected neighbors ---
if ((linkA != -1 && yourId == linkA) || (linkB != -1 && yourId == linkB))
{
float kChain = 5000.0f;
// This force pulls if r > length, pushes if r < length
fMag += -kChain * (length - r_orig);
}
// Apply a cap to the total combined force
float fCap = 1.0e5f; 
if (fMag > fCap) fMag = fCap;
if (fMag < -fCap) fMag = -fCap;

f.x = -fMag * dx / r_orig;
f.y = -fMag * dy / r_orig;
f.z = -fMag * dz / r_orig;

return f;
}

/******************************************************************************
This is the Polymer to micro-plastic interaction function.
Place any comments and papers you used to get parameters for this function here.
*******************************************************************************/
__device__ float4 getPolymerMicroPlasticForce(float4 p0, float4 p1)
{
float4 f = {0.0f, 0.0f, 0.0f, 0.0f};
float dx = p1.x - p0.x;
float dy = p1.y - p0.y;
float dz = p1.z - p0.z;
float r2 = dx*dx + dy*dy + dz*dz + 1.0e-6f;
float r = sqrtf(r2);

float contact = (p0.w + p1.w) * 0.5f;

if (r < contact)
{
float k = 30.0f;
float overlap = contact - r;
float fMag = k * overlap;

float fCap = 200.0f;
if (fMag > fCap) fMag = fCap;

f.x = -fMag * dx / r;
f.y = -fMag * dy / r;
f.z = -fMag * dz / r;
}
return f;
}

/******************************************************************************
This is the micro-plasic to micro-plastic interaction function.
Place any comments and papers you used to get parameters for this function here.
Self-Assembled Plasmonic Nanoparticle Clusters: 
https://www.science.org/doi/10.1126/science.1187949#editor-abstract
*******************************************************************************/
__device__ float4 getMicroPlasticMicroPlasticForce(float4 p0, float4 p1)
{
float4 f = {0.0f, 0.0f, 0.0f, 0.0f};
float dx = p1.x - p0.x;
float dy = p1.y - p0.y;
float dz = p1.z - p0.z;
float r2 = dx*dx + dy*dy + dz*dz + 1.0e-6f;
float r = sqrtf(r2);

// sigma = contact distance (sum of radii).
// LJ minimum sits at r = 2^(1/6) * sigma ≈ 1.122 * sigma,
// just outside the surface — particles stick at near-contact.
float sigma = (p0.w + p1.w) * 0.5f;

// epsilon controls attraction depth.
// 50 fN·um ≈ 50 zJ — comparable to van der Waals at this scale.
// Increase to make aggregates stickier, decrease to break them up.
float epsilon = 30.0f;

// Cut off at 2.5*sigma — standard LJ cutoff, saves compute
float r_cut = 2.5f * sigma;
if (r >= r_cut) return f;

// Soft core: clamp r so force doesn't blow up at very small separations
float r_safe = fmaxf(r, 0.5f * sigma);

float sr = sigma / r_safe;
float sr2 = sr * sr;
float sr6 = sr2 * sr2 * sr2;
float sr12 = sr6 * sr6;

// F = -dU/dr, projected along the separation vector.
// Positive fMag = repulsive (pushes apart), negative = attractive (pulls together).
float fMag = 24.0f * epsilon * (2.0f * sr12 - sr6) / r_safe;

float fCap = 50.0f;
fMag = fmaxf(-fCap, fminf(fCap, fMag));

f.x = -fMag * dx / r;
f.y = -fMag * dy / r;
f.z = -fMag * dz / r;

return f;
}
/******************************************************************************
This is the gravity function add complexity at will.
*******************************************************************************/
__device__ float4 getGravityForces(float density_plastic, float mass, float density_fluid)
{
//calculate this at the beginning so it only happens once 
float4 f;
float G = 9.81f; // acceleration due to gravity

// Volume of the plastic particle: V_plastic = mass / density_plastic
float V_plastic = mass / density_plastic;

// Weight: W = density_plastic * V_plastic * g
float W = density_plastic * V_plastic * G;

// Buoyancy: B = density_fluid * V_plastic * g
float B = density_fluid * V_plastic * G;

//B - W = (density_fluid - density_plastic) * V_plastic * g
float netForce = B - W; // equivalent to (density_fluid - density_plastic) * V_plastic * G

// Apply net force in y-direction (vertical)
f.x = 0.0f;
f.y = netForce; // positive = floats up, negative = sinks down
f.z = 0.0f;
f.w = 0.0f;

return f;
}
/******************************************************************************
This is the fluid drag force function.
*******************************************************************************/
__device__ float4 getDragForces(float4 pos, float4 vel)
{
// Stokes drag: F = -gamma * v, where gamma = 6*pi*eta*R [2]
const float eta = 1.0e3f; // water viscosity pg/(µm·ms)
const float drag_scale = 0.01f; // scale factor to reduce drag for simulation stability
float R = pos.w / 2.0f; // particle radius in meters
float gamma = 6.0f * PI * eta * R * drag_scale; // N*s/m

// float drag_scale = 0.01f;

float4 f;
f.x = -gamma * vel.x;
f.y = -gamma * vel.y;
f.z = -gamma * vel.z;
f.w = 0.0f;
return f;
}

/******************************************************************************
This is the laser force function.
*******************************************************************************/
// Ray-traced laser force [4][8]
// Reads precomputed force from laserForce array (filled by traceRaysKernel)
__device__ float4 getLaserForces(float4 laserF)
{
float4 f;
f.x = laserF.x;
f.y = laserF.y;
f.z = laserF.z;
f.w = 0.0f;
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
float k = 100.0f; // fN/µm, original tuned value
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
float magRand = 10000.0f; //fN
float centerMag;
//float temp;
float magStir = 100000.0f; //fN; 
//float mag2 = 1.0e-14f;
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

but i need to make the diffusion coefficient D = sqrt(7microns) where R = 5microns and this has to happen in 50 seconds with drag
*********************************************************************************/


/******************************************************************************
Brownian Motion using Wiener Process
Physics from:
- Volpe & Volpe (Am. J. Phys. 2013), Section 2:
Finite difference solution (Eq. 10): xi = xi-1 + sqrt(2*D*Dt) * wi
"wi are random numbers with zero mean and unit variance"
- ThorLabs User Guide Section 7.5.2:
Eq. 22: F_R = 6*pi*eta*R*v (Stokes frictional force)
Eq. 23: m = 2*k_B*T / (3*pi*eta*R) (MSD slope)
"The 1 um spheres can be more easily sent into motion by impact 
with the water molecules than larger spheres"
Your requirements:
- Target RMS displacement: sqrt(7) um for R = 5 um particle
- Time interval: 50 seconds
- Tunable coefficient: A
*******************************************************************************/
//divides by zero if dt is too small and everything blows up.
/*__device__ float4 brownian_motion(curandState_t* states, int id, float4 pos, float dt)
{
float4 f;
f.x = 0.0f;
f.y = 0.0f;
f.z = 0.0f;
f.w = 0.0f; 
// Get particle radius from diameter (stored in pos.w)
float radius = pos.w / 2.0f; // micrometers 
// Safety check
if(radius <= 0.0f || dt <= 0.0f)
{
return f;
} 
// ============================================================
// PARAMETERS
// ============================================================ 
// Tunable coefficient - adjust to calibrate sqrt(7) um RMS
float A = 1.0f; 
// Reference particle radius
float R_ref = 5.0f; // micrometers 
// Target RMS displacement and time
float targetRMS = sqrtf(7.0f); // sqrt(7) um
float targetTime = 50.0f * 1000.0f; // 50 seconds in ms 
// ============================================================
// DIFFUSION COEFFICIENT
// From Volpe & Volpe Eq. 10: displacement = sqrt(2*D*dt) * w
// After time t, RMS = sqrt(2*D*t) for 1D, sqrt(4*D*t) for 2D
// So: D = targetRMS^2 / (4 * targetTime)
// ============================================================ 
float D = (targetRMS * targetRMS) / (4.0f * targetTime); 
// Scale D for particle size (larger particles diffuse less)
// ThorLabs Section 7.5.2 [6]
D = D * (R_ref / radius); 
// ============================================================
// WIENER PROCESS
// From Volpe & Volpe Section II [4]:
// "wi are random numbers with zero mean and unit variance"
// curand_normal() returns N(0,1)
// ============================================================ 
float w_x = curand_normal(&states[id]);
float w_y = curand_normal(&states[id]);
float w_z = curand_normal(&states[id]); 
// ============================================================
// CONVERT DISPLACEMENT TO FORCE
// From Volpe Eq. 10: displacement per step = sqrt(2*D*dt) * w
// To get force: F = (displacement / dt) * gamma
// where gamma is drag coefficient from ThorLabs Eq. 22 [6]
// ============================================================ 
float displacementAmplitude = sqrtf(2.0f * D * dt); 
// Apply tunable coefficient A with sqrt(7)/A scaling
float forceAmplitude = A * (targetRMS / A) * displacementAmplitude / dt * curand_normal(&states[id]); //add error check for if dt is too small at the beginning of code
f.x = forceAmplitude * w_x;
f.y = forceAmplitude * w_y;
f.z = forceAmplitude * w_z; 
return f;
}*/

/*__device__ float4 brownian_motion(curandState_t* states, int id, float4 pos, float dt)
{
float4 f;
f.x = 0.0f;
f.y = 0.0f;
f.z = 0.0f;
f.w = 0.0f;

float radius = pos.w / 2.0f;

// Safety check — bail out if invalid
if(radius <= 0.0f || dt <= 1e-9f)
{
return f;
}

// Tunable coefficient
float A = 1.0f;
float R_ref = 5.0f;
float targetRMS = sqrtf(7.0f);
float targetTime = 50.0f * 1000.0f;

// Diffusion coefficient from Volpe & Volpe Eq. 10 [2]
float D = (targetRMS * targetRMS) / (4.0f * targetTime);
D = D * (R_ref / radius); // smaller particles diffuse more [2]

// Wiener process samples - N(0,1) [2]
float w_x = curand_normal(&states[id]);
float w_y = curand_normal(&states[id]);
float w_z = curand_normal(&states[id]);

// Force amplitude scales as 1/sqrt(dt), not 1/dt
// This is the physically correct Langevin thermal force [2]
// F = sqrt(2*D*gamma^2/dt) * w, simplified using A as tunable scale
float forceAmplitude = A * sqrtf(2.0f * D / dt);

// Clamp the force to prevent explosions if dt gets weird
float maxForce = 1.0e4f; // tune this if needed
if(forceAmplitude > maxForce) forceAmplitude = maxForce;

f.x = forceAmplitude * w_x;
f.y = forceAmplitude * w_y;
f.z = forceAmplitude * w_z;

return f;
} */

// brownian_motion - Langevin in scaled units [2]
__device__ float4 brownian_motion(curandState_t* states, int id, float4 pos, float dt)
{
float4 f; f.x=0; f.y=0; f.z=0; f.w=0;
float R = pos.w / 2.0f;
if (R <= 1.0e-6f || dt <= 1.0e-9f) return f;

const float kB = 1.38e-2f; // pg·µm²/ms²/K [2]
const float T = 300.0f; // K
const float eta = 1.0e3f; // pg/(µm·ms)
const float drag_scale = 0.01f; // scale factor to reduce drag for simulation stability
float gamma = 6.0f * PI * eta * R; // pg/ms
float gamma_eff = gamma * drag_scale; // scaled drag for stability

float amp = sqrtf(2.0f * kB * T * gamma_eff / dt); //* drag_scale; // fN

f.x = amp * curand_normal(&states[id]);
f.y = amp * curand_normal(&states[id]);
f.z = amp * curand_normal(&states[id]);
return f;
}

