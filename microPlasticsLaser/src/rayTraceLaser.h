// rayTraceLaser.h
// GPU ray-traced optical force using Ashkin's ray optics model.
//
// PHYSICS REFERENCES:
// [3] ThorLabs Portable Optical Tweezers User Guide, Chapters 4 & 8:
//     - Sec 4.2 Eq 13: Fs = n_m * P_beam * Qs / c  (scattering force, along beam) [3]
//     - Sec 4.2 Eq 15: Fg = n_m * P_beam * Qg / c  (gradient force, perpendicular) [3]
//     - Sec 4.2 Eq 14, 16: Qs, Qg factors from Fresnel reflection [3]
//     - Sec 4.2 Eq 17: Snell's law of refraction n1 sin(th_i) = n2 sin(th_r) [3]
//     - Sec 4.2 Eq 18: Stable trap requires Fg > Fs [3]
//     - Chapter 4 opening: "the laser has a Gaussian intensity profile (TEM00 mode)...
//       the gradient force can act orthogonally to the beam" [3]
//     - Chapter 8: The bead is always pushed toward the point of highest intensity;
//       radially by the Gaussian profile and axially into the focus [3]
// [Ashkin 1992] Forces of a single-beam gradient laser trap on a dielectric sphere
//     in the ray optics regime. Biophys. J. 61, 569-582.

#ifndef RAY_TRACE_LASER_H
#define RAY_TRACE_LASER_H

#define NUM_RAYS 200000
#define MAX_BOUNCES 5
#define SPEED_OF_LIGHT 3.0e11f     // um/ms (scaled units)
#define LASER_POWER_W  1.0e17f     // total beam power in pg*um^2/ms^3 (scaled) 100 mW = 1.0e17 pg*um^2/ms^3 
#define N_MEDIUM       1.33f       // water refractive index [3]
#define N_PARTICLE     1.55f       // generic microplastic (polystyrene ~ 1.57) [3]
#define SIN_THETA_MAX  0.8f        // sin(half-angle) of focusing cone ~ NA/n_medium [3]
#define DEBUG_HITS_PER_RAY 5
#define LASER_WAVELENGTH_UM 0.660f // 660 nm diode laser //alt. 1064 nm Nd:YAG-class trapping laser

// --- Force scaling ---
// Fs (scattering, destabilizing) and Fg (gradient, trapping) are broken out
// so we can enforce Ashkin's stability criterion Fg > Fs [3, Eq 18].
// In real experiments this is achieved via a high-NA objective; here we
// mimic that regime numerically.
#define FG_SCALE 10000.0f    // was 200
#define FS_SCALE 1000.0f     // was 30, keep ratio > 3:1 for Fg > Fs stability [3, Eq 18]

// Gaussian beam waist. In real optics w0 ~ lambda/(pi*NA). We choose a waist
// a few microns wide so real 5-um beads see a meaningful gradient across
// their body [3, Chapter 4 intro].
#define BEAM_WAIST_UM 0.263f; //wide enough to reach across entire beaker

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <math.h>

struct RayData
{
    float4 origin;     // x,y,z = pos, w = power carried by this ray
    float4 direction;  // x,y,z = dir, w = active flag
};

// ------------------------------------------------------------------
// Fresnel reflection coefficient (unpolarized)
// Reference: ThorLabs guide Sec 4.2 (implicit in Qs/Qg definitions) [3]
// ------------------------------------------------------------------
__device__ float fresnelR(float thetaI, float thetaR, float n1, float n2)
{
    float cI = cosf(thetaI), cR = cosf(thetaR);
    float Rs_n = n1*cI - n2*cR, Rs_d = n1*cI + n2*cR;
    float Rp_n = n2*cI - n1*cR, Rp_d = n2*cI + n1*cR;
    float Rs = (fabsf(Rs_d) > 1e-6f) ? (Rs_n/Rs_d)*(Rs_n/Rs_d) : 1.0f;
    float Rp = (fabsf(Rp_d) > 1e-6f) ? (Rp_n/Rp_d)*(Rp_n/Rp_d) : 1.0f;
    return 0.5f * (Rs + Rp);
}

// ------------------------------------------------------------------
// Ashkin Q-factor force equations [3, Sec 4.2, Eqs 13-16]
//   Fs = (n1*P/c) * Qs   [3, Eq 13]  parallel to ray   (scattering)
//   Fg = (n1*P/c) * Qg   [3, Eq 15]  perpendicular     (gradient)
// ------------------------------------------------------------------
__device__ void ashkinForce(float P, float thI, float thR, float n1, float n2,
                             float *Fs_out, float *Fg_out)
{
    float R = fresnelR(thI, thR, n1, n2);
    float T2 = (1.0f - R) * (1.0f - R);
    float c2t = cosf(2.0f*thI);
    float s2t = sinf(2.0f*thI);
    float c2r = cosf(2.0f*thR);
    float c2tmr = cosf(2.0f*thI - 2.0f*thR);
    float s2tmr = sinf(2.0f*thI - 2.0f*thR);

    float denom = 1.0f + R*R + 2.0f*R*c2r;
    if (denom < 1e-6f) denom = 1e-6f;

    // Q factors [3, Eqs 14 and 16]
    float Qs = 1.0f + R*c2t - T2*(c2tmr + R*c2t)/denom;
    float Qg = R*s2t - T2*(s2tmr + R*s2t)/denom;

    float pre = (n1 * P) / SPEED_OF_LIGHT;   // n_m * P_beam / c  [3, Eqs 13,15]
    *Fs_out = pre * Qs;
    *Fg_out = pre * Qg;
}

// ------------------------------------------------------------------
// Ray-sphere intersection: nearest entry point
// ------------------------------------------------------------------
__device__ float raySphereHit(float3 ro, float3 rd, float3 sc, float sr)
{
    float3 oc; oc.x=ro.x-sc.x; oc.y=ro.y-sc.y; oc.z=ro.z-sc.z;
    float a = rd.x*rd.x + rd.y*rd.y + rd.z*rd.z;
    float b = 2.0f*(oc.x*rd.x + oc.y*rd.y + oc.z*rd.z);
    float c = oc.x*oc.x + oc.y*oc.y + oc.z*oc.z - sr*sr;
    float disc = b*b - 4.0f*a*c;
    if (disc < 0.0f) return -1.0f;
    float sd = sqrtf(disc);
    float t1 = (-b - sd) / (2.0f*a);
    float t2 = (-b + sd) / (2.0f*a);
    if (t1 > 0.001f) return t1;
    if (t2 > 0.001f) return t2;
    return -1.0f;
}

__device__ float raySphereExit(float3 ro, float3 rd, float3 sc, float sr)
{
    float3 oc; oc.x=ro.x-sc.x; oc.y=ro.y-sc.y; oc.z=ro.z-sc.z;
    float a = rd.x*rd.x + rd.y*rd.y + rd.z*rd.z;
    float b = 2.0f*(oc.x*rd.x + oc.y*rd.y + oc.z*rd.z);
    float c = oc.x*oc.x + oc.y*oc.y + oc.z*oc.z - sr*sr;
    float disc = b*b - 4.0f*a*c;
    if (disc < 0.0f) return -1.0f;
    return (-b + sqrtf(disc)) / (2.0f*a);
}

// ------------------------------------------------------------------
// Zero laser-force accumulator each timestep
// ------------------------------------------------------------------
__global__ void zeroLaserForceKernel(float4 *laserForce, int nBodies)
{
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if (id >= nBodies) return;
    laserForce[id].x = 0.0f;
    laserForce[id].y = 0.0f;
    laserForce[id].z = 0.0f;
    laserForce[id].w = 0.0f;
}

// ------------------------------------------------------------------
// Initialize a converging cone of rays representing a focused TEM00
// Gaussian beam. Single stationary trap.
//
// Physics from ThorLabs guide [3]:
//   - "the laser has a Gaussian intensity profile (TEM00 mode)" (Ch. 4) [3]
//   - Intensity falls off radially: I(r) ~ exp(-r^2 / (2*sigma^2))
//   - Rays farther from the beam axis carry LESS power; this asymmetry is
//     what creates the gradient force that pulls beads to the beam center
//     (Ch. 8) [3]
//   - Focused beam creates a 3D trap: radial pull by Gaussian profile,
//     axial pull toward the focus (Ch. 8) [3]
// ------------------------------------------------------------------
__global__ void initRaysKernel(RayData *rays, curandState_t *states,
                                float beamRadius, float beamCenterY)
{
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if (id >= NUM_RAYS) return;

    float u1 = curand_uniform(&states[id % 256]);
    float u2 = curand_uniform(&states[id % 256]);
    float u3 = curand_uniform(&states[id % 256]);
    float u4 = curand_uniform(&states[id % 256]);

    // Sample cone direction uniformly in solid angle up to SIN_THETA_MAX
    float sinThetaMax = SIN_THETA_MAX;
    float sinTheta = sinThetaMax * sqrtf(u1);
    float cosTheta = sqrtf(1.0f - sinTheta*sinTheta);
    float phi = 2.0f * PI * u2;

    // Ray direction: pointing downward and inward toward the focus
    float dirX = sinTheta * cosf(phi);
    float dirY = -cosTheta;
    float dirZ = sinTheta * sinf(phi);

    // --- Single stationary trap at (0, focusY, 0) ---
    // Focus is placed at half the beaker height so particles above and
    // below can both be pulled in [3, Ch. 8].
    float focusX = 0.0f;
    float focusY = 10.0f; //just above the beaker floor, matches ThorLabs Ch. 7.2 [3]
    float focusZ = 0.0f;

    // Sub-diffraction focal spot: rays converge to a small disk near the focus,
    // giving the beam a realistic waist. Radius ~ lambda / (2 * NA) [3, Ch. 6.3].
    float focalSpotRadius = LASER_WAVELENGTH_UM / (2.0f * SIN_THETA_MAX);
    float r_spot   = focalSpotRadius * sqrtf(u3);
    float phispot = 2.0f * PI * u4;

        // Offset the ray's focal target within the sub-diffraction spot
    focusX = focusX + r_spot * cosf(phispot);
    focusZ = focusZ + r_spot * sinf(phispot);

    // Back-project the ray origin so all rays truly converge to the focus
    float backDist = (beamCenterY - focusY) / cosTheta;
    rays[id].origin.x = focusX - dirX * backDist;
    rays[id].origin.y = focusY - dirY * backDist;   // = beamCenterY
    rays[id].origin.z = focusZ - dirZ * backDist;

    // --------------------------------------------------------------
    // GAUSSIAN TEM00 POWER WEIGHTING
    //
    // Real TEM00 beams have intensity  I(r) = I0 * exp(-r^2 / (2*sigma^2))
    // [3, Ch. 4]: "Often, a Gaussian profile is used, in which the
    //  intensity decreases in a Gaussian distribution from the center of
    //  the beam outward. This is also the case in our setup."
    //
    // Without radial power weighting every ray carries equal power and
    // there is no transverse intensity gradient, so the gradient force
    // cancels out. Weighting each ray's power by the Gaussian of its
    // focal-plane offset recovers Ashkin's gradient force [3, Eq 15].
    // --------------------------------------------------------------
    
    // Gaussian TEM00 intensity profile applied at the beam cross-section
    // (entry plane), matching ThorLabs Ch. 4.2 [3]: "intensity decreases
    // in a Gaussian distribution from the center of the beam outward".
    // This is what creates the transverse gradient force that pulls off-axis
    // beads toward the beam axis, as described in Ch. 8 [3, Figs 64-67].
    float dxEntry = rays[id].origin.x;   // beam axis at x=0
    float dzEntry = rays[id].origin.z;   // beam axis at z=0
    float r2Entry = dxEntry*dxEntry + dzEntry*dzEntry;

    float sigma_r = BEAM_WAIST_UM;
    float gaussW  = expf(-r2Entry / (2.0f * sigma_r * sigma_r));

    rays[id].origin.w = (LASER_POWER_W / (float)NUM_RAYS) * gaussW;
    rays[id].direction.x = dirX;
    rays[id].direction.y = dirY;
    rays[id].direction.z = dirZ;
    rays[id].direction.w = 1.0f;
}
// (Ashkin ray-optics, [3, Sec 4.2]; Ashkin 1992).
//
// Force decomposition [3, Sec 4.2]:
//   Fs (along ray)    -> [3, Eq 13]  scattering; destabilizing
//   Fg (perpendicular)-> [3, Eq 15]  gradient;   trapping
//
// FS_SCALE < FG_SCALE numerically enforces the Ashkin stability
// criterion Fg > Fs [3, Eq 18].
// ------------------------------------------------------------------
__global__ void traceRaysKernel(RayData *rays, float4 *pos, float4 *laserForce,
                                 float4 *debugInfo, float4 *debugForce,
                                 int nPolymer, int nBodies)
{
    int id = threadIdx.x + blockDim.x*blockIdx.x;
    if (id >= NUM_RAYS) return;
    if (rays[id].direction.w < 0.5f) return;

    float3 ro;
    ro.x = rays[id].origin.x;
    ro.y = rays[id].origin.y;
    ro.z = rays[id].origin.z;
    float P = rays[id].origin.w;   // Gaussian-weighted power

    float3 rd;
    rd.x = rays[id].direction.x;
    rd.y = rays[id].direction.y;
    rd.z = rays[id].direction.z;

    int alreadyHit[20];
    int numHit = 0;

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++)
    {
        float bestT = 1.0e30f;
        int   bestP = -1;

        // Test only microplastics (skip polymer chain beads)
        for (int p = nPolymer; p < nBodies; p++)
        {
            int skip = 0;
            for (int h = 0; h < numHit; h++)
                if (alreadyHit[h] == p) { skip = 1; break; }
            if (skip) continue;

            float3 sc;
            sc.x = pos[p].x;
            sc.y = pos[p].y;
            sc.z = pos[p].z;
            float sr = pos[p].w / 2.0f;

            float t = raySphereHit(ro, rd, sc, sr);
            if (t > 0.0f && t < bestT) { bestT = t; bestP = p; }
        }
        if (bestP < 0) break;

        // Sanity checks
        if (!isfinite(ro.x) || !isfinite(ro.y) || !isfinite(ro.z)) break;
        if (fabsf(ro.x) > 1e6f || fabsf(ro.y) > 1e6f || fabsf(ro.z) > 1e6f) break;

        if (numHit < 20) alreadyHit[numHit] = bestP;
        numHit++;

        // Hit point in world space
        float3 hp;
        hp.x = ro.x + rd.x * bestT;
        hp.y = ro.y + rd.y * bestT;
        hp.z = ro.z + rd.z * bestT;

        float3 sc2;
        sc2.x = pos[bestP].x;
        sc2.y = pos[bestP].y;
        sc2.z = pos[bestP].z;
        float sr2 = pos[bestP].w / 2.0f;

        // Outward normal at hit point
        float3 n;
        n.x = (hp.x - sc2.x) / sr2;
        n.y = (hp.y - sc2.y) / sr2;
        n.z = (hp.z - sc2.z) / sr2;

        // Angle of incidence: cos(thI) = -rd . n  (flip normal if needed)
        float cosT = -(rd.x*n.x + rd.y*n.y + rd.z*n.z);
        if (cosT < 0.0f) {
            n.x = -n.x; n.y = -n.y; n.z = -n.z;
            cosT = -cosT;
        }
        if (cosT > 1.0f) cosT = 1.0f;
        float thI = acosf(cosT);

        // Snell's law [3, Eq 17]:  n1 sin(thI) = n2 sin(thR)
        float sR = (N_MEDIUM / N_PARTICLE) * sinf(thI);
        float thR = (sR > 1.0f) ? (PI/2.0f) : asinf(sR);

        // Ashkin Fs, Fg for this ray [3, Eqs 13-16]
        float Fs, Fg;
        ashkinForce(P, thI, thR, N_MEDIUM, N_PARTICLE, &Fs, &Fg);

        // ---------------------------------------------------------
        // Gradient direction: unit vector in the plane of incidence,
        // perpendicular to the ray direction [3, Sec 4.2; Ashkin 1992].
        // This comes purely from the local hit geometry (normal + ray
        // direction) — NOT from any assumed focus location. The net
        // centering force emerges from summing this over all rays.
        // ---------------------------------------------------------
        float dot_n_d = n.x*rd.x + n.y*rd.y + n.z*rd.z;
        float3 inPlane;
        inPlane.x = n.x - rd.x*dot_n_d;
        inPlane.y = n.y - rd.y*dot_n_d;
        inPlane.z = n.z - rd.z*dot_n_d;
        float ipl = sqrtf(inPlane.x*inPlane.x + inPlane.y*inPlane.y + inPlane.z*inPlane.z);

        float3 gradDir = {0.0f, 0.0f, 0.0f};
        if (ipl > 1e-6f)
        {
            gradDir.x = inPlane.x / ipl;
            gradDir.y = inPlane.y / ipl;
            gradDir.z = inPlane.z / ipl;
        }

        // --------------------------------------------------------------
        // Combine Fs and Fg into a Cartesian force on the particle.
        //   Fs along beam direction  [3, Eq 13]: destabilizing
        //   Fg perpendicular         [3, Eq 15]: trapping
        // Sign of Fg is preserved — this is what lets the axial force
        // flip sign on either side of the focus, which is required for
        // real 3D trapping. FS_SCALE/FG_SCALE are now pure unit-scaling
        // constants, not stability fudge factors.
        // --------------------------------------------------------------
        float fx = Fs*rd.x*FS_SCALE + Fg*gradDir.x*FG_SCALE;
        float fy = Fs*rd.y*FS_SCALE + Fg*gradDir.y*FG_SCALE;
        float fz = Fs*rd.z*FS_SCALE + Fg*gradDir.z*FG_SCALE;

        // Force cap (safety) - 1 nN in scaled units, well above physical
        // optical forces (~ pN range per [3, Ch. 7.5])
        float maxF = 1.0e6f;
        if (fx >  maxF) fx =  maxF;
        if (fx < -maxF) fx = -maxF;
        if (fy >  maxF) fy =  maxF;
        if (fy < -maxF) fy = -maxF;
        if (fz >  maxF) fz =  maxF;
        if (fz < -maxF) fz = -maxF;

        // Accumulate onto this particle's laser-force slot
        atomicAdd(&laserForce[bestP].x, fx);
        atomicAdd(&laserForce[bestP].y, fy);
        atomicAdd(&laserForce[bestP].z, fz);

        // --------------------------------------------------------------
        // Debug logging: record every hit on this ray up to
        // DEBUG_HITS_PER_RAY, so we can trace ray shadowing.
        // --------------------------------------------------------------
        if (bounce < DEBUG_HITS_PER_RAY)
        {
            int slot = id * DEBUG_HITS_PER_RAY + bounce;
            debugInfo[slot].x = (float)bestP;   // particle index
            debugInfo[slot].y = thI;            // incidence angle (rad)
            debugInfo[slot].z = thR;            // refraction angle (rad)
            debugInfo[slot].w = Fs;             // Fs magnitude
            debugForce[slot].x = fx;            // total x force
            debugForce[slot].y = fy;            // total y force (beam axis)
            debugForce[slot].z = fz;            // total z force
            debugForce[slot].w = Fg;            // Fg magnitude
        }

        // --------------------------------------------------------------
        // Ray continues straight through the sphere (shadowing).
        // This lets a single ray transfer momentum to multiple beads
        // it passes through, as in the ray-optics regime [3, Sec 4.2;
        // Ashkin 1992]. Real refraction/reflection at each interface is
        // approximated by the Q-factor formulation, which already
        // accounts for all internal reflections/transmissions.
        // --------------------------------------------------------------
        float t2_exit = raySphereExit(ro, rd, sc2, sr2);
        if (t2_exit < 0.0f) t2_exit = bestT + 2.0f * sr2;
        ro.x = ro.x + rd.x * (t2_exit + 0.01f);
        ro.y = ro.y + rd.y * (t2_exit + 0.01f);
        ro.z = ro.z + rd.z * (t2_exit + 0.01f);
    }
}

#endif // RAY_TRACE_LASER_H
