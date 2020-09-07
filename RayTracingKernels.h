/**
 * @file    RayTracingKernels.h
 * @brief   File containing all kernels needed for ray tracing.
 */

#pragma once

#include <stdio.h>

#include "HPCHelpers.h"
#include "Model.h"
#include "RandomKernels.h"
#include "Vec3.h"

namespace ALGO
{

/**
 * @brief   Kernel function for main simulation loop.
 *          Each thread runs the simulation loop for a single photon until it reaches
 *          the end of the tube or another stopping criterion is fulfilled.
 *
 * @tparam  value_t     The type used for the numerical data (e.g. float).
 * @tparam  index_t     The type used for indexing by the kernel (e.g. uint32_t).
 * @tparam  model_t     The derived model type.
 *
 * @param   startpoints     Individual startpoints for each thread.
 * @param   directions      Individual initial directions for each thread.
 * @param   hitpoints       Stores the final positions of the photons.
 * @param   results         Stores the total travelled distance and angles to
 *                          the end surface of each photon.
 * @param   angles_initial  Stores the initial emission angles of the rays
 * @param   ray_id          Current ray processing.
 * @param   model           The intersection model.
 * @param   alpha_crit      The critical angle of total internal reflection.
 * @param   status_codes    The array storing 1 or 0 for successful or unsuccessful
 *                          run.
 * @param   localstate      The threads random state.
 * @param   lambda_abs
 * @param   lambda_sc
 */
template <typename value_t, typename index_t, typename model_t, bool override = false>
__device__ __forceinline__ void rayTracingLoop(Vec3<value_t> * startpoints, //
                                               Vec3<value_t> * directions,  //
                                               Vec3<value_t> * hitpoints,   //
                                               Vec3<value_t> * results,
											   Vec3<value_t> * angles_initial,     //
                                               index_t ray_id,              //
                                               model_t & model,             //
                                               value_t alpha_crit,          //
                                               int8_t * status_codes,       //
                                               curandState & localstate,    //
                                               value_t lambda_abs,          //
                                               value_t lambda_sc)
{
	
    Vec3<value_t> normal, dir{directions[ray_id]}, hit{startpoints[ray_id]};
    
	angles_initial[ray_id] = {
            value_t(atan2(-dir.x, -dir.y) + M_PI), //
            value_t(acos(dir.z)), 
			dist_abs                              //
        };
        
    if (dir.x == 0 && dir.y == 0 && dir.z == 0)
    {
        status_codes[ray_id] = static_cast<int8_t>(HitResult::Failed);
        hitpoints[ray_id] = hit;
        return;
    }

    // RT Parameters
    HitResult res{HitResult::None};
    value_t part_dist{0}, total_dist{0};
    value_t dist_abs{interactionDist(&localstate, lambda_abs)};
    value_t dist_sc{interactionDist(&localstate, lambda_sc)};

    // normalize just to be sure
    normalize(dir); // later calculations always require normalized vectors

    bool scattered_once{false};
    while (1)
    {
        part_dist = value_t(0);
        res = model(hit, dir, normal, part_dist);
        total_dist += part_dist; // increase total distance

        bool scattered = false;
        if (total_dist >= dist_sc)
        {
            // SCATTERED
            scattered = true;
            scattered_once = true;
            hit -= (total_dist - dist_sc) * dir;
            total_dist = dist_sc;
            dist_sc += interactionDist(&localstate, lambda_sc);
        }

        if (total_dist >= dist_abs)
        {
            // ABSORBED
            hit -= (res == HitResult::Failed ? -1 : 1) * (total_dist - dist_abs) * dir;
            res = HitResult::AbsorbedFailed;
            break;
        }

        if (scattered)
        {
            randomDirection(dir, &localstate); // scattered direction
            continue;
        }

        if (res != HitResult::Continue) break; // == Success, None or Failed

        // apply all possible wall interactions
        // could include surface roughness and refraction in the future
        if (!interactWall(dir, normal, alpha_crit))
        {
            // REFRACTION FAILED
            res = HitResult::None;
            break;
        }
    }

    if (override)
    {
        startpoints[ray_id] = hit;
        directions[ray_id] = dir;
    }
    else
    {
        hitpoints[ray_id] = hit;
        dir = directions[ray_id];
        results[ray_id] = {
            value_t(atan2(-dir.x, -dir.y) + M_PI), //
            value_t(acos(dir.z)),                  //
            dist_abs                               //
        };
    }
    status_codes[ray_id] = static_cast<int8_t>(res) + (scattered_once ? 64 : 0);
}

/**
 * @brief   Kernel function for main simulation loop.
 *          Each thread runs the simulation loop for a single photon until it reaches
 *          the end of the tube or another stopping criterion is fulfilled.
 *
 * @tparam  value_t     The type used for the numerical data (e.g. float).
 * @tparam  index_t     The type used for indexing by the kernel (e.g. uint32_t).
 * @tparam  model_t     The derived model type.
 *
 * @param   num_rays        Length of startpoints and directions.
 * @param   startpoints     Individual startpoints for each thread.
 * @param   directions      Individual initial directions for each thread.
 * @param   hitpoints       Stores the final positions of the photons.
 * @param   results         Stores the total travelled distance and angles to
 *                          the end surface of each photon.
 * @param   angles_initial  Stores the initial emission angles of the rays
 * @param   model           The intersection model.
 * @param   alpha_crit      The critical angle of total internal reflection.
 * @param   state           The random states.
 * @param   status_codes    The array storing 1 or 0 for successful or unsuccessful
 *                          run.
 * @param   lambda_abs
 * @param   lambda_sc
 */
template <typename value_t, typename index_t, typename model_t, bool override = false>
__global__ void startRayTracingLoop(index_t num_rays,            //
                                    Vec3<value_t> * startpoints, //
                                    Vec3<value_t> * directions,  //
                                    Vec3<value_t> * hitpoints,   //
                                    Vec3<value_t> * results,
									Vec3<value_t> * angles_initial,     //
                                    model_t model,               //
                                    value_t alpha_crit,          //
                                    curandState * state,         //
                                    int8_t * status_codes,       //
                                    value_t lambda_abs,          //
                                    value_t lambda_sc,           //
                                    index_t * schedule_counter = nullptr)
{
	
    index_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    curandState localstate = state[idx];

    if (schedule_counter)
    {
        if (idx == 0) *schedule_counter = gridDim.x * blockDim.x;
        __syncthreads();

        index_t ray_id = idx;
        while (ray_id < num_rays)
        {
            rayTracingLoop<value_t, index_t, model_t, override>(
                startpoints, directions, hitpoints, results, angles_initial, ray_id, model, alpha_crit,
                status_codes, localstate, lambda_abs, lambda_sc);
            ray_id = atomicAdd(schedule_counter, 1);
        }
    }
    else
    {
        index_t num_threads = gridDim.x * blockDim.x;

        for (index_t ray_id = idx; ray_id < num_rays; ray_id += num_threads)
        {
            rayTracingLoop<value_t, index_t, model_t, override>(
                startpoints, directions, hitpoints, results, angles_initial, ray_id, model, alpha_crit,
                status_codes, localstate, lambda_abs, lambda_sc);
        }
    }
    state[idx] = localstate;
}

} // namespace ALGO
