/**
 * @file    Simulation.h
 * @brief   File containing all classes for the @ref Simulation class.
 */

#pragma once

#include <curand.h>
#include <curand_kernel.h>
#include <fstream>
#include <iomanip> //stream manipulators
#include <iostream>
#include <math.h>
#include <memory>
#include <regex>
#include <vector>

#include "SimplexFunction.h"
#include "Spline.h"

#include "DefaultPhotonGenerator.h"
#include "HPCHelpers.h"
#include "Model.h"
#include "PhotonGenerator.h"
#include "RayTracingKernels.h"
#include "SumKernels.h"
#include "Tools.h"
#include "Vec3.h"

#define EPS 1E-3

namespace ALGO
{

/**
 * @brief   Simulates an amout of photons through the given Model.
 *
 * @tparam  value_t     The floating point type to operate on.
 * @tparam  index_t     The index type to operate on. (N should fit in)
 * @tparam  model_t     The corresponding Model class. Should be either @ref
 *                      QuadraticSplineModel or EllipticModel (not implemented (yet))
 *
 * @author  Ronja Schnur (rschnur@students.uni-mainz.de) and
 *          Florian Thomas (flthomas@students.uni-mainz.de)
 */
template <typename value_t, typename index_t, typename model_t> class Simulation
{
    // Data
protected:
    model_t * _model{nullptr};
    std::shared_ptr<PhotonGenerator<value_t, index_t>> _generator;

    value_t _alpha_crit;
    value_t _lambda_abs;
    value_t _lambda_sc;
    value_t _n1, _n2;

    // device result data
    Vec3<value_t> * _hitpoints_dev{nullptr};
    Vec3<value_t> * _results_dev{nullptr}; // phi, theta and distance
    int8_t * _status_codes_dev{nullptr};

    index_t * _sum_dev;

    // device RNG states
    curandState * _state{nullptr};

    // Properties
protected:
    void setModel(model_t * model)
    {
        if (!_model && model)
        {
            _model = model;
            init();
        }
    }

public:
    /**
     * @brief Access the hitpoints pointer on the device.
     * @return The pointer.
     */
    Vec3<value_t> * hitpointsDevice() const noexcept { return _hitpoints_dev; }

    /**
     * @brief Access the results pointer on the device.
     * @return The pointer.
     */
    Vec3<value_t> * resultsDevice() const noexcept { return _results_dev; }

    [[deprecated]] int8_t * pmtHitDevice() const noexcept { return _status_codes_dev; }

    /**
     * @brief Access the status_codes pointer on the device.
     * @return The pointer.
     */
    int8_t * statusDevice() const noexcept { return _status_codes_dev; }

    // Constructors
protected:
    Simulation(value_t n1,                                                           //
               value_t n2,                                                           //
               value_t lambda_abs,                                                   //
               value_t lambda_sc,                                                    //
               const std::shared_ptr<PhotonGenerator<value_t, index_t>> & generator) //
        : _generator{generator}, _n1(n1), _n2(n2), _lambda_abs(lambda_abs), _lambda_sc(lambda_sc)
    {}

public:
    /**
     * @brief   Create new @ref Simulation.
     *
     * @param   model           The model with which intersection is performed.
     * @param   n1              The index of refraction arround the model.
     * @param   n2              The index of refraction of the model.
     * @param   lambda_abs      lambda abs
     * @param   lambda_sc       The scattering offset.
     * @param   radius_outer    The outer radius of the ALG. Usually spline(0, intervall_start)
     * @param   generator       The @ref PhotonGenerator instance which generates Photons for the
     *                          simulation.
     */
    Simulation(model_t * model,    // should be configured
               value_t n1,         //
               value_t n2,         //
               value_t lambda_abs, //
               value_t lambda_sc,  //
               const std::shared_ptr<PhotonGenerator<value_t, index_t>> & generator) //
        : _model{model}, _generator{generator}, _n1(n1), _n2(n2), _lambda_abs(lambda_abs),
          _lambda_sc(lambda_sc)
    {
        init();
    }

    ~Simulation() { freeStorage(); }

    Simulation(const Simulation & other) = delete;

    Simulation & operator=(const Simulation & other) = delete;

    Simulation(Simulation && other); // TODO: Implement

    // Properties
public:
    index_t N() const noexcept { return _generator->N(); }

    // METHODS
private:
    void init()
    {
        std::string output = "> Initialize Simulation...";
        DEBUG_OUT(output);

        _generator->generatePhotons();
        _alpha_crit = asin(std::min(_n2, _n1) / std::max(_n1, _n2)); // n2 / n1

        _state = _generator->stateDevice();

        cudaMalloc(&_hitpoints_dev, _generator->N() * sizeof(Vec3<value_t>));
        CUERR

        cudaMalloc(&_results_dev, _generator->N() * sizeof(Vec3<value_t>));
        CUERR

        cudaMalloc(&_status_codes_dev, _generator->N() * sizeof(int8_t));
        CUERR

        cudaMalloc(&_sum_dev, sizeof(index_t));
        CUERR

        _model->preCompute(_generator->N(), _generator->startpointsDevice(),
                           _generator->directionsDevice(), _state);
    }

public:
    /**
     * @brief   Runs the simulation with existing photons and calculates hit result
     * @return  The number of successful photons.
     */
    virtual index_t simulate(index_t * failed = nullptr)
    {
        index_t N = _generator->N();
        Vec3<value_t> * startpoints = _generator->startpointsDevice();
        Vec3<value_t> * directions = _generator->directionsDevice();

        // Invoke Kernel
        startRayTracingLoop<<<BLOCKS, THREADS>>>(
            N, startpoints, directions, _hitpoints_dev, _results_dev, *_model, _alpha_crit, _state,
            _status_codes_dev, _lambda_abs, _lambda_sc, _sum_dev);

        index_t result_arr[4];

        startComputeSum<<<BLOCKS, THREADS>>>(_status_codes_dev, _sum_dev, N,
                                             static_cast<int8_t>(HitResult::Success));
        cudaMemcpy(result_arr, _sum_dev, sizeof(index_t), D2H); // copy result value from gpu

        startComputeSum<<<BLOCKS, THREADS>>>(_status_codes_dev, _sum_dev, N,
                                             static_cast<int8_t>(HitResult::ScatteredSuccess));
        cudaMemcpy(result_arr + 1, _sum_dev, sizeof(index_t), D2H); // copy result value from gpu

        startComputeSum<<<BLOCKS, THREADS>>>(_status_codes_dev, _sum_dev, N,
                                             static_cast<int8_t>(HitResult::SuccessEnd));
        cudaMemcpy(result_arr + 2, _sum_dev, sizeof(index_t), D2H); // copy result value from gpu

        startComputeSum<<<BLOCKS, THREADS>>>(_status_codes_dev, _sum_dev, N,
                                             static_cast<int8_t>(HitResult::ScatteredSuccessEnd));
        cudaMemcpy(result_arr + 3, _sum_dev, sizeof(index_t), D2H); // copy result value from gpu


        index_t result = result_arr[0] + result_arr[1] + result_arr[2] + result_arr[3];

        VERBOSE_OUT(result);

        if (failed)
        {
            startComputeSum<<<BLOCKS, THREADS>>>(_status_codes_dev, _sum_dev, N,
                                                 static_cast<int8_t>(HitResult::Failed));

            cudaMemcpy(failed, _sum_dev, sizeof(index_t), D2H); // copy result value from gpu
            CUERR;

            VERBOSE_OUT(*failed);
        }

        return result;
    }

    /**
     * @brief   Runs a simulation and copies result data into given arrays. Used for Python
     *          Interface.
     *
     * @param   hitpoints       Array of length 3*N() x0, y0, z0, x1, ...
     * @param   results         Array of length 3*N() x0, y0, z0, x1, ...
     * @param   status_codes    Array of length N()
     * @param   result          Array of length 1.
     */
    void simulateData(value_t * hitpoints,    // incoming numpy float array
                      value_t * results,      // incoming numpy float array
                      int32_t * status_codes, // incoming numpy int8_t array
                      index_t * result)       // incoming uint32
    {
        *result = this->simulate();

        cudaMemcpy(reinterpret_cast<Vec3<value_t> *>(hitpoints), _hitpoints_dev,
                   sizeof(Vec3<value_t>) * _generator->N(), D2H);
        CUERR;
        cudaMemcpy(reinterpret_cast<Vec3<value_t> *>(results), _results_dev,
                   sizeof(Vec3<value_t>) * _generator->N(), D2H);
        CUERR;

        std::vector<int8_t> codes(_generator->N());
        cudaMemcpy(codes.data(), _status_codes_dev, sizeof(int8_t) * _generator->N(), D2H);
        CUERR;

        for (size_t i = 0; i < codes.size(); ++i) //
            status_codes[i] = static_cast<int32_t>(codes[i]);
    }

    void simulateHist(index_t * result,        //
                      index_t * hist_data,     //
                      size_t num_spaces,       //
                      value_t intervall_start, //
                      value_t intervall_end,   //
                      bool include_start = false)
    {
        *result = this->simulate();

        std::vector<Vec3<value_t>> hitpoints(N());
        cudaMemcpy(hitpoints.data(), _hitpoints_dev, N() * sizeof(Vec3<value_t>), D2H);
        CUERR

        for (size_t i = 0; i < num_spaces; ++i) hist_data[i] = 0;

        value_t delta = (intervall_end - intervall_start) / num_spaces;

        for (size_t j = 0; j < hitpoints.size(); ++j)
        {
            for (size_t i = 0; i < num_spaces; ++i)
            {
                value_t sec_start{intervall_start + delta * i};
                value_t sec_end{intervall_start + delta * (i + 1)};

                bool ic{(include_start ? true : hitpoints[j].z > intervall_start)};

                // test if in [sec_start, sec_end[
                if (((sec_start <= hitpoints[j].z) && (hitpoints[j].z < sec_end)) && ic)
                {
                    hist_data[i] += 1;
                    break;
                }
            }
        }
    }

private:
    template <typename ptr_t> void freeStorage(ptr_t * ptr) noexcept
    {
        if (ptr)
        {
            cudaFree(ptr);
            CUERR
            ptr = nullptr;
        }
    }

    void freeStorage() noexcept
    {
        freeStorage(_hitpoints_dev);
        freeStorage(_results_dev);
        freeStorage(_status_codes_dev);
        freeStorage(_sum_dev);
    }
};

} // namespace ALGO