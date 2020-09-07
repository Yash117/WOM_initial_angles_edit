/**
 * @file    WOMSimulation.h
 * @brief   Contains the class allowing a full simlation of the WOM.
 */
#pragma once

#include <iterator>
#include <memory>

#include <cuda.h>

#include "EllipticModel.h"
#include "QuadraticSplineModel.h"
#include "Simulation.h"
#include "SortKernels.h"

namespace ALGO
{

/**
 * @brief   Class which allows a full simulation run. Note that it overrides the startpoints and
 *          directions of each Photon. So the simulation can only be run ONCE.
 *
 * @author  Ronja Schnur (rschnur@students.uni-mainz.de)
 */
template <typename value_t, typename index_t>
class WOMSimulation final : public Simulation<value_t, index_t, EllipticModel<value_t, index_t>>
{
    // Data
private:
    std::unique_ptr<QuadraticSplineModel<value_t, index_t>> _spline_model;
    std::unique_ptr<EllipticModel<value_t, index_t>> _elliptic_model;

    // Constructors
public:
    WOMSimulation(value_t n1,                                      //
                  value_t n2,                                      //
                  value_t lambda_abs,                              //
                  value_t lambda_sc,                               //
                  value_t alg_radius_inner,                        //
                  value_t alg_radius_outer,                        //
                  std::shared_ptr<My::Spline<value_t>> alg_spline, //
                  value_t alg_length,                              //
                  value_t elliptic_length,                         //
                  value_t elliptic_a_inner,                        //
                  value_t elliptic_b_inner,                        //
                  value_t elliptic_alpha_inner,                    //
                  value_t elliptic_x_inner,                        //
                  value_t elliptic_y_inner,                        //
                  value_t elliptic_a_outer,                        //
                  value_t elliptic_b_outer,                        //
                  value_t elliptic_alpha_outer,                    //
                  value_t elliptic_x_outer,                        //
                  value_t elliptic_y_outer,                        //
                  std::shared_ptr<PhotonGenerator<value_t, index_t>> generator)
        : Simulation<value_t, index_t, EllipticModel<value_t, index_t>>(n1,         //
                                                                        n2,         //
                                                                        lambda_abs, //
                                                                        lambda_sc,  //
                                                                        generator)
    {
        makeEModel(elliptic_length,      //
                   elliptic_a_inner,     //
                   elliptic_b_inner,     //
                   elliptic_alpha_inner, //
                   elliptic_x_inner,     //
                   elliptic_y_inner,     //
                   elliptic_a_outer,     //
                   elliptic_b_outer,     //
                   elliptic_alpha_outer, //
                   elliptic_x_outer,     //
                   elliptic_y_outer);
        makeQSModel(alg_radius_inner, alg_radius_outer);
        _spline_model->configure(alg_spline, alg_length);
    }

private:
    QuadraticSplineModel<value_t, index_t> * makeQSModel(value_t radius_inner, value_t radius_outer)
    {
        _spline_model =
            std::make_unique<QuadraticSplineModel<value_t, index_t>>(radius_inner, radius_outer);
        return _spline_model.get();
    }

    EllipticModel<value_t, index_t> * makeEModel(value_t length,      //
                                                 value_t a_inner,     //
                                                 value_t b_inner,     //
                                                 value_t alpha_inner, //
                                                 value_t x_inner,     //
                                                 value_t y_inner,     //
                                                 value_t a_outer,     //
                                                 value_t b_outer,     //
                                                 value_t alpha_outer, //
                                                 value_t x_outer,     //
                                                 value_t y_outer)
    {
        _elliptic_model = std::make_unique<EllipticModel<value_t, index_t>>(
            length,                                          //
            a_inner, b_inner, alpha_inner, x_inner, y_inner, //
            a_outer, b_outer, alpha_outer, x_outer, y_outer);

        this->setModel(_elliptic_model.get());

        return _elliptic_model.get();
    }

private:
    template <typename func_t>
    index_t sort(index_t N,                   //
                 Vec3<value_t> * startpoints, //
                 Vec3<value_t> * directions,  //
                 func_t pred)
    {
        partitionZipped<func_t, int8_t, index_t, Vec3<value_t> *, Vec3<value_t> *>
            <<<1, THREADS>>>(pred,                    //
                             this->_status_codes_dev, //
                             N,                       //
                             this->_sum_dev,          //
                             startpoints, directions);
        CUERR;

        index_t result{0};
        cudaMemcpy(&result, this->_sum_dev, sizeof(index_t), D2H); // copy result value from gpu
        CUERR;

        DEBUG(std::cout << "newN: " << result << " oldN: " << N << "\n" << std::flush;)

        return result;
    }

public:
    index_t simulate(index_t * failed = nullptr) override
    {
        index_t N = this->_generator->N();

        Vec3<value_t> * startpoints = this->_generator->startpointsDevice();
        Vec3<value_t> * directions = this->_generator->directionsDevice();

        bool tube = true;
        while (N > 0)
        {
            DEBUG_OUT(N);
            if (tube)
            {
                DEBUG(TIMERSTART(CALC_TUBE);)
                startRayTracingLoop<value_t, index_t, EllipticModel<value_t, index_t>, true>
                    <<<BLOCKS, THREADS>>>(N,                       //
                                          startpoints,             //
                                          directions,              //
                                          this->_hitpoints_dev,    //
                                          this->_results_dev,      
										  this->_angles_init_dev,
                                          *_elliptic_model.get(),  //
                                          this->_alpha_crit,       //
                                          this->_state,            //
                                          this->_status_codes_dev, //
                                          this->_lambda_abs,       //
                                          this->_lambda_sc,        //
                                          this->_sum_dev);
                CUERR;
                DEBUG(TIMERSTOP(CALC_TUBE);)

                DEBUG(TIMERSTART(SORT_TUBE);)
                N = sort(N, startpoints, directions, [] __device__ (uint8_t key) {
                    return (key % 64) == static_cast<uint8_t>(HitResult::Success);
                });
                DEBUG(TIMERSTOP(SORT_TUBE);)
            }
            else
            {
                DEBUG(TIMERSTART(CALC_ALG);)
                startRayTracingLoop<value_t, index_t, QuadraticSplineModel<value_t, index_t>, true>
                    <<<BLOCKS, THREADS>>>(N,                       //
                                          startpoints,             //
                                          directions,              //
                                          this->_hitpoints_dev,    //
                                          this->_results_dev, 
										  this->_angles_init_dev,     
                                          *_spline_model.get(),    //
                                          this->_alpha_crit,       //
                                          this->_state,            //
                                          this->_status_codes_dev, //
                                          this->_lambda_abs,       //
                                          this->_lambda_sc,        //
                                          this->_sum_dev);
                CUERR;
                DEBUG(TIMERSTOP(CALC_ALG);)

                DEBUG(TIMERSTART(SORT_ALG);)
                N = sort(N, startpoints, directions, [] __device__ (uint8_t key) {
                    return (key % 64) == static_cast<uint8_t>(HitResult::Failed);
                }); // sort for returned photons
                DEBUG(TIMERSTOP(SORT_ALG);)
            }
            tube = !tube;
        }

        startComputeSum<<<BLOCKS, THREADS>>>(this->_status_codes_dev, //
                                             this->_sum_dev,          //
                                             this->_generator->N(),
                                             static_cast<int8_t>(HitResult::Success));
        CUERR;

        index_t result{0};
        cudaMemcpy(&result, this->_sum_dev, sizeof(index_t), D2H); // copy result value from gpu
        CUERR;

        if (failed)
        {
            startComputeSum<<<BLOCKS, THREADS>>>(this->_status_codes_dev, //
                                                 this->_sum_dev,          //
                                                 this->_generator->N(),
                                                 static_cast<int8_t>(HitResult::Failed));
            CUERR;

            cudaMemcpy(failed, this->_sum_dev, sizeof(index_t), D2H); // copy result value from gpu
            CUERR;
        }

        return result;
    }
};

} // namespace ALGO
