/**
 * @file    PhotonWriter.h
 * @brief   Contains @ref PhotonWriter class.
 */
#pragma once

#include <fstream>
#include <iomanip>
#include <vector>

#include "HPCHelpers.h"

#include "Vec3.h"

namespace ALGO
{

/**
 * @brief   Writes computed PhotonData into file.
 *
 * @tparam  value_t     The floating point type to operate on.
 * @tparam  index_t     The index type to operate on.
 *
 * @author  Ronja Schnur (rschnur@students.uni-mainz.de)
 */
template <typename value_t, typename index_t> class PhotonWriter
{
    // Data
private:
    std::string _filename;

    index_t _N;
//  Vec3<value_t> *_hitpoints_dev, *_results_dev;
    Vec3<value_t> *_hitpoints_dev, *_results_dev, *_angles_init_dev;
    int8_t * _status_code_dev;

    bool _cartesian;

    // Constructors
public:
    /**
     * @brief   Create new instance.
     *
     * @param   filename             The file where the data will be stored.
     * @param   N                    How many photons are on the GPU
     * @param   hitpoints_dev        The device pointer to hitpoints.
     * @param   results_dev          The device pointer to results.
     * @param   _angles_init_dev     The device pointer to initial emission angles
     * @param   status_code_dev      The device pointer to status_code_dev
     * @param   cartesian            Whether to convert the result direction into cartesian coordinates.
     */
/*  PhotonWriter(std::string filename, index_t N, Vec3<value_t> * hitpoints_dev,
                 Vec3<value_t> * results_dev, int8_t * status_code_dev, bool cartesian = true)
        : _filename{filename}, _N{N}, _hitpoints_dev{hitpoints_dev}, _results_dev{results_dev},
          _status_code_dev{status_code_dev}, _cartesian{cartesian}
    {}
    */

    PhotonWriter(std::string filename, index_t N, Vec3<value_t> * hitpoints_dev,
                 Vec3<value_t> * results_dev, Vec3<value_t> * angles_init_dev, int8_t * status_code_dev, bool cartesian = true)
        : _filename{filename}, _N{N}, _hitpoints_dev{hitpoints_dev}, _results_dev{results_dev}, _angles_init{angles_init},
          _status_code_dev{status_code_dev}, _cartesian{cartesian}
    {}
    
    // Methods
public:
    /**
     * @brief   Writes data into file.
     *
     * @param   print   Whether to print output or not.
     */
    void write(bool print = true)
    {
      //std::vector<Vec3<value_t>> hitpoints(_N), results(_N);
        std::vector<Vec3<value_t>> hitpoints(_N), results(_N), angles_initial(_N);
        std::vector<uint8_t> status_codes(_N);

        if (print) std::cout << "> Writing to file '" << _filename << "' ... ";

        // Copy data from device
        cudaMemcpy(hitpoints.data(), _hitpoints_dev, this->_N * sizeof(Vec3<value_t>), D2H);
        CUERR

        cudaMemcpy(results.data(), _results_dev, this->_N * sizeof(Vec3<value_t>), D2H);
        CUERR
        cudaMemcpy(angles_initial.data(), _angles_init_dev, this->_N * sizeof(Vec3<value_t>), D2H);
        CUERR
        
        cudaMemcpy(status_codes.data(), _status_code_dev, this->_N * sizeof(int8_t), D2H);
        CUERR

        std::fstream fs;
        fs.open(_filename, std::ios_base::out);
        if (!fs) throw FileUnwriteableException(_filename);

        fs << (_cartesian ? 8 : 7) << ' ' << _cartesian << '\n'; // header

        for (size_t i = 0; i < _N; ++i)
        {
            fs << std::setw(10) << std::setprecision(5) << std::right << hitpoints[i].x << ' '
               << std::setw(10) << std::setprecision(5) << std::right << hitpoints[i].y << ' '
               << std::setw(10) << std::setprecision(5) << std::right << hitpoints[i].z << ' ';
            if (_cartesian)
            {
                Vec3<value_t> cart;
                spherical_to_cartesian(value_t(1), results[i].x, results[i].y, cart);
                fs << std::setw(10) << std::setprecision(5) << std::right << cart.x << ' '
                   << std::setw(10) << std::setprecision(5) << std::right << cart.y << ' '
                   << std::setw(10) << std::setprecision(5) << std::right << cart.z << ' '
                   << std::setw(10) << std::setprecision(5) << std::right << results[i].z << ' ';
                   
                Vec3<value_t> cart2;
                spherical_to_cartesian(value_t(1), angles_initial[i].x, angles_initial[i].y, cart2);
                fs << std::setw(10) << std::setprecision(5) << std::right << cart2.x << ' '
                   << std::setw(10) << std::setprecision(5) << std::right << cart2.y << ' '
                   << std::setw(10) << std::setprecision(5) << std::right << cart2.z << ' '
                   << std::setw(10) << std::setprecision(5) << std::right << angles_initial[i].z << ' ';  
                
            }
            else
            {
                fs << std::setw(10) << std::setprecision(5) << std::right << results[i].x << ' '
                   << std::setw(10) << std::setprecision(5) << std::right << results[i].y << ' '
                   << std::setw(10) << std::setprecision(5) << std::right << results[i].z << ' ';
                
                
                fs << std::setw(10) << std::setprecision(5) << std::right << angles_initial[i].x << ' '
                   << std::setw(10) << std::setprecision(5) << std::right << angles_initial[i].y << ' '
                   << std::setw(10) << std::setprecision(5) << std::right << angles_initial[i].z << ' ';
                
            }
            fs << static_cast<int>(status_codes[i]) << '\n';
        }

        if (print) std::cout << "Done.\n";
    }
};

} // namespace ALGO
