/**
 * @file    PhotonWriter.h
 * @brief   Contains @ref PhotonWriter class.
 */
#pragma once

#include <fstream>
#include <iomanip>
#include <vector>

#include "HPCHelpers.h"

#include "Intervall.h"
#include "Spline.h"
#include "Vec3.h"
#include "Exception.h"

namespace ALGO
{

/**
 * @brief   Generates histograms for multiple simulation Batches.
 *
 * @tparam  value_t     The floating point type to operate on.
 * @tparam  index_t     The index type to operate on.
 *
 * @author  Ronja Schnur (rschnur@students.uni-mainz.de)
 */
template <typename value_t, typename index_t> class PhotonHistogrammer
{
    // Data
private:
    std::string _filename;

    index_t _N;
    Vec3<value_t> *_hitpoints_dev, *_results_dev;
    int8_t * _status_code_dev;

    My::Intervall<value_t> _intervall;

    value_t _delta;
    std::vector<std::vector<index_t>> _data;
    std::vector<index_t> _results;
    std::vector<std::vector<value_t>> _spline_data;
    std::vector<value_t> _spaces_x;

    // Constructors
public:
    /**
     * @brief   Create new instance.
     *
     * @param   filename        The file where the data will be stored.
     * @param   N               How many photons are on the GPU
     * @param   hitpoints_dev   The device pointer to hitpoints.
     * @param   results_dev     The device pointer to results.
     * @param   status_code_dev The device pointer to status_code_dev
     * @param   cartesian       Whether to convert the result direction into cartesian coordinates.
     */
    PhotonHistogrammer(std::string filename, My::Intervall<value_t> intervall, size_t num_spaces,
                       index_t N, Vec3<value_t> * hitpoints_dev, Vec3<value_t> * results_dev,
                       int8_t * status_code_dev)
        : _filename{filename}, _N{N}, _hitpoints_dev{hitpoints_dev}, _results_dev{results_dev},
          _status_code_dev{status_code_dev},
          _intervall{intervall}, _delta{(intervall._end - intervall._start) / num_spaces},
          _spaces_x(num_spaces + 1)
    {
        for (value_t i = 0; i < _spaces_x.size(); ++i)
            _spaces_x[i] = _intervall._start + i * _delta; // equally distributed x-values
    }

    // Methods
public:
    /**
     * @brief   Loads photons from GPU and generates an histogram.
     * @param   result  The computed result from the GPU.
     */
    void load(index_t result, std::shared_ptr<My::Spline<value_t>> spline)
    {
        std::vector<Vec3<value_t>> hitpoints(_N);

        // Copy data from device
        cudaMemcpy(hitpoints.data(), _hitpoints_dev, this->_N * sizeof(Vec3<value_t>), D2H);
        CUERR

        std::vector<index_t> data(_spaces_x.size() + 1);

        for (Vec3<value_t> & v : hitpoints)
        {
            if (v.z == _intervall._start)
                data[0] += 1;
            else if (v.z == _intervall._end)
                data[_spaces_x.size()] += 1;
            else
            {
                for (size_t i = 0; i < _spaces_x.size() - 2; ++i)
                {
                    if (_spaces_x[i] <= v.z && _spaces_x[i + 1] > v.z) data[i + 1] += 1;
                }
            }
        }

        std::vector<value_t> spline_data(spline->numKnots());

        for (size_t i = 0; i < spline->numKnots(); ++i) spline_data[i] = spline->knotYData()[i];

        _data.push_back(data);
        _results.push_back(result);
        _spline_data.push_back(spline_data);
    }

    /**
     * @brief   Writes data into file.
     *
     * @param   print   Whether to print output or not.
     */
    void write(bool print = true)
    {
        if (print) std::cout << "> Writing to file '" << _filename << "' ... ";

        std::fstream fs;
        fs.open(_filename, std::ios_base::out);
        if (!fs) throw FileUnwriteableException(_filename);

        // Each row contains:
        // num_back, fst_space, snd_space, ... lst_space, num_end, result # spline_y
        fs << "# ";
        for (size_t i = 0; i < _spaces_x.size() - 1; ++i)
            fs << std::setw(10) << std::setprecision(5) << std::right << _spaces_x[i] << ' ';
        fs << "\n";

        for (size_t i = 0; i < _data.size(); ++i)
        {
            for (index_t entry : _data[i]) fs << std::setw(10) << std::right << entry << ' ';
            fs << "# " << std::setw(10) << std::right << _results[i] << " # ";
            for (value_t spline_entry : _spline_data[i])
                fs << std::setw(10) << std::setprecision(6) << std::right << spline_entry << ' ';
            fs << "\n";
        }

        if (print) std::cout << "Done.\n";
    }
};

} // namespace ALGO
