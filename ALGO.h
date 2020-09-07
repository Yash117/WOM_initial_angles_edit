/**
 * @dir     ALGO
 * @brief   The directory containg all files and classes within @ref ALGO namespace.
 */

/**
 * @file    ALGO.h
 * @brief   File containing the main Adiabatic Light Guide Optimizer class.
 */

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <regex>

#include <external/cxxopts/cxxopts.hpp>

#include "My.h"

#include "CurvatureSpline.h"
#include "DefaultPhotonGenerator.h"
#include "EllipticModel.h"
#include "Exception.h"
#include "PhotonGenerator.h"
#include "PhotonHistogrammer.h"
#include "PhotonLoader.h"
#include "PhotonWriter.h"
#include "QuadraticSplineModel.h"
#include "RandomKernels.h"
#include "RayTracingKernels.h"
#include "SplineSimulation.h"
#include "TubePhotonGenerator.h"
#include "Vec3.h"
#include "WOMSimulation.h"

// #define F_SPLINE

/**
 * @namespace   ALGO
 * @brief       Namespace contains every single class and function of ALGO.
 */
namespace ALGO
{

/**
 * @enum    ALGOMode
 * @brief   Enumaration specifiying the different operation modes of ALGO.
 */
enum class ALGOMode : int
{
    Optimization = 0,
    Info = 1,
    FullSimulation = 2,
    Simulation = 3,
};

/**
 * @enum    ALGOGenerator
 * @brief   Enumaration specifiying the different PhotonGenerators
 */
enum class ALGOGenerator : int
{
    Default = 0,
    Tube = 1
};

enum class ALGOModel : int
{
    ALG_QS = 0,
    Tube_QS = 2,
    Tube_EM = 3
};

/**
 * @brief   Class handling the main program workflow and command line input.
 * @author  Ronja Schnur (rschnur@students.uni-mainz.de)
 */
class ALGO
{
    // Data
private:
    std::unique_ptr<cxxopts::Options> _options{nullptr};

    ALGOMode _mode;
    ALGOGenerator _generator;
    ALGOModel _model;

    uint64_t _N;
    double _n1, _n2;
    double _lambda_abs, _lambda_sc;
    double _length;
    double _simplex_lambda, _simplex_tol;
    bool _precision;
    std::string _input_file, _output_file;
    std::vector<double> _model_arg;

    // Constructors
public:
    /**
     * @brief   Construct new ALGO instance. It requires the command line args to perform.
     *
     * @param   argc    Number of arguments in argv.
     * @param   argv    The command line arguments passed to the program.
     */
    ALGO(int argc, char ** argv);

    // METHODS
private:
    /**
     * @brief   Prints programm title to stdout.
     */
    void title() const noexcept;

    /**
     * @brief   Prints help to stdout.
     */
    void help() const noexcept;

    /**
     * @brief   Initializes a spline on given command line arguments with given function.
     *
     * @tparam  value_t     The floating point type to operate on.
     * @param   f           The function that will be approximated with the spline.
     *
     * @return  Pointer to the new created Spline object.
     */
    template <typename value_t>
    std::shared_ptr<My::Spline<value_t>> initSpline(const std::function<value_t(value_t)> & f) const
    {
        My::Intervall<value_t> intervall{value_t(0), value_t(_length)};

        std::shared_ptr<My::Spline<value_t>> spline =
            std::make_shared<My::QuadraticSpline<value_t>>(size_t(_model_arg[0]), intervall);

        for (size_t i = 0; i < spline->numKnots(); ++i)
            spline->specify(i, f(spline->knotXData()[i]));

        spline->generate();

#ifdef F_SPLINE
        spline = std::make_shared<My::QuadraticSpline<value_t>>(
            std::initializer_list<value_t>{
                -100000.0,          //
                0.8807780707987681, //
                1.7615561415975363, //
                2.6423342123963045, //
                3.5231122831950725, //
                4.403890353993841,  //
                5.284668424792609,  //
                6.165446495591377,  //
                7.046224566390145   //
            },
            std::initializer_list<value_t>{0, 0, 0, 0, 0, 0, 0, 0, 0}, // unknown
            std::initializer_list<value_t>{
                -0.9741290863866998,  -0.007266490218598284, 20.25003501622625,  //
                -0.7636471537128267,  -0.3780422314155787,   20.41332058724148,  //
                -0.4337324652241188,  -1.5403687228366214,   21.437072271993607, //
                -0.11094618157230962, -3.246187204007488,    23.69074353846145,  //
                0.1553114504221099,   -5.122298271555433,    26.995618511819643, //
                0.3544949332686686,   -6.876662709121149,    30.858632823812385, //
                0.49796568799490126,  -8.393053443886922,    34.865443941644756, //
                0.6010422686399557,   -9.664079729718107,    38.78366622153596,  //
                0.6718042706815124,   -10.661289644022439,   42.29694871954546   //
            });
#endif

        return spline;
    }

    /**
     * @brief   Intializes spline depending on given model.
     * @return  nullptr if not required else spline.
     */
    template <typename value_t> std::shared_ptr<My::Spline<value_t>> initSpline() const
    {
        std::shared_ptr<My::Spline<value_t>> spline{nullptr};

        value_t radius_begin = value_t(_model_arg[2]);

        auto falke = [&](value_t x) {                          // Hyperbolic function
            value_t e{value_t(1.4502)},                        //
                d{radius_begin / (e * e - 1)},                 //
                t{radius_begin + d -                           //
                  std::sqrt(d * d + 1 / (e * e - 1) * x * x)}; //
            return t * t;
        };

        value_t radius_end = std::sqrt(falke(_length));

        switch (_model)
        {
            case ALGOModel::ALG_QS:
                // spline = initSpline<value_t>(falke);

                spline = initSpline<value_t>([&](value_t x) {
                    auto r =
                        (radius_end - radius_begin) / (_length * _length) * x * x + radius_begin;
                    // auto r = radius_begin;
                    return r * r;
                });
                break;
            case ALGOModel::Tube_QS:
                spline = initSpline<value_t>(
                    [this](value_t x) { return value_t(_model_arg[2]) * value_t(_model_arg[2]); });
                break;
            default: break;
        }
        return spline;
    }

    /**
     * @brief   Initialized a @ref PhotonGenerator depending on given command line arguments.
     *
     * @tparam  value_t     The floating point type to operate on.
     * @tparam  index_t     The index type to operate on
     *
     * @return Pointer to the generator object.
     */
    template <typename value_t, typename index_t>
    std::shared_ptr<PhotonGenerator<value_t, index_t>> initGenerator() const
    {
        if (_input_file != "")
        {
            try
            {
                return std::make_shared<PhotonLoader<value_t, index_t>>(_input_file);
            }
            catch (const FileIOException & e)
            {
                std::cerr << e.what();
            }
        }

        switch (_generator)
        {
            case ALGOGenerator::Default:
                return std::make_shared<DefaultPhotonGenerator<value_t, index_t>>(index_t(_N), 0.5, 0.0, 5.5, 30.0); //TODO remove hardcoded vals maybe?

            case ALGOGenerator::Tube:
                return std::make_shared<TubePhotonGenerator<value_t, index_t>>(
                    index_t(_N), value_t(_model_arg[1]), value_t(_model_arg[2]));
        }
    }

    /**
     * @brief   Initializes a @ref QuadraticSplineModel depending on given command line
     *          arguments.
     *
     * @tparam  value_t     The floating point type to operate on.
     * @param   spline      The QuadraticSpline which represents the ALG form.
     *
     * @return Pointer to the constructed object.
     */
    template <typename value_t, typename index_t>
    std::shared_ptr<QuadraticSplineModel<value_t, index_t>>
    initQuadraticModel(std::shared_ptr<My::Spline<value_t>> & spline) const
    {
        auto model = std::make_shared<QuadraticSplineModel<value_t, index_t>>(
            value_t(_model_arg[1]), value_t(_model_arg[2]));

        model->configure(spline, value_t(_length));
        return model;
    }

    /**
     * @brief   Performs a Nieder-Meld Simplex Optimization on a @ref SplineSimulation
     *          depending on the given command line arguments.
     *
     * @tparam  value_t     The floating point type to operate on.
     * @tparam  index_t     The index type to operate on.
     */
    template <typename value_t, typename index_t> void performOptimization() const
    {
        if (_model == ALGOModel::Tube_EM || _model == ALGOModel::Tube_QS)
        {
            title();
            std::cout << "\nTube simulation only in mode: Simulate.\n";
            return;
        }

        auto spline = initSpline<value_t>();
        auto splinel = initSpline<value_t>();

        auto lspline = std::static_pointer_cast<My::Spline<value_t>>(spline);
        auto model = initQuadraticModel<value_t, index_t>(lspline);
        auto generator = initGenerator<value_t, index_t>();
        auto simulation = std::make_shared<
            SplineSimulation<value_t, index_t, QuadraticSplineModel<value_t, index_t>>>(
            model.get(), value_t(_n1), value_t(_n2), value_t(_lambda_abs), value_t(_lambda_sc),
            generator);

        std::shared_ptr<SplineSimulationArgument<value_t>> arg =
            std::make_shared<SplineSimulationArgument<value_t>>(spline, value_t(_length));

        arg->disableParameter(0);               // remove first knot
        arg->disableParameter(1);               // remove first knot
        arg->disableParameter(2);               // remove first knot
        arg->disableParameter(3);               // remove first knot
        arg->disableParameter(4);               // remove first knot
        arg->disableParameter(arg->maxN() - 2); // remove last knot
        arg->disableParameter(arg->maxN() - 1); // remove length

        My::SimplexSolver<value_t> solver(simulation, arg, value_t(_simplex_lambda),
                                          value_t(_simplex_tol));

        title();
        std::cout << "\n> Initial spline: " << lspline;

#ifdef _DEBUG
        std::cout << "Run Simplex-Optimization ... (Max.: 200 Iterations).\n";
        auto result = solver.solve(false);

        simulation->simulate();
#else
        auto result = solver.solve();
#endif

        auto res = std::dynamic_pointer_cast<SplineSimulationArgument<value_t>>(result._first);
        std::cout << "> Optimized spline: " << res->spline() << "> Length: " << res->length()
                  << "\n";

        simulation->preCompute(result._first);
        performSimulation(simulation.get(), generator.get());
    }

    /**
     * @brief   Performs a simulation run and writes data if given.
     */
    template <typename value_t, typename index_t, typename model_t>
    void performSimulation(Simulation<value_t, index_t, model_t> * simulation,
                           PhotonGenerator<value_t, index_t> * generator) const
    {
        index_t failed;
        index_t res = simulation->simulate(&failed);

        std::cout << "> Returned Photons: " << failed << "\n";
        std::cout << "> Detected Photons: " << res << "\n";
        std::cout << "> Lost Photons: " << generator->N() - failed - res << "\n";

        if (_output_file != "")
        {
            try
            {
                PhotonWriter<value_t, index_t> writer(
                    _output_file, generator->N(), simulation->hitpointsDevice(),
                    simulation->resultsDevice(), simulation->angles_initial_Device(), simulation->statusDevice(), false); // TODO: Flag
                writer.write();
            }
            catch (const std::exception & e)
            {
                std::cerr << e.what() << '\n';
            }
        }
    }

    /**
     * @brief   Performs a Simulation on a @ref SplineSimulation
     *          depending on the given command line arguments.
     *
     * @tparam  value_t     The floating point type to operate on.
     * @tparam  index_t     The index type to operate on.
     */
    template <typename value_t, typename index_t> void runSimulation() const
    {
        // init spline depending on model or skip
        auto spline = initSpline<value_t>();

        title(); // printing
        if (spline) std::cout << "\n> Computed spline: " << spline;

        // init generator
        auto generator = initGenerator<value_t, index_t>();

        // init model and run
        if (_model == ALGOModel::ALG_QS || _model == ALGOModel::Tube_QS)
        {
            auto model = initQuadraticModel<value_t, index_t>(spline);
            auto simulation = // TODO: Simulation only
                std::make_shared<
                    SplineSimulation<value_t, index_t, QuadraticSplineModel<value_t, index_t>>>(
                    model.get(), value_t(_n1), value_t(_n2), value_t(_lambda_abs),
                    value_t(_lambda_sc), generator);
            performSimulation<value_t, index_t, QuadraticSplineModel<value_t, index_t>>(
                simulation.get(), generator.get());
            return;
        }
        else // ALGOModel::WOM_EM:
        {
            auto model = EllipticModel<value_t, index_t>(value_t(_length),       //
                                                         value_t(_model_arg[0]), //
                                                         value_t(_model_arg[1]), //
                                                         value_t(_model_arg[2]), //
                                                         value_t(_model_arg[3]), //
                                                         value_t(_model_arg[4]), //
                                                         value_t(_model_arg[5]), //
                                                         value_t(_model_arg[6]), //
                                                         value_t(_model_arg[7]), //
                                                         value_t(_model_arg[8]), //
                                                         value_t(_model_arg[9]));

            auto simulation =
                std::make_unique<Simulation<value_t, index_t, EllipticModel<value_t, index_t>>>(
                    &model, value_t(_n1), value_t(_n2), value_t(_lambda_abs), value_t(_lambda_sc),
                    generator);
            performSimulation<value_t, index_t, EllipticModel<value_t, index_t>>(simulation.get(),
                                                                                 generator.get());
            return;
        }
    }

    /**
     * @brief   Starts the program in information mode which simply prints out contributors.
     */
    void performInfo() const
    {
        title();
        std::cout << "Written by Ronja Schnur (rschnur@students.uni-mainz.de)\n\n";
        std::cout << "> Used OpenSource Libraries:\n";
        std::cout << "> cxxopts Copyright © 2014 Jarryd Beck licensed under MIT License.\n";
        std::cout << "> m.css Copyright © 2017, 2018, 2019 Vladimír Vondruš <mosra@centrum.cz> "
                     "licensed under MIT License.\n";
        std::cout << "\n";
    }

    template <typename value_t, typename index_t> void runFullSimulation() const
    {
        title();
        std::cout << "\n> WOMSimulation Mode\n> Start initialization ... " << std::flush;

        auto generator = initGenerator<value_t, index_t>();

        /*
            Factory Pattern would have been a better idea... but ¯\_(ツ)_/¯

            model_arg:
                alg_radius_inner        = 1
                alg_radius_outer        = 2
                alg_length              = 3
                elliptic_length         = 4
                elliptic_a_inner        = 5
                elliptic_b_inner        = 6
                elliptic_alpha_inner    = 7
                elliptic_x_inner        = 8
                elliptic_y_inner        = 9
                elliptic_a_outer        = 10
                elliptic_b_outer        = 11
                elliptic_alpha_outer    = 12
                elliptic_x_outer        = 13
                elliptic_y_outer        = 14
        */

        auto simulation =
            std::make_shared<WOMSimulation<value_t, index_t>>(_n1,                     //
                                                              _n2,                     //
                                                              _lambda_abs,             //
                                                              _lambda_sc,              //
                                                              value_t(_model_arg[1]),  //
                                                              value_t(_model_arg[2]),  //
                                                              initSpline<value_t>(),   //
                                                              value_t(_model_arg[3]),  //
                                                              value_t(_model_arg[4]),  //
                                                              value_t(_model_arg[5]),  //
                                                              value_t(_model_arg[6]),  //
                                                              value_t(_model_arg[7]),  //
                                                              value_t(_model_arg[8]),  //
                                                              value_t(_model_arg[9]),  //
                                                              value_t(_model_arg[10]), //
                                                              value_t(_model_arg[11]), //
                                                              value_t(_model_arg[12]), //
                                                              value_t(_model_arg[13]), //
                                                              value_t(_model_arg[14]), //
                                                              generator);

        std::cout << "Done.\n> Run simulation ... " << std::flush;

        index_t failed{0};
        index_t res = simulation->simulate(&failed);

        std::cout << "Done.\n";

        std::cout << "> Passing Photons: " << failed << "\n";
        std::cout << "> Detected Photons: " << res << "\n";
        std::cout << "> Lost Photons: " << generator->N() - failed - res << "\n";

        if (_output_file != "")
        {
            try
            {
                PhotonWriter<value_t, index_t> writer(
                    _output_file, generator->N(), generator->startpointsDevice(),
                    generator->directionsDevice(), simulation->statusDevice(), false); // TODO: Flag
                writer.write();
            }
            catch (const std::exception & e)
            {
                std::cerr << e.what() << '\n';
            }
        }
    }

    template <typename value_t, typename index_t> int performRun() const
    {
        switch (_mode)
        {
            case ALGOMode::Simulation: runSimulation<value_t, index_t>(); return 0;
            case ALGOMode::Optimization: performOptimization<value_t, index_t>(); return 0;
            case ALGOMode::FullSimulation: runFullSimulation<value_t, index_t>(); return 0;
            case ALGOMode::Info: performInfo(); return 0;
        }
    }

public:
    /**
     * @brief Runs ALGO program depending on the given command line arguments.
     */
    int run() const
    {
        // Wraps template paramters needed.
#if (__CUDA_ARCH__ >= 700)
        if (_N > UINT32_MAX && _precision)
        {
            return performRun<double, uint64_t>();
        }
        else if (N > UINT32_MAX)
        {
            return performRun<float, uint64_t>();
        }
        else
#else
        if (_N > UINT32_MAX)
        {
            std::cerr << "> ERROR: Choose smaller number of photons. Your device does not "
                         "support large values (>2^32-1) (Requires >sm_70).\n";
            exit(-1);
        }
#endif
            if (_precision)
        {
            return performRun<double, uint32_t>();
        }
        else
        {
            return performRun<float, uint32_t>();
        }
    }
};

} // namespace ALGO
