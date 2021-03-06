/**
 ==============================================================================
 Copyright 2019, Jonathan Zrake

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
 of the Software, and to permit persons to whom the Software is furnished to do
 so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 SOFTWARE.

 ==============================================================================
*/




#include "app_config.hpp"
#include "app_control.hpp"
#include "app_hdf5.hpp"
#include "app_hdf5_config.hpp"
#include "app_hdf5_dimensional.hpp"
#include "app_hdf5_rational.hpp"
#include "app_hdf5_ndarray.hpp"
#include "app_hdf5_ndarray_dimensional.hpp"
#include "app_hdf5_numeric_array.hpp"
#include "app_state_templates.hpp"
#include "core_ndarray.hpp"
#include "core_ndarray_ops.hpp"
#include "core_sequence.hpp"
#include "core_util.hpp"
#include "model_wind.hpp"
#include "parallel_thread_pool.hpp"
#include "physics_srhd.hpp"
#include "scheme_mesh_geometry.hpp"
#include "scheme_moving_mesh.hpp"
#include "scheme_plm_gradient.hpp"




//=============================================================================
auto config_template()
{
    return mara::config_template()
    .item("restart",    std::string(), "the name of a checkpoint file to restart from")
    .item("threads",                1, "the number of concurrent threads to execute on (0 for hardware_concurrency)")
    .item("nr",                   512, "number of radial zones, per decade")
    .item("tfinal",            1000.0, "time to stop the simulation")
    .item("router",               1e3, "outer boundary radius")
    .item("print",                 10, "the number of iterations between terminal outputs")
    .item("dfi",                 1.05, "output interval (constant multiplier)")
    .item("rk_order",               2, "Runge-Kutta order (1, 2, or 3)")
    .item("cfl",                 0.25, "courant number")
    .item("mindr",               1e-4, "minimum dr to impose in remeshing")
    .item("maxdr",               1e-3, "maximum dr to impose in remeshing")
    .item("plm_theta",            1.0, "PLM parameter")
    .item("move",                   1, "whether to move the cells")
    .item("power_law_m",          6.0, "Wind model power-law")
    .item("a_0",                240e5, "NS initial separation (in cm)")
    .item("a_f",                 24e5, "NS final separation (in cm)")
    .item("t_f",               1.2e-3, "Time to merger when a=af")
    .item("engine_Gamma0",        1.5, "engine mass rate at a=a0")
    .item("engine_edot0",         1.0, "engine power at a=a0")
    .item("edot_ambient",         1.0, "engine power at a=a0")
    .item("engine_onset",         0.0);  

    //  task.next_time = (task.next_time - reference_time) * factor + reference_time;
    // .item("gamma_ambient",        1.5)   // engine power at a=a0
    // .item("mdot_ambient",        1e-2)   // engine mass rate at a=a0

    // Used as reference point for time stepping. 
    // If 0, then the time-step happens uniformly logarithmically from t=1. 
    // If engine_onset > 0, then the time will be stepped logarithmically until that 
    // point, and after t > engine_onset, the time-stepper resets to use smaller delta 
    // (as used right after t=1).
}




//=============================================================================
template<typename ProviderType>
auto evaluate_on(mara::ThreadPool& pool, nd::array_t<ProviderType, 1> array)
{
    using value_type = typename nd::array_t<ProviderType, 1>::value_type;
    auto nt = pool.size();
    auto futures = std::vector<std::future<int>>();
    auto result = nd::make_unique_array<value_type>(shape(array));

    for (std::size_t t = 0; t < nt; ++t)
    {
        std::size_t start = (t + 0) * size(array) / nt;
        std::size_t final = (t + 1) * size(array) / nt;

        futures.push_back(pool.enqueue([&result, array, start, final]
        {
            for (std::size_t i = start; i < final; ++i)
            {
                result(i) = array(i);
            }
            return 0;
        }));
    }
    for (auto& future : futures)
    {
        future.get();
    }
    return nd::make_shared_array(std::move(result));
}




static const auto gamma_law_index   = 4. / 3;
static const auto temperature_floor = 1e-6;




//=============================================================================
using solution_t            = mara::state_with_vertices_t<srhd::conserved_t>;
using solution_with_tasks_t = std::pair<solution_t, control::task_t>;
using timed_state_pair_t    = control::timed_pair_t<solution_with_tasks_t>;
using mara::spherical_mesh_geometry_t;
using namespace dimensional;

inline auto solution(solution_with_tasks_t p)
{
    return p.first;
}

inline auto tasks(solution_with_tasks_t p)
{
    return p.second;
}




//=============================================================================
auto recover_primitive()
{
    return [] (auto u) { return srhd::recover_primitive(u, gamma_law_index, temperature_floor); };
}




//=============================================================================
void write_diagnostics(const mara::config_t& run_config, solution_t state, unsigned long count)
{
    auto fname     = util::format("sedov.%04lu.h5", count);
    auto dv        = spherical_mesh_geometry_t::cell_volumes(state.vertices);
    auto prim      = state.conserved | nd::divide(dv) | nd::map(recover_primitive()) | nd::to_shared();
    auto file      = h5::File(fname, "w");

    std::printf("Write diagnostics %s\n", fname.data());

    h5::write(file, "vertices", state.vertices);
    h5::write(file, "primitive", prim);
    h5::write(file, "time", state.time);
    h5::write(file, "run_config", run_config);
}

void write_checkpoint(const mara::config_t& run_config, solution_t state, control::task_t task)
{
    auto fname     = util::format("chkpt.%04lu.h5", task.count);
    auto file      = h5::File(fname, "w");

    std::printf("Write checkpoint %s\n", fname.data());

    h5::write(file, "iteration", state.iteration);
    h5::write(file, "time", state.time);
    h5::write(file, "vertices", state.vertices);
    h5::write(file, "conserved", state.conserved);
    h5::write(file, "output_count", task.count);
    h5::write(file, "output_next_time", task.next_time);
    h5::write(file, "run_config", run_config);
}

void read_checkpoint(std::string fname, solution_t& state, control::task_t& task)
{
    auto file = h5::File(fname, "r");

    std::printf("Read checkpoint %s\n", fname.data());

    h5::read(file, "iteration", state.iteration);
    h5::read(file, "time", state.time);
    h5::read(file, "vertices", state.vertices);
    h5::read(file, "conserved", state.conserved);
    h5::read(file, "output_count", task.count);
    h5::read(file, "output_next_time", task.next_time);
}


//=============================================================================
auto semi_major_axis(const mara::config_t & run_config)
{
    return [run_config] (dimensional::unit_time t) -> dimensional::unit_length
    {
        auto a_0          = unit_length(run_config.get_double("a_0"));
        auto a_f          = unit_length(run_config.get_double("a_f"));
        double xi         = a_0 / a_f;
        auto t_f          = unit_time(run_config.get_double("t_f"));
        auto t_merger     = t_f * std::pow(xi, 4.0);
        auto t_mf 	      = t_merger - t_f;
        auto delta_t      = t_merger - t;
        // std::printf("xi= %f, 1/xi = %f\n", xi, 1.0/xi);
        auto a            = a_0 * std::max(1.0 / xi, std::pow(delta_t / t_mf , 0.25));
        return a;
    };
}

auto wind_mass_loss_rate(const mara::config_t & run_config)
{
    auto major_axis = semi_major_axis(run_config);

    return [major_axis, run_config] (dimensional::unit_time t) -> dimensional::unit_mass_rate
    {
        auto a_0          = unit_length(run_config.get_double("a_0"));
        auto a_f          = unit_length(run_config.get_double("a_f"));
        double xi         = a_0 / a_f;
        auto t_f          = unit_time(run_config.get_double("t_f"));
        auto t_merger     = t_f * std::pow(xi, 4.0);
        auto t_mf         = t_merger - t_f;
        auto Edot0        = unit_power(run_config.get_double("engine_edot0"));
        auto Gamma0       = run_config.get_double("engine_Gamma0");
        auto power_law_m  = run_config.get_double("power_law_m");

        auto a            = major_axis(t);
        // std::printf("a= %f\n", a);
        auto smooth       = 0.5 * (1.0 + std::tanh((t - t_mf) / t_f));
        auto Mdot0        = Edot0 / (Gamma0 * srhd::light_speed * srhd::light_speed);
        auto Mdot_ambient = Mdot0;
        auto m            = power_law_m;
        auto Mdot         = Mdot0 * std::max(1.0, std::pow((a / a_0) , -m));
        auto Mdot_smooth  = Mdot * (1.0 - smooth);
        auto Mdot_final   = Mdot_smooth + Mdot_ambient;
        return Mdot_final;
   };
}

auto wind_power(const mara::config_t & run_config)
{
    auto major_axis = semi_major_axis(run_config);

    return [major_axis, run_config] (dimensional::unit_time t) -> dimensional::unit_power
    {
        auto a_0          = unit_length(run_config.get_double("a_0"));
        auto a_f          = unit_length(run_config.get_double("a_f"));
        double xi         = a_0 / a_f;
        auto t_f          = unit_time(run_config.get_double("t_f"));
        auto t_merger     = t_f * std::pow(xi, 4.0);
        auto t_mf         = t_merger - t_f;
        auto a            = major_axis(t);      

        auto smooth       = 0.5 * (1.0 + std::tanh((t - t_mf) / t_f));
        auto Edot0        = unit_power(run_config.get_double("engine_edot0"));
        auto Edot_ambient = unit_power(run_config.get_double("edot_ambient"));
        auto Edot         = Edot0 * std::max(1.0, std::pow((a / a_0) , -7.0));
        auto Edot_smooth  = Edot * (1.0 - smooth);
        return Edot_smooth + Edot_ambient;
    };
}

auto wind_gamma_beta(const mara::config_t & run_config)
{
    auto mass_loss_rate = wind_mass_loss_rate(run_config);
    auto power          = wind_power(run_config);

    return [mass_loss_rate, power, run_config] (dimensional::unit_time t) -> dimensional::unit_scalar
    { 
        auto Edot   = power(t);                     
        auto Mdot   = mass_loss_rate(t);                    
        auto c2     = srhd::light_speed * srhd::light_speed;
        auto gamma  = (Edot / (Mdot * c2)) + 1.01;
        return std::sqrt(gamma * gamma - 1.0);
    };
}




//=============================================================================
auto wind_profile(const mara::config_t& run_config, unit_length r, unit_time t)
{
    return mara::cold_relativistic_wind_t()
    .with_mass_loss_rate(wind_mass_loss_rate(run_config))
    .with_gamma_beta    (wind_gamma_beta    (run_config))
    .primitive(r, t);
}

auto initial_condition(const mara::config_t& run_config)
{
    // For a stationary medium with a given (density, pressure):
    // return [run_config] (unit_length r) { return srhd::primitive(0.01 , 0.001); };

    // For a wind with 1/r^2 density profile:
    return [run_config] (unit_length r) { return wind_profile(run_config, r, 1.0); };
}

auto riemann_solver_for(geometric::unit_vector_t nhat, bool move)
{
    auto contact_mode = srhd::riemann_solver_mode_hllc_fluxes_across_contact_t();

    return util::apply_to([nhat, move, contact_mode] (auto pl, auto pr)
    {
        return move
        ? srhd::riemann_solver(pl, pr, nhat, gamma_law_index, contact_mode)
        : std::make_pair(srhd::riemann_hllc(pl, pr, nhat, gamma_law_index), unit_velocity(0.0));
    });
}

auto time_step(const mara::config_t& run_config, solution_t state)
{
    auto cfl = run_config.get_double("cfl");
    return cfl * nd::min(state.vertices | nd::adjacent_diff()) / srhd::light_speed;
}




//=============================================================================
auto initial_vertices(const mara::config_t& run_config)
{
    auto router     = run_config.get_double("router");
    auto cell_count = run_config.get_int("nr") * int(std::log10(router));

    return nd::linspace(0.0, std::log10(router), cell_count + 1)
    | nd::map([] (auto log10r) { return std::pow(10.0, log10r); })
    | nd::construct<unit_length>();
}

solution_t initial_solution_state(const mara::config_t& run_config)
{
    auto xv = initial_vertices(run_config) | nd::to_shared();
    auto dv = spherical_mesh_geometry_t::cell_volumes(xv);
    auto p0 = xv | nd::adjacent_mean() | nd::map(initial_condition(run_config));
    auto u0 = p0 | nd::map([] (auto p) { return srhd::conserved_density(p, gamma_law_index); });

    // Time starts from T0=0.0
    return {0, 0.0, xv, (u0 * dv) | nd::to_shared()};

    // Time starts from T0=1.0
    // return {0, 1.0, xv, (u0 * dv) | nd::to_shared()};
}

solution_with_tasks_t initial_app_state(const mara::config_t& run_config)
{
    if (! run_config.get_string("restart").empty())
    {
        auto task = control::task("write_diagnostics");
        auto state = solution_t();
        read_checkpoint(run_config.get_string("restart"), state, task);
        return std::pair(state, task);
    }
    return std::pair(initial_solution_state(run_config), control::task("write_diagnostics", 1e-2));
}




//=============================================================================
solution_t add_inner_cell(const mara::config_t& run_config, solution_t solution)
{
    if (front(solution.vertices) > unit_length(1.0 + 1.0 / run_config.get_int("nr")))
    {
        auto x1 = nd::concat(nd::from(unit_length(1.0)), solution.vertices, 0) | nd::to_shared();
        auto xc = spherical_mesh_geometry_t::cell_centers(x1);
        auto dv = spherical_mesh_geometry_t::cell_volumes(x1);
        auto bp = wind_profile(run_config, front(xc), solution.time);
        auto bu = srhd::conserved_density(bp, gamma_law_index) * front(dv);
        auto u1 = nd::concat(nd::from(bu), solution.conserved, 0) | nd::to_shared();

        return {
            solution.iteration,
            solution.time,
            x1,
            u1,
        };
    }
    return solution;
}

solution_t split_join_cells(const mara::config_t& run_config, solution_t solution)
{
    // auto maxdr  = run_config.get_double("maxdr");
    // auto mindr  = run_config.get_double("mindr");
    auto maxdr  = unit_length(run_config.get_double("maxdr"));
    auto mindr  = unit_length(run_config.get_double("mindr"));
    auto uc     = solution.conserved;
    auto rf     = solution.vertices;
    auto rc     = spherical_mesh_geometry_t::cell_centers(solution.vertices);
    auto dr     = spherical_mesh_geometry_t::cell_spacings(solution.vertices);
    // auto aspect = dr / rc;
    auto aspect = dr;
    auto imin   = nd::argmin(aspect)[0];
    auto imax   = nd::argmax(aspect)[0];

    auto construct = [solution] (auto x1, auto u1) -> solution_t
    {
        return {
            solution.iteration,
            solution.time,
            x1 | nd::to_shared(),
            u1 | nd::to_shared(),
        };
    };

    if (aspect(imax) > maxdr)
    {
        return std::apply(construct, nd::add_partition(rf, uc, imax));
    }

    if (aspect(imin) < mindr)
    {
        if (imin == 0)
        {
            return split_join_cells(run_config, std::apply(construct, nd::remove_partition(rf, uc, 1)));
        }
        if (imin + 1 == size(aspect))
        {
            return split_join_cells(run_config, std::apply(construct, nd::remove_partition(rf, uc, imin)));
        }
        if (aspect(imin - 1) <= aspect(imin + 1))
        {
            return split_join_cells(run_config, std::apply(construct, nd::remove_partition(rf, uc, imin)));
        }
        if (aspect(imin - 1) >= aspect(imin + 1))
        {
            return split_join_cells(run_config, std::apply(construct, nd::remove_partition(rf, uc, imin + 1)));
        }
    }
    return solution;
}

solution_t remesh(const mara::config_t& run_config, solution_t solution)
{
    solution = add_inner_cell  (run_config, solution);
    solution = split_join_cells(run_config, solution);
    return solution;
}

solution_t advance(const mara::config_t& run_config, mara::ThreadPool& pool, solution_t solution)
{
    auto base = [&run_config, &pool, dt = time_step(run_config, solution)] (solution_t soln)
    {
        // auto evaluate            = nd::to_shared();
        auto evaluate            = [&pool] (auto array) { return evaluate_on(pool, array); };
        auto move_cells          = run_config.get_int("move");
        auto plm_theta           = run_config.get_double("plm_theta");
        auto xhat                = geometric::unit_vector_on(1);
        auto inner_boundary_prim = wind_profile(run_config, front(soln.vertices), soln.time);
        auto outer_boundary_prim = wind_profile(run_config, back (soln.vertices), 1.0);
        auto mesh_geometry       = spherical_mesh_geometry_t();
        auto source_terms        = util::apply_to([] (auto p, auto x)
        {
            return srhd::spherical_geometry_source_terms(p, x, M_PI / 2, gamma_law_index);
        });
        return mara::advance(
            soln,
            dt,
            inner_boundary_prim,
            outer_boundary_prim,
            riemann_solver_for(xhat, move_cells),
            recover_primitive(),
            source_terms,
            mesh_geometry,
            evaluate,
            plm_theta);
    };
    return remesh(run_config, control::advance_runge_kutta(base, run_config.get_int("rk_order"), solution));
}

control::task_t advance(const mara::config_t& run_config, control::task_t task, unit_time time)
{
    auto engine_onset = unit_time(run_config.get_double("engine_onset"));
    auto ref_time = time < engine_onset ? unit_time(0.0) : engine_onset;
    return jump(task, time, run_config.get_double("dfi"), ref_time);
}

auto advance(const mara::config_t& run_config, mara::ThreadPool& pool)
{
    return util::apply_to([&run_config, &pool] (solution_t state, control::task_t task)
    {
        return std::pair(
            advance(run_config, pool, state),
            advance(run_config, task, state.time));
    });
}




//=============================================================================
auto should_continue(const mara::config_t& run_config)
{
    return [tfinal = run_config.get_double("tfinal")] (timed_state_pair_t p)
    {
        return solution(control::last_state(p)).time <= unit_time(tfinal);
    };
}

void print_run_loop(const mara::config_t& run_config, timed_state_pair_t p)
{
    auto soln = solution(control::this_state(p));
    auto nz = size(soln.vertices);
    auto us = control::microseconds_separating(p);

    std::printf("[%07lu] t=%.3lf dt=%.2e zones=%lu Mzps=%.2lf\n",
        long(soln.iteration),
        soln.time.value,
        time_step(run_config, soln).value,
        nz,
        nz / us);
}

auto side_effects(const mara::config_t& run_config, timed_state_pair_t p)
{
    auto this_soln = solution(control::this_state(p));
    auto last_soln = solution(control::last_state(p));
    auto this_task = tasks(control::this_state(p));
    auto last_task = tasks(control::last_state(p));

    if (this_task.count != last_task.count)
    {
        write_diagnostics(run_config, last_soln, last_task.count);
        write_checkpoint(run_config, last_soln, last_task);
    }

    if (long(this_soln.iteration) % run_config.get_int("print") == 0)
    {
        print_run_loop(run_config, p);            
    }
}

auto time_point_sequence()
{
    using namespace std::chrono;
    return seq::generate(high_resolution_clock::now(), [] (auto) { return high_resolution_clock::now(); });
}

mara::config_parameter_map_t restart_run_config(const mara::config_string_map_t& args)
{
    if (args.count("restart"))
    {
        auto file = h5::File(args.at("restart"), "r");
        return h5::read<mara::config_parameter_map_t>(file, "run_config");
    }
    return {};
}




//=============================================================================
int main(int argc, const char* argv[])
{
    auto args = mara::argv_to_string_map(argc, argv);
    auto run_config = config_template()
    .create()
    .update(restart_run_config(args))
    .update(args);

    auto threads = run_config.get_int("threads");
    auto pool = mara::ThreadPool(threads ? threads : std::thread::hardware_concurrency());

    mara::pretty_print(std::cout, "config", run_config);

    auto simulation = seq::generate(initial_app_state(run_config), advance(run_config, pool))
    | seq::pair_with(time_point_sequence())
    | seq::window()
    | seq::take_while(should_continue(run_config));

    for (auto state_pair : simulation)
    {
        side_effects(run_config, state_pair);
    }
    return 0;
}
