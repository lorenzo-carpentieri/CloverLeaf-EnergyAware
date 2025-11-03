/*
 Crown Copyright 2012 AWE.

 This file is part of CloverLeaf.

 CloverLeaf is free software: you can redistribute it and/or modify it under
 the terms of the GNU General Public License as published by the
 Free Software Foundation, either version 3 of the License, or (at your option)
 any later version.

 CloverLeaf is distributed in the hope that it will be useful, but
 WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 details.

 You should have received a copy of the GNU General Public License along with
 CloverLeaf. If not, see http://www.gnu.org/licenses/.
 */

//  @brief CloverLeaf top level program: Invokes the main cycle
//  @author Wayne Gaudin
//  @details CloverLeaf in a proxy-app that solves the compressible Euler
//  Equations using an explicit finite volume method on a Cartesian grid.
//  The grid is staggered with internal energy, density and pressure at cell
//  centres and velocities on cell vertices.
//
//  A second order predictor-corrector method is used to advance the solution
//  in time during the Lagrangian phase. A second order advective remap is then
//  carried out to return the mesh to an orthogonal state.
//
//  NOTE: that the proxy-app uses uniformly spaced mesh. The actual method will
//  work on a mesh with varying spacing to keep it relevant to it's parent code.
//  For this reason, optimisations should only be carried out on the software
//  that do not change the underlying numerical method. For example, the
//  volume, though constant for all cells, should remain array and not be
//  converted to a scalar.

#include <iostream>

#include "comms.h"
#include "definitions.h"
#include "finalise.h"
#include "hydro.h"
#include "initialise.h"
#include "read_input.h"
#include "report.h"
#include "start.h"
#include <sycl/sycl.hpp>
#include "version.h"
#include "utils.hpp"
#include <vector>
// Output file handler
std::ostream g_out(nullptr);

std::ofstream of;

global_variables initialise(parallel_ &parallel, const std::vector<std::string> &args) {
  global_config config;
  if (parallel.boss) {
    std::cout << "---" << std::endl;
  }
  auto model = create_context(!parallel.boss, args);
  config.dumpDir = model.args.dumpDir;

#ifdef NO_MPI
  bool mpi_enabled = false;
#else
  bool mpi_enabled = true;
#endif

#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
  std::optional<bool> mpi_cuda_aware_header = true;
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
  std::optional<bool> mpi_cuda_aware_header = false;
#else
  std::optional<bool> mpi_cuda_aware_header = {};
#endif

#if defined(MPIX_CUDA_AWARE_SUPPORT)
  std::optional<bool> mpi_cuda_aware_runtime = MPIX_Query_cuda_support() != 0;
#else
  std::optional<bool> mpi_cuda_aware_runtime = {};
#endif

  if (!model.offload) {
    if (model.args.staging_buffer == run_args::staging_buffer::enabled) {
      std::cout << "# WARNING: enabling staging buffer on a non-offload (host) model or device may be no-op" << std::endl;
    }
  }
  switch (model.args.staging_buffer) {
    case run_args::staging_buffer::enabled: config.staging_buffer = true; break;
    case run_args::staging_buffer::disable: config.staging_buffer = false; break;
    case run_args::staging_buffer::automatic:
      config.staging_buffer = !(mpi_cuda_aware_header.value_or(false) && mpi_cuda_aware_runtime.value_or(false));
      break;
  }

  if (parallel.boss) {
    std::cout << "CloverLeaf:\n"
              << " - Ver.:     " << g_version << "\n"
              << " - Deck:     " << model.args.inFile << "\n"
              << " - Out:      " << model.args.outFile << "\n"
              << " - Profiler: " << (model.args.profile ? (*model.args.profile ? "true" : "false") : "deck-specified") << "\n"
              << "MPI:\n"
              << " - Enabled:     " << (mpi_enabled ? "true" : "false") << "\n"
              << " - Total ranks: " << parallel.max_task << "\n"
              << " - Header device-awareness (CUDA-awareness):  "
              << (mpi_cuda_aware_header ? (*mpi_cuda_aware_header ? "true" : "false") : "unknown") << "\n"
              << " - Runtime device-awareness (CUDA-awareness): "
              << (mpi_cuda_aware_runtime ? (*mpi_cuda_aware_runtime ? "true" : "false") : "unknown") << "\n"
              << " - Host-Device halo exchange staging buffer:  " << (config.staging_buffer ? "true" : "false") << "\n"
              << "Model:\n"
              << " - Name:      " << model.name << "\n"
              << " - Execution: " << (model.offload ? "Offload (device)" : "Host") //
              << std::endl;
    report_context(model.context);
    std::cout << "# ---- " << std::endl;
    std::cout << "Output: |+1" << std::endl;
  }

  if (parallel.boss) {
    std::cout << " Output file clover.out opened. All output will go there." << std::endl;
    of.open(model.args.outFile.empty() ? "clover.out" : model.args.outFile);
    if (!of.is_open()) report_error((char *)"initialise", (char *)"Error opening clover.out file.");
    g_out.rdbuf(of.rdbuf());
  } else {
    g_out.rdbuf(std::cout.rdbuf());
  }

  if (parallel.boss) {
    g_out << "Clover Version " << g_version << std::endl     //
          << "Task Count " << parallel.max_task << std::endl //
          << std::endl;
  }

  clover_barrier();

  std::ifstream g_in;
  if (parallel.boss) {
    g_out << "Clover will run from the following input:-" << std::endl << std::endl;
    if (!args.empty()) {
      std::cout << " Args:";
      for (const auto &arg : args)
        std::cout << " " << arg;
      std::cout << std::endl;
    }
  }

  if (!model.args.inFile.empty()) {
    if (parallel.boss) std::cout << " Using input: `" << model.args.inFile << "`" << std::endl;
    g_in.open(model.args.inFile);
    if (g_in.fail()) {
      std::cerr << "Unable to open file: `" << model.args.inFile << "`" << std::endl;
      std::exit(1);
    }
  } else {
    if (parallel.boss) std::cout << "No input file specified, using default input" << std::endl;
    std::ofstream out_unit("clover.in");
    out_unit << "*clover" << std::endl
             << " state 1 density=0.2 energy=1.0" << std::endl
             << " state 2 density=1.0 energy=2.5 geometry=rectangle xmin=0.0 xmax=5.0 ymin=0.0 ymax=2.0" << std::endl
             << " x_cells=10" << std::endl
             << " y_cells=2" << std::endl
             << " xmin=0.0" << std::endl
             << " ymin=0.0" << std::endl
             << " xmax=10.0" << std::endl
             << " ymax=2.0" << std::endl
             << " initial_timestep=0.04" << std::endl
             << " timestep_rise=1.5" << std::endl
             << " max_timestep=0.04" << std::endl
             << " end_time=3.0" << std::endl
             << " test_problem 1" << std::endl
             << "*endclover" << std::endl;
    out_unit.close();
    g_in.open("clover.in");
  }
  //}

  clover_barrier();
  if (parallel.boss) {
    g_out << std::endl << "Initialising and generating" << std::endl << std::endl;
  }
  read_input(g_in, parallel, config);
  if (model.args.profile) {
    config.profiler_on = *model.args.profile;
  }

  clover_barrier();

  //	globals.step = 0;
  config.number_of_chunks = parallel.max_task;

  auto globals = start(parallel, config, model.context);
  clover_barrier(globals);
  if (parallel.boss) {
    g_out << "Starting the calculation" << std::endl;
  }
  g_in.close();
  return globals;
}


void modify_core_freq(synergy::queue &q, int core_freq){
  std::ostringstream freq_info;
  // Change frequency for the target hw so that we can profile the application using a specific core_freq.   
  freq_info << "Frequency " << core_freq << " MHz\n";
  logs::log_device(freq_info.str());
  logs::log_kernel(freq_info.str());

  q.submit(0, core_freq, [&](sycl::handler& cgh) {
    cgh.single_task([=]() {
      // Do nothing
    });
  });
  q.wait();
  polling_freq(q, core_freq, 400);
}

int main(int argc, char *argv[]) {
  
  MPI_Init(&argc, &argv);
  parallel_ parallel;
  // Global rank
  int comm_rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &comm_rank);
 // Local rank
  MPI_Comm local_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, comm_rank,
                      MPI_INFO_NULL, &local_comm);

  int local_comm_rank = -1;
  MPI_Comm_rank(local_comm, &local_comm_rank);
  char node_name[MPI_MAX_PROCESSOR_NAME];
  int node_name_len = 0;
  MPI_Get_processor_name(node_name, &node_name_len);



  // Core frequency used for the entire application
  int core_freq=0;
  // Path to the log directory where device and kernel info will be stored
  std::string log_dir_path;

  // Core freq used for the entire application specified as command line parameter
  if (argc >= 3){
    core_freq = std::stoi(argv[2]);
    log_dir_path = argv[4];
    if (comm_rank == 0){
      std::cout << "Target core freq: " << core_freq << std::endl;
    }
  }
  else{
    std::cout<< "Usage: ./exe <core_freq> <path_to_log_dir>" <<std::endl;
  }
  // Create SYnergy queue
  std::vector<sycl::device> gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  synergy::queue q(gpu_devices[local_comm_rank % gpu_devices.size()]);

  logs::init_log_files(comm_rank, log_dir_path);
  modify_core_freq(q, core_freq); // Change frequency for each gpu associated to a local rank.
  
  double init_energy_start = q.device_energy_consumption();

  auto start = std::chrono::high_resolution_clock::now();
  // Init phase
  global_variables config = initialise(parallel, std::vector<std::string>(argv + 1, argv + argc));
  
  auto end = std::chrono::high_resolution_clock::now();
  double init_energy_end = q.device_energy_consumption();

  auto initialise_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
  
  
  double total_phase_init = aggregate_energy_values(init_energy_start, init_energy_end);
  logs::log_device(logs::gpu_energy_str("init", init_energy_end-init_energy_start)); // Energy for single gpu

  if (parallel.boss) {
    logs::log_device(logs::gpus_energy_all_str("init", total_phase_init)); // Energy for all gpus
  }

  if (parallel.boss) {
    std::cout << " Launching hydro" << std::endl;
  }


  start = std::chrono::high_resolution_clock::now();
  double hydro_energy_start = q.device_energy_consumption();
  hydro(config, parallel);
  finalise(config);
  double hydro_energy_end= q.device_energy_consumption();
  double total_phase_hydro = aggregate_energy_values(hydro_energy_start, hydro_energy_end);
  end = std::chrono::high_resolution_clock::now();
  auto hydro_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); 
  
  logs::log_device(logs::gpu_energy_str("hydro", hydro_energy_end-hydro_energy_start)); // Energy for single gpu
  logs::log_device(logs::gpu_time_str("hydro", hydro_time)); // Energy for single gpu

  double total_time_hydro = aggregate_time_values(hydro_time);  

  if (parallel.boss)
  {
    logs::log_device(logs::gpus_energy_all_str("hydro", total_phase_hydro)); // Energy for all gpus
    logs::log_device(logs::gpus_time_all_str("hydro", total_time_hydro)); // Time for all gpus
  }
  
  

  logs::close_log_files();
  MPI_Finalize();

  if (parallel.boss) {
    std::cout << "Result:\n"
              << " - Problem: " << (config.config.test_problem == 0 ? "none" : std::to_string(config.config.test_problem)) << "\n"
              << " - Outcome: " << (config.report_test_fail ? "FAILED" : "PASSED") << std::endl;
  }


  return config.report_test_fail ? EXIT_FAILURE : EXIT_SUCCESS;
}
