#pragma once
#include <fstream>
#include <memory>
#include <string>
#include <sycl/sycl.hpp>
#include <synergy.hpp>
#include <mpi.h>
namespace logs {
    extern std::unique_ptr<std::ofstream> kernels_log_file;
    extern std::unique_ptr<std::ofstream> device_log_file;
    
    void init_log_files(int myrank, const std::string& base_dir);
    void log_kernel(const std::string& log_string);
    void log_device(const std::string& log_string);
    std::string gpu_energy_str(std::string phase_name, double energy_j);
    std::string gpus_energy_all_str(std::string phase_name, double energy_j);
    std::string gpu_time_str(std::string phase_name, double time_ms);
    std::string gpus_time_all_str(std::string phase_name, double time_ms);
    void close_log_files();
}

void polling_freq(synergy::queue q, int to_set, int polling_time_us);

double aggregate_energy_values(double start_j, double end_j);
double aggregate_time_values(double time_ms);