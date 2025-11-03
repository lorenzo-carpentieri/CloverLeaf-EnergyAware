#include "utils.hpp"
#include <iostream>
#include <filesystem>


// Log time and energy info into two different file:
// device contains info for each device and aggregated info for all devices involved in the execution.
// kernel contains only the information about time and energy of each kernel
namespace logs {
    std::unique_ptr<std::ofstream> kernels_log_file = nullptr;
    std::unique_ptr<std::ofstream> device_log_file  = nullptr;

    void init_log_files(int myrank, const std::string& base_dir) {
        namespace fs = std::filesystem;
        fs::path dir(base_dir);
        if (!fs::exists(dir)) fs::create_directories(dir);

        fs::path kernels_path = dir / ("rank_" + std::to_string(myrank) + "_kernels.log");
        fs::path device_path  = dir / ("rank_" + std::to_string(myrank) + "_device.log");

        kernels_log_file = std::make_unique<std::ofstream>(kernels_path, std::ios::app);
        device_log_file  = std::make_unique<std::ofstream>(device_path,  std::ios::app);

        (*kernels_log_file) << "=== Kernel log started for rank "
                            << myrank <<  " ===" << std::endl;
        (*device_log_file) << "=== Device log started for rank "
                           << myrank << " ===" << std::endl;
    }

    void log_kernel(const std::string& msg) {
        if (kernels_log_file && kernels_log_file->is_open()) {
            (*kernels_log_file) << msg << std::endl;
            kernels_log_file->flush();
        }
    }

    void log_device(const std::string& msg) {
        if (device_log_file && device_log_file->is_open()) {
            (*device_log_file) << msg << std::endl;
            device_log_file->flush();
        }
    }

    void close_log_files() {
        if (kernels_log_file && kernels_log_file->is_open()) {
            (*kernels_log_file) << "=== Kernel log closed ===" << std::endl;
            kernels_log_file->close();
        }
        if (device_log_file && device_log_file->is_open()) {
            (*device_log_file) << "=== Device log closed ===" << std::endl;
            device_log_file->close();
        }
    }

    std::string gpu_energy_str(std::string phase_name, double energy_j){
        std::ostringstream energy_info;
        energy_info << "GPU energy "
        << phase_name 
        << " [J]: "
        << energy_j 
        << std::endl;

        return  energy_info.str();

    }
    std::string gpus_energy_all_str(std::string phase_name, double energy_j){
        std::ostringstream energy_info;
        energy_info << "All GPUs energy "
        << phase_name 
        << " [J]: "
        << energy_j 
        << std::endl;
        return  energy_info.str();
    }


    
    std::string gpu_time_str(std::string phase_name, double time_ms){
        std::ostringstream time_info;
        time_info << "Rank total time "
        << phase_name
        <<" [ms]: " 
        << time_ms 
        << "\n";
        return time_info.str();
    }
    
    std::string gpus_time_all_str(std::string phase_name, double time_ms){
        std::ostringstream time_info;
        time_info << "Max total time "
        << phase_name
        <<" [ms]: " 
        << time_ms 
        << "\n";
        return time_info.str();
    }
}



void polling_freq(synergy::queue q, int to_set, int polling_time_us) {
  int clock_mhz = 0;

  do {
    std::this_thread::sleep_for(std::chrono::microseconds(polling_time_us));
    clock_mhz = q.get_synergy_device().get_core_frequency(false);
    // std::cout<<"Current freq: " << clock_mhz << " / Target freq: " << to_set <<std::endl;
    if (clock_mhz >= to_set - 50 && clock_mhz <= to_set + 50) {
      return ;
    }

  } while (clock_mhz != to_set);
  std::ostringstream poll_info;
  poll_info <<"Current freq: " << clock_mhz << " / Target freq: " << to_set <<std::endl;
  logs::log_device(poll_info.str());
}

double aggregate_energy_values(double start_j, double end_j){
    double all_gpus = 0;
    double single_gpu = end_j - start_j;
    MPI_Reduce(&single_gpu, &all_gpus, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    return all_gpus;
}

double aggregate_time_values(double time_ms){
    double all_gpus = 0;
    MPI_Reduce(&time_ms, &all_gpus, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    return all_gpus;
}
