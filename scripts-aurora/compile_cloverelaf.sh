


# first arg frequency scaling approach APP | KERNEL | PHASE
APPROACH=$1
# paht to the cxx oneApi compiler
cmake \
  -DCMAKE_CXX_COMPILER="icpx" \
  -DCMAKE_CXX_FLAGS="-Wno-deprecated -fsycl -fsycl-targets=intel_gpu_pvc -O3" \
  -DENABLE_MPI=ON \
  -DMPI_C_LIB=/opt/aurora/25.190.0/oneapi/mpi/2021.16/lib/ \
  -DMPI_C_INCLUDE_DIR=/opt/aurora/25.190.0/oneapi/mpi/2021.16/include/ \
  -DMODEL="sycl-usm" \
  -DSYCL_COMPILER="ONEAPI-ICPX" \
  -DSYCL_COMPILER_DIR="/opt/aurora/25.190.0/oneapi/compiler/" \
  ..




  