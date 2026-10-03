# =============================================================================
#  cmake/FindNetCDF.cmake —— 查找 NetCDF C 库（支持 nc-config 与手工路径）
# =============================================================================
include(FindPackageHandleStandardArgs)

set(NetCDF_C_COMPONENTS C)
set(_netcdf_required_vars NetCDF_C_LIBRARY NetCDF_C_INCLUDE_DIR)

find_program(NC_CONFIG_EXECUTABLE NAMES nc-config)
if(NC_CONFIG_EXECUTABLE)
  execute_process(COMMAND ${NC_CONFIG_EXECUTABLE} --libs
                  OUTPUT_VARIABLE _nc_libs OUTPUT_STRIP_TRAILING_WHITESPACE)
  execute_process(COMMAND ${NC_CONFIG_EXECUTABLE} --includedir
                  OUTPUT_VARIABLE _nc_inc OUTPUT_STRIP_TRAILING_WHITESPACE)
  separate_arguments(_nc_lib_list UNIX_COMMAND "${_nc_libs}")
  set(NetCDF_C_LIBRARY ${_nc_lib_list} CACHE STRING "NetCDF C libraries")
  set(NetCDF_C_INCLUDE_DIR ${_nc_inc} CACHE PATH "NetCDF C include directory")
else()
  find_path(NetCDF_C_INCLUDE_DIR NAMES netcdf.h
            HINTS ENV NETCDF_DIR ENV NETCDF_ROOT ${NETCDF_DIR})
  find_library(NetCDF_C_LIBRARY NAMES netcdf
               HINTS ENV NETCDF_DIR ENV NETCDF_ROOT ${NETCDF_DIR}
               PATH_SUFFIXES lib lib64)
endif()

find_package_handle_standard_args(NetCDF
  REQUIRED_VARS ${_netcdf_required_vars}
  VERSION_VAR NetCDF_VERSION)

if(NetCDF_FOUND AND NOT TARGET NetCDF::NetCDF_C)
  add_library(NetCDF::NetCDF_C UNKNOWN IMPORTED)
  set_target_properties(NetCDF::NetCDF_C PROPERTIES
    IMPORTED_LOCATION "${NetCDF_C_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${NetCDF_C_INCLUDE_DIR}")
endif()

mark_as_advanced(NetCDF_C_LIBRARY NetCDF_C_INCLUDE_DIR NC_CONFIG_EXECUTABLE)
