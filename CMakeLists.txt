# CMakeList.txt: plik projektu CMake najwyższego poziomu, wykonaj konfigurację globalną
# i uwzględnij podprojekty w tym miejscu.
#
cmake_minimum_required (VERSION 3.8)

# Włącz przeładowywanie kodu XAML na gorąco dla kompilatorów MSVC, jeśli jest to obsługiwane.
if (POLICY CMP0141)
  cmake_policy(SET CMP0141 NEW)
  set(CMAKE_MSVC_DEBUG_INFORMATION_FORMAT "$<IF:$<AND:$<C_COMPILER_ID:MSVC>,$<CXX_COMPILER_ID:MSVC>>,$<$<CONFIG:Debug,RelWithDebInfo>:EditAndContinue>,$<$<CONFIG:Debug,RelWithDebInfo>:ProgramDatabase>>")
endif()

project ("CurveFittingCPLEX")
set(CMAKE_CXX_STANDARD 20)

# Uwzględnij podprojekty.
add_executable (CurveFittingCPLEX "CurveFittingCPLEX/CurveFittingCPLEX.cpp")

set(CPLEX_ROOT_PATH "C:/Program Files/IBM/ILOG/CPLEX_Studio_Community2212")

target_include_directories(CurveFittingCPLEX PUBLIC "${CMAKE_SOURCE_DIR}/include")
target_include_directories(CurveFittingCPLEX PUBLIC "${CPLEX_ROOT_PATH}/concert/include")
target_include_directories(CurveFittingCPLEX PUBLIC "${CPLEX_ROOT_PATH}/cplex/include")

target_link_directories(CurveFittingCPLEX PUBLIC "${CPLEX_ROOT_PATH}/concert/lib/x64_windows_msvc14/stat_mdd")
target_link_directories(CurveFittingCPLEX PUBLIC "${CPLEX_ROOT_PATH}/cplex/lib/x64_windows_msvc14/stat_mdd")

target_link_libraries(CurveFittingCPLEX PUBLIC concert cplex2212 ilocplex)