# ---------------------------------------------------------------------------
#  islay - Othello / Reversi move generation and perft
# ---------------------------------------------------------------------------
#  Version range: the lower bound (3.16) is old enough to be preinstalled on
#  virtually every still-supported distro / CI image (Ubuntu 20.04, Debian 11,
#  RHEL 8's EPEL, Homebrew, ...), while the upper bound opts in to the newer
#  policy defaults of modern CMake (up to 3.31) without emitting deprecation
#  warnings on brand-new CMake (4.x). This keeps the project buildable on the
#  widest possible set of machines.
# ---------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16...3.31)

project(islay
        VERSION 0.1.0
        DESCRIPTION "Othello/Reversi UCI and perft engine"
        LANGUAGES CXX)

# ---- Language standard ----------------------------------------------------
# C++20 gives us <bit> (std::popcount / std::countr_zero) and is supported by
# every mainstream compiler shipped in the last few years. We deliberately do
# NOT chase C++23/26 so the engine still builds on older-but-modern toolchains.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ---- Default to an optimized build ----------------------------------------
# Perft benefits from optimization, so pick Release unless the user (or a
# multi-config generator like Visual Studio / Xcode) says otherwise.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Debug Release RelWithDebInfo MinSizeRel)
endif()

# ---- User-facing knobs ----------------------------------------------------
option(ISLAY_NATIVE   "Tune the build for the CPU that compiles it (-march/-mcpu=native)"  ON)
option(ISLAY_LTO      "Enable link-time optimization for Release builds"                   ON)
option(ISLAY_ONNX     "Enable CPU ONNX policy/value inference"                            OFF)
set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime distribution with include/ and lib/")
option(ISLAY_USE_NEON "On arm64, use the all-NEON movegen kernels. The default arm64
                       path is the hybrid (2 directions on the integer ALUs, 2 on
                       NEON), which measures ~48% faster than all-NEON on wide
                       out-of-order cores (Apple Silicon). Enable this only for
                       narrow in-order arm cores that cannot co-issue."                   OFF)

option(ISLAY_BATCH_PERFT "x86 only: batch perft's depth-2 children so N sibling boards
                          share one vector (4 lanes AVX2 / 8 lanes AVX-512) -- ~14
                          instructions per board at N=8 vs 83 for a single board.
                          UNVALIDATED FOR SPEED: measured -30% on Apple NEON (only 2
                          lanes, weak vector throughput) and there is no x86 hardware
                          here to time it -- correctness is verified, speed is NOT.
                          Measure with `bench 11` on your x86 box before trusting it."   OFF)

include(CheckCXXCompilerFlag)

add_executable(islay
        main.cpp
        src/movegen.cpp
        src/board.cpp
        src/game.cpp
        src/mcts.cpp
        src/mcts_selftest.cpp
        src/search_controller.cpp
        src/neural.cpp
        src/selfplay.cpp
        src/options.cpp
        src/perft.cpp
        src/uci.cpp)

target_include_directories(islay PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
find_package(Threads REQUIRED)
target_link_libraries(islay PRIVATE Threads::Threads)

if(ISLAY_USE_NEON)
    target_compile_definitions(islay PRIVATE ISLAY_USE_NEON)
endif()

if(ISLAY_BATCH_PERFT)
    target_compile_definitions(islay PRIVATE ISLAY_BATCH_PERFT)
endif()

# ---------------------------------------------------------------------------
#  Architecture / optimization detection
# ---------------------------------------------------------------------------
#  The engine ships three move-generator back-ends selected at compile time via
#  predefined macros:
#     __AVX2__     -> AVX2 vector path      (x86-64)
#     __ARM_NEON   -> NEON vector path      (arm64 / Apple Silicon, always on)
#     (neither)    -> portable scalar path  (any C++20 compiler)
#
#  We do not hard-code any ISA. Instead we let the compiler light up whatever
#  the *build host* supports through -march=native (x86) or -mcpu=native (arm),
#  which in turn defines the macros above. This is the "auto-detect the machine
#  architecture and optimize" behaviour requested: build on the target box and
#  it specializes itself; cross-compile with ISLAY_NATIVE=OFF for a portable
#  binary that still runs everywhere via the scalar fallback.
# ---------------------------------------------------------------------------
if(MSVC)
    target_compile_options(islay PRIVATE
            $<$<CONFIG:Release>:/O2 /Oi /Ot>
            /permissive- /Zc:__cplusplus)
    if(ISLAY_NATIVE)
        # MSVC has no "native"; AVX2 is the sensible modern baseline when asked
        # to optimize. It is a no-op if the CPU running the binary lacks it only
        # in the sense that such CPUs are pre-2013 -- guard anyway.
        check_cxx_compiler_flag("/arch:AVX2" ISLAY_HAS_ARCH_AVX2)
        if(ISLAY_HAS_ARCH_AVX2)
            target_compile_options(islay PRIVATE /arch:AVX2)
        endif()
    endif()
else()
    target_compile_options(islay PRIVATE
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:RelWithDebInfo>:-O2 -g>
            $<$<CONFIG:Debug>:-O0 -g>
            -Wall -Wextra)

    if(ISLAY_NATIVE)
        # x86-64: -march=native picks up AVX2/BMI2/etc. of the build host.
        check_cxx_compiler_flag("-march=native" ISLAY_HAS_MARCH_NATIVE)
        if(ISLAY_HAS_MARCH_NATIVE)
            target_compile_options(islay PRIVATE -march=native)
        else()
            # arm64 / Apple Silicon: -march=native is often unsupported; use the
            # aarch64 spelling. NEON is baseline on arm64 regardless, so even if
            # both fail the vector path still compiles.
            check_cxx_compiler_flag("-mcpu=native" ISLAY_HAS_MCPU_NATIVE)
            if(ISLAY_HAS_MCPU_NATIVE)
                target_compile_options(islay PRIVATE -mcpu=native)
            endif()
        endif()
    endif()
endif()

# ---- Link-time optimization (best-effort) ---------------------------------
if(ISLAY_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT ISLAY_IPO_OK OUTPUT ISLAY_IPO_MSG LANGUAGES CXX)
    if(ISLAY_IPO_OK)
        set_property(TARGET islay PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
        set_property(TARGET islay PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
    else()
        message(STATUS "islay: LTO not supported by this toolchain (${ISLAY_IPO_MSG})")
    endif()
endif()

message(STATUS "islay: build type   = ${CMAKE_BUILD_TYPE}")
message(STATUS "islay: native tuning = ${ISLAY_NATIVE}")

if(ISLAY_ONNX)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
              HINTS ${ONNXRUNTIME_ROOT} PATH_SUFFIXES include include/onnxruntime/core/session)
    find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_ROOT} PATH_SUFFIXES lib)
    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
        message(FATAL_ERROR "ISLAY_ONNX needs ONNXRUNTIME_ROOT with C++ headers and library")
    endif()
    target_compile_definitions(islay PRIVATE ISLAY_ONNX_ENABLED)
    target_include_directories(islay SYSTEM PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(islay PRIVATE ${ONNXRUNTIME_LIBRARY})
    add_executable(islay_nn_probe tools/neural_probe.cpp src/neural.cpp src/board.cpp src/movegen.cpp src/game.cpp)
    target_include_directories(islay_nn_probe PRIVATE src)
    target_include_directories(islay_nn_probe SYSTEM PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_compile_definitions(islay_nn_probe PRIVATE $<TARGET_PROPERTY:islay,COMPILE_DEFINITIONS>)
    target_compile_options(islay_nn_probe PRIVATE $<TARGET_PROPERTY:islay,COMPILE_OPTIONS>)
    target_link_libraries(islay_nn_probe PRIVATE ${ONNXRUNTIME_LIBRARY} Threads::Threads)
    if(ISLAY_LTO AND ISLAY_IPO_OK)
        set_property(TARGET islay_nn_probe PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    endif()
endif()

# Serial self-play/verification share the production rules and PUCT; Python manages data.
add_executable(islay-selfplay src/selfplay_main.cpp src/selfplay.cpp src/mcts.cpp src/neural.cpp
                             src/game.cpp src/board.cpp src/movegen.cpp)
target_include_directories(islay-selfplay PRIVATE src)
target_compile_definitions(islay-selfplay PRIVATE $<TARGET_PROPERTY:islay,COMPILE_DEFINITIONS>)
target_compile_options(islay-selfplay PRIVATE $<TARGET_PROPERTY:islay,COMPILE_OPTIONS>)
target_link_libraries(islay-selfplay PRIVATE Threads::Threads)
if(ISLAY_ONNX)
    target_include_directories(islay-selfplay SYSTEM PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(islay-selfplay PRIVATE ${ONNXRUNTIME_LIBRARY})
endif()
if(ISLAY_LTO AND ISLAY_IPO_OK)
    set_property(TARGET islay-selfplay PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
endif()

# Private POSIX shm inference client: independent trees, no ONNX/model copy in workers.
if(UNIX)
    add_executable(islay-selfplay-worker src/selfplay_worker.cpp src/selfplay.cpp src/mcts.cpp
                                        src/game.cpp src/board.cpp src/movegen.cpp)
    target_include_directories(islay-selfplay-worker PRIVATE src)
    target_compile_definitions(islay-selfplay-worker PRIVATE $<TARGET_PROPERTY:islay,COMPILE_DEFINITIONS>)
    target_compile_options(islay-selfplay-worker PRIVATE $<TARGET_PROPERTY:islay,COMPILE_OPTIONS>)
    target_link_libraries(islay-selfplay-worker PRIVATE Threads::Threads)
    if(ISLAY_LTO AND ISLAY_IPO_OK)
        set_property(TARGET islay-selfplay-worker PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
    endif()
endif()
