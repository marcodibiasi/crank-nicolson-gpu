cmake_minimum_required(VERSION 3.16)
project(CrankNicolson_ocl LANGUAGES C)

# --- Standard C ---
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_compile_options(-Wno-error)

# --- Sorgenti ---
set(SRC
    src/main.c
    src/conjugate_gradient/conjugate_gradient.c
    src/crank_nicolson/crank_nicolson.c
    src/pgm_utils/pgm_utils.c
    src/config/config.c
)

# --- Eseguibile ---
add_executable(cn ${SRC})

# --- Include directories ---
target_include_directories(cn PRIVATE
    src
    src/conjugate_gradient
    src/crank_nicolson
    src/pgm_utils
    src/config
)

# --- Warning e ottimizzazioni ---
target_compile_options(cn PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

# Ottimizzazioni per tipo di build
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(cn PRIVATE -O3)
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(cn PRIVATE -g)
endif()

# --- OpenCL (dipendenza di sistema) ---
find_package(OpenCL REQUIRED)
target_link_libraries(cn PRIVATE OpenCL::OpenCL)

# --- cJSON (FetchContent) ---
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

include(FetchContent)

FetchContent_Declare(
    cjson
    GIT_REPOSITORY https://github.com/DaveGamble/cJSON.git
    GIT_TAG v1.7.18
)

set(ENABLE_CJSON_TEST OFF CACHE BOOL "" FORCE)
set(ENABLE_CJSON_UTILS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(cjson)

target_include_directories(cn PRIVATE ${cjson_SOURCE_DIR}/)
target_link_libraries(cn PRIVATE cjson)

# --- stb_image (FetchContent) ---
FetchContent_Declare(
    stb
    GIT_REPOSITORY https://github.com/nothings/stb.git
    GIT_TAG master
)

FetchContent_MakeAvailable(stb)

# Includi la cartella contenente stb_image_write.h
target_include_directories(cn PRIVATE ${stb_SOURCE_DIR})

