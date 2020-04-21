

#
# A) Package-specific configuration options
#

#
# B) Define the header and source files (and directories)
#

#
# src
#

#SET(HEADERS "")
SET(SOURCES "")

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

#
# Core Files
#

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

APPEND_SET(HEADERS
   BelosKokkosAdapter.hpp
  )

APPEND_SET(SOURCES
   ViewPlay.cpp
   MVTestInternal.cpp
   MVOPTester.cpp
  )

#
# C) Define the targets for package's library(s)
#
TRIBITS_ADD_LIBRARY(
  beloskokkos
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
  DEPLIBS belos
  )

TRIBITS_ADD_EXECUTABLE(
  KokkosViewPlay
  SOURCES ViewPlay.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  KokkosMVOPTester
  SOURCES MVOPTester.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  MVTestInternal
  SOURCES MVTestInternal.cpp
  COMM serial mpi
  )


TRIBITS_COPY_FILES_TO_BINARY_DIR(CopyKokkosMatrixMarketFiles
  SOURCE_DIR ${Belos_SOURCE_DIR}/kokkos/src/
  SOURCE_FILES bcsstk13.mtx
  EXEDEPS KokkosMVOPTester
  ) 
