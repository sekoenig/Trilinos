

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
   PseudoBlockCGKokkosExFile.cpp
   BlockGmresKokkosExFile.cpp
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

TRIBITS_ADD_EXECUTABLE(
  PseudoBlockCGKokkosEx
  SOURCES PseudoBlockCGKokkosExFile.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  BlockGmresKokkosEx
  SOURCES BlockGmresKokkosExFile.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  FLOATBlockGmresKokkosEx
  SOURCES FLOATBlockGmresKokkosExFile.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  MultiGmresKokkosEx
  SOURCES MultiGmresKokkosEx.cpp 
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  GmresIRKokkos
  SOURCES GmresIRKokkos.cpp 
  COMM serial mpi
  )

IF (${PACKAGE_NAME}_ENABLE_Galeri)
TRIBITS_ADD_EXECUTABLE(
  GaleriMatixCreator
  SOURCES GaleriMatrixCreator.cpp 
  COMM serial mpi
  )
ENDIF (${PACKAGE_NAME}_ENABLE_Galeri)

IF (${PACKAGE_NAME}_ENABLE_Epetra)
TRIBITS_ADD_EXECUTABLE(
  PseudoBlockCGEpetraEx
  SOURCES PseudoBlockCGEpetraExFile.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  LabelEpetraTest
  SOURCES LabelEpetraTest.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  BlockGmresEpetraEx
  SOURCES BlockGmresEpetraExFile.cpp
  COMM serial mpi
  )
ENDIF (${PACKAGE_NAME}_ENABLE_Epetra)

IF (${PACKAGE_NAME}_ENABLE_Tpetra)
TRIBITS_ADD_EXECUTABLE(
  PseudoBlockCGTpetraEx
  SOURCES PseudoBlockCGTpetraExFile.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  BlockGmresTpetraEx
  SOURCES BlockGmresTpetraExFile.cpp
  COMM serial mpi
  )
ENDIF (${PACKAGE_NAME}_ENABLE_Tpetra)

TRIBITS_COPY_FILES_TO_BINARY_DIR(CopyKokkosMatrixMarketFiles
  SOURCE_DIR ${Belos_SOURCE_DIR}/kokkos/src/
  SOURCE_FILES bcsstk13.mtx bcsstk12.mtx orsirr_1.mtx
  EXEDEPS KokkosMVOPTester
  ) 
