#
# A) Package-specific configuration options
#
ASSERT_DEFINED(Belos_SOURCE_DIR)
#
# B) Define the header and source files (and directories)
#

#
# src
#

#SET(HEADERS "")
#SET(SOURCES "")

#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

#
# Core Files
#

#INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

#APPEND_SET(HEADERS
#   BelosKokkosAdapter.hpp
#  )

#APPEND_SET(SOURCES
#   ViewPlay.cpp
#   MVTestInternal.cpp
#   MVOPTester.cpp
#   PseudoBlockCGKokkosExFile.cpp
#   BlockGmresKokkosExFile.cpp
#  )

#
# C) Define the targets for package's library(s)
#
#TRIBITS_ADD_LIBRARY(
#  beloskokkos
#  HEADERS ${HEADERS}
#  SOURCES ${SOURCES}
#  DEPLIBS belos
#  )


TRIBITS_ADD_EXECUTABLE(
  FlDoubleSpmvDot
  SOURCES FlDoubleSpmvDot.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  SpmvVsDot
  SOURCES SpmvVsDot.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  TpetraFlDoubleSpmvDot
  SOURCES TpetraFlDoubleSpmvDot.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  TpetraFlDoubleSpmvOnly
  SOURCES TpetraFlDoubleSpmvOnly.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  TpetraSpmvVsDot
  SOURCES TpetraSpmvVsDot.cpp
  COMM serial mpi
  )

IF (Belos_ENABLE_Epetra)
  TRIBITS_ADD_EXECUTABLE(
    GaleriMatrixCreator
    SOURCES GaleriMatrixCreator.cpp
    COMM serial mpi
    )
ENDIF()

TRIBITS_ADD_EXECUTABLE(
  GmresIRStart
  SOURCES TpetraGmresIRStartScript.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  GmresPolyTpetra
  SOURCES BlockGmresPolyTpetraJ.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  CG_Test_J_Belos
  SOURCES test_CG_J.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  TpetraOrthog
  SOURCES TpetraOrthog.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  TpetraOrthogCoeffs
  SOURCES TpetraOrthogCoeffs.cpp
  COMM serial mpi
  )

TRIBITS_COPY_FILES_TO_BINARY_DIR(CopyTpetraMatrixMarketFiles
  SOURCE_DIR ${Belos_SOURCE_DIR}/tpetra/jennifer/
  SOURCE_FILES bcsstk13.mtx bcsstk12.mtx orsirr_1.mtx
  EXEDEPS GmresIRStart
  ) 
