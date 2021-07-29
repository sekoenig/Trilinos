
TRIBITS_ADD_EXECUTABLE(
  GmresIfpackPolyTpetra
  SOURCES BlockGmresIfPackPolyTpetraJ.cpp
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE(
  FLOATGmresIfpackPolyTpetra
  SOURCES FLOATGmresIfPackPolyTpetraJ.cpp
  COMM serial mpi
  )

