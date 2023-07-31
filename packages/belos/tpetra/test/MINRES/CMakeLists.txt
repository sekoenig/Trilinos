ASSERT_DEFINED(Anasazi_SOURCE_DIR)

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Tpetra_Minres_hb_test
  SOURCES test_minres_hb.cpp
  ARGS "--verbose"
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Tpetra_Kokkos_Minres_hb_test
  SOURCES test_kokkos_minres_hb.cpp
  ARGS "--verbose"
  COMM serial mpi
  )

ASSERT_DEFINED(Anasazi_SOURCE_DIR)
TRIBITS_COPY_FILES_TO_BINARY_DIR(Tpetra_Minres_hb_CopyFiles2
  SOURCE_DIR ${Anasazi_SOURCE_DIR}/testmatrices
  SOURCE_FILES bcsstk14.hb
  EXEDEPS Tpetra_Minres_hb_test
  )

