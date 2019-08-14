

ASSERT_DEFINED(Anasazi_SOURCE_DIR)
ASSERT_DEFINED(${PACKAGE_NAME}_ENABLE_Triutils)
ASSERT_DEFINED(Tpetra_INST_FLOAT)

IF (${PACKAGE_NAME}_ENABLE_Triutils)
  IF (Tpetra_INST_FLOAT)

   TRIBITS_ADD_EXECUTABLE_AND_TEST(
     Tpetra_BlockGMRES_hb_DoubleFloat_test
     SOURCES test_bl_gmres_hb_df.cpp 
     ARGS "--verbose"
     COMM serial mpi
     )

  ENDIF()
ENDIF()

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Tpetra_BlockGMRES_hb_test
  SOURCES test_bl_gmres_hb.cpp
  ARGS "--verbose"
  COMM serial mpi
  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  Tpetra_HybridGMRES_hb_test
  SOURCES test_hybrid_gmres_hb.cpp
  ARGS "--verbose --max-degree=10 --poly-type=Arnoldi"
       "--verbose --max-degree=10 --poly-type=Gmres"
       "--verbose --max-degree=10 --poly-type=Roots"
  COMM serial mpi
  )

IF (${PACKAGE_NAME}_ENABLE_Galeri)

 TRIBITS_ADD_EXECUTABLE(
  Tpetra_HybridGMRES_Galeri_multi
  SOURCES hybrid_gmres_galeri_multi.cpp
  COMM serial mpi
  )

ENDIF()

TRIBITS_COPY_FILES_TO_BINARY_DIR(Tpetra_BlockGMRES_hb_CopyFiles
  SOURCE_DIR ${Anasazi_SOURCE_DIR}/testmatrices
  SOURCE_FILES bcsstk14.hb
  EXEDEPS Tpetra_BlockGMRES_hb_test
  )

