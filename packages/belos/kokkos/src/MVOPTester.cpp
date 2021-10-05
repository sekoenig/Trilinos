//@HEADER
// ************************************************************************
//
//                 Belos: Block Linear Solvers Package
//                  Copyright 2004 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
//@HEADER
//
//  This test uses the MVOPTester.hpp functions to test the Belos adapters
//  to Kokkos.
//


#include "BelosConfigDefs.hpp"
#include "BelosMVOPTester.hpp"
#include "BelosOutputManager.hpp"
#include "BelosKokkosAdapter.hpp"

#include "KokkosKernels_IOUtils.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

int main(int argc, char *argv[])
{
  bool ierr;
  bool success = true;
  bool verbose = true;
  Kokkos::initialize();
  {
  //bool verbose = false;
  /*if (argc>1) {
    if (argv[1][0]=='-' && argv[1][1]=='v') {
      verbose = true;
    }
  }*/

  try {
    // number of global elements
    int dim = 50;
    int blockSize = 5;

    // Create an Epetra_Matrix
    //Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp( new Epetra_CrsMatrix(Epetra_DataAccess::Copy, *Map, &NumNz[0]) );

    // Issue several useful typedefs;
    typedef double ScalarType;
    typedef Belos::MultiVec<ScalarType> KMV;
    typedef Belos::Operator<ScalarType> KOP; // unused
    typedef Kokkos::DefaultExecutionSpace     EXSP;

    // Create a Kokkos MultiVec for an initial std::vector to start the solver.
    // Note that this needs to have the same number of columns as the blocksize.
    Teuchos::RCP<Belos::KokkosMultiVec<ScalarType>> ivec = Teuchos::rcp( new Belos::KokkosMultiVec<ScalarType>(dim, blockSize) );
    ivec->MvRandom();
    std::cout << "Printing the random multivec:" << std::endl;
    //ivec->MvPrint(std::cout);
    Teuchos::RCP<Belos::KokkosMultiVec<ScalarType>> ivec2 = Teuchos::rcp( new Belos::KokkosMultiVec<ScalarType>(dim, blockSize) );
    ivec2->MvRandom();
    std::cout << "Printing the random multivec 2:" << std::endl;
    //ivec2->MvPrint(std::cout);
    std::vector<ScalarType> norm1;
    std::vector<ScalarType> norm2;
    for(int i = 0; i<blockSize; i++){
      norm1.push_back(-10.0);
      norm2.push_back(-10.0);
    }
    ivec->MvNorm(norm1);
    ivec2->MvNorm(norm2);
    std::cout << "norms of multivec: " << std::endl;
    for(int i =0; i< blockSize; i++){
      std::cout << norm1[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "norms of multivec 2: " << std::endl;
    for(int i =0; i< blockSize; i++){
      std::cout << norm2[i] << " ";
    }
    std::cout << std::endl;


    // Create an output manager to handle the I/O from the solver
    Teuchos::RCP<Belos::OutputManager<ScalarType> > MyOM = Teuchos::rcp( new Belos::OutputManager<ScalarType>() );
    if (verbose) {
      MyOM->setVerbosity( Belos::Warnings );
    }

    // test the Epetra adapter multivector
    ierr = Belos::TestMultiVecTraits<ScalarType,KMV>(MyOM,ivec);
    if (ierr) {
      MyOM->print(Belos::Warnings,"*** KokkosAdapter PASSED TestMultiVecTraits()\n");
    }
    else {
      MyOM->print(Belos::Warnings,"*** KokkosAdapter FAILED TestMultiVecTraits() ***\n\n");
    }


    // Read in a matrix Market file and use it to test the Kokkos Operator.
    KokkosSparse::CrsMatrix<ScalarType, int, EXSP> crsMat = 
            KokkosKernels::Impl::read_kokkos_crst_matrix<KokkosSparse::CrsMatrix<ScalarType, int, EXSP>>("bcsstk13.mtx"); 
    Teuchos::RCP<Belos::KokkosCrsOperator<ScalarType, int, EXSP>> myOp = 
            Teuchos::rcp(new Belos::KokkosCrsOperator<ScalarType,int,EXSP>(crsMat));
    
    Teuchos::RCP<Belos::KokkosMultiVec<ScalarType>> ivec3 = Teuchos::rcp( new Belos::KokkosMultiVec<ScalarType>(2003, 2) );
    Teuchos::RCP<Belos::KokkosMultiVec<ScalarType>> ivec4 = Teuchos::rcp( new Belos::KokkosMultiVec<ScalarType>(2003, 1) );
      ivec4->MvInit(1.0);
    Teuchos::RCP<Belos::KokkosMultiVec<ScalarType>> ivec5 = Teuchos::rcp( new Belos::KokkosMultiVec<ScalarType>(2003, 1) );

    std::cout << "Testing first op apply:" << std::endl;
    myOp->Apply(*ivec4, *ivec5);
    std::cout << "Op apply 1 resut: " << std::endl;
    //ivec5->MvPrint(std::cout);   
    std::cout << "Testing 2nd op apply: " << std::endl;
    myOp->Apply(*ivec5, *ivec4);
    std::cout << "Op apply 2 resut: " << std::endl;
    //ivec4->MvPrint(std::cout);   

    ierr = Belos::TestOperatorTraits<ScalarType,KMV,KOP>(MyOM,ivec3,myOp);
    if (ierr) {
      MyOM->print(Belos::Warnings,"*** KokkosAdapter PASSED TestOperatorTraits()\n");
    }
    else {
      MyOM->print(Belos::Warnings,"*** KokkosAdapter FAILED TestOperatorTraits() ***\n\n");
    }

    if (!ierr) {
      success = false;
      MyOM->print(Belos::Warnings,"End Result: TEST FAILED\n");
    } else {
      success = true;
      MyOM->print(Belos::Warnings,"End Result: TEST PASSED\n");
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose,std::cerr,success);

  }
  Kokkos::finalize();
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
