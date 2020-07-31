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

#include "Teuchos_StandardCatchMacros.hpp"

using std::cout;
using std::endl;
using Teuchos::RCP;

int main(int argc, char *argv[])
{
  bool ierr;
  bool success = true;
  Kokkos::initialize();
  {
  //bool verbose = false;
  bool verbose = true;
  /*if (argc>1) {
    if (argv[1][0]=='-' && argv[1][1]=='v') {
      verbose = true;
    }
  }*/

//  try {
    // number of global elements
    int dim = 10;
    int blockSize = 5;

    // Create an Epetra_Matrix
    //Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp( new Epetra_CrsMatrix(Epetra_DataAccess::Copy, *Map, &NumNz[0]) );

    // Issue several useful typedefs;
    typedef double ScalarType;
    typedef float ScalarType2;
    typedef Belos::MultiVec<ScalarType> KMV;
    //typedef Belos::Operator<ScalarType> EOP; // unused

    // Create a Kokkos MultiVec for an initial std::vector to start the solver.
    // Note that this needs to have the same number of columns as the blocksize.
    Teuchos::RCP<Belos::KokkosMultiVec<ScalarType>> ivec = Teuchos::rcp( new Belos::KokkosMultiVec<ScalarType>(dim, blockSize) );
    ivec->MvRandom();
    std::cout << "Printing the random multivec:" << std::endl;
    ivec->MvPrint(std::cout);
    Teuchos::RCP<Belos::KokkosMultiVec<ScalarType>> ivec2 = Teuchos::rcp( new Belos::KokkosMultiVec<ScalarType>(dim, blockSize) );
    ivec2->MvRandom();
    std::cout << "Printing the random multivec 2:" << std::endl;
    ivec2->MvPrint(std::cout);
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

    std::cout << "Print ivec2, double." << std::endl;
    ivec2->MvPrint(cout);

    //Test new impl of MvTransMv with Gemv:
    RCP<Teuchos::SerialDenseMatrix<int, ScalarType>> denseMat = Teuchos::rcp( new Teuchos::SerialDenseMatrix<int, ScalarType>(5,1));
    Teuchos::RCP<Belos::KokkosMultiVec<ScalarType>> Avec = Teuchos::rcp( new Belos::KokkosMultiVec<ScalarType>(8,5) );
    Avec->MvInit(2.0);
    Teuchos::RCP<Belos::KokkosMultiVec<ScalarType>> multiVec = Teuchos::rcp( new Belos::KokkosMultiVec<ScalarType>(8,1) );
    multiVec->MvInit(3.0);
    multiVec->MvTransMv(1.0,*Avec,*denseMat);
    cout << "should be all 48: " << endl;
    denseMat->print(std::cout);




    
    Kokkos::View<double**,Kokkos::LayoutLeft> mv("test",4,2);
    Kokkos::Random_XorShift64_Pool<> pool(12371);
    Kokkos::fill_random(mv, pool, -1,1);
    Belos::KokkosMultiVec<ScalarType2> ivecfloat(mv);
    std::cout << "Print ivec, float." << std::endl;
    ivecfloat.MvPrint(cout);

    //Teuchos::RCP<Belos::KokkosMultiVec<ScalarType2>> ivec2float 
    //      = Teuchos::rcp( new Belos::KokkosMultiVec<ScalarType2>(ivec2->myView()) );
    Belos::KokkosMultiVec<ScalarType2> ivec2float(ivec2->myView);
    std::cout << "Print ivec2, float." << std::endl;
    ivec2float.MvPrint(cout);

    cout << "Testing the MV copy construtor: " << endl;
    Belos::KokkosMultiVec<ScalarType> copyvec(*ivec2);
    cout << "Here is the copied MV:" << endl;
    copyvec.MvPrint(cout);
    cout << "Changing orig vec..." << endl;
    ivec2->MvInit(3445.0);
    cout << "Did copyvec change? " << endl;
    copyvec.MvPrint(cout);

    int numvecs2 = 5;
    std::vector<int> ind(numvecs2);
    ind[0] = 0; 
    ind[1] = 4; 
    ind[2] = 2; 
    ind[3] = 2; 
    ind[4] = 3;

    Teuchos::RCP<Belos::KokkosMultiVec<ScalarType>> ivec3 = 
        Teuchos::rcp(dynamic_cast<Belos::KokkosMultiVec<ScalarType> *>(ivec2->CloneCopy(ind)));
    std::cout << "Print cols 0 4 2 2 3 of vec 2:" << std::endl;
    //ivec3->MvPrint(cout);

    ivec->MvInit(-1.0);
    std::cout << "Print ivec, should be -1's." << std::endl;
    //ivec->MvPrint(cout);
    ivec2->MvInit(3.0);
    std::cout << "Print ivec2, should be 3's." << std::endl;
    //ivec2->MvPrint(cout);
    ivec->MvAddMv(-5.0,*ivec,7.0,*ivec2);
    std::cout << "Print ivec, should be 26's." << std::endl;
    //ivec->MvPrint(cout);

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

    if (!ierr) {
      success = false;
      MyOM->print(Belos::Warnings,"End Result: TEST FAILED\n");
    } else {
      success = true;
      MyOM->print(Belos::Warnings,"End Result: TEST PASSED\n");
    }
 // }


  //TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose,std::cerr,success);
  }
  Kokkos::finalize();
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
