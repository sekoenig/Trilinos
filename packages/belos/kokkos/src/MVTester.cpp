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
#include "BelosKokkosMultiVec.hpp"

#include "Teuchos_StandardCatchMacros.hpp"

int main(int argc, char *argv[])
{
  bool ierr;
  Kokkos::initialize();
  //bool verbose = false;
  bool verbose = true;
  /*if (argc>1) {
    if (argv[1][0]=='-' && argv[1][1]=='v') {
      verbose = true;
    }
  }*/

  bool success = true;
  try {
    // number of global elements
    int dim = 100;
    int blockSize = 5;

    // Create an Epetra_Matrix
    //Teuchos::RCP<Epetra_CrsMatrix> A = Teuchos::rcp( new Epetra_CrsMatrix(Epetra_DataAccess::Copy, *Map, &NumNz[0]) );

    // Issue several useful typedefs;
    typedef Belos::MultiVec<double> KMV;
    //typedef Belos::Operator<double> EOP; // unused

    // Create a Kokkos MultiVec for an initial std::vector to start the solver.
    // Note that this needs to have the same number of columns as the blocksize.
    Teuchos::RCP<Belos::KokkosMultiVec<double>> ivec = Teuchos::rcp( new Belos::KokkosMultiVec<double>(dim, blockSize) );
    ivec->MvRandom();

    // Create an output manager to handle the I/O from the solver
    Teuchos::RCP<Belos::OutputManager<double> > MyOM = Teuchos::rcp( new Belos::OutputManager<double>() );
    if (verbose) {
      MyOM->setVerbosity( Belos::Warnings );
    }

    // test the Epetra adapter multivector
    ierr = Belos::TestMultiVecTraits<double,KMV>(MyOM,ivec);
    if (ierr) {
      MyOM->print(Belos::Warnings,"*** EpetraAdapter PASSED TestMultiVecTraits()\n");
    }
    else {
      MyOM->print(Belos::Warnings,"*** EpetraAdapter FAILED TestMultiVecTraits() ***\n\n");
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
  Kokkos::finalize();
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
