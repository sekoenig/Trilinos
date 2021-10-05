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
using Belos::Warnings;
using Belos::OutputManager;

template <class ScalarType>
bool TestKokkosMultiVecOneScalar(const Teuchos::RCP<OutputManager<ScalarType> >& );

int main(int argc, char *argv[])
{
  bool ierr;
  bool success = true;
  Kokkos::initialize();
  {
  bool verbose = false;
  if (argc>1) {
    if (argv[1][0]=='-' && argv[1][1]=='v') {
      verbose = true;
    }
  }
    typedef double ScalarType;
    typedef float ScalarType2;
    typedef Belos::MultiVec<ScalarType> KMV;
    typedef Belos::Operator<ScalarType> KOP; 
    typedef Kokkos::DefaultExecutionSpace     EXSP;

    // Create an output manager to handle the I/O from the solver (defaults to std::cout).
    Teuchos::RCP<Belos::OutputManager<ScalarType> > myOutputMgr = Teuchos::rcp( new Belos::OutputManager<ScalarType>() );
    if (verbose) {
      myOutputMgr->setVerbosity( Warnings );
    }
try {
    // number of global elements
    int dim = 10;
    int blockSize = 5;
    std::vector<ScalarType> norms(blockSize);

    ierr = TestKokkosMultiVecOneScalar<ScalarType>(myOutputMgr);
    if (ierr) {
      myOutputMgr->print(Belos::Warnings,"*** KokkosAdapter PASSED TestKokkosMultiVecOneScalar()\n");
    }
    else {
      myOutputMgr->print(Belos::Warnings,"*** KokkosAdapter FAILED TestKokkosMultiVecOneScalar() ***\n\n");
    }

    // Create a Kokkos MultiVec for an initial std::vector to start the solver.
    // Note that this needs to have the same number of columns as the blocksize.
    Teuchos::RCP<Belos::KokkosMultiVec<ScalarType>> ivec = Teuchos::rcp( new Belos::KokkosMultiVec<ScalarType>(dim, blockSize) );
    ivec->MvRandom();

    // test the Epetra adapter multivector
    ierr = Belos::TestMultiVecTraits<ScalarType,KMV>(myOutputMgr,ivec);
    if (ierr) {
      myOutputMgr->print(Belos::Warnings,"*** KokkosAdapter PASSED TestMultiVecTraits()\n");
    }
    else {
      myOutputMgr->print(Belos::Warnings,"*** KokkosAdapter FAILED TestMultiVecTraits() ***\n\n");
    }

    // Read in a matrix Market file and use it to test the Kokkos Operator.
    KokkosSparse::CrsMatrix<ScalarType, int, EXSP> crsMat = 
            KokkosKernels::Impl::read_kokkos_crst_matrix<KokkosSparse::CrsMatrix<ScalarType, int, EXSP>>("bcsstk13.mtx"); 
    Teuchos::RCP<Belos::KokkosCrsOperator<ScalarType, int, EXSP>> myOp = 
            Teuchos::rcp(new Belos::KokkosCrsOperator<ScalarType,int,EXSP>(crsMat));
    
    Teuchos::RCP<Belos::KokkosMultiVec<ScalarType>> ivec3 = Teuchos::rcp( new Belos::KokkosMultiVec<ScalarType>(2003, 2) );

    ierr = Belos::TestOperatorTraits<ScalarType,KMV,KOP>(myOutputMgr,ivec3,myOp);
    if (ierr) {
      myOutputMgr->print(Belos::Warnings,"*** KokkosAdapter PASSED TestOperatorTraits()\n");
    }
    else {
      myOutputMgr->print(Belos::Warnings,"*** KokkosAdapter FAILED TestOperatorTraits() ***\n\n");
    }
    
    if (!ierr) {
      success = false;
      myOutputMgr->print(Belos::Warnings,"End Result: TEST FAILED\n");
    } else {
      success = true;
      myOutputMgr->print(Belos::Warnings,"End Result: TEST PASSED\n");
    }
    }//end try block


    TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose,std::cerr,success);
  }
  Kokkos::finalize();
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}

template <class ScalarType>
bool TestKokkosMultiVecOneScalar(const Teuchos::RCP<OutputManager<ScalarType> > & outputMgr){
  int dim = 10;
  int blockSize = 5;
  std::vector<ScalarType> norms(blockSize);

  /// Test KokkosMultiVec constructors:
  // Test constructor #1:
  Belos::KokkosMultiVec<ScalarType> myVec1("myLabel", dim, blockSize);
  if ( myVec1.GetNumberVecs() != blockSize ) {
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec constructor 1 returned wrong value "
      << "for GetNumberVecs()." << endl;
    return false;
  }
  if ( myVec1.GetGlobalLength() != dim ) {
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec constructor 1 returned wrong value "
      << "for GetGlobalLength()." << endl;
    return false;
  }
  myVec1.MvNorm(norms); 
  for(int i = 0; i < blockSize; i++){
    if( norms[i] != 0 ){
      outputMgr->stream(Warnings)
        << "*** ERROR *** KokkosMultiVec constructor 1 returned wrong nrm2 value. "
        << "Vector was not initialized to zeros." << endl;
      return false;
    }
  }
  // Test constructor #2:
  Belos::KokkosMultiVec<ScalarType> myVec2(dim, blockSize);
  if ( myVec2.GetNumberVecs() != blockSize ) {
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec constructor 2 returned wrong value "
      << "for GetNumberVecs()." << endl;
    return false;
  }
  if ( myVec2.GetGlobalLength() != dim ) {
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec constructor 2 returned wrong value "
      << "for GetGlobalLength()." << endl;
    return false;
  }
  myVec2.MvNorm(norms); 
  for(int i = 0; i < blockSize; i++){
    if( norms[i] != 0 ){
      outputMgr->stream(Warnings)
        << "*** ERROR *** KokkosMultiVec constructor 2 returned wrong nrm2 value. "
        << "Vector was not initialized to zeros." << endl;
      return false;
    }
  }
  // Test constructor #3:
  Belos::KokkosMultiVec<ScalarType> myVec3(2*dim);
  if ( myVec3.GetNumberVecs() != 1 ) {
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec constructor 3 returned wrong value "
      << "for GetNumberVecs()." << endl;
    return false;
  }
  if ( myVec3.GetGlobalLength() != 2*dim ) {
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec constructor 3 returned wrong value "
      << "for GetGlobalLength()." << endl;
    return false;
  }
  myVec3.MvNorm(norms); 
  if( norms[0] != 0 ){
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec constructor 3 returned wrong nrm2 value. "
      << "Vector was not initialized to zeros." << endl;
    return false;
  }
  // Test copy constructor (should deep copy).
  Belos::KokkosMultiVec<ScalarType> myVecCopy(myVec3);
  if ( myVecCopy.GetNumberVecs() != 1 ) {
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec copy constructor returned wrong value "
      << "for GetNumberVecs()." << endl;
    return false;
  }
  if ( myVecCopy.GetGlobalLength() != 2*dim ) {
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec copy constructor returned wrong value "
      << "for GetGlobalLength()." << endl;
    return false;
  }
  myVecCopy.MvRandom();
  myVecCopy.MvNorm(norms);
  if( norms[0] == 0 ){
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec MvRandom did not fill with random values. " << endl;
    return false;
  }
  myVec3.MvNorm(norms);
  if( norms[0] != 0 ){
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec copy constructor did not deep copy. " << endl;
    return false;
  }
  // Test assignment operator (should also deep copy). 
  myVecCopy = myVec2;
  if ( myVecCopy.GetNumberVecs() != blockSize ) {
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec assignment = returned wrong value "
      << "for GetNumberVecs()." << endl;
    return false;
  }
  if ( myVecCopy.GetGlobalLength() != dim ) {
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec assignment = returned wrong value "
      << "for GetGlobalLength()." << endl;
    return false;
  }
  myVec2.MvInit(3.0);
  myVecCopy.MvNorm(norms); 
  for(int i = 0; i < blockSize; i++){
    if( norms[i] != 0 ){
      outputMgr->stream(Warnings)
        << "*** ERROR *** KokkosMultiVec assignment = returned wrong nrm2 value. "
        << "Vector was not deep copied." << endl;
      return false;
    }
  }
  // Test view to multivec:
  int numCols2 = 4;
  int numRows2 = 60;
  Kokkos::View<ScalarType**> myView("View2MV", numRows2, numCols2);
  typename Kokkos::View<ScalarType**>::HostMirror myView_h("View2MV_host", numRows2, numCols2);
  Kokkos::deep_copy(myView, 42);
  Belos::KokkosMultiVec<ScalarType> myVec4( myView );
  if ( myVec4.GetNumberVecs() != myView.extent(1) ) {
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec view to multivec returned wrong value "
      << "for GetNumberVecs()." << endl;
    return false;
  }
  if ( myVec4.GetGlobalLength() != myView.extent(0) ) {
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec view to multivec returned wrong value "
      << "for GetGlobalLength()." << endl;
    return false;
  }
  Kokkos::deep_copy(myView, 55);
  Kokkos::deep_copy(myView_h, myVec4.GetInternalViewConst());
  if ( myView_h(5,1) != 42 ) {
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultivec view to multivec did not make a deep copy!" << endl;
    return false;
  }
  // Tesst view to multivec with shallow copy:
  Kokkos::deep_copy(myView, 100);
  Belos::KokkosMultiVec<ScalarType> myVec5( myView, false );
  if ( myVec5.GetNumberVecs() != myView.extent(1) ) {
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec view to multivec shallow returned wrong value "
      << "for GetNumberVecs()." << endl;
    return false;
  }
  if ( myVec5.GetGlobalLength() != myView.extent(0) ) {
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultiVec view to multivec shallow returned wrong value "
      << "for GetGlobalLength()." << endl;
    return false;
  }
  Kokkos::deep_copy(myView, 500);
  Kokkos::deep_copy(myView_h, myVec5.GetInternalViewConst());
  if ( myView_h(5,1) != 500 ) {
    outputMgr->stream(Warnings)
      << "*** ERROR *** KokkosMultivec view to multivec shallow made a deep copy!" << endl;
    return false;
  }
  // Test GetInternalViewNonConst:
  auto myView2 = myVec5.GetInternalViewNonConst();
  Kokkos::deep_copy(myView2, 0);
  std::vector<ScalarType> norms2(4);
  myVec5.MvNorm(norms2);
  for(int i = 0; i < myView2.extent(1); i++){
    if( norms[i] != 0 ){
      outputMgr->stream(Warnings)
        << "*** ERROR *** KokkosMultiVec GetInternalViewNonConst returned wrong nrm2 value. "
        << "Vector was not editable." << endl;
      return false;
    }
  }


  return true;
}
