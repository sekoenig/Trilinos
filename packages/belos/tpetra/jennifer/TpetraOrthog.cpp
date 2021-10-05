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
// Questions? Contact Jennifer A. Loe (jloe@sandia.gov)
//
// ************************************************************************
//@HEADER
//
// This driver reads a problem from a file, which can be in Harwell-Boeing (*.hb),
// Matrix Market (*.mtx), or triplet format (*.triU, *.triS).  The right-hand side
// from the problem, if it exists, will be used instead of multiple
// random right-hand-sides.  The initial guesses are all set to zero.
//
//
#include "BelosConfigDefs.hpp"
#include "BelosOutputManager.hpp"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include <Teuchos_StackedTimer.hpp>

#include "BelosTpetraAdapter.hpp"
#include "BelosTpetraOperator.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include <MatrixMarket_Tpetra.hpp>
#include "BelosOrthoManagerFactory.hpp"
#include "BelosOrthoManager.hpp"

using Teuchos::RCP;
using Teuchos::rcp;
using Teuchos::Array;

//Function Forward Declaration:
template <class ScalarType>
void orthogTpMVecs(Tpetra::MultiVector<ScalarType> & inputVecs, std::string orthogType, int blkSize);

int main(int argc, char *argv[]) {
  Tpetra::ScopeGuard tpetraScope(&argc,&argv);
  {
  typedef double                            ScalarType; 
  typedef int                               OT;
  typedef Tpetra::MultiVector<ScalarType>   MV;
  typedef Belos::MultiVecTraits<ScalarType,MV>     MVT;
  typedef Teuchos::SerialDenseMatrix<OT,ScalarType> MAT;


  RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
  int MyPID = Teuchos::rank(*comm);

  bool verbose = true;
  bool proc_verbose = ( verbose && (MyPID==0) ); /* Only print on the zero processor */
  int blockSize = 50; 
  std::string filename("orsirr_1.mtx"); // example matrix
  std::string orthoType("ICGS");
  bool use_stacked_timer = false;              

  Teuchos::CommandLineProcessor cmdp(false,true);
  cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
  cmdp.setOption("filename",&filename,"Filename for test matrix.  Matrix market format only.");
  cmdp.setOption("ortho", &orthoType, "Type of orthogonalization: ICGS, IMGS, DGKS.");
  cmdp.setOption("blkSize",&blockSize,"Number of vectors to orthogonalize at each step."); 
  cmdp.setOption("stacked-timer", "no-stacked-timer", &use_stacked_timer, "Run with or without stacked timer output");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }

  // Create the timer.
  RCP<std::ostream> outputStream = rcp(&std::cout,false);
  RCP<Belos::OutputManager<ScalarType> > printer_ = rcp( new Belos::OutputManager<ScalarType>(Belos::TimingDetails,outputStream) );
  std::string OrthoLabel = "Total Orthog time:";
#ifdef BELOS_TEUCHOS_TIME_MONITOR
    RCP<Teuchos::Time> timerOrtho_ = Teuchos::TimeMonitor::getNewCounter(OrthoLabel);
#endif

  // Set output stream and stacked timer:
  RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
  Teuchos::FancyOStream& out = *fancy;
  out.setOutputToRootOnly(0);
  // Set up timers
  Teuchos::RCP<Teuchos::StackedTimer> stacked_timer;
  if (use_stacked_timer){
    stacked_timer = rcp(new Teuchos::StackedTimer("Main"));
  }
  Teuchos::TimeMonitor::setStackedTimer(stacked_timer);
  

  //-----------------------------------------------------------
  // This code sets up a random multivec to orthogonalize.
  // ----------------------------------------------------------
  //Read CrsMats into Tpetra::Operator
  RCP<Tpetra::CrsMatrix<ScalarType>> A = Tpetra::MatrixMarket::Reader<Tpetra::CrsMatrix<ScalarType>>::readSparseFile(filename,comm);
  RCP<const Tpetra::Map<> > map = A->getDomainMap();

  RCP<Tpetra::MultiVector<ScalarType>> X1 = rcp( new Tpetra::MultiVector<ScalarType>(map, 1008) );
  X1->randomize();
  //-----------------------------------------------------------
  // End random multivec setup.
  //-----------------------------------------------------------

  { //scope guard for timer
  #ifdef BELOS_TEUCHOS_TIME_MONITOR
  Teuchos::TimeMonitor orthotimer(*timerOrtho_);
  #endif
  orthogTpMVecs(*X1, orthoType, blockSize);
  }

  /*// DEBUG: //Verify Orthog:
  RCP<MAT> bigDotAns  = rcp( new MAT(1008,1008));
  MVT::MvTransMv(1.0, *X1, *X1, *bigDotAns);
  std::cout << "Printed dot prod matrix for verification: " << std::endl;
  bigDotAns->print(std::cout);*/

  //Print final timing details:
  Teuchos::TimeMonitor::summarize( printer_->stream(Belos::TimingDetails) );

    if (use_stacked_timer) {
      stacked_timer->stop("Main");
      Teuchos::StackedTimer::OutputOptions options;
      options.output_fraction = options.output_histogram = options.output_minmax = true;
      stacked_timer->report(out, comm, options);
    }

  }// End Tpetra Scope Guard
  return 0;
}

template <class ScalarType>
void orthogTpMVecs(Tpetra::MultiVector<ScalarType> & inputVecs, std::string orthogType, int blkSize){
  typedef int                               OT;
  typedef typename Teuchos::SerialDenseMatrix<OT,ScalarType> MAT;
  typedef Tpetra::MultiVector<ScalarType>   MV;
  typedef Kokkos::DefaultExecutionSpace     EXSP;
  typedef Tpetra::Operator<ScalarType>             OP;
  typedef Belos::OperatorTraits<ScalarType,MV,OP> OPT;
  int numVecs = inputVecs.getNumVectors();
  int numRows = inputVecs.getGlobalLength();

  RCP<MAT> B = rcp(new MAT(blkSize, blkSize)); //Matrix for coeffs of X
  Array<RCP<MAT>> C; 

  Belos::OrthoManagerFactory<ScalarType, MV, OP> factory;
  Teuchos::RCP<Teuchos::ParameterList> paramsOrtho;   // can be null

  //Default OutputManager is std::cout.
  Teuchos::RCP<Belos::OutputManager<ScalarType> > myOutputMgr = Teuchos::rcp( new Belos::OutputManager<ScalarType>() );
  const Teuchos::RCP<Belos::OrthoManager<ScalarType,MV>> orthoMgr = factory.makeOrthoManager (orthogType, Teuchos::null, myOutputMgr, "Tpetra OrthoMgr", paramsOrtho); 
  
  int numLoops = numVecs/blkSize;
  int remainder = numVecs % blkSize;

  RCP<MV> vecBlock = inputVecs.subViewNonConst(Teuchos::Range1D(0,blkSize-1));
  orthoMgr->normalize(*vecBlock, B);
  std::vector<RCP<const MV>> pastVecArray;
  pastVecArray.push_back(vecBlock);
  Teuchos::ArrayView<RCP<const MV>> pastVecArrayView; 

  for(int k=1; k<numLoops; k++){
    pastVecArrayView = arrayViewFromVector(pastVecArray);
    vecBlock = inputVecs.subViewNonConst(Teuchos::Range1D(k*blkSize,k*blkSize + blkSize - 1));
    C.append(rcp(new MAT(blkSize, blkSize)));
    int rank = orthoMgr->projectAndNormalize(*vecBlock, C, B, pastVecArrayView);
    pastVecArray.push_back(vecBlock);
  }
  if( remainder > 0){
    pastVecArrayView = arrayViewFromVector(pastVecArray);
    vecBlock = inputVecs.subViewNonConst(Teuchos::Range1D(numVecs-remainder, numVecs-1));
    B = rcp(new MAT(remainder, remainder));
    C.append(rcp(new MAT(remainder, remainder)));
    int rank = orthoMgr->projectAndNormalize(*vecBlock, C, B, pastVecArrayView);
  }
}
