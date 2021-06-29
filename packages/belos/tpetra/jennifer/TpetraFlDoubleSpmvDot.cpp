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

#include "BelosTpetraAdapter.hpp"
#include "BelosTpetraOperator.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include <MatrixMarket_Tpetra.hpp>

int main(int argc, char *argv[]) {
  Tpetra::ScopeGuard tpetraScope(&argc,&argv);
  {

  typedef double                            ST; //Outer ST
  typedef float                             ST2;
  typedef int                               OT;
  typedef Tpetra::Operator<ST>             OP;
  typedef Tpetra::Operator<ST2>             OP2;
  typedef Tpetra::MultiVector<ST>          MV;
  typedef Tpetra::MultiVector<ST2>          MV2;

  using Teuchos::RCP;
  using Teuchos::rcp;

  RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
  int MyPID = Teuchos::rank(*comm);

  bool verbose = true;
  bool proc_verbose = ( verbose && (MyPID==0) ); /* Only print on the zero processor */
  int maxiters = 1000;         // maximum number of iterations of outer solver allowed per linear system
  std::string filename("orsirr_1.mtx"); // example matrix

  Teuchos::CommandLineProcessor cmdp(false,true);
  cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
  cmdp.setOption("filename",&filename,"Filename for test matrix.  Matrix market format only.");
  cmdp.setOption("iters",&maxiters,"Maximum number of iterations of outer solver."); 

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }


  //Read CrsMats into Tpetra::Operator
  RCP<OP> A = Tpetra::MatrixMarket::Reader<Tpetra::CrsMatrix<ST>>::readSparseFile(filename,comm);
  RCP<const Tpetra::Map<> > map = A->getDomainMap();
  RCP<OP2> A2 = Tpetra::MatrixMarket::Reader<Tpetra::CrsMatrix<ST2>>::readSparseFile(filename,comm);
  RCP<const Tpetra::Map<> > map2 = A2->getDomainMap();

  RCP<MV> X1 = rcp( new MV(map, 50) );
  RCP<MV2> X2 = rcp( new MV2(map2, 50) );
  RCP<MV> B1 = rcp( new MV(map, 50) );
  RCP<MV2> B2 = rcp( new MV2(map2, 50) );
  X1->randomize();
  X2->randomize();
  B1->randomize();
  B2->randomize();
  RCP<Tpetra::Map<>> localMap = rcp( new Tpetra::Map<>(50, map->getIndexBase(), map->getComm(), Tpetra::LocallyReplicated));
  MV bigDotAns(localMap, 50);
  RCP<Tpetra::Map<>> localMap2 = rcp( new Tpetra::Map<>(50, map2->getIndexBase(), map2->getComm(), Tpetra::LocallyReplicated));
  MV2 bigDotAns2(localMap2, 50);

  // Create the timer if we need to.
  RCP<std::ostream> outputStream = rcp(&std::cout,false);
    RCP<Belos::OutputManager<ST> > printer_ = rcp( new Belos::OutputManager<ST>(Belos::TimingDetails,outputStream) );
    std::string SpmvLabel = "SpMV double time:";
    std::string TransLabelBig = "MvTransMv nx50 double time:";
    std::string MvTimesMatLabelBig = "MvTimesMatAddMv big double time:";
    std::string SpmvLabel2 = "SpMV float time:";
    std::string TransLabelBig2 = "MvTransMv nx50 float time:";
    std::string MvTimesMatLabelBig2 = "MvTimesMatAddMv big float time:";
#ifdef BELOS_TEUCHOS_TIME_MONITOR
    RCP<Teuchos::Time> timerSpmv_ = Teuchos::TimeMonitor::getNewCounter(SpmvLabel);
    RCP<Teuchos::Time> timerTransBig_ = Teuchos::TimeMonitor::getNewCounter(TransLabelBig);
    RCP<Teuchos::Time> timerMvTimesMatBig_ = Teuchos::TimeMonitor::getNewCounter(MvTimesMatLabelBig);
    RCP<Teuchos::Time> timerSpmv2_ = Teuchos::TimeMonitor::getNewCounter(SpmvLabel2);
    RCP<Teuchos::Time> timerTransBig2_ = Teuchos::TimeMonitor::getNewCounter(TransLabelBig2);
    RCP<Teuchos::Time> timerMvTimesMatBig2_ = Teuchos::TimeMonitor::getNewCounter(MvTimesMatLabelBig2);
#endif

  //*************************************************************************
  // Main Test loop: Double
  // ***************************************************************************
    for(int iter=0; iter < maxiters; iter++){
      { //scope guard for timer
      #ifdef BELOS_TEUCHOS_TIME_MONITOR
      Teuchos::TimeMonitor slvtimer(*timerSpmv_);
      #endif
      A->apply(*X1, *B1);
      }
      { //scope guard for timer
      #ifdef BELOS_TEUCHOS_TIME_MONITOR
      Teuchos::TimeMonitor slvtimer(*timerTransBig_);
      #endif
      bigDotAns.multiply (Teuchos::CONJ_TRANS, Teuchos::NO_TRANS, 1.0, *X1, *B1, 0.0);
      }
      { //scope guard for timer
      #ifdef BELOS_TEUCHOS_TIME_MONITOR
      Teuchos::TimeMonitor slvtimer(*timerMvTimesMatBig_);
      #endif
      B1->multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, *X1, bigDotAns, 1.0);
      // B2 = 1.0 B2 + 1.0 X2 * bigDotAns
      }
    } // End main loop 
  //*************************************************************************
  // Main Test loop: Float
  // ***************************************************************************
    for(int iter=0; iter < maxiters; iter++){
      { //scope guard for timer
      #ifdef BELOS_TEUCHOS_TIME_MONITOR
      Teuchos::TimeMonitor slvtimer(*timerSpmv2_);
      #endif
      A2->apply(*X2, *B2);
      }
      { //scope guard for timer
      #ifdef BELOS_TEUCHOS_TIME_MONITOR
      Teuchos::TimeMonitor slvtimer(*timerTransBig2_);
      #endif
      bigDotAns2.multiply (Teuchos::CONJ_TRANS, Teuchos::NO_TRANS, 1.0, *X2, *B2, 0.0);
      }
      { //scope guard for timer
      #ifdef BELOS_TEUCHOS_TIME_MONITOR
      Teuchos::TimeMonitor slvtimer(*timerMvTimesMatBig2_);
      #endif
      B2->multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, *X2, bigDotAns2, 1.0);
      // B2 = 1.0 B2 + 1.0 X2 * bigDotAns
      }
    } // End main loop 
  //

  //Print final timing details:
  Teuchos::TimeMonitor::summarize( printer_->stream(Belos::TimingDetails) );

  return 0;
  }// End Tpetra Scope Guard
}
