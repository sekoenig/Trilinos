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

int main(int argc, char *argv[]) {
  Tpetra::ScopeGuard tpetraScope(&argc,&argv);
  {

  typedef double                            ST; //Outer ST
  typedef Tpetra::MultiVector<ST>          MV;

  using Teuchos::RCP;
  using Teuchos::rcp;

  RCP<const Teuchos::Comm<int> > comm = Tpetra::getDefaultComm();
  int MyPID = Teuchos::rank(*comm);

  bool verbose = true;
  bool proc_verbose = ( verbose && (MyPID==0) ); /* Only print on the zero processor */
  int maxiters = 1000;         // maximum number of iterations of outer solver allowed per linear system
  std::string filename("orsirr_1.mtx"); // example matrix
  bool use_stacked_timer = false;              

  Teuchos::CommandLineProcessor cmdp(false,true);
  cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
  cmdp.setOption("filename",&filename,"Filename for test matrix.  Matrix market format only.");
  cmdp.setOption("iters",&maxiters,"Maximum number of iterations of outer solver."); 
    cmdp.setOption("stacked-timer", "no-stacked-timer", &use_stacked_timer, "Run with or without stacked timer output");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }

  // Create the timer if we need to.
  RCP<std::ostream> outputStream = rcp(&std::cout,false);
    RCP<Belos::OutputManager<ST> > printer_ = rcp( new Belos::OutputManager<ST>(Belos::TimingDetails,outputStream) );
    std::string SpmvLabel = "SpMV time:";
    std::string TransLabel = "MvTransMv nx1 time:";
    std::string TransLabelBig = "MvTransMv nx50 time:";
    std::string MvTimesMatLabel = "MvTimesMatAddMv sm time:";
    std::string MvTimesMatLabelBig = "MvTimesMatAddMv big time:";
#ifdef BELOS_TEUCHOS_TIME_MONITOR
    RCP<Teuchos::Time> timerSpmv_ = Teuchos::TimeMonitor::getNewCounter(SpmvLabel);
    RCP<Teuchos::Time> timerTrans_ = Teuchos::TimeMonitor::getNewCounter(TransLabel);
    RCP<Teuchos::Time> timerTransBig_ = Teuchos::TimeMonitor::getNewCounter(TransLabelBig);
    RCP<Teuchos::Time> timerMvTimesMat_ = Teuchos::TimeMonitor::getNewCounter(MvTimesMatLabel);
    RCP<Teuchos::Time> timerMvTimesMatBig_ = Teuchos::TimeMonitor::getNewCounter(MvTimesMatLabelBig);
#endif

    // Set output stream and stacked timer:
    // (see packages/muelu/example/basic/Stratimikos.cpp)
    RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    Teuchos::FancyOStream& out = *fancy;
    out.setOutputToRootOnly(0);
    // Set up timers
    Teuchos::RCP<Teuchos::StackedTimer> stacked_timer;
    if (use_stacked_timer){
      stacked_timer = rcp(new Teuchos::StackedTimer("Main"));
    }
    Teuchos::TimeMonitor::setStackedTimer(stacked_timer);

  //Read CrsMats into Tpetra::Operator
  RCP<Tpetra::CrsMatrix<ST>> A = Tpetra::MatrixMarket::Reader<Tpetra::CrsMatrix<ST>>::readSparseFile(filename,comm);
  RCP<const Tpetra::Map<> > map = A->getDomainMap();

  RCP<MV> X1 = rcp( new MV(map, 1) );
  RCP<MV> X2 = rcp( new MV(map, 50) );
  RCP<MV> B1 = rcp( new MV(map, 1) );
  RCP<MV> B2 = rcp( new MV(map, 50) );
  X1->randomize();
  X2->randomize();
  B1->randomize();
  B2->randomize();
  ST smDotAns;
  RCP<Tpetra::Map<>> localMap = rcp( new Tpetra::Map<>(50, map->getIndexBase(), map->getComm(), Tpetra::LocallyReplicated));
  MV bigDotAns(localMap, 50);


  //*************************************************************************
  // Main Test loop:
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
      Teuchos::TimeMonitor slvtimer(*timerTrans_);
      #endif

      //MVT::MvTransMv(1.0, *X1, *B1, *smDotAns);
      X1->dot (*B1, Teuchos::ArrayView<ST>(&smDotAns,1));
      }
      { //scope guard for timer
      #ifdef BELOS_TEUCHOS_TIME_MONITOR
      Teuchos::TimeMonitor slvtimer(*timerMvTimesMat_);
      #endif
      //X1 and B1 both have one vector here. 
      //MVT::MvTimesMatAddMv(1.0, *X1, smDotAns, 1.0, *B1);
      //B1 = X1*smDotAns + 1.0*B1
      B1->update(smDotAns,*X1,1.0);
      }
      { //scope guard for timer
      #ifdef BELOS_TEUCHOS_TIME_MONITOR
      Teuchos::TimeMonitor slvtimer(*timerTransBig_);
      #endif
      //MVT::MvTransMv(1.0, *X2, *B2, *bigDotAns);

      bigDotAns.multiply (Teuchos::CONJ_TRANS, Teuchos::NO_TRANS, 1.0, *X2, *B2, 0.0);
      }
      { //scope guard for timer
      #ifdef BELOS_TEUCHOS_TIME_MONITOR
      Teuchos::TimeMonitor slvtimer(*timerMvTimesMatBig_);
      #endif
      //MVT::MvTimesMatAddMv(1.0, *X2, *bigDotAns, 1.0, *B2);
      B2->multiply(Teuchos::NO_TRANS, Teuchos::NO_TRANS, 1.0, *X2, bigDotAns, 1.0);
      // B2 = 1.0B2 + 1.0 X2 * bigDotAns
      }
    } // End main loop 
  //

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
