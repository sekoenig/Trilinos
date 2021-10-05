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
#include <Teuchos_Time.hpp>
#include <Teuchos_TimeMonitor.hpp>

#include "BelosKokkosAdapter.hpp"
#include "KokkosKernels_IOUtils.hpp"

int main(int argc, char *argv[]) {
  Kokkos::initialize();
  {

  typedef double                            ST; //Outer ST
  typedef int                               OT;
  typedef Kokkos::DefaultExecutionSpace     EXSP;
  typedef Belos::KokkosMultiVec<ST, EXSP>         MV;
  typedef Belos::KokkosCrsOperator<ST, OT, EXSP>       OP;
  typedef Belos::MultiVec<ST> KMV;
  typedef Belos::Operator<ST> KOP; 
  typedef Belos::MultiVecTraits<ST,KMV>     MVT;
  typedef Belos::OperatorTraits<ST,KMV,KOP>  OPT;

  typedef Teuchos::SerialDenseMatrix<OT,ST> MAT;

  using Teuchos::RCP;
  using Teuchos::rcp;

  int MyPID = 0;

  bool verbose = true;
  bool proc_verbose = false;
  int maxiters = 1000;         // maximum number of iterations of outer solver allowed per linear system
  std::string filename("orsirr_1.mtx"); // example matrix

  Teuchos::CommandLineProcessor cmdp(false,true);
  cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
  cmdp.setOption("filename",&filename,"Filename for test matrix.  Matrix market format only.");
  cmdp.setOption("iters",&maxiters,"Maximum number of iterations of outer solver."); 

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }

  proc_verbose = ( verbose && (MyPID==0) ); /* Only print on the zero processor */

  //Read CrsMats into Kokkos Operator
  KokkosSparse::CrsMatrix<ST, OT, EXSP> crsMat = 
            KokkosKernels::Impl::read_kokkos_crst_matrix<KokkosSparse::CrsMatrix<ST, OT, EXSP>>(filename.c_str()); 
  RCP<Belos::KokkosCrsOperator<ST, OT, EXSP>> A = 
            rcp(new Belos::KokkosCrsOperator<ST,OT,EXSP>(crsMat));
  OT numRows = crsMat.numRows();

  RCP<MV> X1 = rcp( new MV(numRows, 1) );
  RCP<MV> X2 = rcp( new MV(numRows, 50) );
  RCP<MV> B1 = rcp( new MV(numRows, 1) );
  RCP<MV> B2 = rcp( new MV(numRows, 50) );
  MVT::MvRandom( *X1 );
  MVT::MvRandom( *X2 );
  MVT::MvRandom( *B1 );
  MVT::MvRandom( *B2 );
  RCP<MAT> smDotAns  = rcp( new MAT(1,1));
  RCP<MAT> bigDotAns  = rcp( new MAT(50,50));

  int verbosity;
  if (verbose) {
    verbosity = Belos::Errors + Belos::Warnings + Belos::StatusTestDetails + Belos::FinalSummary + Belos::TimingDetails;
  }
  else
    verbosity = Belos::Errors + Belos::Warnings;

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

  //*************************************************************************
  // Warm-up loop:
  // ***************************************************************************
    for(int iter=0; iter < 2; iter++){
      OPT::Apply(*A, *X1, *B1);
      MVT::MvTransMv(1.0, *X1, *B1, *smDotAns);
      MVT::MvTimesMatAddMv(1.0, *X1, *smDotAns, 1.0, *B1);
      MVT::MvTransMv(1.0, *X2, *B2, *bigDotAns);
      MVT::MvTimesMatAddMv(1.0, *X2, *bigDotAns, 1.0, *B2);
      } // End warm-up loop 
  //

  //*************************************************************************
  // Main Test loop:
  // ***************************************************************************
    for(int iter=0; iter < maxiters; iter++){
      { //scope guard for timer
      #ifdef BELOS_TEUCHOS_TIME_MONITOR
      Teuchos::TimeMonitor slvtimer(*timerSpmv_);
      #endif
      OPT::Apply(*A, *X1, *B1);
      }
      { //scope guard for timer
      #ifdef BELOS_TEUCHOS_TIME_MONITOR
      Teuchos::TimeMonitor slvtimer(*timerTrans_);
      #endif
      MVT::MvTransMv(1.0, *X1, *B1, *smDotAns);
      }
      { //scope guard for timer
      #ifdef BELOS_TEUCHOS_TIME_MONITOR
      Teuchos::TimeMonitor slvtimer(*timerMvTimesMat_);
      #endif
      MVT::MvTimesMatAddMv(1.0, *X1, *smDotAns, 1.0, *B1);
      }
      { //scope guard for timer
      #ifdef BELOS_TEUCHOS_TIME_MONITOR
      Teuchos::TimeMonitor slvtimer(*timerTransBig_);
      #endif
      MVT::MvTransMv(1.0, *X2, *B2, *bigDotAns);
      }
      { //scope guard for timer
      #ifdef BELOS_TEUCHOS_TIME_MONITOR
      Teuchos::TimeMonitor slvtimer(*timerMvTimesMatBig_);
      #endif
      MVT::MvTimesMatAddMv(1.0, *X2, *bigDotAns, 1.0, *B2);
      }
    } // End main loop 
  //

  //Print final timing details:
  Teuchos::TimeMonitor::summarize( printer_->stream(Belos::TimingDetails) );

  }
  Kokkos::finalize();
  return 0;
}
