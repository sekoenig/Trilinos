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
// This driver reads a problem from a file, which can be in Harwell-Boeing (*.hb),
// Matrix Market (*.mtx), or triplet format (*.triU, *.triS).  The right-hand side
// from the problem, if it exists, will be used instead of multiple
// random right-hand-sides.  The initial guesses are all set to zero.
//
// NOTE: No preconditioner is used in this example.
//
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosBlockGmresSolMgr.hpp"
#include "BelosOutputManager.hpp"
#include "BelosSolverOp.hpp"

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include "BelosKokkosAdapter.hpp"
#include "KokkosKernels_IOUtils.hpp"
#include "BelosKokkosJacobiOp.hpp"

int main(int argc, char *argv[]) {

bool success = true;
  Kokkos::initialize();
  {

  typedef double                            ST2;
  typedef float                             ST1;
  typedef int                               OT;
  typedef Kokkos::DefaultExecutionSpace     EXSP;
  //typedef Teuchos::ScalarTraits<ST>        SCT;
  //typedef SCT::magnitudeType                MT;
  //typedef Belos::KokkosMultiVec<ST>         MV;
  //typedef Belos::KokkosCrsOperator<ST, OT, EXSP>       OP;
  typedef Belos::MultiVec<ST1> KMV1;
  typedef Belos::Operator<ST1> KOP1; 
  typedef Belos::SolverOp<ST1> SOP1; 
  typedef Belos::MultiVec<ST2> KMV2;
  typedef Belos::Operator<ST2> KOP2; 

  // These only used at end for computing residuals, so use second scalar type for now.
  typedef Belos::MultiVecTraits<ST2,KMV2>     MVT2;
  typedef Belos::OperatorTraits<ST2,KMV2,KOP2>  OPT2;
  typedef Belos::MultiVecTraits<ST1,KMV1>     MVT1;
  typedef Belos::OperatorTraits<ST1,KMV1,KOP1>  OPT1;

  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

bool verbose = true;
//try {
  int maxiters = 40;         // maximum number of iterations allowed per linear system
  std::string filename("orsirr_1.mtx"); // example matrix
  int numrhs = 1;

  Teuchos::CommandLineProcessor cmdp(false,true);
  cmdp.setOption("filename",&filename,"Filename for test matrix.  Acceptable file extensions: *.hb,*.mtx,*.triU,*.triS");
  cmdp.setOption("max-iters",&maxiters,"Maximum number of iterations of outer solver (-1 = adapted to problem/block size).");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }

  // Read in a matrix Market file and use it to test the Kokkos Operator.
  KokkosSparse::CrsMatrix<ST1, OT, EXSP> crsMat = 
    KokkosKernels::Impl::read_kokkos_crst_matrix<KokkosSparse::CrsMatrix<ST1, OT, EXSP>>(filename.c_str()); 
  KokkosSparse::CrsMatrix<ST2, OT, EXSP> crsMat2 = 
    KokkosKernels::Impl::read_kokkos_crst_matrix<KokkosSparse::CrsMatrix<ST2, OT, EXSP>>(filename.c_str()); 

  //Make CrsMats into Belos::Operator
  RCP<Belos::KokkosCrsOperator<ST1, OT, EXSP>> A1 = 
            rcp(new Belos::KokkosCrsOperator<ST1,OT,EXSP>(crsMat));
  RCP<Belos::KokkosCrsOperator<ST2, OT, EXSP>> A2 = 
            rcp(new Belos::KokkosCrsOperator<ST2,OT,EXSP>(crsMat2));
  OT numRows = crsMat.numRows();

  Teuchos::RCP<Belos::KokkosMultiVec<ST1>> B1 = Teuchos::rcp( new Belos::KokkosMultiVec<ST1>(numRows, numrhs) );
  B1->MvInit(1.0);
  Teuchos::RCP<Belos::KokkosMultiVec<ST1>> R1 = Teuchos::rcp( new Belos::KokkosMultiVec<ST1>(numRows, numrhs) );
  Teuchos::RCP<Belos::KokkosMultiVec<ST2>> B2 = Teuchos::rcp( new Belos::KokkosMultiVec<ST2>(numRows, numrhs) );
  B2->MvInit(1.0);
  Teuchos::RCP<Belos::KokkosMultiVec<ST2>> R2 = Teuchos::rcp( new Belos::KokkosMultiVec<ST2>(numRows, numrhs) );


  // Create the timer if we need to.
  Teuchos::RCP<std::ostream> outputStream = Teuchos::rcp(&std::cout,false);
    Teuchos::RCP<Belos::OutputManager<ST2> > printer_ = Teuchos::rcp( new Belos::OutputManager<ST2>(Belos::TimingDetails,outputStream) );
    std::string solveST1Label ="JBelos: Float time";
    std::string solveST2Label ="JBelos: Double time";
#ifdef BELOS_TEUCHOS_TIME_MONITOR
    Teuchos::RCP<Teuchos::Time> timerST1_ = Teuchos::TimeMonitor::getNewCounter(solveST1Label);
    Teuchos::RCP<Teuchos::Time> timerST2_ = Teuchos::TimeMonitor::getNewCounter(solveST2Label);
#endif


  //*************************************************************************
  // SpMV loop:
  // ***************************************************************************
  { //scope guard for timer
#ifdef BELOS_TEUCHOS_TIME_MONITOR
  Teuchos::TimeMonitor slvtimer(*timerST1_);
#endif
    for(int i = 0; i < maxiters; i++){
      OPT1::Apply( *A1, *B1, *R1 ); //R2 = A2*B2
    }
  } //end timer scope guard
  { //scope guard for timer
#ifdef BELOS_TEUCHOS_TIME_MONITOR
  Teuchos::TimeMonitor slvtimer(*timerST2_);
#endif
    for(int i = 0; i < maxiters; i++){
      OPT2::Apply( *A2, *B2, *R2 ); //R2 = A2*B2
    }
  } //end timer scope guard

  //Print final timing details:
  Teuchos::TimeMonitor::summarize( printer_->stream(Belos::TimingDetails) );

  //}
  //TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);
  }
  Kokkos::finalize();
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
