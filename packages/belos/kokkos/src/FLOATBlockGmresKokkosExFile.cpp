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
#include "BelosKokkosJacobiOp.hpp"
#include "BelosKokkosSingleSolverOp.hpp"
#include "KokkosKernels_IOUtils.hpp"

int main(int argc, char *argv[]) {

bool success = true;
  Kokkos::initialize();
  {

  typedef float                            ST;
  typedef float                           ST2;
  typedef int                               OT;
  typedef Kokkos::DefaultExecutionSpace     EXSP;
  //typedef Kokkos::CudaSpace                EXSP;
  //typedef Kokkos::HostSpace               EXSP;
  //typedef Kokkos::CudaUVMSpace               EXSP;
  typedef Teuchos::ScalarTraits<ST>        SCT;
  typedef SCT::magnitudeType                MT;
  typedef Belos::KokkosMultiVec<ST, EXSP>         MV;
  typedef Belos::KokkosOperator<ST, OT, EXSP>       OP;
  typedef Belos::MultiVec<ST> KMV;
  typedef Belos::MultiVec<ST2> KMV2;
  typedef Belos::Operator<ST> KOP; 
  typedef Belos::Operator<ST2> KOP2; 
  typedef Belos::MultiVecTraits<ST,KMV>     MVT;
  typedef Belos::OperatorTraits<ST,KMV,KOP>  OPT;
  //typedef Belos::KokkosSolverOp<ST, OT, EXSP> KSOP; 
  typedef Belos::SolverOp<ST> SOP; 
  typedef Belos::KokkosSingleSolverOp<ST, ST2> SOP2; 

  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using std::cout;
  using std::endl;

bool verbose = true;
//try {
bool proc_verbose = false;
  int frequency = 25;        // frequency of status test output.
  int numrhs = 1;            // number of right-hand sides to solve for
  int maxiters = -1;         // maximum number of iterations allowed per linear system
  int maxsubspace = 50;      // maximum number of blocks the solver can use for the subspace
  int maxrestarts = 25;      // number of restarts allowed
  bool expresidual = false; // use explicit residual
  int blksize = 4;
  int teamsize = -1;
  bool precOn = false;
  std::string polyPrec = "none"; // Use none, poly, or single
  int polyDeg = 25;
  bool polyRandomRhs = true; // if True, poly may be different on each run!
  std::string jacobisolve("GEMV"); //Solve type for Jacobi prec- TRSV or GEMV. 
  std::string filename("bcsstk13.mtx"); // example matrix
  std::string rhsfile("");
  MT tol = 1.0e-6;           // relative residual tolerance

  Teuchos::CommandLineProcessor cmdp(false,true);
  cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
  cmdp.setOption("prec","noprec",&precOn,"Use Jacobi preconditioning.");
  cmdp.setOption("blksize",&blksize,"Block size for Jacobi prec.");
  cmdp.setOption("teamsize",&teamsize,"Team size for Jacobi prec operations.");
  cmdp.setOption("jacobisolve",&jacobisolve,"Solve type for Jacobi prec- TRSV or GEMV.");
  cmdp.setOption("polyprec",&polyPrec,"Use Poly preconditioning. Use 'none', 'poly' or 'single.'");
  cmdp.setOption("randRHS","probRHS",&polyRandomRhs,"Use a random rhs to generate polynomial.");
  cmdp.setOption("poly-deg",&polyDeg,"Degree of poly preconditioner.");
  cmdp.setOption("expres","impres",&expresidual,"Use explicit residual throughout.");
  cmdp.setOption("frequency",&frequency,"Solvers frequency for printing residuals (#iters).");
  cmdp.setOption("filename",&filename,"Filename for test matrix.  Acceptable file extensions: *.hb,*.mtx,*.triU,*.triS");
  cmdp.setOption("rhsfile",&rhsfile,"Filename for right-hand side. (*.mtx file) ");
  cmdp.setOption("tol",&tol,"Relative residual tolerance used by Gmres solver.");
  cmdp.setOption("num-rhs",&numrhs,"Number of right-hand sides to be solved for.");
  cmdp.setOption("max-iters",&maxiters,"Maximum number of iterations per linear system (-1 = adapted to problem/block size).");
  cmdp.setOption("max-subspace",&maxsubspace,"Maximum number of blocks the solver can use for the subspace.");
  cmdp.setOption("max-restarts",&maxrestarts,"Maximum number of restarts allowed for GMRES solver.");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }
  if (!verbose)
    frequency = -1;  // reset frequency if test is not verbose
  
  // Read in a matrix Market file and use it to test the Kokkos Operator.
  KokkosSparse::CrsMatrix<ST, OT, EXSP> crsMat = 
            KokkosKernels::Impl::read_kokkos_crst_matrix<KokkosSparse::CrsMatrix<ST, OT, EXSP>>(filename.c_str()); 
  RCP<Belos::KokkosOperator<ST, OT, EXSP>> A = 
            rcp(new Belos::KokkosOperator<ST,OT,EXSP>(crsMat));
  OT numRows = crsMat.numRows();

  //Test code for Jacobi operator: 
  RCP<Belos::KokkosJacobiOperator<ST, OT, EXSP>> JacobiPrec = 
            rcp(new Belos::KokkosJacobiOperator<ST,OT,EXSP>(crsMat,blksize,jacobisolve,teamsize));

  if(precOn) { 
  std::cout << "Setting up Jacobi prec: " << std::endl;
  JacobiPrec->SetUpJacobi();
  std::cout << "Exited Jacobi prec setup." << std::endl;
  }

  Teuchos::RCP<MV> X = Teuchos::rcp( new MV(numRows, numrhs) );
  X->MvInit(0.0);
  Teuchos::RCP<MV> B;
  if(rhsfile == ""){
    B = Teuchos::rcp( new MV(numRows, numrhs) );
    B->MvInit(1.0);
    std::cout << "Just initialized rhs vecs to 1." << std::endl;
  }
  else{
    Kokkos::View<ST**> bk("kokkos b", numRows, 1);
    auto bksub = Kokkos::subview(bk, Kokkos::ALL, 0);
    KokkosKernels::Impl::kk_read_MM_vector(bksub,rhsfile.c_str());
    B = Teuchos::rcp( new MV(bk) );
    std::cout << "Just read in rhs vecs from file." << std::endl;
  }

  proc_verbose = verbose;  /* Only print on the zero processor */
  // Create the timer for outer iteration:
  Teuchos::RCP<std::ostream> outputStream = Teuchos::rcp(&std::cout,false);
    Teuchos::RCP<Belos::OutputManager<ST> > printer_ = Teuchos::rcp( new Belos::OutputManager<ST>(Belos::TimingDetails,outputStream) );
    std::string solveIRLabel ="JBelos: GmresSolMgr total solve time";
#ifdef BELOS_TEUCHOS_TIME_MONITOR
    Teuchos::RCP<Teuchos::Time> timerIRSolve_ = Teuchos::TimeMonitor::getNewCounter(solveIRLabel);
#endif

  //
  // ********Other information used by block solver***********
  // *****************(can be user specified)******************
  //
  const int NumGlobalElements = B->GetGlobalLength();
  if (maxiters == -1)
    maxiters = NumGlobalElements - 1; // maximum number of iterations to run
  
  ParameterList belosList;
  belosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
  belosList.set( "Convergence Tolerance", tol );         // Relative convergence tolerance requested
  belosList.set( "Num Blocks", maxsubspace);             // Maximum number of blocks in Krylov factorization
  belosList.set( "Maximum Restarts", maxrestarts );      // Maximum number of restarts allowed
  belosList.set( "Explicit Residual Test", expresidual);      // use explicit residual

  if (verbose) {
    belosList.set( "Verbosity", Belos::Errors + Belos::Warnings +
		   Belos::StatusTestDetails + Belos::FinalSummary);// Removed Belos::TimingDetails to add my external timer.
    if (frequency > 0)
      belosList.set( "Output Frequency", frequency );
  }
  else
    belosList.set( "Verbosity", Belos::Errors + Belos::Warnings );
  
  //
  // Construct an unpreconditioned linear problem instance.
  //
  Belos::LinearProblem<ST,KMV,KOP> problem( A, X, B );
  if(polyPrec != "none"){
    std::string innerSolverType = "GmresPoly";
    RCP<Teuchos::ParameterList> innerList = rcp(new Teuchos::ParameterList() );
    innerList->set("Random RHS", polyRandomRhs );           // Use RHS from linear system or random vector
    innerList->set( "Maximum Degree", polyDeg );          // Maximum degree of the GMRES polynomial

    // Has to be KMV and KOP for the Linear Prob specialization.  
    // Can't plug something in here unless it has a Multivector Traits specialization. 
    // Go change solverOp to take a Belos::Multivector. .... Or do we already have this for Belos::Multivectors? 
    // What here is specific to Kokkos??
    if(polyPrec == "poly"){
      RCP<Belos::LinearProblem<ST,KMV,KOP>> innerProblem = rcp( new Belos::LinearProblem<ST,KMV,KOP>());
      innerProblem->setOperator(A);
      if(precOn){
        innerProblem->setRightPrec(JacobiPrec);
      }
      RCP<SOP> myPolyPrec = rcp(new SOP(innerProblem, innerList, innerSolverType));
      problem.setRightPrec(myPolyPrec);
    }
    else if(polyPrec == "single"){
      //Get a single precision version of Matrix A for single poly prec. 
      KokkosSparse::CrsMatrix<ST2, OT, EXSP> crsMatSing = 
        KokkosKernels::Impl::read_kokkos_crst_matrix<KokkosSparse::CrsMatrix<ST2, OT, EXSP>>(filename.c_str()); 
      RCP<Belos::KokkosOperator<ST2, OT, EXSP>> ASing = 
        rcp(new Belos::KokkosOperator<ST2,OT,EXSP>(crsMatSing));
      RCP<Belos::LinearProblem<ST2,KMV2,KOP2>> innerProblem = rcp( new Belos::LinearProblem<ST2,KMV2,KOP2>());
      innerProblem->setOperator(ASing);
      if(precOn){
        std::cout << "Warning: Jacobi not supported with single precision poly prec.  Using poly prec only." << std::endl;
      }
      RCP<SOP2> myPolyPrec = rcp(new SOP2(innerProblem, innerList, innerSolverType));
      problem.setRightPrec(myPolyPrec);
    }
    else{
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
          "BlockGmresKokksExFile: Invalid input for polyPrec.");
    }
  }
  else if(precOn) {
    problem.setRightPrec(JacobiPrec);
  }
  bool set = problem.setProblem();
  if (set == false) {
    if (proc_verbose)
      std::cout << std::endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;
    return -1;
  }
  //
  // *******************************************************************
  // **************Start the block Gmres iteration*************************
  // *******************************************************************
  //
  // Create an iterative solver manager.
  RCP< Belos::SolverManager<ST,KMV,KOP> > newSolver
    = rcp( new Belos::BlockGmresSolMgr<ST,KMV,KOP>(rcp(&problem,false), rcp(&belosList,false)) );

  //
  // **********Print out information about problem*******************
  //
  if (proc_verbose) {
    std::cout << std::endl << std::endl;
    std::cout << "Dimension of matrix: " << NumGlobalElements << std::endl;
    std::cout << "Number of right-hand sides: " << numrhs << std::endl;
    std::cout << "Max number of Gmres iterations: " << maxiters << std::endl;
    std::cout << "Relative residual tolerance: " << tol << std::endl;
    std::cout << std::endl;
  }
  //
  // Perform solve
  //
  Belos::ReturnType ret;
  { //scope guard for timer
#ifdef BELOS_TEUCHOS_TIME_MONITOR
  Teuchos::TimeMonitor slvtimer(*timerIRSolve_);
#endif
  ret = newSolver->solve();
  } //end timer scope guard

  //Print final timing details:
  Teuchos::TimeMonitor::summarize( printer_->stream(Belos::TimingDetails) );

  //
  // Compute actual residuals.
  //
  bool badRes = false;
  std::vector<ST> actual_resids( numrhs );
  std::vector<ST> rhs_norm( numrhs );
  MV resid(numRows, numrhs);
  OPT::Apply( *A, *X, resid );
  MVT::MvAddMv( -1.0, resid, 1.0, *B, resid );
  MVT::MvNorm( resid, actual_resids );
  MVT::MvNorm( *B, rhs_norm );
  if (proc_verbose) {
    std::cout<< "---------- Actual Residuals (normalized) ----------"<<std::endl<<std::endl;
    for ( int i=0; i<numrhs; i++) {
      ST actRes = actual_resids[i]/rhs_norm[i];
      std::cout<<"Problem "<<i<<" : \t"<< actRes <<std::endl;
      if (actRes > tol) badRes = true;
    }
  }

  if (ret!=Belos::Converged || badRes) {
    success = false;
    if (proc_verbose)
      std::cout << std::endl << "ERROR:  Belos did not converge!" << std::endl;
  } else {
    success = true;
    if (proc_verbose)
      std::cout << std::endl << "SUCCESS:  Belos converged!" << std::endl;
  }

#ifdef JENN_DEBUG
  cout << "The JENN_DEBUG variable is working!!!" << endl;
#endif

  //}
  //TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);
  }
  Kokkos::finalize();
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
