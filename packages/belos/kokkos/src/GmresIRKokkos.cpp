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
  typedef float                             ST;
  typedef int                               OT;
  typedef Kokkos::DefaultExecutionSpace     EXSP;
  //typedef Teuchos::ScalarTraits<ST>        SCT;
  //typedef SCT::magnitudeType                MT;
  //typedef Belos::KokkosMultiVec<ST2>         MV2;
  //typedef Belos::KokkosOperator<ST, OT, EXSP>       OP;
  typedef Belos::MultiVec<ST> KMV;
  typedef Belos::Operator<ST> KOP; 
  typedef Belos::SolverOp<ST> SOP; 
  typedef Belos::MultiVec<ST2> KMV2;
  typedef Belos::Operator<ST2> KOP2; 

  // These only used at end for computing residuals, so use second scalar type for now.
  typedef Belos::MultiVecTraits<ST2,KMV2>     MVT;
  typedef Belos::OperatorTraits<ST2,KMV2,KOP2>  OPT;

  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

bool verbose = true;
//try {
bool proc_verbose = false;
  int frequency = 0;       // frequency of status test output.
  int numrhs = 1;            // number of right-hand sides to solve for
  int maxiters = 40;         // maximum number of iterations allowed per linear system
  int maxsubspace = 50;      // maximum number of blocks the solver can use for the subspace
  //int maxrestarts = 50;      // number of restarts allowed
  std::string filename("orsirr_1.mtx"); // example matrix
  std::string rhsfile("");
  double tol = 1.0e-8;           // relative residual tolerance
  bool converged = false;  
  bool verboseTimes = false; //Show timing at every low precision Gmres iter?  
  int blksize = 4;
  int teamsize = -1;
  bool precOn = false;
  std::string polyPrec = "none";
  int polyDeg = 25;
  bool polyRandomRhs = true; // if True, poly may be different on each run!
  std::string jacobisolve("GEMV"); //Solve type for Jacobi prec- TRSV or GEMV. 

  Teuchos::CommandLineProcessor cmdp(false,true);
  cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
  cmdp.setOption("verboseTimes","quietTimes",&verbose,"Print timings at every Gmres run.");
  cmdp.setOption("prec","noprec",&precOn,"Use ILU preconditioning.");
  cmdp.setOption("blksize",&blksize,"Block size for Jacobi prec.");
  cmdp.setOption("teamsize",&teamsize,"Team size for Jacobi prec operations.");
  cmdp.setOption("jacobisolve",&jacobisolve,"Solve type for Jacobi prec- TRSV or GEMV.");
  cmdp.setOption("polyprec",&polyPrec,"Use Poly preconditioning. Options are 'none' or 'poly' ('single'=='poly' for GMRES-IR.)");
  cmdp.setOption("randRHS","probRHS",&polyRandomRhs,"Use a random rhs to generate polynomial.");
  cmdp.setOption("poly-deg",&polyDeg,"Degree of poly preconditioner.");
  cmdp.setOption("frequency",&frequency,"Solvers frequency for printing residuals (#iters).");
  cmdp.setOption("filename",&filename,"Filename for test matrix.  Acceptable file extensions: *.hb,*.mtx,*.triU,*.triS");
  cmdp.setOption("rhsfile",&rhsfile,"Filename for right-hand side. (*.mtx file) ");
  //cmdp.setOption("tol",&tol,"Relative residual tolerance used by Gmres solver.");
  cmdp.setOption("outerTol",&tol,"Relative residual tolerance used by outer iterative refinement solver.");
  cmdp.setOption("num-rhs",&numrhs,"Number of right-hand sides to be solved for.");
  cmdp.setOption("max-iters",&maxiters,"Maximum number of iterations of outer solver (-1 = adapted to problem/block size).");
  cmdp.setOption("max-subspace",&maxsubspace,"Maximum number of blocks the inner solver can use for the subspace.");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }
  if (!verbose)
    frequency = -1;  // reset frequency if test is not verbose

  // Read in a matrix Market file and use it to test the Kokkos Operator.
  KokkosSparse::CrsMatrix<ST, OT, EXSP> crsMat = 
    KokkosKernels::Impl::read_kokkos_crst_matrix<KokkosSparse::CrsMatrix<ST, OT, EXSP>>(filename.c_str()); 
  KokkosSparse::CrsMatrix<ST2, OT, EXSP> crsMat2 = 
    KokkosKernels::Impl::read_kokkos_crst_matrix<KokkosSparse::CrsMatrix<ST2, OT, EXSP>>(filename.c_str()); 

  //Make CrsMats into Belos::Operator
  RCP<Belos::KokkosOperator<ST, OT, EXSP>> A1 = 
            rcp(new Belos::KokkosOperator<ST,OT,EXSP>(crsMat));
  RCP<Belos::KokkosOperator<ST2, OT, EXSP>> A2 = 
            rcp(new Belos::KokkosOperator<ST2,OT,EXSP>(crsMat2));
  OT numRows = crsMat.numRows();

  //Test code for ILU operator: 
  RCP<Belos::KokkosJacobiOperator<ST, OT, EXSP>> JacobiPrec = 
            rcp(new Belos::KokkosJacobiOperator<ST,OT,EXSP>(crsMat,blksize,jacobisolve,teamsize));

  if(precOn) { 
  std::cout << "Setting up Jacobi prec: " << std::endl;
  JacobiPrec->SetUpJacobi();
  std::cout << "Exited Jacobi prec setup." << std::endl;
  }
  
  Teuchos::RCP<Belos::KokkosMultiVec<ST>> X1 = Teuchos::rcp( new Belos::KokkosMultiVec<ST>(numRows, numrhs) );
  X1->MvInit(0.0);
  Teuchos::RCP<Belos::KokkosMultiVec<ST>> R1 = Teuchos::rcp( new Belos::KokkosMultiVec<ST>(numRows, numrhs) );
  Teuchos::RCP<Belos::KokkosMultiVec<ST2>> X2 = Teuchos::rcp( new Belos::KokkosMultiVec<ST2>(numRows, numrhs) );
  Teuchos::RCP<Belos::KokkosMultiVec<ST2>> Xfinal = Teuchos::rcp( new Belos::KokkosMultiVec<ST2>(numRows, numrhs) );
  Xfinal->MvInit(0.0);
  Teuchos::RCP<Belos::KokkosMultiVec<ST2>> R2;

  Teuchos::RCP<Belos::KokkosMultiVec<ST>> B1;
  Teuchos::RCP<Belos::KokkosMultiVec<ST2>> B2;
  if(rhsfile == ""){
    B1 = Teuchos::rcp( new Belos::KokkosMultiVec<ST>(numRows, numrhs) );
    B2 = Teuchos::rcp( new Belos::KokkosMultiVec<ST2>(numRows, numrhs) );
    B1->MvInit(1.0);
    B2->MvInit(1.0);
    R2 = Teuchos::rcp( new Belos::KokkosMultiVec<ST2>(numRows, numrhs) );
    R2->MvInit(1.0);// This is used to make R1 in initial loop. So make it equal B1=B2.  
    std::cout << "Just initialized rhs vecs to 1." << std::endl;
  }
  else{
    Kokkos::View<ST**> b1k("kokkos b1", numRows, 1);
    Kokkos::View<ST2**> b2k("kokkos b2", numRows, 1);
    Kokkos::View<ST2**> r2k("kokkos r2", numRows, 1);
    auto b1ksub = Kokkos::subview(b1k, Kokkos::ALL, 0);
    auto b2ksub = Kokkos::subview(b2k, Kokkos::ALL, 0);
    auto r2ksub = Kokkos::subview(r2k, Kokkos::ALL, 0);
    KokkosKernels::Impl::kk_read_MM_vector(b1ksub,rhsfile.c_str());
    KokkosKernels::Impl::kk_read_MM_vector(b2ksub,rhsfile.c_str());
    KokkosKernels::Impl::kk_read_MM_vector(r2ksub,rhsfile.c_str());
    B1 = Teuchos::rcp( new Belos::KokkosMultiVec<ST>(b1k) );
    B2 = Teuchos::rcp( new Belos::KokkosMultiVec<ST2>(b2k) );
    R2 = Teuchos::rcp( new Belos::KokkosMultiVec<ST2>(r2k) );
    //R2 = MVT::CloneCopy(dynamic_cast<Belos::KokkosMultiVec<ST2>>(*B2)); //Make R2 = B2. //Can't copy line above because constructor owns the view.
    //R2 = KMVT::CloneCopy(*B2); //Make R2 = B2. //Can't copy line above because constructor owns the view.
    std::cout << "Just read in rhs vecs from file." << std::endl;
  }
  proc_verbose = verbose;  /* Only print on the zero processor */

  //
  // ********Other information used by block solver***********
  // *****************(can be user specified)******************
  //
  const int NumGlobalElements = B1->GetGlobalLength();
  if (maxiters == -1)
    maxiters = NumGlobalElements - 1; // maximum number of iterations to run
  //
  ParameterList belosList;
  belosList.set( "Num Blocks", maxsubspace);             // Maximum number of blocks in Krylov factorization
  belosList.set( "Maximum Iterations", maxsubspace );       // Make inner solver stop at each restart.
  belosList.set( "Convergence Tolerance", 1e-10 );         // High convergence criteria- we just want 50 iters.
  //belosList.set("Implicit Residual Scaling", "Norm of RHS"); //Scale residual by b instead of R0 so initial guess works correctly
  //belosList.set("Explicit Residual Scaling","Norm of RHS");

  int verbosity;
  if (verbose) {
    verbosity = Belos::Errors + Belos::Warnings + Belos::StatusTestDetails + Belos::FinalSummary;
    if (frequency > 0)
      belosList.set( "Output Frequency", frequency );
  }
  else
    verbosity = Belos::Errors + Belos::Warnings;
  if(verboseTimes)
    verbosity += Belos::TimingDetails;
  belosList.set( "Verbosity", verbosity);

  // Create the timer if we need to.
  Teuchos::RCP<std::ostream> outputStream = Teuchos::rcp(&std::cout,false);
    Teuchos::RCP<Belos::OutputManager<ST2> > printer_ = Teuchos::rcp( new Belos::OutputManager<ST2>(Belos::TimingDetails,outputStream) );
    std::string solveIRLabel ="JBelos: GmresSolMgr total solve time";
#ifdef BELOS_TEUCHOS_TIME_MONITOR
    Teuchos::RCP<Teuchos::Time> timerIRSolve_ = Teuchos::TimeMonitor::getNewCounter(solveIRLabel);
#endif


  //*************************************************************************
  // Set up initial values:
  // ***************************************************************************

  Belos::LinearProblem<ST,KMV,KOP> problem1;
  problem1.setOperator(A1);
  if(polyPrec == "poly" || polyPrec == "single"){
    std::string innerSolverType = "GmresPoly";

    RCP<Belos::LinearProblem<ST,KMV,KOP>> innerProblem = rcp( new Belos::LinearProblem<ST,KMV,KOP>());
    innerProblem->setOperator(A1);
    RCP<Teuchos::ParameterList> innerList = rcp(new Teuchos::ParameterList() );
    innerList->set("Random RHS", polyRandomRhs );           // Use RHS from linear system or random vector
    innerList->set( "Maximum Degree", polyDeg );          // Maximum degree of the GMRES polynomial
    RCP<SOP> myPolyPrec = rcp(new SOP(innerProblem, innerList, innerSolverType));
    if(precOn){
      innerProblem->setRightPrec(JacobiPrec);
    }
    problem1.setRightPrec(myPolyPrec);
  }
  else if(polyPrec != "none"){
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
        "BlockGmresKokksExFile: Invalid input for polyPrec.");
  }
  else if(precOn) {
    problem1.setRightPrec(JacobiPrec);
  }
  RCP< Belos::SolverManager<ST,KMV,KOP> > Solver1
    = rcp( new Belos::BlockGmresSolMgr<ST,KMV,KOP>(rcp(&problem1,false), rcp(&belosList,false)) );
  int iter = 1; //initialize IR loop counter.

  //Vectors for computing norms:
  std::vector<ST2> actual_resids( numrhs );
  std::vector<ST2> rhs_norm( numrhs );
  MVT::MvNorm( *B2, rhs_norm );


  //*************************************************************************
  // Iterative Refinement loop:
  // ***************************************************************************
  { //scope guard for timer
#ifdef BELOS_TEUCHOS_TIME_MONITOR
  Teuchos::TimeMonitor slvtimer(*timerIRSolve_);
#endif
  while( !converged && iter <= maxiters ){
    X1->MvInit(0.0);
    problem1.setLHS(X1);
    R1 = Teuchos::rcp( new Belos::KokkosMultiVec<ST>(*R2) );//TODO write = operator so no memory allocate!
    problem1.setRHS(R1);
    Solver1->reset(Belos::Problem); //Typically equiv to problem.setProlem().
    Solver1->solve();
    // Copy previous soln into higher precision:
    X2 = Teuchos::rcp( new Belos::KokkosMultiVec<ST2>(*X1) );//TODO write = operator so no memory allocate!
    MVT::MvAddMv( 1.0, *Xfinal, 1.0, *X2, *Xfinal); //Compute new soln. 
    // Compute new residual in double:
    OPT::Apply( *A2, *Xfinal, *R2 );
    MVT::MvAddMv( -1.0, *R2, 1.0, *B2, *R2 );
    MVT::MvNorm( *R2, actual_resids );
    bool badRes = false;
    std::cout<< "---------- Actual Residuals (loop iter: " << iter << " ) ----------"<<std::endl<<std::endl;
    for ( int i=0; i<numrhs; i++) {
      ST2 actRes = actual_resids[i]/rhs_norm[i];
      std::cout<<"Problem "<<i<<" : \t"<< actRes <<std::endl;
      if (actRes > tol) badRes = true;
    }
    if (!badRes) converged = true;
    iter++;
  } //end IR loop
  } //end timer scope guard

  //Print final timing details:
  Teuchos::TimeMonitor::summarize( printer_->stream(Belos::TimingDetails) );

  //
  // Compute actual residuals.
  //
  bool badRes = false;
  if (proc_verbose) {
    std::cout<< "---------- Actual Residuals (normalized) ----------"<<std::endl<<std::endl;
    for ( int i=0; i<numrhs; i++) {
      ST2 actRes = actual_resids[i]/rhs_norm[i];
      std::cout<<"Problem "<<i<<" : \t"<< actRes <<std::endl;
      if (actRes > tol) badRes = true;
    }
  }  

  if (badRes) {
    success = false;
    if (proc_verbose)
      std::cout << std::endl << "ERROR:  Belos did not converge!" << std::endl;
  } else {
    success = true;
    if (proc_verbose)
      std::cout << std::endl << "SUCCESS:  Belos converged!" << std::endl;
  } 
  //}
  //TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);
  }
  Kokkos::finalize();
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
