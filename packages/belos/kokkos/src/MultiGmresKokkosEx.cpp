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

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include "BelosKokkosAdapter.hpp"
#include "KokkosKernels_IOUtils.hpp"

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
  //typedef Belos::KokkosMultiVec<ST>         MV;
  //typedef Belos::KokkosCrsOperator<ST, OT, EXSP>       OP;
  typedef Belos::MultiVec<ST> KMV;
  typedef Belos::Operator<ST> KOP; 
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
  int frequency = 50;        // frequency of status test output.
  int numrhs = 1;            // number of right-hand sides to solve for
  int maxiters = 3000;         // maximum number of iterations allowed per linear system
  int switer = 500;           // Switch to 2nd solve after this num iters
  int maxsubspace = 50;      // maximum number of blocks the solver can use for the subspace
  int maxrestarts = 50;      // number of restarts allowed
  std::string filename("orsirr_1.mtx"); // example matrix
  double tol = 1.0e-6;           // relative residual tolerance

  Teuchos::CommandLineProcessor cmdp(false,true);
  cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
  cmdp.setOption("frequency",&frequency,"Solvers frequency for printing residuals (#iters).");
  cmdp.setOption("filename",&filename,"Filename for test matrix.  Acceptable file extensions: *.hb,*.mtx,*.triU,*.triS");
  cmdp.setOption("tol",&tol,"Relative residual tolerance used by Gmres solver.");
  cmdp.setOption("num-rhs",&numrhs,"Number of right-hand sides to be solved for.");
  cmdp.setOption("max-iters",&maxiters,"Maximum number of iterations per linear system (-1 = adapted to problem/block size).");
  cmdp.setOption("switch-iter",&switer,"Switch to scalarType 2 after this number of iterations.");
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
  KokkosSparse::CrsMatrix<ST2, OT, EXSP> crsMat2 = 
    KokkosKernels::Impl::read_kokkos_crst_matrix<KokkosSparse::CrsMatrix<ST2, OT, EXSP>>(filename.c_str()); 

  // This is how you copy one crs mat to another w/ different scalartype.
  // But you might not want to do this because ST=float -> ST2=double loses precision.
  // You don't know which of ST or ST2 is more precise, so read matrix in twice. 
  // Then GMRES acts funny and doesn't converge and the residuals increase.
  //Kokkos::View<ST2*> newValues("values", crsMat.values.extent(0));
  //Kokkos::deep_copy(newValues, crsMat.values);
  //KokkosSparse::CrsMatrix<ST2, OT, EXSP> crsMat2("Mat", crsMat.numCols(), newValues, crsMat.graph); //Convert scalar types. 
  
  //Make CrsMats into Belos::Operator
  RCP<Belos::KokkosCrsOperator<ST, OT, EXSP>> A1 = 
            rcp(new Belos::KokkosCrsOperator<ST,OT,EXSP>(crsMat));
  RCP<Belos::KokkosCrsOperator<ST2, OT, EXSP>> A2 = 
            rcp(new Belos::KokkosCrsOperator<ST2,OT,EXSP>(crsMat2));
  OT numRows = crsMat.numRows();
  
  Teuchos::RCP<Belos::KokkosMultiVec<ST>> X1 = Teuchos::rcp( new Belos::KokkosMultiVec<ST>(numRows, numrhs) );
  X1->MvInit(0.0);
  Teuchos::RCP<Belos::KokkosMultiVec<ST>> B1 = Teuchos::rcp( new Belos::KokkosMultiVec<ST>(numRows, numrhs) );
  B1->MvInit(1.0);
  Teuchos::RCP<Belos::KokkosMultiVec<ST2>> B2 = Teuchos::rcp( new Belos::KokkosMultiVec<ST2>(numRows, numrhs) );
  B2->MvInit(1.0);

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
  //belosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
  belosList.set( "Maximum Iterations", switer );       // Stop first solver after this many iters. 
  belosList.set( "Convergence Tolerance", tol );         // Relative convergence tolerance requested
  belosList.set( "Num Blocks", maxsubspace);             // Maximum number of blocks in Krylov factorization
  belosList.set( "Maximum Restarts", maxrestarts );      // Maximum number of restarts allowed
  belosList.set("Implicit Residual Scaling", "Norm of RHS"); //Scale residual by b instead of R0 so initial guess works correctly
  belosList.set("Explicit Residual Scaling","Norm of RHS");

  if (verbose) {
    belosList.set( "Verbosity", Belos::Errors + Belos::Warnings +
		   Belos::TimingDetails + Belos::StatusTestDetails + Belos::FinalSummary);
    if (frequency > 0)
      belosList.set( "Output Frequency", frequency );
  }
  else
    belosList.set( "Verbosity", Belos::Errors + Belos::Warnings );


  //*************************************************************************
  // Solve part 1:
  // ***************************************************************************

  Belos::LinearProblem<ST,KMV,KOP> problem1( A1, X1, B1 );
  bool set = problem1.setProblem();
  if (set == false) {
    if (proc_verbose)
      std::cout << std::endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;
    return -1;
  }
  RCP< Belos::SolverManager<ST,KMV,KOP> > Solver1
    = rcp( new Belos::BlockGmresSolMgr<ST,KMV,KOP>(rcp(&problem1,false), rcp(&belosList,false)) );
  Belos::ReturnType ret = Solver1->solve();


  //*************************************************************************
  // Solve part 2:
  // ***************************************************************************

  // Copy previous soln into new problem:
  Teuchos::RCP<Belos::KokkosMultiVec<ST2>> X2 = Teuchos::rcp( new Belos::KokkosMultiVec<ST2>(*X1) );
  //Teuchos::RCP<Belos::KokkosMultiVec<ST2>> R2 = Teuchos::rcp( new Belos::KokkosMultiVec<ST2>(numRows, numrhs) );
  // Compute new residual in double, so second solve starts where we want:
//  OPT::Apply( *A2, *X2, R2 );
//  MVT::MvAddMv( -1.0, R2, 1.0, *B2, R2 );
 // Teuchos::RCP<Belos::KokkosMultiVec<ST2>> X2 = Teuchos::rcp( new Belos::KokkosMultiVec<ST2>(numRows, numrhs) );
 // X2->MvInit(0.0);
  Belos::LinearProblem<ST2,KMV2,KOP2> problem2( A2, X2, B2 );
  set = problem2.setProblem();
  if (set == false) {
    if (proc_verbose)
      std::cout << std::endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;
    return -1;
  }
  belosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
  belosList.set( "Timer Label", "Solve2: " ); //new label second solve
  RCP< Belos::SolverManager<ST2,KMV2,KOP2> > Solver2
    = rcp( new Belos::BlockGmresSolMgr<ST2,KMV2,KOP2>(rcp(&problem2,false), rcp(&belosList,false)) );
  ret = Solver2->solve();


  //
  // Compute actual residuals.
  //
  bool badRes = false;
  std::vector<ST2> actual_resids( numrhs );
  std::vector<ST2> rhs_norm( numrhs );
  Belos::KokkosMultiVec<ST2> resid(numRows, numrhs);
  OPT::Apply( *A2, *X2, resid );
  MVT::MvAddMv( -1.0, resid, 1.0, *B2, resid );
  MVT::MvNorm( resid, actual_resids );
  MVT::MvNorm( *B2, rhs_norm );
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
  //}
  //TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);
  }
  Kokkos::finalize();
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
