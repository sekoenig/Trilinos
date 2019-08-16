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
#include "BelosEpetraAdapter.hpp"
#include "BelosGmresPolySolMgr.hpp"
#include "BelosBlockGmresSolMgr.hpp"

#include "EpetraExt_readEpetraLinearSystem.h"
#include <EpetraExt_CrsMatrixIn.h>
#include <EpetraExt_MultiVectorIn.h>
#include "Epetra_Map.h"
#ifdef EPETRA_MPI
  #include "Epetra_MpiComm.h"
#else
  #include "Epetra_SerialComm.h"
#endif
#include "Epetra_CrsMatrix.h"

#include "Ifpack.h"
#include <Ifpack_IlukGraph.h>
#include <Ifpack_CrsRiluk.h>

#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"

#include <Epetra_LinearProblem.h>
#include <Isorropia_Exception.hpp>
#include <Isorropia_Epetra.hpp>
#include <Isorropia_EpetraPartitioner.hpp>
#include <Isorropia_EpetraRedistributor.hpp>

int main(int argc, char *argv[]) {
  //
  int MyPID = 0;
#ifdef EPETRA_MPI
  // Initialize MPI
  MPI_Init(&argc,&argv);
  Epetra_MpiComm Comm(MPI_COMM_WORLD);
  MyPID = Comm.MyPID();
#else
  Epetra_SerialComm Comm;
#endif
  //
  typedef double                            ST;
  typedef Teuchos::ScalarTraits<ST>        SCT;
  typedef SCT::magnitudeType                MT;
  typedef Epetra_MultiVector                MV;
  typedef Epetra_Operator                   OP;
  typedef Belos::MultiVecTraits<ST,MV>     MVT;
  typedef Belos::OperatorTraits<ST,MV,OP>  OPT;

  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;

  bool verbose = false;
  bool success = true;
  try {
    bool proc_verbose = false;
    bool debug = false;
    bool userandomrhs = true;
    int frequency = -1;        // frequency of status test output.
    int blocksize = 1;         // blocksize
    int numrhs = 1;            // number of right-hand sides to solve for
    int maxiters = -1;         // maximum number of iterations allowed per linear system
    int maxdegree = 25;        // maximum degree of polynomial
    int maxsubspace = 50;      // maximum number of blocks the newSolver can use for the subspace
    int maxrestarts = 15;      // number of restarts allowed
    std::string outernewSolver("Block Gmres");
    std::string polytype("Arnoldi");
    std::string filename("orsirr1.hb");
    std::string rhsfile;
    std::string precond("right");
    MT tol = 1.0e-5;           // relative residual tolerance
    MT polytol = tol/10;       // relative residual tolerance for polynomial construction

    Teuchos::CommandLineProcessor cmdp(false,true);
    cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
    cmdp.setOption("debug","nondebug",&debug,"Print debugging information from newSolver.");
    cmdp.setOption("use-random-rhs","use-rhs",&userandomrhs,"Use linear system RHS or random RHS to generate polynomial.");
    cmdp.setOption("frequency",&frequency,"Solvers frequency for printing residuals (#iters).");
    cmdp.setOption("filename",&filename,"Filename for test matrix.  Acceptable file extensions: *.hb,*.mtx,*.triU,*.triS");
    cmdp.setOption("rhsfile",&rhsfile,"Filename for test rhs b. ");
    cmdp.setOption("outernewSolver",&outernewSolver,"Name of outer newSolver to be used with GMRES poly");
    cmdp.setOption("poly-type",&polytype,"Name of the polynomial to be generated.");
    cmdp.setOption("precond",&precond,"Preconditioning type (none, left, right).");
    cmdp.setOption("tol",&tol,"Relative residual tolerance used by GMRES newSolver.");
    cmdp.setOption("poly-tol",&polytol,"Relative residual tolerance used to construct the GMRES polynomial.");
    cmdp.setOption("num-rhs",&numrhs,"Number of right-hand sides to be solved for.");
    cmdp.setOption("block-size",&blocksize,"Block size used by GMRES.");
    cmdp.setOption("max-iters",&maxiters,"Maximum number of iterations per linear system (-1 = adapted to problem/block size).");
    cmdp.setOption("max-degree",&maxdegree,"Maximum degree of the GMRES polynomial.");
    cmdp.setOption("max-subspace",&maxsubspace,"Maximum number of blocks the newSolver can use for the subspace.");
    cmdp.setOption("max-restarts",&maxrestarts,"Maximum number of restarts allowed for GMRES newSolver.");
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      return -1;
    }
    if (!verbose)
      frequency = -1;  // reset frequency if test is not verbose
    //
    // Get the problem
    //
    const std::string::size_type ext_dot = filename.rfind(".");
    TEUCHOS_TEST_FOR_EXCEPT( ext_dot == std::string::npos );
    std::string ext = filename.substr(ext_dot+1);

    bool generate_rhs = (numrhs>1);
    RCP<Epetra_Map> Map;
    Epetra_CrsMatrix* ptrA = 0;
    RCP<Epetra_CrsMatrix> A;
    RCP<Epetra_MultiVector> B, X;
    RCP<Epetra_Vector> vecB, vecX;
    if ( ext != "mtx" && ext != "mm" )
    {
      EpetraExt::readEpetraLinearSystem(filename, Comm, &A, &Map, &vecX, &vecB);
    }
    else
    {
      int ret = EpetraExt::MatrixMarketFileToCrsMatrix(filename.c_str(), Comm, ptrA, false, false);
      A = Teuchos::rcp( ptrA );
      Map = Teuchos::rcp( new Epetra_Map( A->RowMap() ) );
      generate_rhs = true;
    }

    A->OptimizeStorage();
    proc_verbose = verbose && (MyPID==0);  /* Only print on the zero processor */

    if(rhsfile.compare("") != 0)
    {
      if(proc_verbose) std::cout << "Reading in rhs" << std::endl;
      Epetra_MultiVector *b=0;
      int finfo = EpetraExt::MatrixMarketFileToMultiVector( rhsfile.c_str() , *Map, b );
      if (finfo!=0 && MyPID==0)
        std::cout << "First rhs file could not be read in!!!, info = "<< finfo << std::endl;
      X = rcp( new Epetra_MultiVector( *Map, numrhs ) );
      X->PutScalar( 0.0 );
      B = rcp(b);
    }
    else 
    {
      // Check to see if the number of right-hand sides is the same as requested.
      if (generate_rhs) {
        X = rcp( new Epetra_MultiVector( *Map, numrhs ) );
        B = rcp( new Epetra_MultiVector( *Map, numrhs ) );
        X->Random();
        OPT::Apply( *A, *X, *B );
        X->PutScalar( 0.0 );
      }
      else {
       X = Teuchos::rcp_implicit_cast<Epetra_MultiVector>(vecX);
       B = Teuchos::rcp_implicit_cast<Epetra_MultiVector>(vecB);
      }
    }

    std::vector<double> norm_x(1), norm_b(1);
    MVT::MvNorm( *X, norm_x );
    MVT::MvNorm( *B, norm_b );
    if (proc_verbose) std::cout << "The norm of x is " << norm_x[0] << ", the norm of b is " << norm_b[0] << std::endl;

#ifdef EPETRA_MPI
    //Redistribute system with Zoltan:
    // Create the parameter list and fill it with values.
    Teuchos::ParameterList paramlist;
    Teuchos::ParameterList& sublist = paramlist.sublist("ZOLTAN");
    paramlist.set("PARTITIONING METHOD", "HYPERGRAPH");
    sublist.set("LB_APPROACH", "PARTITION");

    Epetra_LinearProblem origProblem( A.get(), X.get(), B.get() );
    if ( Comm.NumProc() > 1 ) {

      Teuchos::RCP<const Epetra_CrsGraph> graph =  Teuchos::rcp( &(A->Graph()), false);

      Teuchos::RCP<Isorropia::Epetra::Partitioner> partitioner =
        Teuchos::rcp(new Isorropia::Epetra::Partitioner(graph, paramlist));

      Isorropia::Epetra::Redistributor rd(partitioner);

      Teuchos::RCP<Epetra_CrsMatrix> bal_matrix = rd.redistribute(*origProblem.GetMatrix());
      Teuchos::RCP<Epetra_MultiVector> bal_x = rd.redistribute(*origProblem.GetLHS());
      Teuchos::RCP<Epetra_MultiVector> bal_b = rd.redistribute(*origProblem.GetRHS());

      A = bal_matrix;
      B = bal_b;
      X = bal_x;
      Map = Teuchos::rcp( new Epetra_Map( bal_matrix->RowMatrixRowMap() ) );

    }
    //End redistribute
#endif

    //
    // ************Construct preconditioner*************
    //
    RCP<Belos::EpetraPrecOp> belosPrec;

    // Ifpack preconditioning classes.
    Teuchos::RCP<Ifpack_IlukGraph> ilukGraph_;
    Teuchos::RCP<Ifpack_CrsRiluk>  rILUK_;

    if (precond != "none") {
      // Initialize the graph if we need to.
      if ( Teuchos::is_null( ilukGraph_ ) )
      {
        const Epetra_CrsGraph & Graph = A->Graph();
        ilukGraph_ = Teuchos::rcp( new Ifpack_IlukGraph( Graph, 1, 0 ) );
        int graphRet = ilukGraph_->ConstructFilledGraph();
        TEUCHOS_TEST_FOR_EXCEPT( graphRet != 0 );

        // Create the preconditioner if one doesn't exist.
        if ( Teuchos::is_null( rILUK_ ) )
        {
          rILUK_ = Teuchos::rcp( new Ifpack_CrsRiluk( *ilukGraph_ ) );
          rILUK_->SetAbsoluteThreshold( 0.0001 );
          rILUK_->SetRelativeThreshold( 1.0001 );
        }

        int initErr = rILUK_->InitValues( *A );
        TEUCHOS_TEST_FOR_EXCEPT( initErr != 0 );

        int factErr = rILUK_->Factor();
        TEUCHOS_TEST_FOR_EXCEPT( factErr != 0 );
      }

      // Create the Belos preconditioned operator from the Ifpack preconditioner.
      // NOTE:  This is necessary because Belos expects an operator to apply the
      //        preconditioner with Apply() NOT ApplyInverse().
      belosPrec = rcp( new Belos::EpetraPrecOp( rILUK_ ) );
    }
    //
    // ********Other information used by block newSolver***********
    // *****************(can be user specified)******************
    //
    const int NumGlobalElements = B->GlobalLength();
    if (maxiters == -1)
      maxiters = NumGlobalElements/blocksize - 1; // maximum number of iterations to run
    //
    ParameterList belosList;
    belosList.set( "Num Blocks", maxsubspace);             // Maximum number of blocks in Krylov factorization
    belosList.set( "Block Size", blocksize );              // Blocksize to be used by iterative newSolver
    belosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
    belosList.set( "Maximum Restarts", maxrestarts );      // Maximum number of restarts allowed
    belosList.set( "Convergence Tolerance", tol );         // Relative convergence tolerance requested
    int verbosity = Belos::Errors + Belos::Warnings;
    if (verbose) {
      verbosity += Belos::FinalSummary + Belos::TimingDetails + Belos::StatusTestDetails;
      if (frequency > 0)
        belosList.set( "Output Frequency", frequency );
    }
    if (debug) {
      verbosity += Belos::Debug;
    }
    belosList.set( "Verbosity", verbosity );

    ParameterList polyList;
    polyList.set( "Polynomial Type", polytype );          // Type of polynomial to be generated
    polyList.set( "Maximum Degree", maxdegree );          // Maximum degree of the GMRES polynomial
    polyList.set( "Polynomial Tolerance", polytol );      // Polynomial convergence tolerance requested
    polyList.set( "Verbosity", verbosity );               // Verbosity for polynomial construction
    polyList.set( "Random RHS", userandomrhs );           // Use RHS from linear system or random vector
    if ( outernewSolver != "" ) {
      polyList.set( "Outer Solver", outernewSolver );
      polyList.set( "Outer Solver Params", belosList );
    }
    //
    // Construct an preconditioned linear problem instance.
    //
    Belos::LinearProblem<double,MV,OP> problem( A, X, B );
    problem.setInitResVec( B );
    if (precond == "left") {
      problem.setLeftPrec( belosPrec );
    }
    if (precond == "right") {
      problem.setRightPrec( belosPrec );
    }
    bool set = problem.setProblem();
    if (set == false) {
      if (proc_verbose)
        std::cout << std::endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;
      return -1;
    }
    //
    // *******************************************************************
    // *************Start the block Gmres iteration*************************
    // *******************************************************************
    //
    // Create an iterative newSolver manager.
    RCP< Belos::SolverManager<double,MV,OP> > newSolver
      = rcp( new Belos::GmresPolySolMgr<double,MV,OP>(rcp(&problem,false), rcp(&polyList,false)));

    //
    // **********Print out information about problem*******************
    //
    if (proc_verbose) {
      std::cout << std::endl << std::endl;
      std::cout << "Dimension of matrix: " << NumGlobalElements << std::endl;
      std::cout << "Number of right-hand sides: " << numrhs << std::endl;
      std::cout << "Block size used by newSolver: " << blocksize << std::endl;
      std::cout << "Max number of restarts allowed: " << maxrestarts << std::endl;
      std::cout << "Max number of Gmres iterations per restart cycle: " << maxiters << std::endl;
      std::cout << "Relative residual tolerance: " << tol << std::endl;
      std::cout << std::endl;
    }
    //
    // Perform solve
    //
    Belos::ReturnType ret = newSolver->solve();
    int numIters = newSolver->getNumIters();
    if (proc_verbose)
      std::cout << "Number of iterations performed for the initial solve: " << numIters << std::endl;
    //
    // Compute actual residuals.
    //
    bool badRes = false;
    std::vector<double> actual_resids( numrhs );
    std::vector<double> rhs_norm( numrhs );
    Epetra_MultiVector resid(*Map, numrhs);
    OPT::Apply( *A, *X, resid );
    MVT::MvAddMv( -1.0, resid, 1.0, *B, resid );
    MVT::MvNorm( resid, actual_resids );
    MVT::MvNorm( *B, rhs_norm );
    if (proc_verbose) {
      std::cout<< "---------- Actual Residuals (normalized) ----------"<<std::endl<<std::endl;
      for ( int i=0; i<numrhs; i++) {
        double actRes = actual_resids[i]/rhs_norm[i];
        std::cout<<"Problem "<<i<<" : \t"<< actRes <<std::endl;
        if (actRes > tol) badRes = true;
      }
    }
    //
    // -----------------------------------------------------------------
    // Resolve the first problem without resetting the newSolver manager.
    // -----------------------------------------------------------------
    X->PutScalar( 0.0 );
    //
    // Perform solve (again)
    //
    ret = newSolver->solve();
    //
    // Get the number of iterations for this solve.
    //
    numIters = newSolver->getNumIters();
    if (proc_verbose)
      std::cout << "Number of iterations performed for this solve (without manager reset): " << numIters << std::endl;
    // -----------------------------------------------------------------
    // Resolve the first matrix, with different RHS, without resetting the newSolver manager.
    // -----------------------------------------------------------------
    X->Random();
    OPT::Apply( *A, *X, *B );
    X->PutScalar( 0.0 );
    //
    // Perform solve (again, again)
    //
    ret = newSolver->solve();
    //
    // Get the number of iterations for this solve.
    //
    numIters = newSolver->getNumIters();
    if (proc_verbose)
      std::cout << "Number of iterations performed for this solve (new RHS, without manager reset): " << numIters << std::endl;
    //
    // -----------------------------------------------------------------
    // Resolve the first matrix, with different RHS, by resetting the newSolver manager.
    // -----------------------------------------------------------------
    X->PutScalar( 0.0 );
    newSolver->reset( Belos::Problem );
    //
    // Perform solve (again, again, again)
    //
    ret = newSolver->solve();
    //
    // Get the number of iterations for this solve.
    //
    numIters = newSolver->getNumIters();
    if (proc_verbose)
      std::cout << "Number of iterations performed for this solve (new RHS, with manager reset): " << numIters << std::endl;
    //
    if (ret!=Belos::Converged || badRes) {
      success = false;
      if (proc_verbose)
        std::cout << std::endl << "ERROR:  Belos did not converge!" << std::endl;
    } else {
      success = true;
      if (proc_verbose)
        std::cout << std::endl << "SUCCESS:  Belos converged!" << std::endl;
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

#ifdef EPETRA_MPI
  MPI_Finalize();
#endif

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
