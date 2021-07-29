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
// This driver tests variations of Hybrid GMRES using the Belos GMRES 
// Polynomial solver manager.  One can specify an 'outer solver'
// to which the GMRES Polynomial is applied as a preconditioner.  
// If the 'outer solver' string is empty, the polynomial will be applied
// alone.
//
// Alternately, one can apply the GMRES polynomial without the 'outer
// solver' functionality by using the BelosTpetraOperator interface.
// See the FGMRES test driver for an example of its use.  
//
// This driver reads a problem from a Harwell-Boeing (HB) file.
// The right-hand-side corresponds to a randomly generated solution.
// The initial guesses are all set to zero.
//
//
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosGmresPolySolMgr.hpp"

// I/O for Harwell-Boeing files
#define HIDE_TPETRA_INOUT_IMPLEMENTATIONS
#include <Tpetra_MatrixIO.hpp>
#include <MatrixMarket_Tpetra.hpp>

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardCatchMacros.hpp>
#include <Teuchos_StackedTimer.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>

#include "Ifpack2_Factory.hpp"

using namespace Teuchos;
using Tpetra::Operator;
using Tpetra::CrsMatrix;
using Tpetra::MultiVector;
using std::endl;
using std::cout;
using std::vector;
using Teuchos::tuple;

int main(int argc, char *argv[]) {

  typedef float                           ST;
  typedef ScalarTraits<ST>                SCT;
  typedef SCT::magnitudeType               MT;
  typedef Tpetra::Operator<ST>             OP;
  typedef Tpetra::MultiVector<ST>          MV;
  typedef Belos::OperatorTraits<ST,MV,OP> OPT;
  typedef Belos::MultiVecTraits<ST,MV>    MVT;

  typedef Teuchos::TimeMonitor time_monitor_type;

  Tpetra::ScopeGuard tpetraScope(&argc,&argv);

  bool success = false;
  bool verbose = true;
  try {
    const ST one  = SCT::one();

    int MyPID = 0;

    RCP<const Comm<int> > comm = Tpetra::getDefaultComm();

    bool proc_verbose = false;
    bool userandomrhs = false;                // use linear problem RHS or random RHS to generate poly
    int frequency = -1;                       // frequency of status test output.
    int blocksize = 1;                        // blocksize
    int numrhs = 1;                           // number of right-hand sides to solve for
    int maxiters = -1;                        // maximum number of iterations allowed per linear system
    int maxdegree = 25;                       // maximum degree of polynomial
    bool use_stacked_timer = false;              
    int maxsubspace = 50;                     // maximum number of blocks the solver can use for the subspace
    int maxrestarts = 25;                     // number of restarts allowed
    std::string outersolver("Block Gmres");   // name of outer solver
    std::string polytype("Roots");            // polynomial configuration.  
    std::string filename("bcsstk13.mtx");      // name of matrix file
    MT tol = 1.0e-8;                          // relative residual tolerance
    MT polytol = tol/10;                      // relative residual tolerance for polynomial construction

    struct CmndLinePrecOpts {
      std::string preconditioner {""};
      std::string relaxation {"SGS MT"};
      // Schwarz
      std::string subdomainSolver = {"AMESOS2"};
      // Relaxation
      double dampFactor {1.0};
      double innerDampFactor {1.0};
      int numRelaxationSweeps {1};
      int numInnerRelaxationSweeps {1};
      int numOuterRelaxationSweeps {1};
      bool doBackwardSweep {false};
      bool useInnerSptrsv {false};
      bool useCompactForm {false};
      int clusterSize {16};
      int evalMaxIters {10};
      ST evalRatio {30.0};
    } clpOpts; 

    //**************************************************************************
    // Read paramters from command line:
    //**************************************************************************
    Teuchos::CommandLineProcessor cmdp(false,true);
    cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
    cmdp.setOption("use-random-rhs","use-rhs",&userandomrhs,"Use linear system RHS or random RHS to generate polynomial.");
    cmdp.setOption("frequency",&frequency,"Solvers frequency for printing residuals (#iters).");
    cmdp.setOption("stacked-timer", "no-stacked-timer", &use_stacked_timer, "Run with or without stacked timer output");
    cmdp.setOption("filename",&filename,"Filename for test matrix.  Acceptable file extensions: *.hb,*.mtx,*.triU,*.triS");
    cmdp.setOption("outersolver",&outersolver,"Name of outer solver to be used with GMRES poly");
    cmdp.setOption("poly-type",&polytype,"Name of the polynomial to be generated. Arnoldi, Gmres, or Roots.");
    cmdp.setOption("tol",&tol,"Relative residual tolerance used by GMRES solver.");
    cmdp.setOption("poly-tol",&polytol,"Relative residual tolerance used to construct the GMRES polynomial.");
    cmdp.setOption("num-rhs",&numrhs,"Number of right-hand sides to be solved for.");
    cmdp.setOption("block-size",&blocksize,"Block size used by GMRES.");
    cmdp.setOption("max-iters",&maxiters,"Maximum number of iterations per linear system (-1 = adapted to problem/block size).");
    cmdp.setOption("max-degree",&maxdegree,"Maximum degree of the GMRES polynomial.");
    cmdp.setOption("max-subspace",&maxsubspace,"Maximum number of blocks the solver can use for the subspace.");
    cmdp.setOption("max-restarts",&maxrestarts,"Maximum number of restarts allowed for GMRES solver.");

    //IfPack2 options:
    cmdp.setOption ("preconditioner", &clpOpts.preconditioner, "IfPack2 factory name: RELAXATION, CHEBYSHEV, or SCHWARZ.");
    cmdp.setOption ("evalMaxIters", &clpOpts.evalMaxIters, "Max power iter steps for Cheby poly eval est. ");
    cmdp.setOption ("evalRatio", &clpOpts.evalRatio, "Eigenvalue ratio for Cheby poly.");
    cmdp.setOption ("relaxation", &clpOpts.relaxation, "Type of relaxation ");
    cmdp.setOption ("dampFactor", &clpOpts.dampFactor, "Damp factor for relaxation");
    cmdp.setOption ("innerDampFactor", &clpOpts.innerDampFactor, "Damp factor for inner relaxation");
    cmdp.setOption ("numRelaxationSweeps", &clpOpts.numRelaxationSweeps, "Num relaxation sweeps");
    cmdp.setOption ("numOuterRelaxationSweeps", &clpOpts.numOuterRelaxationSweeps, "Num outer relaxation sweeps");
    cmdp.setOption ("numInnerRelaxationSweeps", &clpOpts.numInnerRelaxationSweeps, "Num inner relaxation sweeps");
    cmdp.setOption ("doBackwardSweep", "noBackwardSweep", &clpOpts.doBackwardSweep, "Whether to do backward or forward sweeps");
    cmdp.setOption ("useInnerSptrsv", "noInnerSptrsv", &clpOpts.useInnerSptrsv, "Whether to use Sptrsv or iteration");
    cmdp.setOption ("useCompactForm", "noCompactForm", &clpOpts.useCompactForm, "Whether to use solution-based GS recurrence");
    cmdp.setOption ("clusterSize", &clpOpts.clusterSize, "Size of cluster for clustered GS");
    cmdp.setOption ("subdomainSolver", &clpOpts.subdomainSolver, "Name of subdomain solver: RILUK, AMESOS2, or RELAXATION");  

    if (cmdp.parse(argc,argv) != CommandLineProcessor::PARSE_SUCCESSFUL) {
      return EXIT_FAILURE;
    }
    if (!verbose) {
      frequency = -1;  // reset frequency if test is not verbose
    }

    MyPID = rank(*comm);
    proc_verbose = ( verbose && (MyPID==0) );

    //**************************************************************************
    // Read in matrix and initialze vectors:
    //**************************************************************************
    RCP<CrsMatrix<ST> > A;
    //Tpetra::Utils::readHBMatrix(filename,comm,A);
    A = Tpetra::MatrixMarket::Reader<CrsMatrix<ST> >::readSparseFile(filename,comm);
    RCP<const Tpetra::Map<> > map = A->getDomainMap();

    // Create initial vectors
    RCP<MV> B, X;
    X = rcp( new MV(map,numrhs) );
    B = rcp( new MV(map,numrhs) );
    MVT::MvInit( *X, 0.0 );
    MVT::MvInit( *B, 1.0 );

    //**************************************************************************
    // Set solver and poly prec paramters:
    //**************************************************************************
    const int NumGlobalElements = B->getGlobalLength();
    if (maxiters == -1) {
      maxiters = NumGlobalElements/blocksize - 1; // maximum number of iterations to run
    }

    //Parameter list used by the outer solver:
    ParameterList belosList;
    belosList.set( "Num Blocks", maxsubspace);             // Maximum number of blocks in Krylov factorization
    belosList.set( "Block Size", blocksize );              // Blocksize to be used by iterative solver
    belosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
    belosList.set( "Maximum Restarts", maxrestarts );      // Maximum number of restarts allowed
    belosList.set( "Convergence Tolerance", tol );         // Relative convergence tolerance requested
    int verbosity = Belos::Errors + Belos::Warnings;
    if (verbose) {
      verbosity += Belos::FinalSummary + Belos::TimingDetails + Belos::StatusTestDetails;
      if (frequency > 0)
        belosList.set( "Output Frequency", frequency );
    }
    belosList.set( "Verbosity", verbosity );

    // Parameter list used by the GMRES Polynomial (inner) solver
    ParameterList polyList;
    polyList.set( "Polynomial Type", polytype );          // Type of polynomial to be generated
    polyList.set( "Maximum Degree", maxdegree );          // Maximum degree of the GMRES polynomial
    polyList.set( "Polynomial Tolerance", polytol );      // Polynomial convergence tolerance requested
    polyList.set( "Verbosity", verbosity );               // Verbosity for polynomial construction
    polyList.set( "Random RHS", userandomrhs );           // Use RHS from linear system or random vector
                                                          // to generate the polynomial.
    //
    // Pass the outer solver name and parameters to the polynomial solver manager.
    //
    if ( outersolver != "" ) {
      polyList.set( "Outer Solver", outersolver );
      polyList.set( "Outer Solver Params", belosList );
    }

    //**************************************************************
    // Begin preconditioning setup.
    //**************************************************************

    RCP< Teuchos::Time > Ifpack2InitTimer = time_monitor_type::getNewCounter ("Ifpack2: initialize");
    RCP< Teuchos::Time > Ifpack2CompTimer = time_monitor_type::getNewCounter ("Ifpack2: compute");
    Teuchos::RCP<Ifpack2::Preconditioner<ST>> M;
    if (clpOpts.preconditioner != "") {
      M = Ifpack2::Factory::create (clpOpts.preconditioner, 
            Teuchos::rcp_const_cast<const Tpetra::CrsMatrix<ST>>(A), 0);

      RCP<ParameterList> PrecParams = parameterList ("Preconditioner");
      if (clpOpts.preconditioner == "RELAXATION") {
        if (clpOpts.relaxation == "Richardson") {
          PrecParams->set ("relaxation: type", "Richardson");
        }
        //
        else if (clpOpts.relaxation == "GS MT") {
          PrecParams->set ("relaxation: type", "MT Gauss-Seidel");
        }
        else if (clpOpts.relaxation == "GS CL") {
          PrecParams->set ("relaxation: type", "MT Gauss-Seidel");
          PrecParams->set ("relaxation: mtgs cluster size", clpOpts.clusterSize);
        }
        else if (clpOpts.relaxation == "GS") {
          PrecParams->set ("relaxation: type", "Gauss-Seidel");
        }
        else if (clpOpts.relaxation == "GS2") {
          PrecParams->set ("relaxation: type", "Two-stage Gauss-Seidel");
          PrecParams->set ("relaxation: inner sweeps", clpOpts.numInnerRelaxationSweeps);
          PrecParams->set ("relaxation: outer sweeps", clpOpts.numOuterRelaxationSweeps);
          PrecParams->set ("relaxation: inner sparse-triangular solve", clpOpts.useInnerSptrsv);
          PrecParams->set ("relaxation: compact form", clpOpts.useCompactForm);
          PrecParams->set ("relaxation: inner damping factor", clpOpts.innerDampFactor);
          PrecParams->set ("relaxation: backward mode", clpOpts.doBackwardSweep);
        }
        //
        else if (clpOpts.relaxation == "SGS MT") {
          PrecParams->set ("relaxation: type", "MT Symmetric Gauss-Seidel");
        }
        else if (clpOpts.relaxation == "SGS CL") {
          PrecParams->set ("relaxation: type", "MT Symmetric Gauss-Seidel");
          PrecParams->set ("relaxation: mtgs cluster size", clpOpts.clusterSize);
        }
        else if (clpOpts.relaxation == "SGS2") {
          PrecParams->set ("relaxation: type", "Two-stage Symmetric Gauss-Seidel");
          PrecParams->set ("relaxation: inner sweeps", clpOpts.numInnerRelaxationSweeps);
          PrecParams->set ("relaxation: outer sweeps", clpOpts.numOuterRelaxationSweeps);
          PrecParams->set ("relaxation: inner sparse-triangular solve", clpOpts.useInnerSptrsv);
          PrecParams->set ("relaxation: compact form", clpOpts.useCompactForm);
          PrecParams->set ("relaxation: inner damping factor", clpOpts.innerDampFactor);
        }
        else if (clpOpts.relaxation == "SGS") {
          PrecParams->set ("relaxation: type", "Symmetric Gauss-Seidel");
        } else {
          std::cout << " Invalide relaxation type " << std::endl;
          exit(0);
        }
        PrecParams->set ("relaxation: sweeps", clpOpts.numRelaxationSweeps);
        PrecParams->set ("relaxation: damping factor", clpOpts.dampFactor);
      }
      else if (clpOpts.preconditioner == "SCHWARZ") {
        //int overlap = 0;
        //PrecParams->set ("schwarz: overlap level", overlap);
        if (clpOpts.subdomainSolver == "AMESOS2") {
          PrecParams->set ("relaxation: sweeps", clpOpts.numRelaxationSweeps);

          PrecParams->set ("subdomain solver name", "AMESOS2");
          /*if (clpOpts.useSuperLU) {
            ParameterList &SubdomainParams = PrecParams->sublist("subdomain solver parameters");
            SubdomainParams.set ("Amesos2 solver name", "Superlu");

            //ParameterList &Amesos2Params = SubdomainParams.sublist("Amesos2");
            //ParameterList &SuperluParams = Amesos2Params.sublist("SuperLU");
            //SuperLUParams->set ("Equil", true);
          }*/
        }
        else if (clpOpts.subdomainSolver == "RILUK") {
          PrecParams->set ("subdomain solver name",   "RILUK");
          PrecParams->set ("schwarz: use reordering", true);
        }
        else {
          PrecParams->set ("subdomain solver name",  "RELAXATION");
          PrecParams->set ("relaxation: type",       "Symmetric Gauss-Seidel");
          PrecParams->set ("relaxation: sweeps",     clpOpts.numRelaxationSweeps);
        }
      }
      else if (clpOpts.preconditioner == "CHEBYSHEV") {
          PrecParams->set ("relaxation: sweeps",     clpOpts.numRelaxationSweeps); // Degree of poly
          PrecParams->set ("chebyshev: eigenvalue max iterations", clpOpts.evalMaxIters); //Max iters power method for eval
          PrecParams->set ("chebyshev: ratio eigenvalue", clpOpts.evalRatio); 
      }
      M->setParameters (*PrecParams);
      {
        time_monitor_type LocalTimer (*Ifpack2InitTimer);
        M->initialize ();
      }
      {
        time_monitor_type LocalTimer (*Ifpack2CompTimer);
        M->compute ();
      }
      if (proc_verbose) {
        std::cout << "Preconditioner:" << endl;
        std::cout << PrecParams->currentParametersString() << endl;
      }

    }
    //**************************************************************
    // End preconditioning setup.
    //**************************************************************

    //**************************************************************
    // Construct preconditioned linear problem instance. 
    //**************************************************************
    //
    // The polynomial preconditioning will be handled by the solver manager.
    // One could pass another preconditioner such as ILU to the linear problem
    // to be used in combination with the polynomial preconditioner.
    //
    Belos::LinearProblem<ST,MV,OP> problem( A, X, B );
    problem.setInitResVec( B );
    if (clpOpts.preconditioner != "") {
      problem.setRightPrec(M);
    }
    bool set = problem.setProblem();
    if (set == false) {
      if (proc_verbose)
        std::cout << std::endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;
      return EXIT_FAILURE;
    }
    //
    // *******************************************************************
    // *************Start the Hybrid Gmres iteration***********************
    // ********* Using the GMRES Poly Solver Manager *********************
    // *******************************************************************
    //
    Belos::GmresPolySolMgr<ST,MV,OP> solver( rcpFromRef(problem), rcpFromRef(polyList) );

    //
    // **********Print out information about problem*******************
    //
    if (proc_verbose) {
      std::cout << std::endl << std::endl;
      std::cout << "Dimension of matrix: " << NumGlobalElements << std::endl;
      std::cout << "Number of right-hand sides: " << numrhs << std::endl;
      std::cout << "Block size used by solver: " << blocksize << std::endl;
      std::cout << "Max number of Gmres iterations: " << maxiters << std::endl;
      std::cout << "Relative residual tolerance: " << tol << std::endl;
      std::cout << std::endl;
    }

    // *******************************************************************
    // Set output stream and stacked timer:
    // (see packages/muelu/example/basic/Stratimikos.cpp)
    // *******************************************************************
    RCP<Teuchos::FancyOStream> fancy = Teuchos::fancyOStream(Teuchos::rcpFromRef(std::cout));
    Teuchos::FancyOStream& out = *fancy;
    out.setOutputToRootOnly(0);
    // Set up timers
    Teuchos::RCP<Teuchos::StackedTimer> stacked_timer;
    if (use_stacked_timer){
      stacked_timer = rcp(new Teuchos::StackedTimer("Main"));
    }
    TimeMonitor::setStackedTimer(stacked_timer);

    // *******************************************************************
    // Perform solve
    // *******************************************************************
    Belos::ReturnType ret = solver.solve();

    if (use_stacked_timer) {
      stacked_timer->stop("Main");
      Teuchos::StackedTimer::OutputOptions options;
      options.output_fraction = options.output_histogram = options.output_minmax = true;
      stacked_timer->report(out, comm, options);
    }

    // *******************************************************************
    // Compute actual residuals.
    // *******************************************************************
    bool badRes = false;
    std::vector<MT> actual_resids( numrhs );
    std::vector<MT> rhs_norm( numrhs );
    MV resid(map, numrhs);
    OPT::Apply( *A, *X, resid );
    MVT::MvAddMv( -one, resid, one, *B, resid );
    MVT::MvNorm( resid, actual_resids );
    MVT::MvNorm( *B, rhs_norm );
    if (proc_verbose) {
      std::cout<< "---------- Actual Residuals (normalized) ----------"<<std::endl<<std::endl;
    }
    for ( int i=0; i<numrhs; i++) {
      MT actRes = actual_resids[i]/rhs_norm[i];
      if (proc_verbose) {
        std::cout<<"Problem "<<i<<" : \t"<< actRes <<std::endl;
      }
      if (actRes > tol) badRes = true;
    }

    success = (ret==Belos::Converged && !badRes);

    if (success) {
      if (proc_verbose)
        std::cout << "\nEnd Result: TEST PASSED" << std::endl;
    } else {
      if (proc_verbose)
        std::cout << "\nEnd Result: TEST FAILED" << std::endl;
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

  return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
} // end Jennifer's GMRES test driver with IfPack2
