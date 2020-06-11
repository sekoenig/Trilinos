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
// This driver reads a problem from a Harwell-Boeing (HB) file.
// The right-hand-side corresponds to a randomly generated solution.
// The initial guesses are all set to zero.
//
// NOTE: No preconditioner is used in this case.
//
#include "BelosConfigDefs.hpp"
#include "BelosLinearProblem.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosGmresPolySolMgr.hpp"
#include "BelosSolverFactory.hpp"

// I/O for Harwell-Boeing files
#define HIDE_TPETRA_INOUT_IMPLEMENTATIONS
#include <Tpetra_MatrixIO.hpp>
#include <MatrixMarket_Tpetra.hpp>

#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_StandardCatchMacros.hpp>
#include <Tpetra_Core.hpp>
#include <Tpetra_CrsMatrix.hpp>

namespace BelosTpetra {
  namespace Impl {

    extern void register_CgPipeline (const bool verbose);
    extern void register_CgSingleReduce (const bool verbose);
    extern void register_GmresPipeline (const bool verbose);
    extern void register_GmresSingleReduce (const bool verbose);
    extern void register_GmresSstep (const bool verbose);
      
  } // namespace Impl
} // namespace BelosTpetra 

// Xpetra
#include <Xpetra_MultiVectorFactory.hpp>
#include <Xpetra_ImportFactory.hpp>
#include <Xpetra_Operator.hpp>
#include <Xpetra_IO.hpp>

// Galeri
#include <Galeri_XpetraParameters.hpp>
#include <Galeri_XpetraProblemFactory.hpp>
#include <Galeri_XpetraUtils.hpp>
#include <Galeri_XpetraMaps.hpp>

//#include <Ifpack2_Factory.hpp>
#include <Zoltan2_PartitioningProblem.hpp>
#include <Zoltan2_XpetraCrsMatrixAdapter.hpp>
#include <Zoltan2_XpetraMultiVectorAdapter.hpp>

using namespace Teuchos;
using Tpetra::Operator;
using Tpetra::CrsMatrix;
using Tpetra::MultiVector;
using std::endl;
using std::cout;
using std::vector;
using Teuchos::tuple;

int main(int argc, char *argv[]) {

  typedef double                           ST;
  typedef ScalarTraits<ST>                SCT;
  typedef SCT::magnitudeType               MT;
  typedef Tpetra::Operator<ST>             OP;
  typedef Tpetra::MultiVector<ST>          MV;
  typedef Belos::OperatorTraits<ST,MV,OP> OPT;
  typedef Belos::MultiVecTraits<ST,MV>    MVT;
  typedef Tpetra::Map<>              map_type;
  typedef map_type::local_ordinal_type     LO;
  typedef map_type::global_ordinal_type    GO;

  Tpetra::ScopeGuard tpetraScope(&argc,&argv);

  bool success = false;
  bool verbose = true;
  try {
    const ST one  = SCT::one();

    int MyPID = 0;

    RCP<const Comm<int> > comm = Tpetra::getDefaultComm();
    //
    // Get test parameters from command-line processor
    //
    bool proc_verbose = false;
    bool userandomrhs = true; //NOTE: Polys will be different between runs unless problem rhs is used for poly.
    bool damppoly = false;
    bool addRoots = true;
    int frequency = 50;        // frequency of status test output.
    int blocksize = 1;         // blocksize
    int numrhs = 1;            // number of right-hand sides to solve for
    int maxiters = 3000;         // maximum number of iterations allowed per linear system
    int maxdegree;        // maximum degree of polynomial
    int maxsubspace = 50;      // maximum number of blocks the solver can use for the subspace
    int maxrestarts = 50;      // number of restarts allowed
    int OverlapLevel = 1;   //Overlap level for non-poly preconditioners must be >= 0. If Comm.NumProc() == 1,it is ignored.   
    int ilutFill_ = 1;      //Fill level for ILU factorization
    int nx = 10;               // number of discretization points in each direction
    double aThresh_ = 0.0001;
    double rThresh_ = 1.0001;
    double dropTol_ = 1e-3;
    double diff = 1e-5; //Diffusion term
    double conv = 1.0; //Convection term
    std::string outersolver("Block Gmres");
    std::string polytype("Roots");
    std::string filename;
    std::string rhsfile;
    std::string precond("right");
    std::string partition("zoltan");
    std::string PrecType = "ILU"; // incomplete LU
    std::string orthog("ICGS");
    std::string MatrixType("Laplace3D");
    std::string degreesToTest("25");
    MT tol = 1.0e-8;           // relative residual tolerance
    MT polytol = tol/10;       // relative residual tolerance for polynomial construction

    Teuchos::CommandLineProcessor cmdp(false,true);
    cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
    cmdp.setOption("use-random-rhs","use-rhs",&userandomrhs,"Use linear system RHS or random RHS to generate polynomial.");
    cmdp.setOption("damp-poly","no-damp",&damppoly,"Damp the polynomial.");
    cmdp.setOption("add-roots","no-add-roots",&addRoots,"Add extra roots as needed to stabilize the polynomial.");
    cmdp.setOption("frequency",&frequency,"Solvers frequency for printing residuals (#iters).");
    cmdp.setOption("filename",&filename,"Filename for test matrix.  Acceptable file extensions: *.hb,*.mtx,*.triU,*.triS. If empty, will generate Galeri problem.");
    cmdp.setOption("rhsfile",&rhsfile,"Filename for test rhs b. ");
    cmdp.setOption("outersolver",&outersolver,"Name of outer solver to be used with GMRES poly");
    cmdp.setOption("poly-type",&polytype,"Name of the polynomial to be generated.");
    cmdp.setOption("precond",&precond,"Preconditioning placement (none, left, right).");
    cmdp.setOption("prec-type",&PrecType,"Preconditioning type (Amesos, ILU, ILUT, ILUK2, none).");
    cmdp.setOption("overlap",&OverlapLevel,"Overlap level for non-poly preconditioners.");
    cmdp.setOption("fill",&ilutFill_,"Fill level for ILU-type preconditioners.");
    cmdp.setOption("athresh",&aThresh_,"Absolute Threshold for ILU-type preconditioners.");
    cmdp.setOption("rthresh",&rThresh_,"Relative Threshold for ILU-type preconditioners.");
    cmdp.setOption("droptol",&dropTol_,"Drop tolerance for ILU-type preconditioners.");
    cmdp.setOption("tol",&tol,"Relative residual tolerance used by GMRES solver.");
    cmdp.setOption("poly-tol",&polytol,"Relative residual tolerance used to construct the GMRES polynomial.");
    cmdp.setOption("num-rhs",&numrhs,"Number of right-hand sides to be solved for.");
    cmdp.setOption("block-size",&blocksize,"Block size used by GMRES.");
    cmdp.setOption("orthog",&orthog,"Orthogonalization: DGKS, ICGS, IMGS");
    cmdp.setOption("max-iters",&maxiters,"Maximum number of iterations per linear system (-1 = adapted to problem/block size).");
    cmdp.setOption("degrees",&degreesToTest,"Polynomial degrees to test. Separated by spaces. e.g. \"5 10 20\"");
    cmdp.setOption("max-subspace",&maxsubspace,"Maximum number of blocks the solver can use for the subspace.");
    cmdp.setOption("max-restarts",&maxrestarts,"Maximum number of restarts allowed for GMRES solver.");
    cmdp.setOption("partition",&partition,"Partitioning type (zoltan, linear, none).");
    cmdp.setOption("nx",&nx,"Number of discretization points in each direction of PDE.");
    cmdp.setOption("matrix-type",&MatrixType,"Matrix type. See Galeri documentation. (Default: Laplace3D)");
    cmdp.setOption("diff",&diff,"Diffusion term.  Default: 1e-5");
    cmdp.setOption("conv",&conv,"Convection term.  Default: 1.0");

    if (cmdp.parse(argc,argv) != CommandLineProcessor::PARSE_SUCCESSFUL) {
      return EXIT_FAILURE;
    }
    if (!verbose) {
      frequency = -1;  // reset frequency if test is not verbose
    }

    MyPID = rank(*comm);
    proc_verbose = ( verbose && (MyPID==0) );

    if( precond != "none" && precond != "right" && precond != "left")
    {
      if(MyPID==0) std::cout << "Error: Invalid string for precond param." << std::endl;
      return -1;
    }
    if( PrecType != "Amesos" && PrecType!= "ILU" && PrecType != "ILUT" && PrecType != "ILUK2" && PrecType != "none")
    {
      if(MyPID==0) std::cout << "Error: Invalid string for prec-type param." << std::endl;
      return -1;
    }

    //Parse polynomial degrees
    std::istringstream istr( degreesToTest );
    std::vector<int> degreeVector{ std::istream_iterator<int>( istr ), std::istream_iterator<int>() };

    //Add new Tpetra solvers to solver factory:
    BelosTpetra::Impl::register_CgPipeline (false);
    BelosTpetra::Impl::register_GmresSstep (false);

    Teuchos::RCP<Tpetra::CrsMatrix<ST>> A;
    RCP< const map_type> map;
    if(filename.compare("") != 0){//Read in problem
    //
    // Get the data from the HB file and build the Map,Matrix
    //
    Tpetra::Utils::readHBMatrix(filename,comm,A);
    map = A->getDomainMap();
    }
    else{ //Create Galeri Matrix

      //Set Params for Galeri:
      Teuchos::ParameterList GaleriList;
      GaleriList.set ("nx", nx);
      GaleriList.set ("ny", nx);
      GaleriList.set ("nz", nx);
      GaleriList.set ("diff", diff);
      GaleriList.set ("conv", conv);
      Tpetra::global_size_t nGlobalElts;
      if( MatrixType == "Laplace3D" || MatrixType == "Cross3D"|| MatrixType == "Star3D"|| MatrixType == "Elasticity3D"){
        nGlobalElts = nx * nx * nx;
      }
      else{ //Assume 2D map
        nGlobalElts = nx * nx;
      }
      if(proc_verbose){ std::cout << "Creating Map " << std::endl; }
      map = rcp(new map_type(nGlobalElts, 0, comm));

      if(proc_verbose){ std::cout << "Building Problem" << std::endl; }
      typedef Galeri::Xpetra::Problem<map_type, Tpetra::CrsMatrix<ST>, MV> Galeri_t;
      RCP<Galeri_t> galeriProblem = Galeri::Xpetra::BuildProblem<ST, LO, GO, 
        map_type, Tpetra::CrsMatrix<ST>, MV> (MatrixType, map, GaleriList);
      if(proc_verbose){ std::cout << "Building Matrix" << std::endl; }
      A = galeriProblem->BuildMatrix();
    }
    if(proc_verbose) {
      std::cout << "Matrix Size is: " << A->getGlobalNumRows() << std::endl;
      std::cout << "Number of Entries: " << A->getGlobalNumEntries() << std::endl;
    }  


    // Read RHS from a file
    RCP<MV> B, X; 
    if (rhsfile != "") {
      if (proc_verbose) std::cout << "RHS read from " << rhsfile << std::endl;
      RCP<const map_type> rhsMap = A->getRangeMap(); 
      B = Tpetra::MatrixMarket::Reader<Tpetra::CrsMatrix<ST>>::readDenseFile (rhsfile, comm, rhsMap, false, false);
    } else {
      B = rcp (new MV(A->getRangeMap(), numrhs));
      B->randomize();
    }

    // Create initial vectors
    X = rcp (new MV(A->getRangeMap(), numrhs));
    MVT::MvInit( *X, 0.0 );

    //Double-check norms for debugging:
    std::vector<double> tempnorm(numrhs), temprhs(numrhs);
    MVT::MvNorm(*B, temprhs);
    MVT::MvNorm(*X, tempnorm);
    {if(MyPID==0) std::cout << "Norm of B: " << temprhs[0] << " Norm of X: " << tempnorm[0] << std::endl;}

    //
    // ********Other information used by block solver***********
    // *****************(can be user specified)******************
    //
    const int NumGlobalElements = B->getGlobalLength();
    if (maxiters == -1) {
      maxiters = NumGlobalElements/blocksize - 1; // maximum number of iterations to run
    }
    //
    ParameterList belosList;
    belosList.set( "Num Blocks", maxsubspace);             // Maximum number of blocks in Krylov factorization
    belosList.set( "Block Size", blocksize );              // Blocksize to be used by iterative solver
    belosList.set( "Maximum Iterations", maxiters );       // Maximum number of iterations allowed
    belosList.set( "Maximum Restarts", maxrestarts );      // Maximum number of restarts allowed
    belosList.set( "Convergence Tolerance", tol );         // Relative convergence tolerance requested
    belosList.set( "Orthogonalization", orthog);           // Type of orthogonalizaion: DGKS, IMGS, ICGS
    int verbosity = Belos::Errors + Belos::Warnings;
    if (verbose) {
      verbosity += Belos::FinalSummary + Belos::TimingDetails + Belos::StatusTestDetails;
      if (frequency > 0)
        belosList.set( "Output Frequency", frequency );
    }
    belosList.set( "Verbosity", verbosity );

    ParameterList polyList;
    polyList.set( "Polynomial Type", polytype );          // Type of polynomial to be generated
    polyList.set( "Polynomial Tolerance", polytol );      // Polynomial convergence tolerance requested
    polyList.set( "Verbosity", verbosity );               // Verbosity for polynomial construction
    polyList.set( "Random RHS", userandomrhs );           // Use RHS from linear system or random vector
    polyList.set( "Damped Poly", damppoly );              // Option to damp polynomial
    polyList.set( "Add Roots", addRoots );                // Option to add roots to stabilize poly 
    polyList.set( "Orthogonalization", orthog);           // Type of orthogonalizaion: DGKS, IMGS, ICGS

    // Construct an unpreconditioned linear problem instance.
    //
    Belos::LinearProblem<ST,MV,OP> problem( A, X, B );
    problem.setInitResVec( B );
    bool set = problem.setProblem();
    if (set == false) {
      if (proc_verbose)
        std::cout << std::endl << "ERROR:  Belos::LinearProblem failed to set up correctly!" << std::endl;
      return EXIT_FAILURE;
    }

    Belos::GenericSolverFactory<ST, MV, OP> factory;
    //
    // *******************************************************************
    // *************Start the block Gmres iteration***********************
    // *******************************************************************
    //
    for( unsigned int i=0; i < degreeVector.size(); i++){
      maxdegree = degreeVector[i];
      polyList.set( "Maximum Degree", maxdegree );          // Maximum degree of the GMRES polynomial
      if(proc_verbose){std::cout << "Next poly degree is: " << maxdegree << std::endl;}

      Belos::GmresPolySolMgr<ST,MV,OP> innerSolver( rcpFromRef(problem), rcpFromRef(polyList) );
      RCP<Belos::SolverManager<ST, MV, OP> > solver = factory.create( outersolver, rcpFromRef(belosList) );
      TEUCHOS_TEST_FOR_EXCEPTION(solver == Teuchos::null, std::invalid_argument, "Selected solver is not valid.");
      solver->setProblem(rcpFromRef(problem)); 

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
      //
      // Perform solve
      //
      Belos::ReturnType ret = solver->solve();
      //
      // Compute actual residuals.
      //
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
      // Zero out previous timers
      Teuchos::TimeMonitor::zeroOutTimers();
      //Reset solution for next solve:
      MVT::MvInit( *X, 0.0 );
      solver->reset( Belos::Problem );
      //newSolver = Teuchos::null; //Delete old solver so we can start with new one. 
    }//end for loop over poly degrees

    }
    TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

    return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
  } // end test_hybrid_gmres_hb.cpp
