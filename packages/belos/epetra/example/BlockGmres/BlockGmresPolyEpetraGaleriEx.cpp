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

// The Trilinos package Galeri has many example problems.
#include "Galeri_Maps.h"
#include "Galeri_CrsMatrices.h"

#include "EpetraExt_readEpetraLinearSystem.h"
#include "Epetra_Map.h"
#include "Epetra_Import.h"
#ifdef EPETRA_MPI
  #include "Epetra_MpiComm.h"
#else
  #include "Epetra_SerialComm.h"
#endif
#include "Epetra_CrsMatrix.h"
#include <EpetraExt_MultiVectorIn.h>
#include <EpetraExt_CrsMatrixIn.h>

#include "Ifpack.h"
#include <Ifpack_IlukGraph.h>
#include <Ifpack_CrsRiluk.h>
#include "Ifpack_Chebyshev.h"
#include "Ifpack_PointRelaxation.h"

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

  bool verbose = true;
  bool success = true;
  try {
    bool proc_verbose = false;
    bool debug = false;
    bool userandomrhs = true;
    bool damppoly = false;
    bool addRoots = true;
    int frequency = 50;        // frequency of status test output.
    int blocksize = 1;         // blocksize
    int numrhs = 1;            // number of right-hand sides to solve for
    int maxiters = 3000;         // maximum number of iterations allowed per linear system
    int maxdegree = 25;        // maximum degree of polynomial
    int maxsubspace = 50;      // maximum number of blocks the solver can use for the subspace
    int maxrestarts = 50;      // number of restarts allowed
    int OverlapLevel = 1;   //Overlap level for non-poly preconditioners must be >= 0. If Comm.NumProc() == 1,it is ignored.   
    int ilutFill_ = 1;      //Fill level for ILU factorization
    int nx = 10;               // number of discretization points in each direction
    int chebyPowerIters = 10;
    double aThresh_ = 0.0001;
    double rThresh_ = 1.0001;
    double dropTol_ = 1e-3;
    double diff = 1e-5; //Diffusion term
    double conv = 1.0; //Convection term
    std::string outersolver("Block Gmres");
    std::string polytype("Roots");
    std::string filename("");
    std::string rhsfile;
    std::string precond("right");
    std::string partition("zoltan");
    std::string PrecType = "ILU"; // incomplete LU
    std::string orthog("ICGS");
    std::string MatrixType("Laplace3D");
    MT tol = 1.0e-8;           // relative residual tolerance
    MT polytol = tol/10;       // relative residual tolerance for polynomial construction

    Teuchos::CommandLineProcessor cmdp(false,true);
    cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
    cmdp.setOption("debug","nondebug",&debug,"Print debugging information from solver.");
    cmdp.setOption("use-random-rhs","use-rhs",&userandomrhs,"Use linear system RHS or random RHS to generate polynomial.");
    cmdp.setOption("damp-poly","no-damp",&damppoly,"Damp the polynomial.");
    cmdp.setOption("add-roots","no-add-roots",&addRoots,"Add extra roots as needed to stabilize the polynomial.");
    cmdp.setOption("frequency",&frequency,"Solvers frequency for printing residuals (#iters).");
    cmdp.setOption("filename",&filename,"Filename for test matrix.  Acceptable file extensions: *.hb,*.mtx,*.triU,*.triS. If empty, will generate Galeri problem.");
    cmdp.setOption("rhsfile",&rhsfile,"Filename for test rhs b. ");
    cmdp.setOption("outersolver",&outersolver,"Name of outer solver to be used with GMRES poly");
    cmdp.setOption("poly-type",&polytype,"Name of the polynomial to be generated.");
    cmdp.setOption("precond",&precond,"Preconditioning placement (none, left, right).");
    cmdp.setOption("prec-type",&PrecType,"Preconditioning type (Amesos, ILU, ILUT, ILUK2, Cheby, Diag, none).");
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
    cmdp.setOption("max-degree",&maxdegree,"Maximum degree of the GMRES polynomial.");
    cmdp.setOption("max-subspace",&maxsubspace,"Maximum number of blocks the solver can use for the subspace.");
    cmdp.setOption("max-restarts",&maxrestarts,"Maximum number of restarts allowed for GMRES solver.");
    cmdp.setOption("partition",&partition,"Partitioning type (zoltan, linear, none).");
    cmdp.setOption("nx",&nx,"Number of discretization points in each direction of PDE.");
    cmdp.setOption("matrix-type",&MatrixType,"Matrix type. See Galeri documentation. (Default: Laplace3D)");
    cmdp.setOption("diff",&diff,"Diffusion term.  Default: 1e-5");
    cmdp.setOption("conv",&conv,"Convection term.  Default: 1.0");
    cmdp.setOption("cheby-iters",&chebyPowerIters,"Iterations of power method to estimate large eval for Cheby poly.  Default: 10");

    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      return -1;
    }
    if (!verbose)
      frequency = -1;  // reset frequency if test is not verbose
    if( precond != "none" && precond != "right" && precond != "left")
    {
      if(MyPID==0) std::cout << "Error: Invalid string for precond param." << std::endl;
      return -1; 
    }
    if( PrecType != "Amesos" && PrecType!= "ILU" && PrecType != "ILUT" && PrecType != "ILUK2" && PrecType != "Cheby" && PrecType != "none"&& PrecType != "Diag")
    {
      if(MyPID==0) std::cout << "Error: Invalid string for prec-type param." << std::endl;
      return -1; 
    }

    proc_verbose = verbose && (MyPID==0);  /* Only print on the zero processor */
    //
    // Get the problem
    //
    RCP<Epetra_Map> Map;
    RCP<Epetra_CrsMatrix> A;
    if(filename.compare("") != 0){//Read in problem
      RCP<Epetra_Vector> vecB, vecX;
      EpetraExt::readEpetraLinearSystem(filename, Comm, &A, &Map, &vecX, &vecB);
    }
    else{//Generate Galeri problem
      std::string MapType("Cartesian2D");
      if( MatrixType == "Laplace3D" || MatrixType == "Cross3D"){
        MapType = "Cartesian3D";
      }

      Teuchos::ParameterList GaleriList;
      GaleriList.set ("n", nx * nx * nx);
      GaleriList.set ("nx", nx);
      GaleriList.set ("ny", nx);
      GaleriList.set ("nz", nx);
      GaleriList.set ("diff", diff);
      GaleriList.set ("conv", conv);
      Map = rcp (Galeri::CreateMap (MapType, Comm, GaleriList));
      A = rcp (Galeri::CreateCrsMatrix (MatrixType, &*Map, GaleriList));
    }

    A->OptimizeStorage();
    if(proc_verbose) {
      std::cout << "Matrix Size is: " << A->NumGlobalRows() << std::endl;
      std::cout << "NNZ : " << A->NumGlobalNonzeros() << std::endl;
    }

    // If file given, read in RHS.  Else, make it random.
    RCP<Epetra_MultiVector> B, X;
    X = rcp( new Epetra_MultiVector( *Map, numrhs ) );
    B = rcp( new Epetra_MultiVector( *Map, numrhs ) );
    X->PutScalar( 0.0 );
    if(rhsfile.compare("") != 0){
      {if(MyPID==0) std::cout << "Reading in rhs" << std::endl;}
      Epetra_MultiVector *b=0;
      int finfo = EpetraExt::MatrixMarketFileToMultiVector( rhsfile.c_str() , *Map, b );
      if (finfo!=0)
      {if(MyPID==0)std::cout << "First rhs file could not be read in!!!, info = "<< finfo << std::endl;}
      else
      {if(MyPID==0) std::cout << "rhs read was successful!" << std::endl;}
      B = rcp(b);
    }
    else{
      B->Random();
    }

#ifdef EPETRA_MPI
    if(partition == "linear"){
      // Rebalance linear system across multiple processors.
      // NOTE: After this section A, X, B, and Map will all be rebalanced over multiple MPI processors.
      if ( Comm.NumProc() > 1 ) { 
        RCP<Epetra_Map> newMap = Teuchos::rcp( new Epetra_Map( Map->NumGlobalElements(), Map->IndexBase(), Comm ) );
        RCP<Epetra_Import> newImport = Teuchos::rcp( new Epetra_Import( *newMap, *Map ) );

        // Create rebalanced versions of the linear system.
        RCP<Epetra_CrsMatrix> newA = Teuchos::rcp( new Epetra_CrsMatrix( Copy, *newMap, 0 ) );
        newA->Import( *A, *newImport, Insert );
        newA->FillComplete();
        RCP<Epetra_MultiVector> newB = Teuchos::rcp( new Epetra_MultiVector( *newMap, numrhs ) );
        newB->Import( *B, *newImport, Insert );
        RCP<Epetra_MultiVector> newX = Teuchos::rcp( new Epetra_MultiVector( *newMap, numrhs ) );
        newX->Import( *X, *newImport, Insert );

        // Set the pointers to the new rebalance linear system.
        A = newA;
        B = newB;
        X = newX;
        Map = newMap;
      }  
    } else if(partition == "zoltan"){
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
    }
    //End redistribute
#endif
  

    std::vector<double> tempnorm(numrhs), temprhs(numrhs);
    MVT::MvNorm(*B, temprhs);
    MVT::MvNorm(*X, tempnorm);
    {if(MyPID==0) std::cout << "Norm of B: " << temprhs[0] << " Norm of X: " << tempnorm[0] << std::endl;}
    //X->PutScalar( 0.0 );
    //B->PutScalar( 1.0 );
    //
    // ************Construct preconditioner*************
    //
    RCP<Belos::EpetraPrecOp> belosPrec;

    if (precond != "none" && PrecType != "none") {
      if(PrecType == "ILUK2"){
        // Ifpack preconditioning classes.
        Teuchos::RCP<Ifpack_IlukGraph> ilukGraph_;
        Teuchos::RCP<Ifpack_CrsRiluk>  rILUK_;

        // Initialize the graph if we need to.
        if ( Teuchos::is_null( ilukGraph_ ) )
        {
          const Epetra_CrsGraph & Graph = A->Graph();
          ilukGraph_ = Teuchos::rcp( new Ifpack_IlukGraph( Graph, ilutFill_, OverlapLevel ) );
          int graphRet = ilukGraph_->ConstructFilledGraph();
          TEUCHOS_TEST_FOR_EXCEPT( graphRet != 0 );

          // Create the preconditioner if one doesn't exist.
          if ( Teuchos::is_null( rILUK_ ) )
          {
            rILUK_ = Teuchos::rcp( new Ifpack_CrsRiluk( *ilukGraph_ ) );
            rILUK_->SetAbsoluteThreshold( aThresh_ );
            rILUK_->SetRelativeThreshold( rThresh_ );
          }

          if(MyPID==0) std::cout << "Prec paremeters set. Overlap is: " << OverlapLevel << std::endl;
          int initErr = rILUK_->InitValues( *A );
          TEUCHOS_TEST_FOR_EXCEPT( initErr != 0 );
          if(MyPID==0) std::cout << "Prec Initialized." << std::endl;

          int factErr = rILUK_->Factor();
          TEUCHOS_TEST_FOR_EXCEPT( factErr != 0 );
          if(proc_verbose) std::cout << "Prec Computed." << std::endl;
        }

        // Create the Belos preconditioned operator from the Ifpack preconditioner.
        // NOTE:  This is necessary because Belos expects an operator to apply the
        //        preconditioner with Apply() NOT ApplyInverse().
        belosPrec = rcp( new Belos::EpetraPrecOp( rILUK_ ) );
      }
      else if(PrecType == "Cheby"){

        ParameterList ChebyList;
        ChebyList.set("chebyshev: degree", maxdegree); //Do NOT need to subtract 1 to make it correspond to our poly Ap(A).
        ChebyList.set("chebyshev: eigenvalue max iterations", chebyPowerIters);
        maxdegree = 0; //Turn off our poly when using Cheby poly. 

        Teuchos::RCP<Ifpack_Chebyshev> Cheby = Teuchos::rcp( new Ifpack_Chebyshev(&*A) );
        Cheby->SetParameters(ChebyList);
        // Build a Chebyshev Polynomial
        Cheby->Initialize();
        if(MyPID==0){
          std::cout << "Prec Initialized." << std::endl;
          std::cout << "Initialization time: " << Cheby->InitializeTime() << std::endl;
        }
        // Cheby.SetParameters(ChebyList);
        Cheby->Compute();
        if(MyPID==0){
          std::cout << "Prec computed." << std::endl;
          std::cout << "Pre compute time: " << Cheby->ComputeTime() << std::endl;
        }
        belosPrec = rcp( new Belos::EpetraPrecOp( Cheby ) );
        
        /*Ifpack_Chebyshev Cheby(&*A);
        Cheby.SetParameters(ChebyList);
        // Build a Chebyshev Polynomial
        Cheby.Initialize();
        if(MyPID==0){
          std::cout << "Prec Initialized." << std::endl;
          std::cout << "Initialization time: " << Cheby.InitializeTime() << std::endl;
        }
        // Cheby.SetParameters(ChebyList);
        Cheby.Compute();
        if(MyPID==0){
          std::cout << "Prec computed." << std::endl;
          std::cout << "Pre compute time: " << Cheby.ComputeTime() << std::endl;
        }
        belosPrec = rcp( new Belos::EpetraPrecOp( Cheby ) );*/
      }
      else{
        ParameterList ifpackList;

        // allocates an IFPACK factory. No data is associated
        // to this object (only method Create()).
        Ifpack Factory;

        // create the preconditioner. For valid PrecType values,
        // please check the documentation
        // Pass solver type to Amesos.

        if (PrecType == "Amesos")
        {
          ifpackList.set("amesos: solver type", "Amesos_Klu");
          //if (diagPerturb_ != 0.0)
          //  ifpackList.set("AddToDiag", diagPerturb_);
        }
        else if (PrecType == "ILUT")
        {
          ifpackList.set("fact: absolute threshold", aThresh_);
          ifpackList.set("fact: relative threshold", rThresh_);
          ifpackList.set("fact: ilut level-of-fill", (double)ilutFill_);
          ifpackList.set("fact: drop tolerance", dropTol_);
        }
        else if (PrecType == "ILU")
        {
          ifpackList.set("fact: absolute threshold", aThresh_);
          ifpackList.set("fact: relative threshold", rThresh_);
          ifpackList.set("fact: level-of-fill", (int)ilutFill_);
          ifpackList.set("fact: drop tolerance", dropTol_);
        }
        else if (PrecType == "Diag")
        {
          ifpackList.set("relaxation: type", "Jacobi");
          PrecType = "point relaxation";
        }

        if (OverlapLevel)
        {
          // the combine mode is on the following:
          // "Add", "Zero", "Insert", "InsertAdd", "Average", "AbsMax"
          // Their meaning is as defined in file Epetra_CombineMode.h
          ifpackList.set("schwarz: combine mode", "Add"); //Add is defualt combine mode.
        }

        RCP<Ifpack_Preconditioner> Prec = Teuchos::rcp( Factory.Create(PrecType, &*A, OverlapLevel) );
        assert(Prec != Teuchos::null);

        // sets the parameters
        IFPACK_CHK_ERR(Prec->SetParameters(ifpackList));
        if(MyPID==0) std::cout << "Prec paremeters set. Overlap is: " << OverlapLevel << std::endl;
        // initialize the preconditioner. At this point the matrix must
        // have been FillComplete()'d, but actual values are ignored.
        IFPACK_CHK_ERR(Prec->Initialize());
        if(MyPID==0){
          std::cout << "Prec Initialized." << std::endl;
          std::cout << "Initialization time: " << Prec->InitializeTime() << std::endl;
        }

        // Builds the preconditioners, by looking for the values of
        // the matrix.
        IFPACK_CHK_ERR(Prec->Compute());
        if(MyPID==0){
          std::cout << "Prec computed." << std::endl;
          std::cout << "Pre compute time: " << Prec->ComputeTime() << std::endl;
        }

        //Check conditioner number of preconditioner:
        double condEst;
        condEst = Prec->Condest();
        if(proc_verbose ) std::cout << "ILU condition est:" << condEst << std::endl;

        // Create the Belos preconditioned operator from the Ifpack preconditioner.
        // NOTE:  This is necessary because Belos expects an operator to apply the
        //        preconditioner with Apply() NOT ApplyInverse().
        belosPrec = rcp( new Belos::EpetraPrecOp( Prec ) );
        if(MyPID==0) std::cout << "ILU Preconditioner has been created." << std::endl;

      }//End else for Ifpack factory precs
    }//End if prec exists


    //
    // ********Other information used by block solver***********
    // *****************(can be user specified)******************
    //
    const int NumGlobalElements = B->GlobalLength();
    if (maxiters == -1)
      maxiters = NumGlobalElements/blocksize - 1; // maximum number of iterations to run
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
    polyList.set( "Damped Poly", damppoly );              // Option to damp polynomial
    polyList.set( "Add Roots", addRoots );                // Option to add roots to stabilize poly 
    polyList.set( "Orthogonalization", orthog);           // Type of orthogonalizaion: DGKS, IMGS, ICGS
    if ( outersolver != "" ) {
      polyList.set( "Outer Solver", outersolver );
      polyList.set( "Outer Solver Params", belosList );
    }
    //
    // Construct an unpreconditioned linear problem instance.
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

    // Create an iterative solver manager.
    RCP< Belos::SolverManager<double,MV,OP> > newSolver
      = rcp( new Belos::GmresPolySolMgr<double,MV,OP>(rcp(&problem,false), rcp(&polyList,false)));

    //
    // **********Print out information about problem*******************
    //
    if (proc_verbose) {
      std::cout << std::endl << std::endl;
      std::cout << "Dimension of matrix: " << NumGlobalElements << std::endl;
      std::cout << "Number of right-hand sides: " << numrhs << std::endl;
      std::cout << "Block size used by solver: " << blocksize << std::endl;
      std::cout << "Max number of restarts allowed: " << maxrestarts << std::endl;
      std::cout << "Max number of Gmres iterations per restart cycle: " << maxiters << std::endl;
      std::cout << "Relative residual tolerance: " << tol << std::endl;
      std::cout << std::endl;
    }
    //
    // Perform solve
    //
    if(MyPID==0) std::cout << "Performing Solve: " << std::endl;
    Belos::ReturnType ret = newSolver->solve();
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
        std::cout << "Act resid: " << actual_resids[i] << "  rhs norm " << rhs_norm[i] << std::endl;
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
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(verbose, std::cerr, success);

#ifdef EPETRA_MPI
  MPI_Finalize();
#endif

  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
