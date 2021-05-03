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

// The Trilinos package Galeri has many example problems.
#include "Galeri_Maps.h"
#include "Galeri_CrsMatrices.h"

#include "Epetra_Map.h"
#include "Epetra_Import.h"
#ifdef EPETRA_MPI
  #include "Epetra_MpiComm.h"
#else
  #include "Epetra_SerialComm.h"
#endif
#include "Epetra_CrsMatrix.h"
#include "EpetraExt_RowMatrixOut.h"


#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include "Teuchos_YamlParameterListCoreHelpers.hpp"

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
  //typedef Teuchos::ScalarTraits<ST>        SCT;
  //typedef SCT::magnitudeType                MT;
  typedef Epetra_MultiVector                MV;
  typedef Epetra_Operator                   OP;

  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using Teuchos::rcp;


    bool verbose = true;
    bool proc_verbose = false;
    bool debug = false;
    int nx = 10;               // number of discretization points in each direction
    double diff = 1e-5; //Diffusion term
    double conv = 1.0; //Convection term
    std::string MatrixType("Laplace3D");
    std::string filename("");
    std::string yamlFile("");

    Teuchos::CommandLineProcessor cmdp(false,true);
    cmdp.setOption("verbose","quiet",&verbose,"Print messages and results.");
    cmdp.setOption("debug","nondebug",&debug,"Print debugging information from solver.");
    cmdp.setOption("filename",&filename,"Name of the file to which matrix is saved.  Default is MatrixType+nx.");
    cmdp.setOption("yamlfile",&yamlFile, "Name of the YAML file from which to read Galeri matrix paramters.");
    cmdp.setOption("nx",&nx,"Number of discretization points in each direction of PDE.");
    cmdp.setOption("matrix-type",&MatrixType,"Matrix type. See Galeri documentation. (Default: Laplace3D)");
    cmdp.setOption("diff",&diff,"Diffusion term.  Default: 1e-5");
    cmdp.setOption("conv",&conv,"Convection term.  Default: 1.0");

    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      return -1;
    }
    RCP<ParameterList> GaleriList;
    if(yamlFile.compare("") != 0){//string::compare returns 0 when equal.
      GaleriList = Teuchos::getParametersFromYamlFile(yamlFile);
    }
    else{
      GaleriList = rcp(new Teuchos::ParameterList);
    }

  proc_verbose = verbose && (MyPID==0);  /* Only print on the zero processor */


    //Generate Galeri problem
    RCP<Epetra_Map> Map;
    RCP<Epetra_CrsMatrix> A;
      std::string MapType("Cartesian2D");
      if( MatrixType == "Laplace3D" || MatrixType == "Cross3D"){
        MapType = "Cartesian3D";
      }

      if(!GaleriList->isParameter("nx"))
        GaleriList->set ("nx", nx);
      if(!GaleriList->isParameter("ny"))
        GaleriList->set ("ny", nx);
      if(!GaleriList->isParameter("nz"))
        GaleriList->set ("nz", nx);
      if(!GaleriList->isParameter("diff"))
        GaleriList->set ("diff", diff);
      if(!GaleriList->isParameter("conv"))
        GaleriList->set ("conv", conv);
      GaleriList->print(std::cout);
      Map = rcp (Galeri::CreateMap (MapType, Comm, *GaleriList));
      A = rcp (Galeri::CreateCrsMatrix (MatrixType, &*Map, *GaleriList));
    

    A->OptimizeStorage();
    if(proc_verbose) {
      std::cout << "Matrix Size is: " << A->NumGlobalRows() << std::endl;
      std::cout << "NNZ : " << A->NumGlobalNonzeros() << std::endl;
    }


    if(filename.compare("") == 0){//string::compare returns 0 when equal.
      filename = MatrixType + std::to_string(nx) + ".mtx";
    }

    std::cout << "filename is: " << filename << std::endl;
      

    //int finfo = EpetraExt::MatrixMarketFileToMultiVector( rhsfile.c_str() , *Map, b );
    //
    //Finally write the matrix:
    int ret = EpetraExt::RowMatrixToMatrixMarketFile(filename.c_str(), *A);
   if( ret != 0 ){
     throw std::runtime_error("Something went wrong when writing the file");
   }

}
