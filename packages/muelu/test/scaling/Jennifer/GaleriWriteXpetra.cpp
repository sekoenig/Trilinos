#include <Tpetra_Core.hpp>
// Xpetra
#include <Xpetra_Operator.hpp>
#include <Xpetra_Map.hpp>
#include <Xpetra_IO.hpp>
#include <Xpetra_TpetraOperator.hpp>
// Galeri
#include <Galeri_XpetraParameters.hpp>
#include <Galeri_XpetraProblemFactory.hpp>
#include <Galeri_XpetraUtils.hpp>
#include <Galeri_XpetraMaps.hpp>
// Read Matrix Utility
#include <JMatrixLoad.hpp>

using Teuchos::RCP;
using Teuchos::rcp;

int main(int argc, char *argv[]) {
  typedef double ST;
  typedef int LO;
  typedef long long GO;
  typedef KokkosClassic::DefaultNode::DefaultNodeType Node;
  bool success = true;
  Tpetra::ScopeGuard tpetraScope (&argc, &argv);
  {
  // MPI initialization using Teuchos
  //Teuchos::GlobalMPISession mpiSession(&argc, &argv, NULL);
  //Kokkos::initialize(argc, argv);

  int nx = 10;               // number of discretization points in each direction
  std::string MatrixType("Laplace3D");
  std::string filename("");

  Teuchos::CommandLineProcessor cmdp(false,true);
  cmdp.setOption("filename",&filename,"Name of the file to which matrix is saved.  Default is MatrixType+nx.");
  cmdp.setOption("nx",&nx,"Number of discretization points in each direction of PDE.");
  cmdp.setOption("matrix-type",&MatrixType,"Matrix type. See Galeri documentation. (Default: Laplace3D)");

  if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
    return -1;
  }
  Xpetra::Parameters xpetraParams(cmdp);
  Galeri::Xpetra::Parameters<GO> galeriParams(cmdp, nx, nx, nx, MatrixType); 

    if(filename.compare("") == 0){//string::compare returns 0 when equal.
      filename = MatrixType + std::to_string(nx) + ".mtx";
    }   
    std::cout << "filename is: " << filename << std::endl;

  RCP<const Teuchos::Comm<int>> comm = rcp(new Teuchos::SerialComm<int>);
  RCP<Xpetra::Matrix<ST,LO,GO,Node>> A;

  MatrixLoad<ST,LO,GO,Node>( comm, A, galeriParams, xpetraParams);
  std::cout << "Matrix Size is: " << A->getGlobalNumRows() << std::endl;
  std::cout << "NNZ : " << A->getGlobalNumEntries() << std::endl;

  Xpetra::IO<ST,LO,GO,Node>::Write( filename, *A);

  //Kokkos::finalize();
  } //End Tpetra scope guard. 

  return ( success ? EXIT_SUCCESS : EXIT_FAILURE );
}
