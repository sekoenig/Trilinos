#include<Kokkos_Core.hpp>
#include<KokkosBlas.hpp>
#include<BelosTypes.hpp>
#include<Kokkos_Random.hpp>
#include "Teuchos_CommandLineProcessor.hpp"
#include "BelosKokkosAdapter.hpp"
#include "KokkosKernels_IOUtils.hpp"

#include <math.h> //For ceiling and floor functions.  

int main(int argc, char *argv[]) {
  typedef double                            ST;
  typedef int                               OT;
  //typedef Kokkos::DefaultExecutionSpace     EXSP;
  //typedef Kokkos::CudaSpace                EXSP;
  typedef Kokkos::HostSpace               EXSP; //for development purposes. 


  using ViewBlocksType = Kokkos::View<ST***,Kokkos::LayoutRight, EXSP>;
  using std::cout;
  using std::endl;

  Kokkos::initialize();
  {
    std::string filename("bcsstk01.mtx"); // example matrix
    int blkSize = 4; // Size of diagonal blocks. 

    Teuchos::CommandLineProcessor cmdp(false,true);
    cmdp.setOption("filename",&filename,"Filename for test matrix.  Acceptable file extensions: *.hb,*.mtx,*.triU,*.triS");
    cmdp.setOption("blkSize",&blkSize,"Size of each block.  Matrix size must be evenly divisible by block size.");
    if (cmdp.parse(argc,argv) != Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL) {
      return -1;
    }

    cout << "Filename is: " << filename << endl;
    cout << "Block size is: " << blkSize << endl;
    // Read in a matrix Market file and use it to test the Kokkos Operator.
    KokkosSparse::CrsMatrix<ST, OT, EXSP> crsMat = 
      KokkosKernels::Impl::read_kokkos_crst_matrix<KokkosSparse::CrsMatrix<ST, OT, EXSP>>(filename.c_str()); 
    OT numRows = crsMat.numRows();

    //Right now, we only support if the block size evenly divides the size of the matrix.
    if(numRows % blkSize != 0){
      throw std::runtime_error ("Num rows is not divisible by block size.");
    }
    int numBlks = ceil(numRows/blkSize ); // Need ceil for later if use uneven blocks.
    ViewBlocksType blocksView("diagBlcks", numBlks, blkSize, blkSize);

    auto values = crsMat.values; 
    auto colIdxView = crsMat.graph.entries;
    auto rowPtrView = crsMat.graph.row_map;

    int colPtr = 0;
    for (int row = 0; row < numRows; row++){
      int rowBlk = floor(row/blkSize);
      while( colPtr < rowPtrView(row+1) ) { // If we are still in the same row...
        int colBlk = floor( colIdxView(colPtr)/blkSize );
          if( rowBlk == colBlk) { // Then we are in one of the blocks to extract.
            blocksView( rowBlk, row % blkSize, colIdxView(colPtr) % blkSize ) = values(colPtr);
          }
        colPtr++;
      }
    }

    //Print the blocks so we can see if we did it right:
    for(int b = 0; b < blocksView.extent(0); b++){
      std::cout << "Block number " << b << std::endl;
      for(int i = 0; i < blocksView.extent(1); i++){
        for (int j = 0; j < blocksView.extent(2); j++){
          std::cout << blocksView(b, i , j) << "  ";
        }
        std::cout << std::endl;
      } 
      std::cout << std::endl;
    }

  }
  Kokkos::finalize();
}
