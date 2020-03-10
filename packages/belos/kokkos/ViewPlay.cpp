#include<Kokkos_Core.hpp>
#include<KokkosSparse_spmv.hpp>


void printView(Kokkos::View<double**> A);
void printView(Kokkos::View<double*> A);
int GetNumberVecs(Kokkos::View<double**> A);
std::ptrdiff_t GetGlobalLength(Kokkos::View<double**> A);

int main(int argc, char* argv[]) {
   Kokkos::initialize();

   //int N = atoi(argv[1]);
   int N = 5;
   int M = 7;

   Kokkos::View<double**> A("A",M,N);
   Kokkos::View<double*> x("X",N);
//   Kokkos::View<double*> y("Y",N);
   Kokkos::deep_copy(A,1.0);
   Kokkos::deep_copy(x,-1.0);
   double alpha = 1.0;
   double beta  = 1.0;

   printView(A);

   printView(x);

   int numVecs = GetNumberVecs(A);
   std::cout << "Num Vecs " << numVecs << std::endl;

   std::ptrdiff_t length = GetGlobalLength(A);
   std::cout << "Vector length: " << length << std::endl;

   Kokkos::finalize();
}

int GetNumberVecs(Kokkos::View<double**> A){
  return A.extent(1);
}

std::ptrdiff_t GetGlobalLength(Kokkos::View<double**> A){
  return A.extent(0);
}

void printView(Kokkos::View<double**> A){
  for(int i = 0; i < A.extent(0); i++){
    for (int j = 0; j < A.extent(1); j++){
      std::cout << A(i , j);
      }
    std::cout << std::endl;
  } 
  std::cout << std::endl;
}

void printView(Kokkos::View<double*> A){
  for(int i = 0; i < A.extent(0); i++){
      std::cout << A(i);
  } 
  std::cout << std::endl;
}

