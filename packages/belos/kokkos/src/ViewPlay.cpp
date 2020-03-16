#include<Kokkos_Core.hpp>
#include<KokkosBlas.hpp>
#include<BelosTypes.hpp>
//#include<KokkosSparse_spmv.hpp>

using ScalarType = double;
using MV = Kokkos::View<ScalarType**>;

MV Clone(const MV &mv, const int numvecs);
int GetNumberVecs(MV A);
std::ptrdiff_t GetGlobalLength(MV A);
void MvScale(MV A, ScalarType alpha);
void MvPrint(MV A);
void MvPrint(Kokkos::View<ScalarType*> A);
void Assign(const MV &A, MV &mv);
void MvInit(MV &A, ScalarType alpha);

int main(int argc, char* argv[]) {
   Kokkos::initialize();
  {

   //int N = atoi(argv[1]);
   int N = 5;
   int M = 7;

   MV A("A",M,N);
   Kokkos::View<ScalarType*> x("X",N);
//   Kokkos::View<ScalarType*> y("Y",N);
   Kokkos::deep_copy(A,1.0);
   Kokkos::deep_copy(x,-1.0);
   ScalarType alpha = 1.0;
   ScalarType beta  = 1.0;

   MvPrint(A);

   MvPrint(x);

   int numVecs = GetNumberVecs(A);
   std::cout << "Num Vecs " << numVecs << std::endl;

   std::ptrdiff_t length = GetGlobalLength(A);
   std::cout << "Vector length: " << length << std::endl;
  }
   Kokkos::finalize();
}

//Creation Methods
MV Clone(const MV &mv, const int numvecs){
   MV temp("MV",mv.extent(0),numvecs);
   return temp;
}

//MV CloneCopy(const MV &mv){
//   
//}

//Attribute Methods

int GetNumberVecs(MV A){
  return A.extent(1);
}

std::ptrdiff_t GetGlobalLength(MV A){
  return A.extent(0);
}

//Update Methods:

void MvAddMv(const ScalarType alpha, const MV &A, const ScalarType beta, const MV &B, MV&mv){
  Kokkos::deep_copy(mv, B);
  KokkosBlas::axpby(alpha, A, beta, mv);
}

void MvScale(MV A, ScalarType alpha){
  //Later- Can we do this better with less copying?  TODO
  MV temp = Clone( A, A.extent(1));
  KokkosBlas::scal(temp, alpha, A); 
  Kokkos::deep_copy(A, temp);
}

//Norm Method
void MvNorm(const MV &mv, std::vector<ScalarType> &normvec, Belos::NormType type=Belos::NormType::TwoNorm)
{
  Kokkos::View<double*> normView("Norm",mv.extent(1));
  if(type == Belos::NormType::TwoNorm){
    KokkosBlas::nrm2(normView, mv);
  }
  else if(type == Belos::NormType::OneNorm){
    KokkosBlas::nrm1(normView, mv);
  }
  else if(type == Belos::NormType::InfNorm){
    KokkosBlas::nrminf(normView, mv);
  }
  //TODO precond norm- same as inf norm??
  for(int i=0; i < mv.extent(1); i++){
    normvec.push_back(normView(i)); //Is there a better way to do this?
    //TODO: will probably have to mirror the normView to the host space. 
  }
}

//Initialization Methods:
void MvInit(MV &mv, ScalarType alpha){
   Kokkos::deep_copy(mv,alpha);
}

void Assign(const MV &A, MV &mv){
   Kokkos::deep_copy(mv,A);
}

//Print Method:
void MvPrint(MV A){
  for(int i = 0; i < A.extent(0); i++){
    for (int j = 0; j < A.extent(1); j++){
      std::cout << A(i , j);
      }
    std::cout << std::endl;
  } 
  std::cout << std::endl;
}

void MvPrint(Kokkos::View<ScalarType*> A){
  for(int i = 0; i < A.extent(0); i++){
      std::cout << A(i) << std::endl;
  } 
  std::cout << std::endl;
}

