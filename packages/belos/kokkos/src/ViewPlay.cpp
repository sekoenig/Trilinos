#include<Kokkos_Core.hpp>
#include<KokkosBlas.hpp>
#include<BelosTypes.hpp>
#include<Kokkos_Random.hpp>

#include <Teuchos_SerialDenseMatrix.hpp>
//#include<KokkosSparse_spmv.hpp>

using ScalarType = double;
using MV = Kokkos::View<ScalarType**>;
using ScalarType2 = float;

void Kokkos2TeuchosMat(const Kokkos::View<const ScalarType**> & K,  Teuchos::SerialDenseMatrix<int, ScalarType> &T);
MV Clone(const MV &mv, const int numvecs);
MV CloneCopy(const MV &mv);
MV CloneViewNonConst( MV & mv, const std::vector<int> & index);
int GetNumberVecs(MV A);
std::ptrdiff_t GetGlobalLength(MV A);
void MvScale(MV A, ScalarType alpha);
void MvScale(MV & A, const std::vector<ScalarType> & alpha);
void MvTransMv( const ScalarType alpha, const MV & A, const MV & mv, Teuchos::SerialDenseMatrix<int, ScalarType> &B);
void MvPrint(MV A);
void SetBlock (const MV &A, const std::vector< int > &index, MV &mv);
void Assign(const MV &A, MV &mv);
void MvRandom(MV &mv);
void MvInit(MV &A, ScalarType alpha);

template<class ST>
void MvPrint(Kokkos::View<ST*> A){
  for(int i = 0; i < A.extent(0); i++){
      std::cout << A(i) << std::endl;
  } 
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
   Kokkos::initialize();
   {

   //int N = atoi(argv[1]);
   int N = 5;
   int M = 7;

   MV A("A",M,N);
   MV A2("A2",M,N);
   Kokkos::View<ScalarType*> x("X",N);
   Kokkos::View<ScalarType*> y("Y",N);
   Kokkos::deep_copy(A,4.0);
   Kokkos::deep_copy(A2,-2.1);
   Kokkos::deep_copy(x,-1.0);
   ScalarType alpha = 1.0;
   ScalarType beta  = 1.0;
  for(int i = 0; i<5; i++){
    y(i) = i;
  }

   MvPrint(A);

   MvPrint(x);

   Kokkos::View<ScalarType2*> xfloat;
   Kokkos::deep_copy(xfloat, x);

  KokkosBlas::scal(A2,y,A);
  std::cout << "Scaled matrix:" << std::endl;
  MvPrint(A2);

   MvRandom(A);
   std::cout << "Print random A:" << std::endl;
   MvPrint(A);

    KokkosBlas::axpy(-4.0,x,y);
    std::cout << "MV print -4.0*(-2.1) plus 0 1 2 3 4:" << std::endl;
    MvPrint(y);


   int numVecs = GetNumberVecs(A);
   std::cout << "Num Vecs " << numVecs << std::endl;

   std::ptrdiff_t length = GetGlobalLength(A);
   std::cout << "Vector length: " << length << std::endl;

  Teuchos::SerialDenseMatrix<int,double> myTeuchosMat(M,N);

  Kokkos2TeuchosMat(A,myTeuchosMat);
  myTeuchosMat.print(std::cout);

  Kokkos::View<ScalarType**> B = CloneCopy(A);
  std::cout << "Here is a copy of A: " << std::endl;
  MvPrint(B);

  //Test a few complex things:

  typedef Kokkos::complex<double> Kcomplex;
  Kokkos::View<Kcomplex*> x2("X2",N);
  Kokkos::View<Kcomplex*> y2("Y2",N);
  Kokkos::View<Kcomplex> z2("Z2");
  Kcomplex a,b;
  a.imag() = -2.0; a.real() = 1.0;
  b.imag() = 3.0; b.real() = -1.2;
  Kokkos::deep_copy(x2,a);
  Kokkos::deep_copy(y2,b);
  std::cout << "Here is complex x and y: " << std::endl;
  MvPrint(x2);
  MvPrint(y2);
  KokkosBlas::dot(z2,x2,y2);
  std::cout << "Here is the dot prod: " << std::endl;
  std::cout << z2() << std::endl;
  }
  Kokkos::finalize();
}

//Creation Methods
MV Clone(const MV &mv, const int numvecs){
  MV temp("MV",mv.extent(0),numvecs);
  return temp;
}

MV CloneCopy(const MV &mv){
  MV temp("MV",mv.extent(0),mv.extent(1));
  Kokkos::deep_copy(temp,mv);
  return temp;
}
//Kokkos::View<double*[2]> B("B",A.extent(0)); Kokkos::deep_copy(B,Kokkos::subview(A,Kokkos::ALL,std::make_pair(4,5)));
MV CloneCopy( MV & mv, const std::vector<int> & index){
  int numvecs = index.at(1) + 1 - index.at(0);
  Kokkos::View<ScalarType**> B("B", mv.extent(0), numvecs);
  // Be careful with indexing- need to add 1 to last index value b/c Belos includes value at last index while Kokkos doesn't.
  // TODO might need to check that index bounds are valid. 
  Kokkos::deep_copy(B,Kokkos::subview(mv, Kokkos::ALL, std::make_pair(index.at(0), index.at(1)+1)));
  return B; 
}

MV CloneViewNonConst( MV & mv, const std::vector<int> & index){
  Kokkos::View<ScalarType**> B = Kokkos::subview(mv, Kokkos::ALL, std::make_pair(index.at(0), index.at(1)+1));
  return B; 
}

//Attribute Methods

int GetNumberVecs(MV A){
  return A.extent(1);
}

std::ptrdiff_t GetGlobalLength(MV A){
  return A.extent(0);
}

//Update Methods:
//
void Kokkos2TeuchosMat(const Kokkos::View<const ScalarType**> & K,  Teuchos::SerialDenseMatrix<int, ScalarType> &T){
TEUCHOS_TEST_FOR_EXCEPTION(K.extent(0) != T.numRows() || K.extent(1) != T.numCols(), std::runtime_error, "Error: Matrix dimensions do not match!");
//This is all on host, so there's no use trying to use parallel_for, right?... Well, host could have openMP... TODO improve this?
  for(int i = 0; i < K.extent(0); i++){
    for (int j = 0; j < K.extent(1); j++){
      T(i,j) = K(i,j);
    }
  } 
}

void Teuchos2KokkosMat(const Teuchos::SerialDenseMatrix<int, ScalarType> &T, Kokkos::View<ScalarType**> & K){
TEUCHOS_TEST_FOR_EXCEPTION(K.extent(0) != T.numRows() || K.extent(1) != T.numCols(), std::runtime_error, "Error: Matrix dimensions do not match!");
//This is all on host, so there's no use trying to use parallel_for, right?... Well, host could have openMP... TODO improve this?
  for(int i = 0; i < K.extent(0); i++){
    for (int j = 0; j < K.extent(1); j++){
      K(i,j) = T(i,j);
    }
  } 
}

//Example of error throwing within Kokkos without Teuchos:
//if (entriesB.extent(0) > max_integer|| entriesA.extent(0) > max_integer){
//        throw std::runtime_error ("MKL requires integer values for size type for SPGEMM. Copying to integer will cause overflow.\n"); 
//        return;
//              }

void MvTimesMatAddMv( const ScalarType alpha, const MV &A, const Teuchos::SerialDenseMatrix<int, ScalarType> & B, const ScalarType beta, MV & mv){
  Kokkos::View<ScalarType**> mat("mat", A.extent(1), mv.extent(1));
  Teuchos2KokkosMat(B,mat);
  KokkosBlas::gemm("N", "N", alpha, A, mat, beta, mv);
}
// In normal use of this method, does the correct size of the Teuchos SerialDenseMat have to be defined ahead of time?? 
// Or is that determined based on the size of the multiplied matrices?? 

void MvAddMv(const ScalarType alpha, const MV &A, const ScalarType beta, const MV &B, MV&mv){
  Kokkos::deep_copy(mv, B);
  KokkosBlas::axpby(alpha, A, beta, mv);
}

void MvScale(MV & A, ScalarType alpha){
  //Later- Can we do this better with less copying?  TODO
  MV temp = Clone( A, A.extent(1));
  KokkosBlas::scal(temp, alpha, A); 
  Kokkos::deep_copy(A, temp);
}

void MvScale(MV & A, const std::vector<ScalarType> & alpha){
  //Later- Can we do this better with less copying?  TODO
  MV temp = Clone( A, A.extent(1));
  Kokkos::View<ScalarType*> scalars("alpha", alpha.size());
  for( int i = 0 ; i < alpha.size(); i++){
    scalars(i) = alpha.at(i);
  } 
  KokkosBlas::scal(temp, scalars, A); 
  Kokkos::deep_copy(A, temp);
}

void MvTransMv( const ScalarType alpha, const MV & A, const MV & mv, Teuchos::SerialDenseMatrix<int, ScalarType> &B){
  Kokkos::View<ScalarType**> soln("soln", A.extent(1), mv.extent(1));
  KokkosBlas::gemm("C", "N", alpha, A, mv, ScalarType(0.0), soln);
  Kokkos2TeuchosMat(soln, B);
}

void MvDot(const MV & mv, const MV & A, std::vector<ScalarType> &b){
  Kokkos::View<ScalarType*> dotView("Dot",mv.extent(1));
  KokkosBlas::dot(dotView, A, mv); //TODO check- it should be A that is conjugate transposed, not mv.  Is it??
  for(int i=0; i < mv.extent(1); i++){
    b.push_back(dotView(i)); //Is there a better way to do this?
    //TODO: will probably have to mirror the normView to the host space. 
  }
}

//Norm Method
void MvNorm(const MV &mv, std::vector<ScalarType> &normvec, Belos::NormType type=Belos::NormType::TwoNorm)
{
  Kokkos::View<ScalarType*> normView("Norm",mv.extent(1));
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

void MvRandom(MV &mv){
  Kokkos::Random_XorShift64_Pool<> pool(12371);
  Kokkos::fill_random(mv, pool, -1,1);
}

void SetBlock (const MV &A, const std::vector< int > &index, MV &mv){
//TODO skip subview if wants whole multivec?
//TODO check bounds of index??
  Kokkos::View<ScalarType**> Asub = Kokkos::subview(A, Kokkos::ALL, std::make_pair(index.at(0), index.at(1)+1));
  Kokkos::View<ScalarType**> MVsub = Kokkos::subview(mv, Kokkos::ALL, std::make_pair(index.at(0), index.at(1)+1));
  Kokkos::deep_copy(MVsub, Asub);
}

void Assign(const MV &A, MV &mv){
   Kokkos::deep_copy(mv,A);
}

//Print Method:
void MvPrint(MV A){
  for(int i = 0; i < A.extent(0); i++){
    for (int j = 0; j < A.extent(1); j++){
      std::cout << A(i , j) << "  ";
      }
    std::cout << std::endl;
  } 
  std::cout << std::endl;
}

