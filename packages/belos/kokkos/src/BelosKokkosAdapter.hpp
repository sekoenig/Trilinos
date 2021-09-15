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
//@HEADER

/*! \file BelosKokkosAdapter.hpp
    \brief Implementation of the interface between Belos virtual classes and Kokkos concrete classes.
*/

#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<KokkosBlas.hpp>
#include<KokkosSparse_spmv.hpp>

#include<BelosTypes.hpp>
#include<BelosMultiVec.hpp>
#include<BelosOperator.hpp>

#include <Teuchos_SerialDenseMatrix.hpp>


#ifndef BELOS_KOKKOS_ADAPTER_HPP
#define BELOS_KOKKOS_ADAPTER_HPP
namespace Belos {

//Forward class declaration of KokkosOperator:
template<class ScalarType, class OrdinalType, class Device>
class KokkosOperator; 

/// \class KokkosMultiVec
/// \brief Implementation of Belos::MultiVec using Kokkos::View.
/// \author Jennifer Loe
/// 
/// \tparam ScalarType The type of entries of the multivector.
/// \tparam Device The Kokkos::ExecutionSpace where the multivector
/// should reside.
///
/// Belos::MultiVec offers a simple abstract interface for
/// multivector operations in Belos solver algorithms.  This class
/// implements Belos::MultiVec using Kokkos::View.
template<class ScalarType, class Device = Kokkos::DefaultExecutionSpace >
class KokkosMultiVec : public MultiVec<ScalarType> {

//TODO T- Did we make myView public because of this??
  //Think it is okay for ScalarType to not match ScalarType2 b/c eventually we might want
  //MV and OP to have different precisions
//   template<class ScalarType2, class OrdinalType, class Device>
//   friend class KokkosOperator; 
public: //TODO: should these typedefs be public? 
  using ViewVectorType = Kokkos::View<ScalarType*,Kokkos::LayoutLeft, Device>;
  using ConstViewVectorType = Kokkos::View<const ScalarType*,Kokkos::LayoutLeft, Device>;
  using ViewMatrixType = Kokkos::View<ScalarType**,Kokkos::LayoutLeft, Device>;
  using ConstViewMatrixType = Kokkos::View<const ScalarType**,Kokkos::LayoutLeft, Device>;
  
  //Unmanaged view types: 
  //Not used now.  //using UMViewVectorType = Kokkos::View<ScalarType*,Kokkos::LayoutLeft, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using UMHostViewVectorType = Kokkos::View<ScalarType*,Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  //Not used now. //using UMConstViewVectorType = Kokkos::View<const ScalarType*,Kokkos::LayoutLeft, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using UMHostConstViewVectorType = Kokkos::View<const ScalarType*,Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  //using UMViewMatrixType = Kokkos::View<ScalarType**,Kokkos::LayoutLeft, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using UMHostViewMatrixType = Kokkos::View<ScalarType**,Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  //using UMConstViewMatrixType = Kokkos::View<const ScalarType**,Kokkos::LayoutLeft, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using UMHostConstViewMatrixType = Kokkos::View<const ScalarType**,Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  // constructors
  // TODO: Heidi: Should the views in these first 3 constructors be initialized?  Or not?
  KokkosMultiVec<ScalarType, Device> (const std::string label, const int numrows, const int numvecs) :
    myView (label,numrows,numvecs) { }

  KokkosMultiVec<ScalarType, Device> (const int numrows, const int numvecs) :
    myView ("MV",numrows,numvecs) { }

  KokkosMultiVec<ScalarType, Device> (const int numrows) :
    myView("MV",numrows,1) { }

  // Make so that copy constructor of MV gives deep copy.  
  template < class ScalarType2 >
    KokkosMultiVec<ScalarType, Device> (const KokkosMultiVec<ScalarType2, Device> &sourceVec) : 
    myView(Kokkos::ViewAllocateWithoutInitializing("MV"),(int)sourceVec.GetGlobalLength(),sourceVec.GetNumberVecs())
  { Kokkos::deep_copy(myView,sourceVec.myView); }

  //Need this explicitly, else compiler makes its own with shallow copy. 
  KokkosMultiVec<ScalarType, Device> (const KokkosMultiVec<ScalarType, Device> &sourceVec) : 
    myView(Kokkos::ViewAllocateWithoutInitializing("MV"),(int)sourceVec.GetGlobalLength(),sourceVec.GetNumberVecs())
  { Kokkos::deep_copy(myView,sourceVec.myView); }

  // Compiler default should give shallow copy.
  KokkosMultiVec<ScalarType, Device> & operator=(const KokkosMultiVec<ScalarType, Device> & sourceVec) {
    Kokkos::deep_copy(myView,sourceVec.myView);
    return *this;
  }
  
  template < class ScalarType2 >
  KokkosMultiVec<ScalarType, Device> & operator=(const KokkosMultiVec<ScalarType2, Device> & sourceVec) {
    Kokkos::deep_copy(myView,sourceVec.myView);
    return *this;
  }
  
  //TODO: Would it be better if this was deep copy?  So we can't change the user's original data?  
  // And so the user can't change ours?
  // But doing so would add a bunch of deep copy's to clone view... Maybe make it an option?  
  KokkosMultiVec<ScalarType, Device> (const ViewMatrixType & sourceView) : 
    myView(sourceView){ }
  
  // This version allows us to make exclusive changes to the view and convert between scalar types:
  template < class ScalarType2 > //TODO: Fix this so that passing in a view without device specified actually compiles...
  KokkosMultiVec<ScalarType, Device> (const Kokkos::View<ScalarType2**,Kokkos::LayoutLeft, Device> & sourceView) : 
    myView(Kokkos::ViewAllocateWithoutInitializing("MV"),sourceView.extent(0),sourceView.extent(1))
  { Kokkos::deep_copy(myView,sourceView); }

  //This function specialization makes things compile...else compiler can't deduce template type?
  //KokkosMultiVec<ScalarType> (const ViewMatrixType & sourceView) : 
  //    myView(Kokkos::ViewAllocateWithoutInitializing("MV"),sourceView.extent(0),sourceView.extent(1)) {Kokkos::deep_copy(myView,sourceView);}
  //If we've already made a view and want it to be a multivec... is this the right way to do it?? TODO
  ~KokkosMultiVec<ScalarType, Device>(){}

  //! @name Member functions inherited from Belos::MultiVec
  //@{

  /// A virtual "copy constructor" that returns a pointer to a new
  /// object of the pure virtual class.  This vector's entries are
  /// not copied; instead, a new MultiVec is created with the same
  /// data distribution, but with numvecs columns (numvecs > 0).
  ///
  /// \param numvecs [in] The number of columns in the output
  ///   multivector.  Must be positive.
  MultiVec<ScalarType> * Clone ( const int numvecs ) const{
    KokkosMultiVec<ScalarType, Device> * ptr = new KokkosMultiVec<ScalarType, Device>(myView.extent(0),numvecs);
    return ptr;
  }

  /// A virtual "copy constructor" returning a pointer to a new
  /// object of the pure virtual class.  This vector's entries are
  /// copied and a new stand-alone multivector is created.  (deep
  /// copy).
  MultiVec<ScalarType> * CloneCopy () const{
    KokkosMultiVec<ScalarType, Device> * ptr = new KokkosMultiVec<ScalarType, Device>(myView.extent(0),myView.extent(1));
    Kokkos::deep_copy(ptr->myView,myView);
    return ptr;
  }

  /// A virtual "copy constructor" returning a pointer to the pure
  /// virtual class.  This vector's entries are copied and a new
  /// stand-alone MultiVector is created where only selected columns
  /// are chosen.  (deep copy).
  MultiVec<ScalarType> * CloneCopy ( const std::vector<int>& index ) const{
    // Be careful with indexing- need to add 1 to last index value b/c Belos includes value at last index while Kokkos doesn't.
    // TODO might need to check that index bounds are valid. 
    int numvecs = index.size();
    KokkosMultiVec<ScalarType, Device> * B = new KokkosMultiVec<ScalarType, Device>("B",myView.extent(0),numvecs);
    bool isAscending = true;//Also checks if is contiguous.
    for(unsigned int i=0; i< (index.size()-1); i++){
      if( index[i+1] != index[i]+1 ){
        isAscending = false;
      }
    }
    if(isAscending && index.size()==(unsigned)this->GetNumberVecs()){ //Copy entire multivec.
      Kokkos::deep_copy(B->myView,myView);
    }
    else if (isAscending){ //Copy contiguous subset
      ViewMatrixType ThisSub = Kokkos::subview(myView, Kokkos::ALL, std::make_pair(index.front(), index.back()+1));
      Kokkos::deep_copy(B->myView,ThisSub);
    } 
    else{ //Copy columns one by one
      for(unsigned int i=0; i<index.size(); i++){
        //Both versions work; second requires auto while first doesn't?
        //
        //First version:
        //Kokkos::View<ScalarType**> Bsub = Kokkos::subview(B->myView, Kokkos::ALL, std::make_pair(i, i+1));
        //Kokkos::View<ScalarType**> ThisSub = Kokkos::subview(myView, Kokkos::ALL, std::make_pair(index[i], index[i]+1));

        //Second version:
        auto Bsub = Kokkos::subview(B->myView, Kokkos::ALL, i);
        auto ThisSub = Kokkos::subview(myView, Kokkos::ALL, index[i]);
        Kokkos::deep_copy(Bsub, ThisSub);
      }
    }
    return B; 
  }

  /// A virtual view "constructor" returning a pointer to the pure
  /// virtual class.  This vector's entries are shared and hence no
  /// memory is allocated for the columns.
  MultiVec<ScalarType> * CloneViewNonConst ( const std::vector<int>& index ){ //TODO this won't work for non-contiguous!
    bool isAscending = true;//Also checks if is contiguous.
    for(unsigned int i=0; i< (index.size()-1); i++){
      if( index[i+1] != index[i]+1 ){
        isAscending = false;
      }
    }
    if(isAscending ){ //Copy entire multivec.
    KokkosMultiVec<ScalarType, Device> * B = 
        new KokkosMultiVec<ScalarType, Device>(Kokkos::subview(myView, Kokkos::ALL, std::make_pair(index.front(), index.back()+1)));
      return B; 
    }
    else{
      throw std::runtime_error("CloneViewNonConst asked for non-contiguous subset. \n This feature is not supported in Belos for Kokkos.");
    }
  }

  /// A virtual view constructor returning a pointer to the pure
  /// virtual class.  This vector's entries are shared and hence no
  /// memory is allocated for the columns.
  const MultiVec<ScalarType> * CloneView ( const std::vector<int>& index ) const { //TODO implement this!! This isn't const!!
    bool isAscending = true;//Also checks if is contiguous.
    for(unsigned int i=0; i< (index.size()-1); i++){
      if( index[i+1] != index[i]+1 ){
        isAscending = false;
      }
    }
    if(isAscending ){ //Copy entire multivec.
    const KokkosMultiVec<ScalarType, Device> * B = 
        new KokkosMultiVec<ScalarType, Device>(Kokkos::subview(myView, Kokkos::ALL, std::make_pair(index.front(), index.back()+1)));
      return B; 
    }
    else{
      throw std::runtime_error("CloneView asked for non-contiguous subset. \n This feature is not supported in Belos for Kokkos.");
    }
  }


  /// Set a subblock of the multivector, which need not be
  /// contiguous, and is given by the indices.
  void SetBlock ( const MultiVec<ScalarType>& A, const std::vector<int>& index ){
    //TODO check bounds of index?? 
    KokkosMultiVec<ScalarType, Device> *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(A));
    bool isAscending = true;// Also checks for contiguous
    for(unsigned int i=0; i< (index.size()-1); i++){
      if( index[i+1] != index[i]+1 ){
        isAscending = false;
      }
    }
    if(isAscending && index.size()==(unsigned)this->GetNumberVecs()){ //Copy entire multivec.
      Kokkos::deep_copy(myView,A_vec->myView);
    }
    else if (isAscending){ //Copy contiguous subset
      ViewMatrixType Asub = Kokkos::subview(A_vec->myView, Kokkos::ALL, std::make_pair(index.front(), index.back()+1));
      ViewMatrixType ThisSub = Kokkos::subview(myView, Kokkos::ALL, std::make_pair(index.front(), index.back()+1));
      Kokkos::deep_copy(ThisSub, Asub);
    } 
    else{ //Copy columns one by one
      for(unsigned int i=0; i<index.size(); i++){
        ViewMatrixType Asub = Kokkos::subview(A_vec->myView, Kokkos::ALL, std::make_pair(i, i+1));
        ViewMatrixType ThisSub = Kokkos::subview(myView, Kokkos::ALL, std::make_pair(index[i], index[i]+1));
        Kokkos::deep_copy(ThisSub, Asub);
      }
    }
  }

  //! The number of rows in the multivector.
  ptrdiff_t GetGlobalLength () const {
    return static_cast<ptrdiff_t>(myView.extent(0));
  }

  //! The number of columns in the multivector.
  int GetNumberVecs () const { return myView.extent(1); }

  //! *this <- alpha * A * B + beta * (*this)
  void MvTimesMatAddMv ( const ScalarType alpha, const MultiVec<ScalarType>& A,
                         const Teuchos::SerialDenseMatrix<int,ScalarType>& B, const ScalarType beta ){
    KokkosMultiVec<ScalarType, Device> *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(A));
    if( myView.extent(1) == 1 && A_vec->myView.extent(1) == 1){
      // Then B is a scalar...
      ScalarType scal1 = alpha*B(0,0);
      ViewVectorType mysub = Kokkos::subview(myView, Kokkos::ALL, 0);
      ViewVectorType Asub = Kokkos::subview(A_vec->myView, Kokkos::ALL, 0);
      KokkosBlas::axpby(scal1, Asub, beta, mysub); 
    }
    else{
      UMHostConstViewMatrixType mat_h(B.values(), A_vec->myView.extent(1), myView.extent(1));
      ViewMatrixType mat_d(Kokkos::ViewAllocateWithoutInitializing("mat"), A_vec->myView.extent(1), myView.extent(1));
      Kokkos::deep_copy(mat_d, mat_h);
      if( myView.extent(1) == 1 ){ // Only 1 col
          ConstViewVectorType Bsub = Kokkos::subview(mat_d, Kokkos::ALL, 0);
          ViewVectorType mysub = Kokkos::subview(myView, Kokkos::ALL, 0);
          KokkosBlas::gemv("N", alpha, A_vec->myView, Bsub, beta, mysub);
      }
      else{
        KokkosBlas::gemm("N", "N", alpha, A_vec->myView, mat_d, beta, myView);
      }
    }
  }

  //Version not used by belos that has multivecs everywhere.
  //! *this <- alpha * A * B + beta * (*this)
  void MvTimesMatAddMv ( const ScalarType alpha, const MultiVec<ScalarType>& A,
                         const MultiVec<ScalarType>& B, const ScalarType beta ){
    KokkosMultiVec<ScalarType, Device> *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(A));
    KokkosMultiVec<ScalarType, Device> *B_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(B));

    if( myView.extent(1) == 1 ){ // Only 1 col
      ConstViewVectorType Bsub = Kokkos::subview(B_vec->myView, Kokkos::ALL, 0);
      ViewVectorType mysub = Kokkos::subview(myView, Kokkos::ALL, 0);
      KokkosBlas::gemv("N", alpha, A_vec->myView, Bsub, beta, mysub);
    }
    else{
      KokkosBlas::gemm("N", "N", alpha, A_vec->myView, B_vec->myView, beta, myView);
    }
  }

  //! *this <- alpha * A + beta * B
  void MvAddMv ( const ScalarType alpha, const MultiVec<ScalarType>& A, const ScalarType beta,
                 const MultiVec<ScalarType>& B){
    KokkosMultiVec<ScalarType, Device> *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(A));
    KokkosMultiVec<ScalarType, Device> *B_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(B));

    KokkosBlas::update(alpha, A_vec->myView, beta, B_vec->myView, (ScalarType) 0.0, myView);
  }

  //! Scale each element of the vectors in \c *this with \c alpha.
  void MvScale ( const ScalarType alpha ) {
    KokkosBlas::scal(myView, alpha, myView); 
  }

  //! Scale each element of the \c i-th vector in \c *this with \c alpha[i].
  void MvScale ( const std::vector<ScalarType>& alpha ){
    UMHostConstViewVectorType scalars_h(alpha.data(), alpha.size());
    ViewVectorType scalars_d(Kokkos::ViewAllocateWithoutInitializing("scalars_d"), alpha.size());
    Kokkos::deep_copy(scalars_d, scalars_h);

    KokkosBlas::scal(myView, scalars_d, myView); 
  }

  //! B <- alpha * A^T * (*this)
  void MvTransMv ( const ScalarType alpha, const MultiVec<ScalarType>& A, Teuchos::SerialDenseMatrix<int,ScalarType>& B ) const{
    KokkosMultiVec<ScalarType, Device> *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(A));
    if(A_vec->myView.extent(1) == 1 && myView.extent(1) == 1){
      ViewVectorType Asub = Kokkos::subview(A_vec->myView, Kokkos::ALL, 0);
      ViewVectorType mysub = Kokkos::subview(myView, Kokkos::ALL, 0);
      ScalarType soln = KokkosBlas::dot(Asub, mysub);
      soln = alpha*soln;
      B(0,0) = soln;
    }
   // else if( myView.extent(1) == 1 ){ // Only 1 col in soln vec
   //   ViewVectorType soln(Kokkos::ViewAllocateWithoutInitializing("soln"), A_vec->myView.extent(1));
   //   ViewVectorType mysub = Kokkos::subview(myView, Kokkos::ALL, 0);
   //   KokkosBlas::gemv("C", alpha, A_vec->myView, mysub, ScalarType(0.0), soln);
   //   for( unsigned int i = 0; i < soln.extent(0); i++){
   //     B(i,0) = soln(i);
   //   }
   // }
    else{
    //TODO denote internal data by _
      UMHostViewMatrixType soln_h(B.values(), A_vec->myView.extent(1), myView.extent(1));
      //Kokkos::Timer timeAlloc;
      ViewMatrixType soln_d(Kokkos::ViewAllocateWithoutInitializing("mat"), A_vec->myView.extent(1), myView.extent(1)); //TODO don't allocate each iteration!  
      //Kokkos::fence();
      //double time1 = timeAlloc.seconds();
      //std::cout << "Seconds for alloc: " << time1 << std::endl;        
      KokkosBlas::gemm("C", "N", alpha, A_vec->myView, myView, ScalarType(0.0), soln_d);
      //std::cout << "A has size " << A_vec->myView.extent(0) << " by " << A_vec->myView.extent(1) << " to be transposed." << std::endl; 
      //std::cout << "this has size " << myView.extent(0) << " by " << myView.extent(1) << std::endl;
      Kokkos::deep_copy(soln_h, soln_d);
    }
  }


  //! b[i] = A[i]^T * this[i]
  void MvDot ( const MultiVec<ScalarType>& A, std::vector<ScalarType>& b ) const{
    UMHostViewVectorType dotView_h(b.data(),myView.extent(1)); 
    ViewVectorType dotView_d(Kokkos::ViewAllocateWithoutInitializing("Dot"),myView.extent(1));
    KokkosMultiVec<ScalarType, Device> *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(A));
    KokkosBlas::dot(dotView_d, A_vec->myView, myView); 
    Kokkos::deep_copy(dotView_h, dotView_d);
  }

  //! alpha[i] = norm of i-th column of (*this)
  void MvNorm ( std::vector<ScalarType>& normvec, NormType norm_type = TwoNorm ) const{
    UMHostViewVectorType normView_h(normvec.data() ,myView.extent(1));
    ViewVectorType normView_d(Kokkos::ViewAllocateWithoutInitializing("Norm"),myView.extent(1));
    switch( norm_type ) { 
      case ( OneNorm ) : 
        KokkosBlas::nrm1(normView_d, myView);
        break;
      case ( TwoNorm ) : 
        KokkosBlas::nrm2(normView_d, myView);
        break;
      case ( InfNorm ) : 
        KokkosBlas::nrminf(normView_d, myView);
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
            "Belos::KokkosMultiVec::MvNorm: Invalid norm_type "
            << norm_type << ".  The current list of valid norm "
            "types is {OneNorm, TwoNorm, InfNorm}.");
    }   
    Kokkos::deep_copy(normView_h, normView_d);
  }

  //! Fill all columns of *this with random values.
  void MvRandom() {
    int rand_seed = std::rand();
    Kokkos::Random_XorShift64_Pool<> pool(rand_seed); 
    Kokkos::fill_random(myView, pool, -1,1);
  }

  //! Initialize each element of (*this) to the scalar value alpha.
  void MvInit ( const ScalarType alpha ) {
     Kokkos::deep_copy(myView,alpha);
  }

  //! Print (*this) to the given output stream.//TODO use printf so works on device?? 
  void MvPrint( std::ostream& os ) const {
    for(unsigned int i = 0; i < (myView.extent(0)); i++){
      for (unsigned int j = 0; j < (myView.extent(1)); j++){
        os << myView(i , j) << "  ";
      }
      os << std::endl;
    } 
  os << std::endl;
  }

//private: 
//This var should be private, but I can't get friend class templaty stuff to work, so...
  ViewMatrixType myView;
private:

};

///
/// \class KokkosOperator
/// \brief Implementation of Belos::Operator using KokkosSparse::CrsMatrix.
///
template<class ScalarType, class OrdinalType=int, class Device=Kokkos::DefaultExecutionSpace>
class KokkosOperator : public Operator<ScalarType> {
private:
  KokkosSparse::CrsMatrix<ScalarType, OrdinalType, Device> myMatrix;

public:
// Shallow copy for mat of same scalar type:
  KokkosOperator<ScalarType, OrdinalType, Device> (const KokkosSparse::CrsMatrix<ScalarType, OrdinalType, Device> mat) 
   : myMatrix(mat) {}

  void Apply (const MultiVec<ScalarType>& x,  MultiVec<ScalarType>& y,  ETrans trans=NOTRANS) const{
  //Note: Do NOT make x and y the same multivector!  You will get NaNs...
    char mode[] = "X";
    switch(trans){
      case NOTRANS:
        mode[0]='N';
        break;
      case TRANS:
        mode[0]='T';
        break;
      case CONJTRANS:
        mode[0]='C';
        break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
            "Belos::KokkosOperator::Apply: Invalid ETrans type ");
    }
    // Note: spmv computes y = beta*y + alpha*Op(A)*x  spmv(mode,alpha,A,x,beta,y);
    ScalarType alpha = 1.0;
    ScalarType beta = 0;
    KokkosMultiVec<ScalarType, Device> *x_vec = 
            dynamic_cast<KokkosMultiVec<ScalarType, Device> *>(&const_cast<MultiVec<ScalarType> &>(x));
    KokkosMultiVec<ScalarType, Device> *y_vec = dynamic_cast<KokkosMultiVec<ScalarType, Device> *>(&y);

   //  using RANK_SPECIALISE =
   //   typename std::conditional<static_cast<int> (x_vec->myView.rank) == 2, KokkosSparse::RANK_TWO, KokkosSparse::RANK_ONE>::type;
   // KokkosSparse::spmv(mode, alpha, myMatrix, x_vec->myView(), beta, y_vec->myView(), RANK_SPECIALISE());
    KokkosSparse::spmv(mode, alpha, myMatrix, x_vec->myView, beta, y_vec->myView);
  }

  bool HasApplyTranspose () const {
    return true;
  }

  ~KokkosOperator<ScalarType, OrdinalType, Device>(){}

};
}// end namespace Belos
#endif
