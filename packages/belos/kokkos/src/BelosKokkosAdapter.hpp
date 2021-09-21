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
  using UMHostViewVectorType = Kokkos::View<ScalarType*,Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using UMHostConstViewVectorType = Kokkos::View<const ScalarType*,Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using UMHostViewMatrixType = Kokkos::View<ScalarType**,Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
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

  //! Copy the vectors in A to the vectors in (*this) specified by index.
  ///
  /// Sets a subblock of vectors of this multivector, 
  /// which need not be contiguous, and is given by the indices.
  /// Result is (*this)[:,index[i]] = A[:,i].
  ///
  void SetBlock ( const MultiVec<ScalarType>& A, const std::vector<int>& index ){
    KokkosMultiVec<ScalarType, Device> *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(A));
    
    if( index.size() > myView.extent(1) ){
      throw std::runtime_error("Error in KokkosMultiVec::SetBlock. A cannot have more vectors than (*this).");
    }
    //Check if the indices are contiguous and increasing by 1
    bool isAscending = true;
    for(unsigned int i=0; i< (index.size()-1); i++){
      if( index[i+1] != index[i]+1 ){
        isAscending = false;
      }
    }

    //Perform deep copy of sub block:
    if(isAscending && index.size()==(unsigned)this->GetNumberVecs()){ //Copy entire multivec.
      Kokkos::deep_copy(myView,A_vec->myView);
    }
    else if (isAscending){ //Copy contiguous subset
      ViewMatrixType Asub = Kokkos::subview(A_vec->myView, Kokkos::ALL, std::make_pair(0,(int)index.size()));
      ViewMatrixType ThisSub = Kokkos::subview(myView, Kokkos::ALL, std::make_pair(index.front(), index.back()+1));
      Kokkos::deep_copy(ThisSub, Asub);
    } 
    else{ //Copy columns one by one
      for(unsigned int i=0; i<index.size(); i++){
        ViewVectorType Asub = Kokkos::subview(A_vec->myView, Kokkos::ALL, i);
        ViewVectorType ThisSub = Kokkos::subview(myView, Kokkos::ALL, index[i]);
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
    if( myView.extent(1) == 1 && A_vec->myView.extent(1) == 1){ //B is a scalar.
      ScalarType scal1 = alpha*B(0,0);
      ViewVectorType mysub = Kokkos::subview(myView, Kokkos::ALL, 0);
      ViewVectorType Asub = Kokkos::subview(A_vec->myView, Kokkos::ALL, 0);
      KokkosBlas::axpby(scal1, Asub, beta, mysub); 
    }
    else{
      UMHostConstViewMatrixType mat_h(B.values(), A_vec->myView.extent(1), myView.extent(1));
      ViewMatrixType mat_d(Kokkos::ViewAllocateWithoutInitializing("mat"), A_vec->myView.extent(1), myView.extent(1));
      Kokkos::deep_copy(mat_d, mat_h);
      if( myView.extent(1) == 1 ){ // B has only 1 col
          ConstViewVectorType Bsub = Kokkos::subview(mat_d, Kokkos::ALL, 0);
          ViewVectorType mysub = Kokkos::subview(myView, Kokkos::ALL, 0);
          KokkosBlas::gemv("N", alpha, A_vec->myView, Bsub, beta, mysub);
      }
      else{
        KokkosBlas::gemm("N", "N", alpha, A_vec->myView, mat_d, beta, myView);
      }
    }
  }

  //! *this <- alpha * A + beta * B
  void MvAddMv ( const ScalarType alpha, const MultiVec<ScalarType>& A, const ScalarType beta,
                 const MultiVec<ScalarType>& B){
    KokkosMultiVec<ScalarType, Device> *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(A));
    KokkosMultiVec<ScalarType, Device> *B_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(B));

    KokkosBlas::update(alpha, A_vec->myView, beta, B_vec->myView, (ScalarType) 0.0, myView);
  }

  //! Scale (multiply) each element of the vectors in \c *this with \c alpha.
  void MvScale ( const ScalarType alpha ) {
    KokkosBlas::scal(myView, alpha, myView); 
  }

  //! Scale (multiply) each element of the \c i-th vector in \c *this with \c alpha[i].
  void MvScale ( const std::vector<ScalarType>& alpha ){

    //Move scalar values to a Kokkos View:
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
   // ***
   // For MvTransMv, this option runs slower than GEMM on NVIDIA V100.  
   // Do not enable for now. 
   // ****
   // else if( myView.extent(1) == 1 ){ // Only 1 col in soln vec
   //   ViewVectorType soln(Kokkos::ViewAllocateWithoutInitializing("soln"), A_vec->myView.extent(1));
   //   ViewVectorType mysub = Kokkos::subview(myView, Kokkos::ALL, 0);
   //   KokkosBlas::gemv("C", alpha, A_vec->myView, mysub, ScalarType(0.0), soln);
   //   for( unsigned int i = 0; i < soln.extent(0); i++){
   //     B(i,0) = soln(i);
   //   }
   // }
    else{
      UMHostViewMatrixType soln_h(B.values(), A_vec->myView.extent(1), myView.extent(1));
      ViewMatrixType soln_d(Kokkos::ViewAllocateWithoutInitializing("mat"), A_vec->myView.extent(1), myView.extent(1)); 
      KokkosBlas::gemm("C", "N", alpha, A_vec->myView, myView, ScalarType(0.0), soln_d);
      Kokkos::deep_copy(soln_h, soln_d);
    }
  }


  //! b[i] = A[i]^T * this[i]
  ///
  /// Performs a dot product between A and (*this).
  /// Uses conjugate transpose when appropriate. 
  void MvDot ( const MultiVec<ScalarType>& A, std::vector<ScalarType>& b ) const{
    //Put output vector in unmanaged Kokkos view:
    UMHostViewVectorType dotView_h(b.data(),myView.extent(1)); 
    ViewVectorType dotView_d(Kokkos::ViewAllocateWithoutInitializing("Dot"),myView.extent(1));

    KokkosMultiVec<ScalarType, Device> *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(A));

    KokkosBlas::dot(dotView_d, A_vec->myView, myView); 
    Kokkos::deep_copy(dotView_h, dotView_d);
  }

  //! alpha[i] = norm of i-th column of (*this)
  ///
  /// Valid norm types are Belos::TwoNorm, Belos::OneNorm,
  /// and Belos::InfNorm.
  void MvNorm ( std::vector<ScalarType>& normvec, NormType norm_type = TwoNorm ) const{

    //Put output vector in unmanaged Kokkos view:
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

  //! Print (*this) to the given output stream.
  ///
  /// (This function will first copy the multivector to host space 
  /// if needed.)
  void MvPrint( std::ostream& os ) const {
    typename ViewMatrixType::HostMirror hostView("myViewMirror", myView.extent(0), myView.extent(1));
    Kokkos::deep_copy(hostView, myView);
    for(unsigned int i = 0; i < (hostView.extent(0)); i++){
      for (unsigned int j = 0; j < (hostView.extent(1)); j++){
        os << hostView(i , j) << "  ";
      }
      os << std::endl;
    } 
    os << std::endl;
  }

//private:  //TODO fix this??
//This var should be private, but I can't get friend class templaty stuff to work, so...
  ViewMatrixType myView;
private:

};

///
/// \class KokkosOperator
/// \brief Implementation of Belos::Operator using KokkosSparse::CrsMatrix.
///
//TODO: Should this name change to reflect that it isn't generic?  e.g. KokkosCRSOperator??
template<class ScalarType, class OrdinalType=int, class Device=Kokkos::DefaultExecutionSpace>
class KokkosOperator : public Operator<ScalarType> {

private:
  // Shallow copy of the CrsMatrix used for SpMV. 
  KokkosSparse::CrsMatrix<ScalarType, OrdinalType, Device> myMatrix;

public:
  //! @name Constructor/Destructor
  //@{ 
  //! Constructor obtains a shallow copy of the given CrsMatrix.
  KokkosOperator<ScalarType, OrdinalType, Device> (const KokkosSparse::CrsMatrix<ScalarType, OrdinalType, Device> mat) 
   : myMatrix(mat) {}

  //! Destructor.
  ~KokkosOperator<ScalarType, OrdinalType, Device>(){}
  //@}

  //! @name Methods relating to applying the operator
  //@{ 

  /// \brief Apply the operator to x, putting the result in y.
  ///
  /// Take the KokkosMultiVec \c x and apply the operator (or its
  /// transpose or Hermitian transpose) to it, writing the result
  /// into the KokkosMultiVec \c y. (This performs a sparse matrix-
  /// vector product.)
  ///
  /// \param x [in] The input multivector.
  ///
  /// \param y [out] The output multivector.  x and y may not alias
  ///   (i.e., be views of) one another.
  ///
  /// \param trans [in] Whether to apply the operator (Belos::NOTRANS), its
  ///   transpose (Belos::TRANS), or its Hermitian transpose (Belos::CONJTRANS).
  ///   The default is Belos::NOTRANS. (Defined in BelosTypes.hpp.)
  ///
  void Apply (const MultiVec<ScalarType>& x,  MultiVec<ScalarType>& y,  ETrans trans=NOTRANS) const{

    // Determine transpose mode:
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

    //Use dynamic_cast to tell the compiler these are Kokkos Multivecs.
    KokkosMultiVec<ScalarType, Device> *x_vec = 
            dynamic_cast<KokkosMultiVec<ScalarType, Device> *>(&const_cast<MultiVec<ScalarType> &>(x));
    KokkosMultiVec<ScalarType, Device> *y_vec = dynamic_cast<KokkosMultiVec<ScalarType, Device> *>(&y);

    //KokkosSparse::spmv computes y = beta*y + alpha*Op(A)*x
    ScalarType alpha = 1.0;
    ScalarType beta = 0;
    KokkosSparse::spmv(mode, alpha, myMatrix, x_vec->myView, beta, y_vec->myView);
  }

  /// \brief Whether this operator implements applying the transpose.
  ///
  /// This function returns true since we can always apply the transpose
  /// of a Kokkos::CrsMatrix. 
  ///
  bool HasApplyTranspose () const {
    return true;
  }
  //@}
};
}// end namespace Belos
#endif
