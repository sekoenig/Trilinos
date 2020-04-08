#include<Kokkos_Core.hpp>
#include<KokkosBlas.hpp>
#include<BelosTypes.hpp>
#include<Kokkos_Random.hpp>
#include<BelosMultiVec.hpp>

#include <Teuchos_SerialDenseMatrix.hpp>
//#include<KokkosSparse_spmv.hpp>
//
//
 namespace Belos {

  /// \class KokkosMultiVec
  /// \brief Implementation of Belos::MultiVec using Kokkos_MultiVector.
  ///
  /// Belos::MultiVec offers a simple abstract interface for
  /// multivector operations in Belos solver algorithms.  This class
  /// implements Belos::MultiVec by extending Kokkos_MultiVector.
  template<class ScalarType >
  class KokkosMultiVec : public MultiVec<ScalarType>, public Kokkos::View<ScalarType**> {
  public:
    // constructors
//    KokkosMultiVec(const Kokkos_BlockMap& Map_in, double * array, const int numvecs, const int stride=0);
//    KokkosMultiVec(const Kokkos_BlockMap& Map_in, const int numvecs, bool zeroOut=true);
//    KokkosMultiVec(Kokkos_DataAccess CV_in, const Kokkos_MultiVector& P_vec, const std::vector<int>& index);
    KokkosMultiVec<ScalarType> (const std::string label, const int numrows, const int numvecs) : Kokkos::View<ScalarType**>(label,numrows,numvecs){}
    KokkosMultiVec<ScalarType> (const int numrows, const int numvecs) : Kokkos::View<ScalarType**>("MV",numrows,numvecs){}
    KokkosMultiVec<ScalarType> (const int numrows) : Kokkos::View<ScalarType**>("MV",numrows,1){}
    //KokkosMultiVec& operator=(const KokkosMultiVec& pv) { Kokkos_MultiVector::operator=(pv); return *this; }
    KokkosMultiVec<ScalarType> (const Kokkos::View<ScalarType**> & sourceView) : Kokkos::View<ScalarType**>(sourceView){} 
    //If we've already made a view and want it to be a multivec... is this the right way to do it?? TODO
    ~KokkosMultiVec();

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
      KokkosMultiVec<ScalarType> * ptr = new KokkosMultiVec<ScalarType>(this->extent(0),numvecs);
      return ptr;
    }

    /// A virtual "copy constructor" returning a pointer to a new
    /// object of the pure virtual class.  This vector's entries are
    /// copied and a new stand-alone multivector is created.  (deep
    /// copy).
    MultiVec<ScalarType> * CloneCopy () const{
      //KokkosMultiVec<ScalarType>  temp("MV",this->extent(0),this->extent(1));
      KokkosMultiVec<ScalarType> * ptr = new KokkosMultiVec<ScalarType>(this->extent(0),this->extent(1));
      Kokkos::deep_copy(*ptr,*this);
      return ptr;
    }

    /// A virtual "copy constructor" returning a pointer to the pure
    /// virtual class.  This vector's entries are copied and a new
    /// stand-alone MultiVector is created where only selected columns
    /// are chosen.  (deep copy).
    MultiVec<ScalarType> * CloneCopy ( const std::vector<int>& index ) const{
      // Be careful with indexing- need to add 1 to last index value b/c Belos includes value at last index while Kokkos doesn't.
      // TODO might need to check that index bounds are valid. 
      int numvecs = index.size() + 1;
      KokkosMultiVec<ScalarType> * B = new KokkosMultiVec<ScalarType>("B",this->extent(0),numvecs);
      //KokkosMultiVec<ScalarType> B("B", this->extent(0), numvecs);
      bool isAscending = true;
      for(int i=0; i< (index.size()+1); i++){
        if( index[i+1] <= index[i] ){
          isAscending = false;
        }
      }
      if(isAscending && index.size()==this->GetNumberVecs()){ //Copy entire multivec.
        Kokkos::deep_copy(*B,*this);
      }
      else if (isAscending){ //Copy contiguous subset
        Kokkos::View<ScalarType**> ThisSub = Kokkos::subview(*this, Kokkos::ALL, std::make_pair(index.front(), index.back()+1));
        Kokkos::deep_copy(*B,ThisSub);
      } 
      else{ //Copy columns one by one
        for(int i=0; i<index.size(); i++){
          Kokkos::View<ScalarType**> Bsub = Kokkos::subview(*B, Kokkos::ALL, std::make_pair(i, i+1));
          Kokkos::View<ScalarType**> ThisSub = Kokkos::subview(*this, Kokkos::ALL, std::make_pair(index[i], index[i]+1));
          Kokkos::deep_copy(Bsub, ThisSub);
        }
      }
      return B; 
    }

    /// A virtual view "constructor" returning a pointer to the pure
    /// virtual class.  This vector's entries are shared and hence no
    /// memory is allocated for the columns.
    MultiVec<ScalarType> * CloneViewNonConst ( const std::vector<int>& index ){
      KokkosMultiVec<ScalarType> * B = new KokkosMultiVec<ScalarType>(Kokkos::subview(*this, Kokkos::ALL, std::make_pair(index.at(0), index.at(1)+1)));
      return B; 
    }

    /// A virtual view constructor returning a pointer to the pure
    /// virtual class.  This vector's entries are shared and hence no
    /// memory is allocated for the columns.
    const MultiVec<ScalarType> * CloneView ( const std::vector<int>& index ) const;

    /// Set a subblock of the multivector, which need not be
    /// contiguous, and is given by the indices.
    void SetBlock ( const MultiVec<ScalarType>& A, const std::vector<int>& index ){
      //TODO check bounds of index?? 
      KokkosMultiVec<ScalarType> *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<double> &>(A));
      bool isAscending = true;
      for(int i=0; i< (index.size()+1); i++){
        if( index[i+1] <= index[i] ){
          isAscending = false;
        }
      }
      if(isAscending && index.size()==this->GetNumberVecs()){ //Copy entire multivec.
        Kokkos::deep_copy(*this,*A_vec);
      }
      else if (isAscending){ //Copy contiguous subset
        Kokkos::View<ScalarType**> Asub = Kokkos::subview(*A_vec, Kokkos::ALL, std::make_pair(index.front(), index.back()+1));
        Kokkos::View<ScalarType**> ThisSub = Kokkos::subview(*this, Kokkos::ALL, std::make_pair(index.front(), index.back()+1));
        Kokkos::deep_copy(ThisSub, Asub);
      } 
      else{ //Copy columns one by one
        for(int i=0; i<index.size(); i++){
          Kokkos::View<ScalarType**> Asub = Kokkos::subview(*A_vec, Kokkos::ALL, std::make_pair(i, i+1));
          Kokkos::View<ScalarType**> ThisSub = Kokkos::subview(*this, Kokkos::ALL, std::make_pair(index[i], index[i]+1));
          Kokkos::deep_copy(ThisSub, Asub);
        }
      }
    }

    //! The number of rows in the multivector.
    ptrdiff_t GetGlobalLength () const {
      return static_cast<ptrdiff_t>(this->extent(0));
    }

    //! The number of columns in the multivector.
    int GetNumberVecs () const { return this->extent(1); }

    //! *this <- alpha * A * B + beta * (*this)
    void MvTimesMatAddMv ( const ScalarType alpha, const MultiVec<ScalarType>& A,
                           const Teuchos::SerialDenseMatrix<int,ScalarType>& B, const ScalarType beta ){
      KokkosMultiVec<ScalarType> *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<double> &>(A));
      Kokkos::View<ScalarType**> mat("mat", A_vec->extent(1), this->extent(1));
      Teuchos2KokkosMat(B,mat);
      KokkosBlas::gemm("N", "N", alpha, *A_vec, mat, beta, *this);
    }

    //! *this <- alpha * A + beta * B
    void MvAddMv ( const ScalarType alpha, const MultiVec<ScalarType>& A, const ScalarType beta,
                   const MultiVec<ScalarType>& B){
      KokkosMultiVec<ScalarType> *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<double> &>(A));
      KokkosMultiVec<ScalarType> *B_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<double> &>(B));
      Kokkos::deep_copy(*this, *B_vec);
      KokkosBlas::axpby(alpha, *A_vec, beta, *this);
    }

    //! Scale each element of the vectors in \c *this with \c alpha.
    void MvScale ( const ScalarType alpha ) {
      //Later- Can we do this better with less copying?  TODO
      //KokkosMultiVec<ScalarType> * temp = Clone(this->extent(1));
      //KokkosMultiVec<ScalarType> * ptr = new KokkosMultiVec<ScalarType>(this->extent(0),numvecs);
      KokkosMultiVec<ScalarType> temp(this->extent(0),this->extent(1));
      KokkosBlas::scal(temp, alpha, *this); 
      Kokkos::deep_copy(*this, temp);
    }

    //! Scale each element of the \c i-th vector in \c *this with \c alpha[i].
    void MvScale ( const std::vector<ScalarType>& alpha ){
      //Later- Can we do this better with less copying?  TODO
      //KokkosMultiVec<ScalarType> * temp = Clone(this->extent(1));
      KokkosMultiVec<ScalarType> temp(this->extent(0),this->extent(1));
      Kokkos::View<ScalarType*> scalars("alpha", alpha.size());
      for( int i = 0 ; i < alpha.size(); i++){
        scalars(i) = alpha.at(i);
      } 
      KokkosBlas::scal(temp, scalars, *this); 
      Kokkos::deep_copy(*this, temp);
    }

    //! B <- alpha * A^T * (*this)
    void MvTransMv ( const ScalarType alpha, const MultiVec<ScalarType>& A, Teuchos::SerialDenseMatrix<int,ScalarType>& B ) const{
      KokkosMultiVec<ScalarType> *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<double> &>(A));
      Kokkos::View<ScalarType**> soln("soln", A_vec->extent(1), this->extent(1));
      KokkosBlas::gemm("C", "N", alpha, *A_vec, *this, ScalarType(0.0), soln);
      Kokkos2TeuchosMat(soln, B);
    }


    //! b[i] = A[i]^T * this[i]
    void MvDot ( const MultiVec<ScalarType>& A, std::vector<ScalarType>& b ) const{
      Kokkos::View<ScalarType*> dotView("Dot",this->extent(1));
      KokkosMultiVec<ScalarType> *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<double> &>(A));
      KokkosBlas::dot(dotView, *A_vec, *this); //TODO check- it should be A that is conjugate transposed, not mv.  Is it??
      for(int i=0; i < this->extent(1); i++){
        b.push_back(dotView(i)); //Is there a better way to do this?
        //TODO: will probably have to mirror the normView to the host space. 
      }
    }

    //! alpha[i] = norm of i-th column of (*this)
    void MvNorm ( std::vector<ScalarType>& normvec, NormType norm_type = TwoNorm ) const{
      Kokkos::View<ScalarType*> normView("Norm",this->extent(1));
      switch( norm_type ) { 
        case ( OneNorm ) : 
          KokkosBlas::nrm1(normView, *this);
          break;
        case ( TwoNorm ) : 
          KokkosBlas::nrm2(normView, *this);
          break;
        case ( InfNorm ) : 
          KokkosBlas::nrminf(normView, *this);
          break;
        default:
          TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
              "Belos::KokkosMultiVec::MvNorm: Invalid norm_type "
              << norm_type << ".  The current list of valid norm "
              "types is {OneNorm, TwoNorm, InfNorm}.");
      }   
      for(int i=0; i < this->extent(1); i++){
        normvec.push_back(normView(i)); //Is there a better way to do this?
        //TODO: will probably have to mirror the normView to the host space. 
      }
    }

    //! Fill all columns of *this with random values.
    void MvRandom() {
      Kokkos::Random_XorShift64_Pool<> pool(12371);
      Kokkos::fill_random(*this, pool, -1,1);
    }

    //! Initialize each element of (*this) to the scalar value alpha.
    void MvInit ( const ScalarType alpha ) {
       Kokkos::deep_copy(*this,alpha);
    }

    //! Print (*this) to the given output stream.
    void MvPrint( std::ostream& os ) const {
      for(int i = 0; i < (this->extent(0)); i++){
        for (int j = 0; j < (this->extent(1)); j++){
          os << (*this)(i , j) << "  ";
        }
        os << std::endl;
      } 
    os << std::endl;
    }

private:
    void Kokkos2TeuchosMat(const Kokkos::View<const ScalarType**> & K,  Teuchos::SerialDenseMatrix<int, ScalarType> &T) const {
      TEUCHOS_TEST_FOR_EXCEPTION(K.extent(0) != T.numRows() || K.extent(1) != T.numCols(), std::runtime_error, "Error: Matrix dimensions do not match!");
  //This is all on host, so there's no use trying to use parallel_for, right?... Well, host could have openMP... TODO improve this?
      for(int i = 0; i < K.extent(0); i++){
        for (int j = 0; j < K.extent(1); j++){
          T(i,j) = K(i,j);
        }
      } 
    }

    void Teuchos2KokkosMat(const Teuchos::SerialDenseMatrix<int, ScalarType> &T, Kokkos::View<ScalarType**> & K) const {
      TEUCHOS_TEST_FOR_EXCEPTION(K.extent(0) != T.numRows() || K.extent(1) != T.numCols(), std::runtime_error, "Error: Matrix dimensions do not match!");
      //This is all on host, so there's no use trying to use parallel_for, right?... Well, host could have openMP... TODO improve this?
      for(int i = 0; i < K.extent(0); i++){
        for (int j = 0; j < K.extent(1); j++){
          K(i,j) = T(i,j);
        }
      } 
    }

  };
}// end namespace Belos

//using MV = Kokkos::View<ScalarType**>;
//Creation Methods



