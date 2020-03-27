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
//    KokkosMultiVec& operator=(const KokkosMultiVec& pv) { Kokkos_MultiVector::operator=(pv); return *this; }
//    KokkosMultiVec(const Kokkos_MultiVector & P_vec);
    KokkosMultiVec();
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
    MultiVec<double> * Clone ( const int numvecs ) const;

    /// A virtual "copy constructor" returning a pointer to a new
    /// object of the pure virtual class.  This vector's entries are
    /// copied and a new stand-alone multivector is created.  (deep
    /// copy).
    MultiVec<double> * CloneCopy () const;

    /// A virtual "copy constructor" returning a pointer to the pure
    /// virtual class.  This vector's entries are copied and a new
    /// stand-alone MultiVector is created where only selected columns
    /// are chosen.  (deep copy).
    MultiVec<double> * CloneCopy ( const std::vector<int>& index ) const;

    /// A virtual view "constructor" returning a pointer to the pure
    /// virtual class.  This vector's entries are shared and hence no
    /// memory is allocated for the columns.
    MultiVec<double> * CloneViewNonConst ( const std::vector<int>& index );

    /// A virtual view constructor returning a pointer to the pure
    /// virtual class.  This vector's entries are shared and hence no
    /// memory is allocated for the columns.
    const MultiVec<double> * CloneView ( const std::vector<int>& index ) const;

    /// Set a subblock of the multivector, which need not be
    /// contiguous, and is given by the indices.
    void SetBlock ( const MultiVec<double>& A, const std::vector<int>& index );

    //! The number of rows in the multivector.
    ptrdiff_t GetGlobalLength () const
    {
//#ifndef EPETRA_NO_64BIT_GLOBAL_INDICES
//       if ( Map().GlobalIndicesLongLong() )
//          return static_cast<ptrdiff_t>( GlobalLength64() );
//       else
//          return static_cast<ptrdiff_t>( GlobalLength() );
//#else
//          return static_cast<ptrdiff_t>( GlobalLength() );
//#endif
      return 0;
    }

    //! The number of columns in the multivector.
    int GetNumberVecs () const { return 0; }
//return NumVectors(); }

    //! *this <- alpha * A * B + beta * (*this)
    void MvTimesMatAddMv ( const double alpha, const MultiVec<double>& A,
                           const Teuchos::SerialDenseMatrix<int,double>& B, const double beta );
    //! *this <- alpha * A + beta * B
    void MvAddMv ( const double alpha, const MultiVec<double>& A, const double beta,
                   const MultiVec<double>& B);

    //! Scale each element of the vectors in \c *this with \c alpha.
    void MvScale ( const double alpha ) {
//      TEUCHOS_TEST_FOR_EXCEPTION( this->Scale( alpha )!=0, KokkosMultiVecFailure,
//                          "Belos::KokkosMultiVec::MvScale() call to Scale() returned a nonzero value."); }
                          }

    //! Scale each element of the \c i-th vector in \c *this with \c alpha[i].
    void MvScale ( const std::vector<double>& alpha );

    //! B <- alpha * A^T * (*this)
    void MvTransMv ( const double alpha, const MultiVec<double>& A, Teuchos::SerialDenseMatrix<int,double>& B ) const;

    //! b[i] = A[i]^T * this[i]
    void MvDot ( const MultiVec<double>& A, std::vector<double>& b ) const;

    //! alpha[i] = norm of i-th column of (*this)
    void MvNorm ( std::vector<double>& normvec, NormType norm_type = TwoNorm ) const;

    //! Fill all columns of *this with random values.
    void MvRandom() {
      //TEUCHOS_TEST_FOR_EXCEPTION( Random()!=0, KokkosMultiVecFailure,
                          //"Belos::KokkosMultiVec::MvRandom() call to Random() returned a nonzero value."); }
                          }

    //! Initialize each element of (*this) to the scalar value alpha.
    void MvInit ( const double alpha ) {
      //TEUCHOS_TEST_FOR_EXCEPTION( PutScalar(alpha)!=0, KokkosMultiVecFailure,
                          //"Belos::KokkosMultiVec::MvInit() call to PutScalar() returned a nonzero value."); }
                          }

    //! Print (*this) to the given output stream.
    void MvPrint( std::ostream& os ) const { os << *this << std::endl; };
  private:
  };
}// end namespace Belos
