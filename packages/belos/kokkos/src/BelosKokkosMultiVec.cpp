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


/*! \file BelosKokkosAdapter.cpp
    \brief Implementation of the interfaces between Belos virtual classes and Kokkos concrete classes.
*/

#include "BelosKokkosMultiVec.hpp"


namespace Belos {

  // An anonymous namespace restricts the scope of its definitions to
  // this file.
  //namespace {

    //! Return a string representation of the given transpose enum value.
    //std::string
    //etransToString (const ETrans trans)
    //{
    //  if (trans == NOTRANS) {
    //    return "NOTRANS";
    //  } else if (trans == TRANS) {
    //    return "TRANS";
    //  } else if (trans == CONJTRANS) {
    //    return "CONJTRANS";
    //  }
    //  TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
    //                     "Invalid ETrans value trans = " << trans << ".");
    //}

    /// \fn implementsApplyTranspose
    /// \brief Whether Op implements applying the transpose.
    ///
    /// Kokkos_Operator instances are not required to implement
    /// applying the transpose (or conjugate transpose, if
    /// applicable).  This function lets you ask.
    ///
    /// We need this function because Kokkos_Operator doesn't have a
    /// direct query method.  We query by asking it to set using the
    /// transpose, and examine the returned error code.  Kokkos
    /// operators have a persistent "use the transpose" state that you
    /// can set or unset using Kokkos_Operator::SetUseTranspose().
    ///
    /// \note This function is not thread-safe, because it makes no
    ///   effort to protect Op against simultaneous calls to its
    ///   Apply() or SetUseTranspose() methods.
    ///
    /// We use this function multiple times in this file, so it's
    /// worthwhile breaking it out into a separate function, rather
    /// than copying and pasting.
  //  bool
  //  implementsApplyTranspose (const Kokkos_Operator& Op)
  //  {
  //    return true; //Can't you always use apply Transpose with a Kokos crs matrix?  
  //                // So this is fine unless we interface to some other class.
  //  }

  //} // namespace (anonymous)


KokkosMultiVec::KokkosMultiVec (const Kokkos_BlockMap& Map_in,
                                double * array,
                                const int numvecs,
                                const int stride)
  : Kokkos_MultiVector(BELOSKOKKOSCOPY, Map_in, array, stride, numvecs)
{
}


KokkosMultiVec::KokkosMultiVec (const Kokkos_BlockMap& Map_in,
                                const int numvecs,
                                bool zeroOut)
  : Kokkos_MultiVector(Map_in, numvecs, zeroOut)
{
}


KokkosMultiVec::KokkosMultiVec (Kokkos_DataAccess CV_in,
                                const Kokkos_MultiVector& P_vec,
                                const std::vector<int>& index )
  : Kokkos_MultiVector(CV_in, P_vec, &(const_cast<std::vector<int> &>(index))[0], index.size())
{
}


KokkosMultiVec::KokkosMultiVec(const Kokkos_MultiVector& P_vec)
  : Kokkos_MultiVector(P_vec)
{
}


KokkosMultiVec::~KokkosMultiVec()
{
}
//
//  member functions inherited from Belos::MultiVec
//
//
//  Simulating a virtual copy constructor. If we could rely on the co-variance
//  of virtual functions, we could return a pointer to KokkosMultiVec
//  (the derived type) instead of a pointer to the pure virtual base class.
//

MultiVec<ScalarType>* KokkosMultiVec::Clone ( const int numvecs ) const
{
  KokkosMultiVec * ptr_apv = new KokkosMultiVec(Map(), numvecs, false);
  return ptr_apv; // safe upcast.
}
//
//  the following is a virtual copy constructor returning
//  a pointer to the pure virtual class. vector values are
//  copied.
//

MultiVec<ScalarType>* KokkosMultiVec::CloneCopy() const
{
  KokkosMultiVec *ptr_apv = new KokkosMultiVec(*this);
  return ptr_apv; // safe upcast
}


MultiVec<ScalarType>* KokkosMultiVec::CloneCopy ( const std::vector<int>& index ) const
{
  KokkosMultiVec * ptr_apv = new KokkosMultiVec(BELOSKOKKOSCOPY, *this, index);
  return ptr_apv; // safe upcast.
}


MultiVec<ScalarType>* KokkosMultiVec::CloneViewNonConst ( const std::vector<int>& index )
{
  KokkosMultiVec * ptr_apv = new KokkosMultiVec(BELOSKOKKOSVIEW, *this, index);
  return ptr_apv; // safe upcast.
}


const MultiVec<ScalarType>*
KokkosMultiVec::CloneView (const std::vector<int>& index) const
{
  KokkosMultiVec * ptr_apv = new KokkosMultiVec(BELOSKOKKOSVIEW, *this, index);
  return ptr_apv; // safe upcast.
}


void
KokkosMultiVec<ScalarType>::SetBlock (const MultiVec<ScalarType>& A,
                          const std::vector<int>& index)
{
  KokkosMultiVec<ScalarType> temp_vec(BELOSKOKKOSVIEW, *this, index);

  int numvecs = index.size();
  if ( A.GetNumberVecs() != numvecs ) {
    std::vector<int> index2( numvecs );
    for(int i=0; i<numvecs; i++)
      index2[i] = i;
    KokkosMultiVec<ScalarType> *tmp_vec =
      dynamic_cast<KokkosMultiVec<ScalarType> *>(&const_cast<MultiVec<ScalarType> &>(A));
    TEUCHOS_TEST_FOR_EXCEPTION(tmp_vec==NULL, KokkosMultiVecFailure,
                       "Belos::KokkosMultiVec::SetBlock: Dynamic cast from "
                       "Belos::MultiVec<ScalarType> to Belos::KokkosMultiVec "
                       "failed.");
    KokkosMultiVec<ScalarType> A_vec(BELOSKOKKOSVIEW, *tmp_vec, index2);
    temp_vec.MvAddMv( 1.0, A_vec, 0.0, A_vec );
  }
  else {
    temp_vec.MvAddMv( 1.0, A, 0.0, A );
  }
}

//-------------------------------------------------------------
//
// *this <- alpha * A * B + beta * (*this)
//
//-------------------------------------------------------------

void KokkosMultiVec<ScalarType>::MvTimesMatAddMv ( const ScalarType alpha, const MultiVec<ScalarType>& A,
                                       const Teuchos::SerialDenseMatrix<int,ScalarType>& B, const ScalarType beta )
{
  KokkosMultiVec *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(A));
  TEUCHOS_TEST_FOR_EXCEPTION(A_vec==NULL, KokkosMultiVecFailure,
                     "Belos::KokkosMultiVec::MvTimesMatAddMv cast from Belos::MultiVec<> to Belos::KokkosMultiVec failed.");

  int info = Multiply( 'N', 'N', alpha, *A_vec, B_Pvec, beta );
  TEUCHOS_TEST_FOR_EXCEPTION(info!=0, KokkosMultiVecFailure,
                     "Belos::KokkosMultiVec::MvTimesMatAddMv call to Multiply() returned a nonzero value.");

}
void MvTimesMatAddMv( const ScalarType alpha, const MV &A, const Teuchos::SerialDenseMatrix<int, ScalarType> & B, const ScalarType beta, MV & mv){
  Kokkos::View<ScalarType**> mat("mat", A.extent(1), mv.extent(1));
  Teuchos2KokkosMat(B,mat);
  KokkosBlas::gemm("N", "N", alpha, A, mat, beta, mv);
}
// In normal use of this method, does the correct size of the Teuchos SerialDenseMat have to be defined ahead of time?? 

//-------------------------------------------------------------
//
// *this <- alpha * A + beta * B
//
//-------------------------------------------------------------

void KokkosMultiVec::MvAddMv ( const ScalarType alpha , const MultiVec<ScalarType>& A,
                               const ScalarType beta, const MultiVec<ScalarType>& B)
{
  KokkosMultiVec *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(A));
  TEUCHOS_TEST_FOR_EXCEPTION( A_vec==NULL, KokkosMultiVecFailure,
                     "Belos::KokkosMultiVec::MvAddMv cast from Belos::MultiVec<> to Belos::KokkosMultiVec failed.");
  KokkosMultiVec *B_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(B));
  TEUCHOS_TEST_FOR_EXCEPTION( B_vec==NULL, KokkosMultiVecFailure,
                     "Belos::KokkosMultiVec::MvAddMv cast from Belos::MultiVec<> to Belos::KokkosMultiVec failed.");

  int info = Update( alpha, *A_vec, beta, *B_vec, 0.0 );
  TEUCHOS_TEST_FOR_EXCEPTION(info!=0, KokkosMultiVecFailure,
                     "Belos::KokkosMultiVec::MvAddMv call to Update() returned a nonzero value.");
}

//-------------------------------------------------------------
//
// this[i] = alpha[i] * this[i]
//
//-------------------------------------------------------------
void KokkosMultiVec::MvScale ( const std::vector<ScalarType>& alpha )
{
  // Check to make sure the input vector of scale factors has the same
  // number of entries as the number of columns in the multivector.
  int numvecs = this->NumVectors();
  TEUCHOS_TEST_FOR_EXCEPTION((int)alpha.size() != numvecs, KokkosMultiVecFailure,
                     "Belos::MultiVecTraits<ScalarType,Kokkos_MultiVec>::MvScale: "
                     "Vector (alpha) of scaling factors has length "
                     << alpha.size() << ", which is different than the number "
                     "of columns " << numvecs << " in the input multivector "
                     "(mv).");
  int ret = 0;
  std::vector<int> tmp_index( 1, 0 );
  for (int i=0; i<numvecs; i++) {
    Kokkos_MultiVector temp_vec(BELOSKOKKOSVIEW, *this, &tmp_index[0], 1);
    ret = temp_vec.Scale( alpha[i] );
    TEUCHOS_TEST_FOR_EXCEPTION(ret!=0, KokkosMultiVecFailure,
                       "Belos::MultiVecTraits<ScalarType,Kokkos_MultiVec>::MvScale: "
                       "Call to Kokkos_MultiVector::Scale() returned a nonzero "
                       "error code " << ret << ".");
    tmp_index[0]++;
  }
}

//-------------------------------------------------------------
//
// dense B <- alpha * A^T * (*this)
//
//-------------------------------------------------------------

void KokkosMultiVec::MvTransMv ( const ScalarType alpha, const MultiVec<ScalarType>& A,
                                 Teuchos::SerialDenseMatrix<int,ScalarType>& B) const
{
  KokkosMultiVec *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(A));

  if (A_vec) {
    Kokkos_LocalMap LocalMap(B.numRows(), 0, Map().Comm());
    Kokkos_MultiVector B_Pvec(BELOSKOKKOSVIEW, LocalMap, B.values(), B.stride(), B.numCols());

    int info = B_Pvec.Multiply( 'T', 'N', alpha, *A_vec, *this, 0.0 );
    TEUCHOS_TEST_FOR_EXCEPTION(info!=0, KokkosMultiVecFailure,
                       "Belos::KokkosMultiVec::MvTransMv call to Multiply() returned a nonzero value.");
  }
}

//-------------------------------------------------------------
//
// b[i] = A[i]^T * this[i]
//
//-------------------------------------------------------------

void KokkosMultiVec::MvDot ( const MultiVec<ScalarType>& A, std::vector<ScalarType>& b ) const
{
  KokkosMultiVec *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(A));
  TEUCHOS_TEST_FOR_EXCEPTION(A_vec==NULL, KokkosMultiVecFailure,
                     "Belos::KokkosMultiVec::MvDot: Dynamic cast from Belos::"
                     "MultiVec<ScalarType> to Belos::KokkosMultiVec failed.");
  if (A_vec && ( (int)b.size() >= A_vec->NumVectors() ) ) {
     int info = this->Dot( *A_vec, &b[0] );
     TEUCHOS_TEST_FOR_EXCEPTION(info!=0, KokkosMultiVecFailure,
                        "Belos::KokkosMultiVec::MvDot: Call to Kokkos_Multi"
                        "Vector::Dot() returned a nonzero error code "
                        << info << ".");
  }
}

//-------------------------------------------------------------
//
// alpha[i] = norm of i-th column of (*this)
//
//-------------------------------------------------------------

void KokkosMultiVec::MvNorm ( std::vector<ScalarType>& normvec, NormType norm_type ) const {
  if ((int)normvec.size() >= GetNumberVecs()) {
    int info = 0;
    switch( norm_type ) {
    case ( OneNorm ) :
      info = Norm1(&normvec[0]);
      break;
    case ( TwoNorm ) :
      info = Norm2(&normvec[0]);
      break;
    case ( InfNorm ) :
      info = NormInf(&normvec[0]);
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument,
                         "Belos::KokkosMultiVec::MvNorm: Invalid norm_type "
                         << norm_type << ".  The current list of valid norm "
                         "types is {OneNorm, TwoNorm, InfNorm}.");
    }
    TEUCHOS_TEST_FOR_EXCEPTION(info!=0, KokkosMultiVecFailure,
                       "Belos::KokkosMultiVec::MvNorm: Call to Kokkos_Multi"
                       "Vector::Norm() returned a nonzero error code "
                       << info << ".");
  }
}

/*
///////////////////////////////////////////////////////////////
//
// Implementation of the Belos::KokkosOp class.
//
///////////////////////////////////////////////////////////////

KokkosOp::KokkosOp( const Teuchos::RCP<Kokkos_Operator> &Op )
  : Kokkos_Op(Op)
{
#ifdef BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
  using std::endl;

  std::ostream& err = *(getErrStream (*Kokkos_Op));
  err << "Belos::KokkosOp constructor" << endl;
  err << "-- On input: Kokkos_Op->UseTranspose() = "
      << Kokkos_Op->UseTranspose() << endl;
#endif // BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
}

void
KokkosOp::Apply (const MultiVec<double>& x,
                 MultiVec<double>& y,
                 ETrans trans) const
{
#ifdef BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
  using std::endl;

  std::ostream& err = *(getErrStream (*Kokkos_Op));
  err << "-- Belos::KokkosOp::Apply:" << endl
      << "---- Implements Belos::Operator<double>::Apply()" << endl
      << "---- trans input = " << etransToString (trans) << endl
      << "---- On input: Kokkos_Op->UseTranspose() = "
      << Kokkos_Op->UseTranspose() << endl;
#endif // BELOS_KOKKOS_AGGRESSIVE_DEBUGGING

  MultiVec<double> & temp_x = const_cast<MultiVec<double> &>(x);
  Kokkos_MultiVector* vec_x = dynamic_cast<Kokkos_MultiVector* >(&temp_x);
  Kokkos_MultiVector* vec_y = dynamic_cast<Kokkos_MultiVector* >(&y);

  TEUCHOS_TEST_FOR_EXCEPTION(vec_x==NULL || vec_y==NULL, KokkosOpFailure,
                     "Belos::KokkosOp::Apply: x and/or y could not be "
                     "dynamic cast to an Kokkos_MultiVector.");

  // Temporarily set the transpose state of Op, if it's not the same
  // as trans, and restore it on exit of this scope.
  KokkosOperatorTransposeScopeGuard guard (*Kokkos_Op, trans);

#ifdef BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
  err << "---- Before calling Apply: Kokkos_Op->UseTranspose() = "
      << Kokkos_Op->UseTranspose() << endl;
#endif // BELOS_KOKKOS_AGGRESSIVE_DEBUGGING

  // Apply the operator to x and put the result in y.
  const int info = Kokkos_Op->Apply (*vec_x, *vec_y);
  TEUCHOS_TEST_FOR_EXCEPTION(info!=0, KokkosOpFailure,
                     "Belos::KokkosOp::Apply: The Kokkos_Operator's Apply() "
                     "method returned a nonzero error code of " << info << ".");
}

bool
KokkosOp::HasApplyTranspose() const
{
#ifdef BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
  std::ostream& err = *(getErrStream (*Kokkos_Op));
  err << "-- Belos::KokkosOp::HasApplyTranspose" << std::endl;
#endif // BELOS_KOKKOS_AGGRESSIVE_DEBUGGING

  return implementsApplyTranspose (*Kokkos_Op);
}
*/

/*
// ///////////////////////////////////////////////////////////////////
//
// Implementation of the Belos::KokkosPrecOp class.
//
// ///////////////////////////////////////////////////////////////////

KokkosPrecOp::KokkosPrecOp (const Teuchos::RCP<Kokkos_Operator> &Op)
  : Kokkos_Op(Op)
{
#ifdef BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
  using std::endl;

  std::ostream& err = *(getErrStream (*Kokkos_Op));
  err << "-- Belos::KokkosPrecOp constructor" << endl
      << "---- On input: Kokkos_Op->UseTranspose() = "
      << Kokkos_Op->UseTranspose() << endl;
#endif // BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
}

// The version of Apply() that takes an optional 'trans' argument and
// returns void implements the Belos::Operator interface.
void
KokkosPrecOp::Apply (const MultiVec<double>& x,
                     MultiVec<double>& y,
                     ETrans trans) const
{
#ifdef BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
  using std::endl;

  std::ostream& err = *(getErrStream (*Kokkos_Op));
  err << "-- Belos::KokkosPrecOp::Apply:" << endl
      << "---- Implements Belos::Operator<double>::Apply()" << endl
      << "---- trans input = " << etransToString (trans) << endl
      << "---- On input: Kokkos_Op->UseTranspose() = "
      << Kokkos_Op->UseTranspose() << endl;
#endif // BELOS_KOKKOS_AGGRESSIVE_DEBUGGING

  MultiVec<double>&  temp_x = const_cast<MultiVec<double> &>(x);
  Kokkos_MultiVector* vec_x = dynamic_cast<Kokkos_MultiVector* >(&temp_x);
  TEUCHOS_TEST_FOR_EXCEPTION(vec_x == NULL, KokkosOpFailure,
                     "Belos::KokkosPrecOp::Apply: The MultiVec<double> input x "
                     "cannot be dynamic cast to an Kokkos_MultiVector.");
  Kokkos_MultiVector* vec_y = dynamic_cast<Kokkos_MultiVector* >(&y);
  TEUCHOS_TEST_FOR_EXCEPTION(vec_x == NULL, KokkosOpFailure,
                     "Belos::KokkosPrecOp::Apply: The MultiVec<double> input y "
                     "cannot be dynamic cast to an Kokkos_MultiVector.");

  // Temporarily set the transpose state of Op, if it's not the same
  // as trans, and restore it on exit of this scope.
  KokkosOperatorTransposeScopeGuard guard (*Kokkos_Op, trans);

#ifdef BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
  err << "---- Before calling ApplyInverse: Kokkos_Op->UseTranspose() = "
      << Kokkos_Op->UseTranspose() << endl;
#endif // BELOS_KOKKOS_AGGRESSIVE_DEBUGGING

  // KokkosPrecOp's Apply() methods apply the inverse of the
  // underlying operator.  This may not succeed for all
  // implementations of Kokkos_Operator, so we have to check.
  const int info = Kokkos_Op->ApplyInverse (*vec_x, *vec_y);
  TEUCHOS_TEST_FOR_EXCEPTION(info != 0, KokkosOpFailure,
                     "Belos::KokkosPrecOp::Apply: Calling ApplyInverse() on the "
                     "underlying Kokkos_Operator object failed, returning a "
                     "nonzero error code of " << info << ".  This probably means"
                     " that the underlying Kokkos_Operator object doesn't know "
                     "how to apply its inverse.");
}

// The version of Apply() that takes two arguments and returns int
// implements the Kokkos_Operator interface.
int
KokkosPrecOp::Apply (const Kokkos_MultiVector &X,
                     Kokkos_MultiVector &Y) const
{
#ifdef BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
  using std::endl;

  std::ostream& err = *(getErrStream (*Kokkos_Op));
  err << "-- Belos::KokkosPrecOp::Apply:" << endl
      << "---- Implements Kokkos_Operator::Apply()" << endl
      << "---- On input: Kokkos_Op->UseTranspose() = "
      << Kokkos_Op->UseTranspose() << endl
      << "---- Calling Kokkos_Op->ApplyInverse (X, Y)" << endl;
#endif // BELOS_KOKKOS_AGGRESSIVE_DEBUGGING

  // This operation computes Y = A^{-1}*X.
  const int info = Kokkos_Op->ApplyInverse( X, Y );

#ifdef BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
  err << "---- Kokkos_Op->ApplyInverse (X, Y) returned info = " << info << endl;
#endif // BELOS_KOKKOS_AGGRESSIVE_DEBUGGING

  return info;
}

// This implements Kokkos_Operator::ApplyInverse().
int
KokkosPrecOp::ApplyInverse (const Kokkos_MultiVector &X,
                            Kokkos_MultiVector &Y) const
{
#ifdef BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
  using std::endl;

  std::ostream& err = *(getErrStream (*Kokkos_Op));
  err << "-- Belos::KokkosPrecOp::ApplyInverse:" << endl
      << "---- Implements Kokkos_Operator::ApplyInverse()" << endl
      << "---- On input: Kokkos_Op->UseTranspose() = "
      << Kokkos_Op->UseTranspose() << endl
      << "---- Calling Kokkos_Op->Apply (X, Y)" << endl;
#endif // BELOS_KOKKOS_AGGRESSIVE_DEBUGGING

  // This operation computes Y = A*X.
  const int info = Kokkos_Op->Apply( X, Y );

#ifdef BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
  err << "---- Kokkos_Op->Apply (X, Y) returned info = " << info << endl;
#endif // BELOS_KOKKOS_AGGRESSIVE_DEBUGGING

  return info;
}

bool
KokkosPrecOp::HasApplyTranspose() const
{
#ifdef BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
  std::ostream& err = *(getErrStream (*Kokkos_Op));
  err << "-- Belos::KokkosPrecOp::HasApplyTranspose" << std::endl;
#endif // BELOS_KOKKOS_AGGRESSIVE_DEBUGGING

  return implementsApplyTranspose (*Kokkos_Op);
}


// ///////////////////////////////////////////////////////////////////
//
// Specialization of Belos::OperatorTraits for Kokkos_Operator.
//
// ///////////////////////////////////////////////////////////////////

void
OperatorTraits<double, Kokkos_MultiVector, Kokkos_Operator>::
Apply (const Kokkos_Operator& Op,
       const Kokkos_MultiVector& x,
       Kokkos_MultiVector& y,
       ETrans trans)
{
#ifdef BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
  using std::endl;

  std::ostream& err = *(getErrStream (Op));
  err << "Belos::OperatorTraits<double, Kokkos_MultiVector, Kokkos_Operator>::Apply:"
      << endl
      << "-- trans input = " << etransToString (trans) << endl
      << "-- On input: Op.UseTranspose() = "
      << Op.UseTranspose() << endl;
#endif // BELOS_KOKKOS_AGGRESSIVE_DEBUGGING

  // Temporarily set the transpose state of Op, if it's not the same
  // as trans, and restore it on exit of this scope.
  KokkosOperatorTransposeScopeGuard guard (Op, trans);

#ifdef BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
  err << "-- Before calling Op.Apply: Op.UseTranspose() = "
      << Op.UseTranspose() << endl;
#endif // BELOS_KOKKOS_AGGRESSIVE_DEBUGGING

  const int info = Op.Apply (x, y);
  TEUCHOS_TEST_FOR_EXCEPTION(info != 0, KokkosOpFailure,
                     "Belos::OperatorTraits::Apply (Kokkos specialization): "
                     "Calling the Kokkos_Operator object's Apply() method "
                     "failed, returning a nonzero error code of " << info
                     << ".");
}

bool
OperatorTraits<double, Kokkos_MultiVector, Kokkos_Operator>::
HasApplyTranspose (const Kokkos_Operator& Op)
{
#ifdef BELOS_KOKKOS_AGGRESSIVE_DEBUGGING
  std::ostream& err = *(getErrStream (Op));
  err << "Belos::OperatorTraits<double, Kokkos_MultiVector, Kokkos_Operator>::"
    "HasApplyTranspose" << std::endl;
#endif // BELOS_KOKKOS_AGGRESSIVE_DEBUGGING

  return implementsApplyTranspose (Op);
}
*/

}  // end namespace Belos
