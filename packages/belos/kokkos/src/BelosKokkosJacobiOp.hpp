#include<Kokkos_Core.hpp>
#include<KokkosBlas.hpp>
#include<BelosTypes.hpp>
#include<BelosOperator.hpp>
#include<BelosKokkosAdapter.hpp>

//All the headers you need for different batched functions:
#include "Kokkos_ArithTraits.hpp"
#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Copy_Decl.hpp"
#include "KokkosBatched_Copy_Impl.hpp"
#include "KokkosBatched_SetIdentity_Decl.hpp"
#include "KokkosBatched_SetIdentity_Impl.hpp"
#include "KokkosBatched_Gemv_Decl.hpp"
#include "KokkosBatched_Gemv_Team_Impl.hpp"
#include "KokkosBatched_Trsm_Decl.hpp"
#include "KokkosBatched_Trsm_Team_Impl.hpp"
#include "KokkosBatched_Trsv_Decl.hpp"
#include "KokkosBatched_Trsv_Team_Impl.hpp"
#include "KokkosBatched_LU_Decl.hpp"
#include "KokkosBatched_LU_Team_Impl.hpp"


#ifndef BELOS_KOKKOS_JACOBI_HPP
#define BELOS_KOKKOS_JACOBI_HPP
 namespace Belos {
  /// \class KokkosOperator
  /// \brief Implementation of Belos::Operator using Kokkos::Crs matrix.
  ///
  template<class ScalarType, class OrdinalType=int, class Device=Kokkos::DefaultExecutionSpace>
  class KokkosJacobiOperator : public Operator<ScalarType> {
  private:

  //TODO: really check if these typedefs match the template. 
    using scalar_t  = ScalarType;
    using Layout          = Kokkos::LayoutLeft;
    using ViewVectorType  = Kokkos::View<ScalarType*, Layout, Device>;
    using ManyVectorType  = Kokkos::View<ScalarType**, Layout, Device>;
    using ManyMatrixType  = Kokkos::View<ScalarType***, Layout, Device>;
    using execution_space = typename ViewVectorType::device_type::execution_space;
    using memory_space    = typename ViewVectorType::device_type::memory_space;
    using crsMat_t = KokkosSparse::CrsMatrix<ScalarType, OrdinalType, Device>;
    using size_type       = typename crsMat_t::size_type; 
    using policy_type = Kokkos::TeamPolicy<execution_space>;
    using member_type = typename policy_type::member_type;

    crsMat_t A_;// Matrix from which to create Jacobi preconditioner. 
    ManyMatrixType Ablocks_; //To hold the Jacobi blocks. 
    ManyMatrixType Sblocks_; //To hold block inverses if GEMV option chosen.
    int blockSize_;
    int numBlocks_;
    int teamSize_;
    std::string solve_; // TRSV or store inverse and use GEMV. 

  public:
  // Shallow copy for mat of same scalar type:
    KokkosJacobiOperator<ScalarType, OrdinalType, Device> (const KokkosSparse::CrsMatrix<ScalarType, OrdinalType, Device> A, int blockSize, std::string solve="TRSV", int teamSize=-1) 
     : A_(A), 
     Ablocks_("Ablocks",0,0,0),
     Sblocks_("Sblocks",0,0,0),
     blockSize_(blockSize),
     numBlocks_(-1),
     solve_(solve),
     teamSize_(teamSize)
    {}

    void SetUpJacobi() {
    const size_type numRows = A_.graph.numRows();
    //Right now, we only support if the block size evenly divides the size of the matrix.
    if(numRows % blockSize_ != 0){
      throw std::runtime_error ("Num rows is not divisible by block size.");
    }
    int numBlocks_ = ceil(numRows/blockSize_ ); // Need ceil for later if use uneven blocks.
    Kokkos::resize(Ablocks_, numBlocks_, blockSize_, blockSize_);
    if(solve_ == "GEMV"){
      Kokkos::resize(Sblocks_, numBlocks_, blockSize_, blockSize_);
    }
    else if (solve_ != "TRSV"){
      solve_ = "TRSV";
      printf("Block Jacobi Prec: Invalid solve type given. Using TRSV. \n \n");
    }

      printf("started extracting Jacobi blocks. \n \n");

    auto values = A_.values; 
    auto colIdxView = A_.graph.entries;
    auto rowPtrView = A_.graph.row_map;

    int colPtr = 0;
    for (int row = 0; row < numRows; row++){
      int rowBlk = floor(row/blockSize_);
      while( colPtr < rowPtrView(row+1) ) { // If we are still in the same row...
        int colBlk = floor( colIdxView(colPtr)/blockSize_ );
          if( rowBlk == colBlk) { // Then we are in one of the blocks to extract.
            Ablocks_( rowBlk, row % blockSize_, colIdxView(colPtr) % blockSize_ ) = values(colPtr);
          }
        colPtr++;
      }
    }

      printf("Finished extracting Jacobi blocks. \n \n");
      //TODO: should we store policy as part of the object so we dont' have to recreate for apply?
	policy_type policy(numBlocks_, Kokkos::AUTO());	
        if (teamSize_ > 0) 
          policy = policy_type(numBlocks_, teamSize_);
  Kokkos::Impl::Timer timer;
	timer.reset();
	    const int  one (1);
    if(solve_ == "GEMV" ){
    //TODO: Should we save the "subBlocks" variables as part of the object so we don't have to recreate for apply?
	Kokkos::parallel_for
	  ("task2.factorize-invert",
	   policy, KOKKOS_LAMBDA(const member_type &member) {
      const int i = member.league_rank();
      auto Asub = Kokkos::subview(Ablocks_, i, Kokkos::ALL(), Kokkos::ALL());

      KokkosBatched::TeamLU<member_type,KokkosBatched::Algo::Level3::Unblocked>::invoke(member,Asub);
      auto Ssub = Kokkos::subview(Sblocks_, i, Kokkos::ALL(), Kokkos::ALL());
      KokkosBatched::TeamCopy<member_type,KokkosBatched::Trans::NoTranspose>::invoke(member, Asub, Ssub);
      KokkosBatched::TeamSetIdentity<member_type>::invoke(member, Asub);
      KokkosBatched::TeamTrsm<member_type, KokkosBatched::Side::Left,KokkosBatched::Uplo::Lower,
            KokkosBatched::Trans::NoTranspose,KokkosBatched::Diag::Unit,
            KokkosBatched::Algo::Level3::Unblocked> ::invoke(member, one, Ssub, Asub);
      KokkosBatched::TeamTrsm<member_type,
      KokkosBatched::Side::Left,KokkosBatched::Uplo::Upper,KokkosBatched::Trans::NoTranspose,KokkosBatched::Diag::NonUnit,
      KokkosBatched::Algo::Level3::Unblocked>
      ::invoke(member, one, Ssub, Asub);
      });}
    else if (solve_ == "TRSV"){
      Kokkos::parallel_for
        ("task2.factorize-invert",
         policy, KOKKOS_LAMBDA(const member_type &member) {
         const int i = member.league_rank();
         auto Asub = Kokkos::subview(Ablocks_, i, Kokkos::ALL(), Kokkos::ALL());
         KokkosBatched::TeamLU<member_type,KokkosBatched::Algo::Level3::Unblocked>::invoke(member,Asub);
         });
    }
    Kokkos::fence();
	const double t = timer.seconds();
	printf("task 2: construction of jacobi time = %f , # of constructions per min = %.0f \n", t, 1.0/t*60);

    }

    //template<typename ScalarType2>  //Tried ST2. Doesn't work b/c apply interface has all same ST. 
    // y = A*x
    void Apply (const MultiVec<ScalarType>& x,  MultiVec<ScalarType>& y,  ETrans trans=NOTRANS) const{
    //Note: Do NOT make x and y the same multivector!  You will get NaNs...
      // Note: spmv computes y = beta*y + alpha*Op(A)*x  spmv(mode,alpha,A,x,beta,y);
      //ScalarType alpha = 1.0;
      //ScalarType beta = 0;
      KokkosMultiVec<ScalarType, Device> *x_vec = dynamic_cast<KokkosMultiVec<ScalarType, Device> *>
            (&const_cast<MultiVec<ScalarType> &>(x));
      KokkosMultiVec<ScalarType, Device> *y_vec = dynamic_cast<KokkosMultiVec<ScalarType, Device> *>(&y);

      // Get a rank-1 subview of our rank-2 view, so don't fail asserts on sptrsv. 
      Kokkos::View<ScalarType*, Kokkos::LayoutLeft, Device> xsub = Kokkos::subview(x_vec->myView, Kokkos::ALL, 0);
      Kokkos::View<ScalarType*, Kokkos::LayoutLeft, Device> ysub = Kokkos::subview(y_vec->myView, Kokkos::ALL, 0);

      // KokkosSparse::Experimental::sptrsv_solve(handle, rowmap, entries, values, b, x);
      // x = U\b, x = L\b, Ux=b, etc. 
      // TODO did I pass in correct vectors? for solve??
      // LUy = x by this layout
      // Uy = L\x = tmp
      // y = U\tmp
  //    KokkosSparse::Experimental::sptrsv_solve( &kh_sptrsv_L, L_row_map, L_entries, L_values, xsub, tmp); 
   //   KokkosSparse::Experimental::sptrsv_solve( &kh_sptrsv_U, U_row_map, U_entries, U_values, tmp, ysub);
      //TODO: should we store policy as part of the object so we dont' have to recreate for apply?
	policy_type policy(numBlocks_, Kokkos::AUTO());	
        if (teamSize_ > 0) 
          policy = policy_type(numBlocks_, teamSize_);
  Kokkos::Impl::Timer timer;
	timer.reset();
	    const int  one (1);
    if(solve_ == "GEMV" ){
    //TODO: Should we save the "subBlocks" variables as part of the object so we don't have to recreate for apply?
	Kokkos::parallel_for
	  ("GEMV apply LU inv",
	   policy, KOKKOS_LAMBDA(const member_type &member) {
      const int i = member.league_rank();
      auto Asub = Kokkos::subview(Ablocks_, i, Kokkos::ALL(), Kokkos::ALL());

      KokkosBatched::TeamLU<member_type,KokkosBatched::Algo::Level3::Unblocked>::invoke(member,Asub);
      auto Ssub = Kokkos::subview(Sblocks_, i, Kokkos::ALL(), Kokkos::ALL());
      KokkosBatched::TeamCopy<member_type,KokkosBatched::Trans::NoTranspose>::invoke(member, Asub, Ssub);
      KokkosBatched::TeamSetIdentity<member_type>::invoke(member, Asub);
      KokkosBatched::TeamTrsm<member_type, KokkosBatched::Side::Left,KokkosBatched::Uplo::Lower,
            KokkosBatched::Trans::NoTranspose,KokkosBatched::Diag::Unit,
            KokkosBatched::Algo::Level3::Unblocked> ::invoke(member, one, Ssub, Asub);
      KokkosBatched::TeamTrsm<member_type,KokkosBatched::Side::Left,KokkosBatched::Uplo::Upper,
            KokkosBatched::Trans::NoTranspose,KokkosBatched::Diag::NonUnit,
            KokkosBatched::Algo::Level3::Unblocked>::invoke(member, one, Ssub, Asub);
      });}
    else if (solve_ == "TRSV"){
    //Must deep copy here because TRSV uses same input and output vector.
    Kokkos::deep_copy(y_vec->myView, x_vec->myView);
      Kokkos::parallel_for
        ("TRSV apply LU inv",
         policy, KOKKOS_LAMBDA(const member_type &member) {
         const int i = member.league_rank();
         auto Asub = Kokkos::subview(Ablocks_, i, Kokkos::ALL(), Kokkos::ALL());
      Kokkos::View<ScalarType*, Kokkos::LayoutLeft, Device> ysub = 
            Kokkos::subview(y_vec->myView, Kokkos::make_pair(i*blockSize_,(i+1)*blockSize_), 0);
      KokkosBatched::TeamTrsv<member_type,KokkosBatched::Uplo::Lower,
            KokkosBatched::Trans::NoTranspose,KokkosBatched::Diag::Unit,
            KokkosBatched::Algo::Trsv::Unblocked> ::invoke(member, one, Asub, ysub);
      KokkosBatched::TeamTrsv<member_type,KokkosBatched::Uplo::Upper,
            KokkosBatched::Trans::NoTranspose,KokkosBatched::Diag::NonUnit,
            KokkosBatched::Algo::Trsv::Unblocked>::invoke(member, one, Asub, ysub);
      });
    }
    Kokkos::fence();
	const double t = timer.seconds();
	printf("task 2: construction of jacobi time = %f , # of constructions per min = %.0f \n", t, 1.0/t*60);

    }

    bool HasApplyTranspose () const {
      return false;
    }

    ~KokkosJacobiOperator<ScalarType, OrdinalType, Device>(){}

  };
}// end namespace Belos
#endif
