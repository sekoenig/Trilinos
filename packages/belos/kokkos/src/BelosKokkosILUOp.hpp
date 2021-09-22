#include<Kokkos_Core.hpp>
#include<KokkosBlas.hpp>
#include<BelosTypes.hpp>
#include<BelosOperator.hpp>
#include<BelosKokkosAdapter.hpp>

#include<KokkosSparse_spmv.hpp>
#include<KokkosSparse_spiluk.hpp>
#include<KokkosSparse_sptrsv.hpp>

#define EXPAND_FACT 6 // a factor used in expected sizes of L and U
enum {DEFAULT, LVLSCHED_RP, LVLSCHED_TP1};

#ifndef BELOS_KOKKOS_ILU_HPP
#define BELOS_KOKKOS_ILU_HPP
 namespace Belos {
  /// \class KokkosOperator
  /// \brief Implementation of Belos::Operator using Kokkos::Crs matrix.
  ///
  template<class ScalarType, class OrdinalType=int, class Device=Kokkos::DefaultExecutionSpace>
  class KokkosILUOperator : public Operator<ScalarType> {
  private:

  //TODO: really check if these typedefs match the template. 
  //Typedefs from ILUEx:
    using scalar_t  = ScalarType;
    using lno_t     = int;
    using Layout          = Kokkos::LayoutLeft;
    using ViewVectorType  = Kokkos::View<ScalarType*, Layout, Device>;
    using execution_space = typename ViewVectorType::device_type::execution_space;
    using memory_space    = typename ViewVectorType::device_type::memory_space;
    using crsMat_t = KokkosSparse::CrsMatrix<ScalarType, OrdinalType, Device>;
    //using crsmat_t        = KokkosSparse::CrsMatrix<scalar_t, lno_t, Kokkos::DefaultExecutionSpace>;
    using size_type       = typename crsMat_t::size_type; 
    using graph_t         = typename crsMat_t::StaticCrsGraphType;
    using lno_view_t      = typename graph_t::row_map_type::non_const_type; //row map view type
    using lno_nnz_view_t  = typename graph_t::entries_type::non_const_type; //entries view type
    using scalar_view_t   = typename crsMat_t::values_type::non_const_type; //values view type
    using KernelHandle    = KokkosKernels::Experimental::KokkosKernelsHandle <size_type, lno_t, scalar_t,
                                                                              execution_space, memory_space, memory_space>;

    //KokkosSparse::CrsMatrix<ScalarType, OrdinalType, Device> A;
    crsMat_t A;// Matrix from which to create ILU factorization. 
    mutable KernelHandle kh_sptrsv_L, kh_sptrsv_U; //Kernel handles for ILU factorization. 
    // Allocate row map, entries, values views (1-D) for L and U
    // We don't know how big to make these until ILU setup, so will resize them later.  
    lno_view_t     L_row_map;
    lno_nnz_view_t L_entries;
    scalar_view_t  L_values;
    lno_view_t     U_row_map;
    lno_nnz_view_t U_entries;
    scalar_view_t  U_values;	
    // one-time allocate temp vector for applying L and U solve:
    ViewVectorType tmp;

  public:
  // Shallow copy for mat of same scalar type:
    KokkosILUOperator<ScalarType, OrdinalType, Device> (const KokkosSparse::CrsMatrix<ScalarType, OrdinalType, Device> mat) 
     : A(mat), 
    L_row_map("L_row_map",0),
    L_entries("L_entries",0),
    L_values ("L_values",0),
    U_row_map("U_row_map",0),
    U_entries("U_entries",0),
    U_values ("U_values",0),
    tmp ("tmp",0)
    {}

  //This doesn't work yet!! That is view notation!
  // Deep copy for changing scalar types: 
  //  template <class ScalarType2 > 
  //  KokkosILUOperator<ScalarType, OrdinalType, Device> (const KokkosSparse::CrsMatrix<ScalarType2, OrdinalType, Device> mat) 
  //   : myMatrix(Kokkos::ViewAllocateWithoutInitializing("Mat"),mat.extent(0),mat.extent(1)) {Kokkos::deep_copy(myMatrix,mat);}
  //
  //TODO add all the options for ILU.  Take out the extra junk. 
    int SetUpILU() {
    int success = 1;
    //Defaulting to team policy; Nathan says this usually exposes more parallelism.  
  int algo_spiluk = LVLSCHED_TP1; // SPILUK kernel implementation
  int algo_sptrsv = LVLSCHED_TP1; // SPTRSV kernel implementation
  int k = 0;                     // fill level
  int team_size = -1;            // team size //TODO I bet in the original driver, this has an option to read from cmnd line

    //scalar_t tolerance = 0.0000000001;
    //scalar_t one  = scalar_t(1.0);
    //scalar_t zero = scalar_t(0.0);
    //scalar_t mone = scalar_t(-1.0);

    graph_t  graph    = A.graph; // in_graph
    const size_type N = graph.numRows();
    typename KernelHandle::const_nnz_lno_t fill_lev = lno_t(k) ;
    const size_type nnzA = A.graph.entries.extent(0);
    std::cout << "Matrix size: " << N << " x " << N << ", nnz = " << nnzA << std::endl;
    // Create SPILUK handle and SPTRSV handles (for L and U)
    KernelHandle kh_spiluk; //Kernel handles for ILU factorization. 

    std::cout << "Create SPILUK handle ..." << std::endl;
    switch(algo_spiluk) {
      case LVLSCHED_RP: //Using range policy (KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_RP)
        kh_spiluk.create_spiluk_handle(KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_RP, N, EXPAND_FACT*nnzA*(fill_lev+1), EXPAND_FACT*nnzA*(fill_lev+1));
        std::cout << "Kernel implementation type: "; kh_spiluk.get_spiluk_handle()->print_algorithm();
        break;
      case LVLSCHED_TP1: //Using team policy (KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1)
        kh_spiluk.create_spiluk_handle(KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_TP1, N, EXPAND_FACT*nnzA*(fill_lev+1), EXPAND_FACT*nnzA*(fill_lev+1));
        std::cout << "TP1 set team_size = " << team_size << std::endl;
        if (team_size != -1) kh_spiluk.get_spiluk_handle()->set_team_size(team_size);
        std::cout << "Kernel implementation type: "; kh_spiluk.get_spiluk_handle()->print_algorithm();
        break;
      default: //Using range policy
        kh_spiluk.create_spiluk_handle(KokkosSparse::Experimental::SPILUKAlgorithm::SEQLVLSCHD_RP, N, EXPAND_FACT*nnzA*(fill_lev+1), EXPAND_FACT*nnzA*(fill_lev+1));
        std::cout << "Kernel implementation type: "; kh_spiluk.get_spiluk_handle()->print_algorithm();
    }

    // EXERCISE: Create a SPTRSV handle
    // EXERCISE hint: input arguments include implementation type (i.e. KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP or KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1), number of matrix rows, lower or upper matrix flag (boolean)
    std::cout << "Create SPTRSV handle for L ..." << std::endl;
    //TODO can use  SPTRSV_CUSPARSE if CUsparse is enabled... add this option.
    bool is_lower_tri = true;
    switch(algo_sptrsv) {
      case LVLSCHED_RP:
        // EXERCISE hint: kh.create_sptrsv_handle(implementation_type, number_of_matrix_rows, is_lower_tri)
        kh_sptrsv_L.create_sptrsv_handle(KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP, N, is_lower_tri);
        kh_sptrsv_L.get_sptrsv_handle()->print_algorithm();
        break;
      case LVLSCHED_TP1:
        // EXERCISE hint: kh.create_sptrsv_handle(implementation_type, number_of_matrix_rows, is_lower_tri)
        kh_sptrsv_L.create_sptrsv_handle(KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1, N, is_lower_tri);
        std::cout << "TP1 set team_size = " << team_size << std::endl;
        if (team_size != -1) kh_sptrsv_L.get_sptrsv_handle()->set_team_size(team_size);
        kh_sptrsv_L.get_sptrsv_handle()->print_algorithm();
        break;
      default:
        // EXERCISE hint: kh.create_sptrsv_handle(implementation_type, number_of_matrix_rows, is_lower_tri)
        kh_sptrsv_L.create_sptrsv_handle(KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP, N, is_lower_tri);
        kh_sptrsv_L.get_sptrsv_handle()->print_algorithm();
    }

    std::cout << "Create SPTRSV handle for U ..." << std::endl;
    is_lower_tri = false;
    switch(algo_sptrsv) {
      case LVLSCHED_RP:
        // EXERCISE hint: kh.create_sptrsv_handle(implementation_type, number_of_matrix_rows, is_lower_tri)
        kh_sptrsv_U.create_sptrsv_handle(KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP, N, is_lower_tri);
        kh_sptrsv_U.get_sptrsv_handle()->print_algorithm();
        break;
      case LVLSCHED_TP1:
        // EXERCISE hint: kh.create_sptrsv_handle(implementation_type, number_of_matrix_rows, is_lower_tri)
        kh_sptrsv_U.create_sptrsv_handle(KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_TP1, N, is_lower_tri);
        std::cout << "TP1 set team_size = " << team_size << std::endl;
        if (team_size != -1) kh_sptrsv_U.get_sptrsv_handle()->set_team_size(team_size);
        kh_sptrsv_U.get_sptrsv_handle()->print_algorithm();
        break;
      default:
        // EXERCISE hint: kh.create_sptrsv_handle(implementation_type, number_of_matrix_rows, is_lower_tri)
        kh_sptrsv_U.create_sptrsv_handle(KokkosSparse::Experimental::SPTRSVAlgorithm::SEQLVLSCHD_RP, N, is_lower_tri);
        kh_sptrsv_U.get_sptrsv_handle()->print_algorithm();
    }
	
    auto spiluk_handle  = kh_spiluk.get_spiluk_handle();
    auto sptrsvL_handle = kh_sptrsv_L.get_spiluk_handle();
    auto sptrsvU_handle = kh_sptrsv_U.get_spiluk_handle();

    /*// Allocate row map, entries, values views (1-D) for L and U
    lno_view_t     L_row_map("L_row_map", N + 1);
    lno_nnz_view_t L_entries("L_entries", spiluk_handle->get_nnzL());
    scalar_view_t  L_values ("L_values",  spiluk_handle->get_nnzL());
    lno_view_t     U_row_map("U_row_map", N + 1);
    lno_nnz_view_t U_entries("U_entries", spiluk_handle->get_nnzU());
    scalar_view_t  U_values ("U_values",  spiluk_handle->get_nnzU());	
    */

    Kokkos::resize(tmp, N);
    Kokkos::resize(L_row_map, N + 1);
    Kokkos::resize(L_entries, spiluk_handle->get_nnzL());
    Kokkos::resize(L_values, spiluk_handle->get_nnzL());
    Kokkos::resize(U_row_map, N + 1);
    Kokkos::resize(U_entries, spiluk_handle->get_nnzU());
    Kokkos::resize(U_values, spiluk_handle->get_nnzU());

    // ILU(k) Symbolic phase
    std::cout << "Run ILU(k) symbolic phase ..." << std::endl;
    KokkosSparse::Experimental::spiluk_symbolic( &kh_spiluk, fill_lev, 
                                                 A.graph.row_map, A.graph.entries, 
                                                 L_row_map, L_entries, U_row_map, U_entries );
    // Resize L and U to their actual sizes
    Kokkos::resize(L_entries, spiluk_handle->get_nnzL());
    Kokkos::resize(L_values,  spiluk_handle->get_nnzL());
    Kokkos::resize(U_entries, spiluk_handle->get_nnzU());
    Kokkos::resize(U_values,  spiluk_handle->get_nnzU());

    std::cout << "L_row_map size = " << L_row_map.extent(0) << std::endl;
    std::cout << "L_entries size = " << L_entries.extent(0) << std::endl;
    std::cout << "L_values size  = " << L_values.extent(0)  << std::endl;
    std::cout << "U_row_map size = " << U_row_map.extent(0) << std::endl;
    std::cout << "U_entries size = " << U_entries.extent(0) << std::endl;
    std::cout << "U_values size  = " << U_values.extent(0)  << std::endl;
    std::cout << "ILU(k) fill_level: "   << fill_lev << std::endl;
    std::cout << "ILU(k) fill-factor: "  << (spiluk_handle->get_nnzL() + spiluk_handle->get_nnzU() - N)/(double)nnzA << std::endl;
    std::cout << "num levels: "          << spiluk_handle->get_num_levels() << std::endl;
    std::cout << "max num rows levels: " << spiluk_handle->get_level_maxrows() << std::endl << std::endl;

    // ILU(k) Numeric phase
    std::cout << "Run ILU(k) numeric phase ..." << std::endl;
    KokkosSparse::Experimental::spiluk_numeric( &kh_spiluk, fill_lev, 
                                                 A.graph.row_map, A.graph.entries, A.values, 
                                                 L_row_map, L_entries, L_values, U_row_map, U_entries, U_values );
    // Tri-solve Symbolic phase
    KokkosSparse::Experimental::sptrsv_symbolic( &kh_sptrsv_L, L_row_map, L_entries );
    KokkosSparse::Experimental::sptrsv_symbolic( &kh_sptrsv_U, U_row_map, U_entries );

    // Allocate vectors needed for CGSolve
     return success;

    }

    //template<typename ScalarType2>  //Tried ST2. Doesn't work b/c apply interface has all same ST. 
    // y = A*x
    void Apply (const MultiVec<ScalarType>& x,  MultiVec<ScalarType>& y,  ETrans trans=NOTRANS) const{
    //Note: Do NOT make x and y the same multivector!  You will get NaNs...
      // Note: spmv computes y = beta*y + alpha*Op(A)*x  spmv(mode,alpha,A,x,beta,y);
      //ScalarType alpha = 1.0;
      //ScalarType beta = 0;
      KokkosMultiVec<ScalarType, Device> *x_vec = dynamic_cast<KokkosMultiVec<ScalarType, Device> *>(&const_cast<MultiVec<ScalarType> &>(x));
      KokkosMultiVec<ScalarType, Device> *y_vec = dynamic_cast<KokkosMultiVec<ScalarType, Device> *>(&y);

      // Get a rank-1 subview of our rank-2 view, so don't fail asserts on sptrsv. 
      Kokkos::View<const ScalarType*, Kokkos::LayoutLeft, Device> xsub = Kokkos::subview(x_vec->GetInternalViewConst(), Kokkos::ALL, 0);
      Kokkos::View<ScalarType*, Kokkos::LayoutLeft, Device> ysub = Kokkos::subview(y_vec->GetInternalViewNonConst(), Kokkos::ALL, 0);

      // KokkosSparse::Experimental::sptrsv_solve(handle, rowmap, entries, values, b, x);
      // x = U\b, x = L\b, Ux=b, etc. 
      // TODO did I pass in correct vectors? for solve??
      // LUy = x by this layout
      // Uy = L\x = tmp
      // y = U\tmp
      KokkosSparse::Experimental::sptrsv_solve( &kh_sptrsv_L, L_row_map, L_entries, L_values, xsub, tmp); 
      KokkosSparse::Experimental::sptrsv_solve( &kh_sptrsv_U, U_row_map, U_entries, U_values, tmp, ysub);

    }

    bool HasApplyTranspose () const {
      return false;
    }

    ~KokkosILUOperator<ScalarType, OrdinalType, Device>(){}

  };
}// end namespace Belos
#endif
