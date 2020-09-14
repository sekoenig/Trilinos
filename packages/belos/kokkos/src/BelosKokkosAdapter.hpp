#include<Kokkos_Core.hpp>
#include<KokkosBlas.hpp>
#include<BelosTypes.hpp>
#include<Kokkos_Random.hpp>
#include<BelosMultiVec.hpp>
#include<BelosOperator.hpp>

#include<KokkosSparse_spmv.hpp>
#include <Teuchos_SerialDenseMatrix.hpp>


#ifndef BELOS_KOKKOS_ADAPTER_HPP
#define BELOS_KOKKOS_ADAPTER_HPP
 namespace Belos {
  //Forward class declaration of KokkosOperator:
  template<class ScalarType, class OrdinalType, class Device>
  class KokkosOperator; // : public Operator<ScalarType>;

  /// \class KokkosMultiVec
  /// \brief Implementation of Belos::MultiVec using Kokkos_MultiVector.
  ///
  /// Belos::MultiVec offers a simple abstract interface for
  /// multivector operations in Belos solver algorithms.  This class
  /// implements Belos::MultiVec by extending Kokkos_MultiVector.
  template<class ScalarType, class Device =Kokkos::DefaultExecutionSpace >
  class KokkosMultiVec : public MultiVec<ScalarType> {
    //Think it is okay for ScalarType to not match ScalarType2 b/c eventually we might want
    //MV and OP to have different precisions
 //   template<class ScalarType2, class OrdinalType, class Device>
 //   friend class KokkosOperator; 
  public:
    using size_type = typename Device::size_type;
    using ViewVectorType = Kokkos::View<ScalarType*,Kokkos::LayoutLeft, Device>;
    using ConstViewVectorType = Kokkos::View<const ScalarType*,Kokkos::LayoutLeft, Device>;
    using ViewMatrixType = Kokkos::View<ScalarType**,Kokkos::LayoutLeft, Device>;
    using ConstViewMatrixType = Kokkos::View<const ScalarType**,Kokkos::LayoutLeft, Device>;
    
    //Unmanaged view types: 
    using UMViewVectorType = Kokkos::View<ScalarType*,Kokkos::LayoutLeft, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UMHostViewVectorType = Kokkos::View<ScalarType*,Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UMConstViewVectorType = Kokkos::View<const ScalarType*,Kokkos::LayoutLeft, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UMHostConstViewVectorType = Kokkos::View<const ScalarType*,Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UMViewMatrixType = Kokkos::View<ScalarType**,Kokkos::LayoutLeft, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UMHostViewMatrixType = Kokkos::View<ScalarType**,Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UMConstViewMatrixType = Kokkos::View<const ScalarType**,Kokkos::LayoutLeft, Device, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
    using UMHostConstViewMatrixType = Kokkos::View<const ScalarType**,Kokkos::LayoutLeft, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

    static int multivecCount;
    static int resizeScratchCount;

    // constructors
    // TODO: Should the views in these first 3 constructors be initialized?  Or not?
    KokkosMultiVec<ScalarType, Device> (const std::string label, const int numrows, const int numvecs) :
      myView (label,numrows,numvecs),
      scratchView(Kokkos::ViewAllocateWithoutInitializing("SCRATCH"),0,0)
    { multivecCount++; }

    KokkosMultiVec<ScalarType, Device> (const int numrows, const int numvecs) :
      myView ("MV",numrows,numvecs),
      scratchView(Kokkos::ViewAllocateWithoutInitializing("SCRATCH"),0,0)
    { multivecCount++; }

    KokkosMultiVec<ScalarType, Device> (const int numrows) :
      myView("MV",numrows,1),
      scratchView(Kokkos::ViewAllocateWithoutInitializing("SCRATCH"),0,0)
    { multivecCount++; }

    // Make so that copy constructor of MV gives deep copy.  
    template < class ScalarType2 >
      KokkosMultiVec<ScalarType, Device> (const KokkosMultiVec<ScalarType2, Device> &sourceVec) : 
      myView(Kokkos::ViewAllocateWithoutInitializing("MV"),(int)sourceVec.GetGlobalLength(),sourceVec.GetNumberVecs()),
      scratchView(Kokkos::ViewAllocateWithoutInitializing("SCRATCH"),0,0) //New scratch space since deep copy.
    {
      Kokkos::deep_copy(myView,sourceVec.myView);
      multivecCount++;
    }

    //Need this explicitly, else compiler makes its own with shallow copy. 
    KokkosMultiVec<ScalarType, Device> (const KokkosMultiVec<ScalarType, Device> &sourceVec) : 
      myView(Kokkos::ViewAllocateWithoutInitializing("MV"),(int)sourceVec.GetGlobalLength(),sourceVec.GetNumberVecs()),
      scratchView(Kokkos::ViewAllocateWithoutInitializing("SCRATCH"),0,0) //New scratch space since deep copy.
    {
      Kokkos::deep_copy(myView,sourceVec.myView);
      multivecCount++;
    }

    //TODO: Do we need to implement this operator?  Like this??
    // Compiler default should give shallow copy.
    //KokkosMultiVec& operator=(const KokkosMultiVec& pv) { Kokkos_MultiVector::operator=(pv); return *this; }
    
    //TODO: Would it be better if this was deep copy?  So we can't change the user's original data?  
    // And so the user can't change ours?
    // But doing so would add a bunch of deep copy's to clone view... Maybe make it an option?  
    KokkosMultiVec<ScalarType, Device> (const ViewMatrixType & sourceView) : 
      myView(sourceView),
      scratchView(Kokkos::ViewAllocateWithoutInitializing("SCRATCH"),0,0)
    { multivecCount++; }
    
    // Here is a new version of the constructor that allows us to reuse an already existing scratch space.
    // Use this so that cloneViews can use same scratch space of original vector. 
    KokkosMultiVec<ScalarType, Device> (const ViewMatrixType & sourceView,
      ViewMatrixType & userScratchView ) : 
      myView(sourceView),
      //scratchView(Kokkos::ViewAllocateWithoutInitializing("SCRATCH"),0,0)
      scratchView(userScratchView)
    { multivecCount++; }


    // This version allows us to make exclusive changes to the view and convert between scalar types:
    template < class ScalarType2 > //TODO: Fix this so that passing in a view without device specified actually compiles...
    KokkosMultiVec<ScalarType, Device> (const Kokkos::View<ScalarType2**,Kokkos::LayoutLeft, Device> & sourceView) : 
      myView(Kokkos::ViewAllocateWithoutInitializing("MV"),sourceView.extent(0),sourceView.extent(1)),
      scratchView(Kokkos::ViewAllocateWithoutInitializing("SCRATCH"),0,0)
    {
      Kokkos::deep_copy(myView,sourceView);
      multivecCount++;
    }

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
      //KokkosMultiVec<ScalarType>  temp("MV",this->extent(0),this->extent(1));
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
      //KokkosMultiVec<ScalarType> B("B", this->extent(0), numvecs);
      bool isAscending = true;//Also checks if is contiguous.
      for(unsigned int i=0; i< (index.size()-1); i++){
        if( index[i+1] != index[i]+1 ){
          isAscending = false;
        }
      }
      if(debug && isAscending == false){ std::cout << "isAscending = false" << std::endl; }
      if(isAscending && index.size()==(unsigned)this->GetNumberVecs()){ //Copy entire multivec.
        Kokkos::deep_copy(B->myView,myView);
      //  if(debug){ std::cout << "copy entire MV" << std::endl; } //TODO remove debug stuff
      }
      else if (isAscending){ //Copy contiguous subset
        ViewMatrixType ThisSub = Kokkos::subview(myView, Kokkos::ALL, std::make_pair(index.front(), index.back()+1));
        Kokkos::deep_copy(B->myView,ThisSub);
     //   if(debug){ std::cout << "copy contiguous subset of  MV" << std::endl; }
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
          new KokkosMultiVec<ScalarType, Device>(Kokkos::subview(myView, Kokkos::ALL, std::make_pair(index.front(), index.back()+1)),
            scratchView);
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
          new KokkosMultiVec<ScalarType, Device>(Kokkos::subview(myView, Kokkos::ALL, std::make_pair(index.front(), index.back()+1)),
            scratchView);
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
      UMHostConstViewMatrixType mat_h(B.values(), A_vec->myView.extent(1), myView.extent(1));
      ViewMatrixType mat_d(Kokkos::ViewAllocateWithoutInitializing("mat"), A_vec->myView.extent(1), myView.extent(1));
      Kokkos::deep_copy(mat_d, mat_h);

      //UMConstViewMatrixType mat(B.values(), A_vec->myView.extent(1), myView.extent(1));
      //Teuchos2KokkosMat(B,mat);
      if( myView.extent(1) == 1 ){ // Only 1 col
        ConstViewVectorType Bsub = Kokkos::subview(mat_d, Kokkos::ALL, 0);
        ViewVectorType mysub = Kokkos::subview(myView, Kokkos::ALL, 0);
        KokkosBlas::gemv("N", alpha, A_vec->myView, Bsub, beta, mysub);
      //  std::cout << "A has size: " << A_vec->myView.extent(0) << " by " << A_vec->myView.extent(1) << std::endl;
      }
      else{
        KokkosBlas::gemm("N", "N", alpha, A_vec->myView, mat_d, beta, myView);
      }
    }

    //! *this <- alpha * A + beta * B
    void MvAddMv ( const ScalarType alpha, const MultiVec<ScalarType>& A, const ScalarType beta,
                   const MultiVec<ScalarType>& B){
      KokkosMultiVec<ScalarType, Device> *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(A));
      KokkosMultiVec<ScalarType, Device> *B_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(B));
      //ViewMatrixType temp(Kokkos::ViewAllocateWithoutInitializing("tmp"), myView.extent(0), myView.extent(1));
      //KokkosMultiVec<ScalarType> temp(myView.extent(0),myView.extent(1));
        //Kokkos::Timer timeCopy1;
      //Kokkos::deep_copy(temp, B_vec->myView);
        //Kokkos::fence();
        //double time1 = timeCopy1.seconds();
        //std::cout << "Seconds for Copy1: " << time1 << std::endl;        
      //Kokkos::deep_copy(myView, B_vec->myView); // BAD!  Assumes myVec != A.  
      //KokkosBlas::axpby(alpha, A_vec->myView, beta, temp);

      // myView = alpha A + beta B + 0 * myView
      KokkosBlas::update(alpha, A_vec->myView, beta, B_vec->myView, (ScalarType) 0.0, myView);
        //Kokkos::Timer timeCopy2;
      //Kokkos::deep_copy(myView,temp);
        //Kokkos::fence();
        //double time2 = timeCopy2.seconds();
        //std::cout << "Seconds for Copy2: " << time2 << std::endl;        
    }

    //! Scale each element of the vectors in \c *this with \c alpha.
    void MvScale ( const ScalarType alpha ) {
      //Later- Can we do this better with less copying?  TODO
      //ViewMatrixType temp(Kokkos::ViewAllocateWithoutInitializing("tmp"), myView.extent(0), myView.extent(1));
      //KokkosBlas::scal(temp, alpha, myView); 
      KokkosBlas::scal(myView, alpha, myView); 
      //Kokkos::deep_copy(myView, temp);
    }

    //! Scale each element of the \c i-th vector in \c *this with \c alpha[i].
    void MvScale ( const std::vector<ScalarType>& alpha ){
      //Later- Can we do this better with less copying?  TODO
      //ViewMatrixType temp(Kokkos::ViewAllocateWithoutInitializing("tmp"), myView.extent(0), myView.extent(1));
      //ViewVectorType scalars(Kokkos::ViewAllocateWithoutInitializing("alpha"), alpha.size());
      
      UMHostConstViewVectorType scalars_h(alpha.data(), alpha.size());
      ViewVectorType scalars_d(Kokkos::ViewAllocateWithoutInitializing("scalars_d"), alpha.size());
      Kokkos::deep_copy(scalars_d, scalars_h);

      //UMConstViewVectorType scalars(alpha.data(), alpha.size());
      //for(unsigned int i = 0 ; i < alpha.size(); i++){
        //scalars(i) = alpha.at(i);
      //} 
      //KokkosBlas::scal(temp, scalars, myView); 
      KokkosBlas::scal(myView, scalars_d, myView); 
      //Kokkos::deep_copy(myView, temp);
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
        //ViewMatrixType soln(Kokkos::ViewAllocateWithoutInitializing("soln"), A_vec->myView.extent(1), myView.extent(1));
        //UMViewMatrixType soln(B.values(), A_vec->myView.extent(1), myView.extent(1));
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
        //Kokkos2TeuchosMat(soln, B);
      }
    }


    //! b[i] = A[i]^T * this[i]
    void MvDot ( const MultiVec<ScalarType>& A, std::vector<ScalarType>& b ) const{
      UMHostViewVectorType dotView_h(b.data(),myView.extent(1)); //Make an unmanaged view of b. 
      ViewVectorType dotView_d(Kokkos::ViewAllocateWithoutInitializing("Dot"),myView.extent(1));
      //ViewVectorType dotView = getScratchView(myView.extent(1));
      //std::cerr << "Now in Dot.  dotView extent(0) is: " << dotView.extent(0) << " extent(1) is: " << dotView.extent(1) << std::endl;
      KokkosMultiVec<ScalarType, Device> *A_vec = dynamic_cast<KokkosMultiVec *>(&const_cast<MultiVec<ScalarType> &>(A));
      KokkosBlas::dot(dotView_d, A_vec->myView, myView); //TODO check- it should be A that is conjugate transposed, not mv.  Is it??
      Kokkos::deep_copy(dotView_h, dotView_d);
      //for(unsigned int i=0; i < myView.extent(1); i++){
        //b[i] = dotView(i); //Is there a better way to do this?
      //}
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
      //for(unsigned int i=0; i < myView.extent(1); i++){
        //normvec[i] = normView(i); 
        //TODO: will probably have to mirror the normView to the host space. 
      //}
    }

    //! Fill all columns of *this with random values.
    void MvRandom() {
      int rand_seed = std::rand();
      Kokkos::Random_XorShift64_Pool<> pool(rand_seed); //initially used seed 12371
      Kokkos::fill_random(myView, pool, -1,1);
      //TODO: check that this random matches the scalar type...
     // rand_seed += 5;
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

  //private: //This var should be private, but I can't get friend class templaty stuff to work, so...
    //static int rand_seed; //C++ initializes to zero.  Need this so new MVs in program will have different 
                          // results with MVRandom. 
    ViewMatrixType myView;
    bool debug = true; 
 private:
    const int SCRATCH_SIZE = 60; 
    mutable ViewMatrixType scratchView;

    ViewMatrixType getScratchView( size_type m, size_type n ) const {
      if( scratchView.extent(0) < m || scratchView.extent(1) < n ){
        size_type tmpSize = scratchView.extent(0);
        while ( tmpSize < m || tmpSize < n ){
          tmpSize += 50; 
        }
        Kokkos::resize(scratchView, tmpSize, tmpSize);
        resizeScratchCount++;
//        std::cerr << "WARNING: Kokkos scratchView has been resized!!" << std::endl 
//            << "Requested size is m = " << m << "  n = " << n << std::endl;
      }
      return Kokkos::subview(scratchView, std::make_pair(0,(int) m), std::make_pair(0,(int) n));  
    }

    ViewVectorType getScratchView( size_type m ) const {
      if( scratchView.extent(0) < m ){
        size_type tmpSize = scratchView.extent(0);
        while ( tmpSize < m ){
          tmpSize += 50; 
        }
        Kokkos::resize(scratchView, tmpSize, tmpSize);
        resizeScratchCount++;
//        std::cerr << "WARNING: Kokkos scratchView has been resized!!" << std::endl 
//            << "Requested size is m = " << m << std::endl;
      }
      return Kokkos::subview(scratchView, std::make_pair(0,(int) m), 0);   //TODO: More effective to put the null in the first dimension?? 
    }
    
    
    void Kokkos2TeuchosMat(const ViewMatrixType & K,  Teuchos::SerialDenseMatrix<int, ScalarType> &T) const {
      TEUCHOS_TEST_FOR_EXCEPTION(K.extent(0) != (unsigned)T.numRows() || K.extent(1) != (unsigned)T.numCols(), std::runtime_error, "Error: Matrix dimensions do not match!");
  //This is all on host, so there's no use trying to use parallel_for, right?... Well, host could have openMP... TODO improve this?
      for(unsigned int i = 0; i < K.extent(0); i++){
        for (unsigned int j = 0; j < K.extent(1); j++){
          T(i,j) = K(i,j);
        }
      } 
    }

    void Teuchos2KokkosMat(const Teuchos::SerialDenseMatrix<int, ScalarType> &T, ViewMatrixType & K) const {
      TEUCHOS_TEST_FOR_EXCEPTION(K.extent(0) != (unsigned)T.numRows() || K.extent(1) != (unsigned)T.numCols(), std::runtime_error, "Error: Matrix dimensions do not match!");
      //This is all on host, so there's no use trying to use parallel_for, right?... Well, host could have openMP... TODO improve this?
      for(unsigned int i = 0; i < K.extent(0); i++){
        for (unsigned int j = 0; j < K.extent(1); j++){
          K(i,j) = T(i,j);
        }
      } 
    }

  };

  //Must include a templated definition of our static variable to compile:
  template<class ScalarType, class Device> 
  int KokkosMultiVec< ScalarType, Device >::multivecCount = 0;

  template<class ScalarType, class Device> 
  int KokkosMultiVec< ScalarType, Device >::resizeScratchCount = 0;


  /// \class KokkosOperator
  /// \brief Implementation of Belos::Operator using Kokkos::Crs matrix.
  ///
  template<class ScalarType, class OrdinalType=int, class Device=Kokkos::DefaultExecutionSpace>
  class KokkosOperator : public Operator<ScalarType> {
  private:
    KokkosSparse::CrsMatrix<ScalarType, OrdinalType, Device> myMatrix;

  public:
  // Shallow copy for mat of same scalar type:
    KokkosOperator<ScalarType, OrdinalType, Device> (const KokkosSparse::CrsMatrix<ScalarType, OrdinalType, Device> mat) 
     : myMatrix(mat) {}

  //This doesn't work yet!! That is view notation!
  // Deep copy for changing scalar types: 
  //  template <class ScalarType2 > 
  //  KokkosOperator<ScalarType, OrdinalType, Device> (const KokkosSparse::CrsMatrix<ScalarType2, OrdinalType, Device> mat) 
  //   : myMatrix(Kokkos::ViewAllocateWithoutInitializing("Mat"),mat.extent(0),mat.extent(1)) {Kokkos::deep_copy(myMatrix,mat);}

    //template<typename ScalarType2>  //Tried ST2. Doesn't work b/c apply interface has all same ST. 
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
      KokkosMultiVec<ScalarType, Device> *x_vec = dynamic_cast<KokkosMultiVec<ScalarType, Device> *>(&const_cast<MultiVec<ScalarType> &>(x));
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
