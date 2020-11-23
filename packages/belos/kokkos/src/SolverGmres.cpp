#include"KokkosKernels_IOUtils.hpp"
#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>
#include<KokkosBlas.hpp>
#include<KokkosBlas3_trsm.hpp>
#include<KokkosSparse_spmv.hpp>


template<typename exec_space, typename lno_t, typename Matrix>
void mgsQR(Matrix Q, Matrix R);

int main(int argc, char *argv[]) {

  typedef double                             ST;
  typedef int                               OT;
  typedef Kokkos::DefaultExecutionSpace     EXSP;

  Kokkos::initialize();
  {//TODO: Should these really be layout left?
  using ViewVectorType = Kokkos::View<ST*,Kokkos::LayoutLeft, EXSP>;
  using ViewMatrixType = Kokkos::View<ST**,Kokkos::LayoutLeft, EXSP>; 

  std::string filename("Laplace3D100.mtx"); // example matrix
  //std::string filename("Identity50.mtx"); // example matrix
  bool converged = false;
  int m = 50; //Max subspace size.
  double convTol = 1e-10; //Keep in double.
  int cycLim = 100;
  int cycle = 0;
  
  // Read in a matrix Market file and use it to test the Kokkos Operator.
  KokkosSparse::CrsMatrix<ST, OT, EXSP> A = 
    KokkosKernels::Impl::read_kokkos_crst_matrix<KokkosSparse::CrsMatrix<ST, OT, EXSP>>(filename.c_str()); 

  int n = A.numRows();
  ViewVectorType x("x",n); //Should init to zeros, right?
  ViewVectorType b(Kokkos::ViewAllocateWithoutInitializing("b"),n);
  ViewVectorType r(Kokkos::ViewAllocateWithoutInitializing("r"),n);
  ViewVectorType wj(Kokkos::ViewAllocateWithoutInitializing("w_j"),n);
  ViewVectorType tmpVec(Kokkos::ViewAllocateWithoutInitializing("tmpVec"),n);
  ViewVectorType lsVec("lsVec",m+1);
  ViewVectorType::HostMirror lsVec_h = Kokkos::create_mirror_view(lsVec);
  ViewMatrixType lsTmp("lsTmp",m,1);

  //ViewMatrixType H("H",m+1,m);
  ViewMatrixType Q("Q",m+1,m);
  ViewMatrixType::HostMirror H = Kokkos::create_mirror_view(Q); //Make H into a host view of Q. 
  ViewMatrixType R("R",m,m);
  ViewMatrixType V(Kokkos::ViewAllocateWithoutInitializing("V"),n,m+1);
  ViewMatrixType VSub = Kokkos::subview(V,Kokkos::ALL,Kokkos::make_pair(0,m)); //Subview of 1st m cols for updating soln.

  ST tmpScalar;

  double trueRes; //Keep this in double regardless so we know how small error gets.


  // Make rhs random.
  /*int rand_seed = std::rand();
  Kokkos::Random_XorShift64_Pool<> pool(rand_seed); //initially used seed 12371
  Kokkos::fill_random(b, pool, -1,1);*/

  // Make rhs ones to replicate:
  Kokkos::deep_copy(b,1.0);
  Kokkos::deep_copy(r,b);

  KokkosSparse::spmv("N", 1.0, A, x, 0.0, wj); // wj = Ax
  KokkosBlas::axpy(-1.0, wj, r); // r = b-Ax. //TODO do we really need to store r separately?
  trueRes = KokkosBlas::nrm2(r);
  std::cout << "Initial trueRes is : " << trueRes << std::endl;
    
  while( trueRes > convTol && cycle < cycLim){
    lsVec_h(0) = trueRes;

    //DEBUG: Print lsVec (rhs of ls prob)
    //std::cout << "lsVec elements: " << std::endl;
    //for (int i1 = 0; i1 < m+1; i1++){ std::cout << lsVec_h(i1) << " " ; }

    Kokkos::deep_copy(lsVec, lsVec_h);
    auto V0 = Kokkos::subview(V,Kokkos::ALL,0);
    Kokkos::deep_copy(V0,r);
    KokkosBlas::scal(V0,1.0/trueRes,V0); //V0 = V0/norm(V0)

    //Might need to move v0 normalize to here??
    //
    // Run Arnoldi iteration:

    // DEBUG: Print elts of H:
    /*for (int i1 = 0; i1 < m+1; i1++){
      for (int j1 = 0; j1 < m; j1++){
        std::cout << H(i1,j1);
      }
      std::cout << std::endl;
    }*/
    for (int j = 0; j < m; j++){
      auto Vj = Kokkos::subview(V,Kokkos::ALL,j); //TODO Could skip this one and use the v0 earlier and vj at end??
      KokkosSparse::spmv("N", 1.0, A, Vj, 0.0, wj); //wj = A*Vj
      // Think this is MGS ortho, but 1 vector at a time?
      for (int i = 0; i <= j; i++){
        auto Vi = Kokkos::subview(V,Kokkos::ALL,i); 
        H(i,j) = KokkosBlas::dot(Vi,wj);  //Host or device //TODO is this the right order for cmplx dot product?
        KokkosBlas::axpy(-H(i,j),Vi,wj);//wj = wj-Hij*Vi //Host
      }
      //Re-orthog:
/*      for (int i = 0; i <= j; i++){
        auto Vi = Kokkos::subview(V,Kokkos::ALL,i); 
        tmpScalar = KokkosBlas::dot(Vi,wj);
        KokkosBlas::axpy(-tmpScalar,Vi,wj);//wj = wj-tmpScalar*Vi
        H(i,j) = H(i,j) + tmpScalar; //Host
        KokkosBlas::scal(tmpVec,H(i,j),Vi);//tmpVec = H(i,j)*Vi //Host
      }*/
      
      //auto Hlast = Kokkos::subview(H,j+1,j);//TODO is this the right subview?? How does that indexing work?
      H(j+1,j) = KokkosBlas::nrm2(wj); //Host or device
      //std::cout << "Hlast is " << H(j+1,j) << std::endl;
      //bool myBool = H(j+1,j)<1e-14;
      //std::cout << "test bool is: " << myBool << std::endl;
      if(H(j+1,j) < 1e-14){ //Host
        //std::cout << "In the breakdonw if statement! " << std::endl;
        throw std::runtime_error("Lucky breakdown");
      }
      Vj = Kokkos::subview(V,Kokkos::ALL,j+1); 
      KokkosBlas::scal(Vj,1.0/H(j+1,j),wj); //Host or maybe device?

    }//end Arnoldi iter.

    //DEBUG: Check orthogonality of V:
    /*ViewMatrixType Vsm("Vsm", m+1, m+1);
    KokkosBlas::gemm("T","N", 1.0, V, V, 0.0, Vsm);
    ViewVectorType nrmV("nrmV",m+1);
    KokkosBlas::nrm2(nrmV, Vsm);
    std::cout << "Norm of V^T V: " << std::endl;
    ViewVectorType::HostMirror nrmV_h = Kokkos::create_mirror_view(nrmV); 
    Kokkos::deep_copy(nrmV_h, nrmV);
    //for (int i1 = 0; i1 < m+1; i1++){ std::cout << nrmV_h(i1) << " " ; } */



    //Compute least squares soln:
    Kokkos::deep_copy(Q,H); //TODO Do we really need a copy, or can we reuse H? //copies to something on device....
    //Yes this ^^ is needed, now we made H a mirror view.  

    /*//DEBUG: Check Arn Rec AV=VH
    ViewMatrixType AV("AV", n, m);
    ViewMatrixType VH("VH", n, m);
    KokkosSparse::spmv("N", 1.0, A, VSub, 0.0, AV); 
    KokkosBlas::gemm("N","N", 1.0, V, Q, 0.0, VH);
    KokkosBlas::axpy(-1.0, AV, VH); //VH = VH-AV
    ViewVectorType nrmARec("ARNrm", m);
    ViewVectorType::HostMirror nrmARec_h = Kokkos::create_mirror_view(nrmARec); 
    KokkosBlas::nrm2( nrmARec, VH);
    Kokkos::deep_copy(nrmARec_h, nrmARec);
    std::cout << "ArnRec norm check: " << std::endl;
    for (int i1 = 0; i1 < m; i1++){ std::cout << nrmARec_h(i1) << " " ; }
    std::cout << std::endl; */

    //Compute QR factorization:
    mgsQR<EXSP, int, ViewMatrixType> (Q,R);
    //DEBUG: Check QR = H
    //ViewMatrixType C("C",m+1,m); //To test Q*R=H
    //ViewMatrixType C2("C",m+1,m); //To test Q*R=H
    //Kokkos::deep_copy(C2,H); //Need H on device to compare
    /*KokkosBlas::gemm("N","N", 1.0, Q, R, 0.0, C);
    KokkosBlas::axpy(-1.0,C2,C); //C = C-H
    ViewVectorType nrmC("nrmC",m);
    ViewVectorType::HostMirror nrmC_h = Kokkos::create_mirror_view(nrmC); //Make H into a host view of Q. 
    KokkosBlas::nrm2(nrmC, C);
    Kokkos::deep_copy(nrmC_h, nrmC);
    for (int i1 = 0; i1 < m; i1++){ std::cout << nrmC_h(i1) << " " ; }
    std::cout << std::endl;*/

    auto lsTmpSub = Kokkos::subview(lsTmp,Kokkos::ALL,0); //Original view has rank 2, need a rank 1 here. 
    KokkosBlas::gemv("T",1.0,Q,lsVec,0.0,lsTmpSub); 
    KokkosBlas::trsm("L", "U", "N", "N", 1.0, R, lsTmp);

    //DEBUG: Check Short residual norm
    ViewMatrixType C3("C",m+1,m); //To test Q*R=H
    Kokkos::deep_copy(C3,H); //Need H on device to compare
    ViewVectorType lsVecCpy("lsVecCpy",m+1);
    Kokkos::deep_copy(lsVecCpy,lsVec);
    KokkosBlas::gemv("N",-1.0,C3,lsTmpSub,1.0,lsVecCpy); 
    ST shortRes = KokkosBlas::nrm2(lsVecCpy);
    std::cout << "Short residual is: " << shortRes << std::endl;

    //Update long solution and residual:
    KokkosBlas::gemv ("N", 1.0, VSub, lsTmpSub, 1.0, x); //x = x + V(1:m)lsSoln

    //TODO Could avoid repeating this with a do-while loop?
    KokkosSparse::spmv("N", 1.0, A, x, 0.0, wj); // wj = Ax
    Kokkos::deep_copy(r,b); // Reset r=b.
    KokkosBlas::axpy(-1.0, wj, r); // r = b-Ax. //TODO do we really need to store r separately?
    trueRes = KokkosBlas::nrm2(r);
    std::cout << "Next trueRes is : " << trueRes << std::endl;

    cycle++;
  }

  std::cout << "Ending residual is: " << trueRes << std::endl;
  if( trueRes < convTol )
    std::cout << "Solver converged! " << std::endl;
  else
    std::cout << "Solver did not converge. :( " << std::endl;
  std::cout << "Number of cycles completed is " << cycle << std::endl;
  std::cout << "which corresponds to " << cycle*m << " iterations." << std::endl;

  }
  Kokkos::finalize();

}


template<typename exec_space, typename lno_t, typename Matrix>
void mgsQR(Matrix Q, Matrix R)
{
  lno_t k = Q.extent(1);
  //Set R = I(k)
  auto Rhost = Kokkos::create_mirror_view(R);
  for(lno_t i = 0; i < k; i++)
  {
    for(lno_t j = 0; j < k; j++)
      Rhost(i, j) = 0;
    Rhost(i, i) = 1;
  }
  Kokkos::deep_copy(R, Rhost);
  for(lno_t i = 0; i < k; i++)
  {
    auto QcolI = Kokkos::subview(Q, Kokkos::ALL(), i);
    //normalize column i
    double colNorm = KokkosBlas::nrm2(QcolI);
    KokkosBlas::scal(QcolI, 1.0 / colNorm, QcolI);
    //scale up R row i by inorm
    auto RrowI = Kokkos::subview(R, i, Kokkos::ALL());
    KokkosBlas::scal(RrowI, colNorm, RrowI);
    for(lno_t j = i + 1; j < k; j++)
    {
      auto QcolJ = Kokkos::subview(Q, Kokkos::ALL(), j);
      auto RrowJ = Kokkos::subview(R, j, Kokkos::ALL());
      //orthogonalize QcolJ against QcolI
      double d = KokkosBlas::dot(QcolI, QcolJ);
      KokkosBlas::axpby(-d, QcolI, 1, QcolJ);
      KokkosBlas::axpby(d, RrowJ, 1, RrowI);
    }
  }
}
