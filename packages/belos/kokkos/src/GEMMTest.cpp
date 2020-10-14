#include<Kokkos_Core.hpp>
#include<KokkosBlas.hpp>
#include<Kokkos_Random.hpp>


typedef Kokkos::DefaultExecutionSpace     EXSP;

//Note: this has the same fct call as gemv, but only computes case beta=0, alpha=1. 
void gemv2(std::string trans,double alpha, Kokkos::View<double**>A, Kokkos::View<double*>x, double beta, Kokkos::View<double*> y ){
  typedef typename Kokkos::TeamPolicy<EXSP>::member_type member_type;
  typedef typename Kokkos::TeamPolicy<EXSP> team_policy;
  // Application: y  = beta*y + alpha*A*x
  int N = A.extent(0);
  int M = A.extent(1);

  if(trans == "N"){
  //Kokkos::parallel_for("Ax", N, KOKKOS_LAMBDA ( int i ){
  Kokkos::parallel_for("Ax NoTrans", team_policy(N, Kokkos::AUTO), KOKKOS_LAMBDA ( const member_type & teamMember ){
    int i = teamMember.league_rank();
    //for(int i = 0; i < N; i++ ) {
    //Kokkos::parallel_reduce( "dot", M, [=] ( int j, double &temp ) { 
    Kokkos::parallel_reduce( Kokkos::TeamThreadRange(teamMember, M), [=] ( int j, double &temp ) { 
        temp += A( i, j ) * x( j ); 
        }, y(i) );
    });
  }
  else if(trans == "T"){
  //TODO must switch indices for transpose version!!!
    /*for(int i = 0; i < M; i++ ) {
    Kokkos::parallel_reduce( "dot", N, KOKKOS_LAMBDA ( int j, double &temp ) { 
        temp += A( j, i ) * x( j ); 
        }, y(i) );
    }*/
  Kokkos::parallel_for("Ax Trans", team_policy(M, Kokkos::AUTO), KOKKOS_LAMBDA ( const member_type & teamMember ){
    int i = teamMember.league_rank();
    Kokkos::parallel_reduce( Kokkos::TeamThreadRange(teamMember, N), [=] ( int j, double &temp ) { 
        temp += A( j, i ) * x( j ); 
        }, y(i) );
    });
  }
  else{
  throw std::runtime_error("Invalide transpose argument in gemv2.");
  }
}

int main(int argc, char* argv[]) {
  Kokkos::initialize();
  {
     
    typedef double                            ST;
    //typedef Kokkos::CudaSpace                EXSP;
    typedef Kokkos::LayoutLeft                Layout;

    const int N = 2250000;
    const int M = 50;

    Kokkos::View<ST**,Layout,EXSP> A("A",N,M);
    //Kokkos::View<ST**,Layout,EXSP> Bgemm("Bgemm", N, 1);
    //Kokkos::View<ST**,Layout,EXSP> Cgemm("Cgemm", M, 1);
    Kokkos::View<ST*,Layout,EXSP> Bgemm("Bgemm", N);
    Kokkos::View<ST*,Layout,EXSP> Cgemm("Cgemm", M);

    for (int k = 1; k <= 200; k++){
      for (int i = 2; i <= 50; i++){
        Kokkos::View<ST**,Layout,EXSP> Asub = Kokkos::subview(A, Kokkos::ALL, Kokkos::make_pair(0,i));
        //Kokkos::View<ST**,Layout,EXSP> Cgemmsub = Kokkos::subview(Cgemm, Kokkos::make_pair(0,i), Kokkos::ALL);
        Kokkos::View<ST*,Layout,EXSP> Cgemmsub = Kokkos::subview(Cgemm, Kokkos::make_pair(0,i));
        for (int rep = 1; rep <= 2; rep++){
          //Try to avoid cache reuse by resetting A and B with a different value:
          Kokkos::deep_copy(Asub,i+1*rep);
          Kokkos::deep_copy(Bgemm,i+1*rep);
          //KokkosBlas::gemm("T","N",1.0,Asub,Bgemm,0.0,Cgemmsub);
          gemv2("T",1.0,Asub,Bgemm,0.0,Cgemmsub);
        }
      }
    }

    Kokkos::View<ST*,Layout,EXSP> Bgemv("Bgemv", M);
    Kokkos::View<ST*,Layout,EXSP> Cgemv("Cgemv", N);

    for (int k = 1; k <= 200; k++){
      for (int i = 2; i <= 50; i++){
        Kokkos::View<ST**,Layout,EXSP> Asub = Kokkos::subview(A, Kokkos::ALL, Kokkos::make_pair(0,i));
        Kokkos::View<ST*,Layout,EXSP> Bgemvsub = Kokkos::subview(Bgemv, Kokkos::make_pair(0,i));
        for (int rep = 1; rep <= 2; rep++){
          Kokkos::deep_copy(Asub,i+1*rep);
          Kokkos::deep_copy(Bgemvsub,i+1*rep);
       //   KokkosBlas::gemv("N",1.0,Asub,Bgemvsub,0.0,Cgemv); 
          gemv2("N",1.0,Asub,Bgemvsub,0.0,Cgemv); 
        }
      }
      //For the last two runs at the restart:
      for (int rep = 1; rep <= 2; rep++){
        Kokkos::deep_copy(A,rep);
        Kokkos::deep_copy(Bgemv,rep);
        //KokkosBlas::gemv("N",1.0,A,Bgemv,0.0,Cgemv); 
        gemv2("N",1.0,A,Bgemv,0.0,Cgemv); 
      }
    }

  }
  Kokkos::finalize();
}

