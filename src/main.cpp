#include <iostream>
#include <sstream>
#include "mat_mul.h"
#include "timing.h"
#include <raft>

unsigned x = 1, y = 4, z = 7, w = 13;

unsigned rando() {
  unsigned t = x;
  t ^= t << 11;
  t ^= t >> 8;
  x = y;
  y = z;
  z = w;
  w ^= w >> 19;
  w ^= t;
  return w;
}

int main(int argc, char** argv) {
  if(argc != 6)
  {
    std::cerr << "Exactly 6 arguments are needed" << std::endl;
    return -1;
  }
  std::istringstream sn (argv[1]);
  std::istringstream sm (argv[2]);
  std::istringstream so (argv[3]);
  std::istringstream sp (argv[4]);
  std::istringstream sq (argv[5]);
  unsigned n;
  unsigned m;
  unsigned o;
  unsigned p;
  unsigned q;
  if(!(sn >> n) || !(sm >> m) || !(so >> o) || !(sp >> p) || !(sq >> q))
  {
    std::cerr << "One of your arguments was malformed:" << std::endl;
    for(int i = 1; i < 6; i++) std::cerr << i << "\t" << argv[i] << std::endl;
    return -2;
  }

  unsigned* a = (unsigned *) malloc(n * m * sizeof(unsigned));
  unsigned* b = (unsigned *) malloc(m * o * sizeof(unsigned));
  unsigned* c = (unsigned *) malloc(o * p * sizeof(unsigned));
  unsigned* d = (unsigned *) malloc(p * q * sizeof(unsigned));

  fill<unsigned>(n,m,a, []{return rando() % 1000;});
  fill<unsigned>(m,o,b, []{return rando() % 1000;});
  fill<unsigned>(o,p,c, []{return rando() % 1000;});
  fill<unsigned>(p,q,d, []{return rando() % 1000;});

  Timing<true> t;

  t.s();
  unsigned* axb = cache_oblivious_mat_mul<unsigned>(n,m,o,a,b);
  unsigned* cxd = cache_oblivious_mat_mul<unsigned>(o,p,q,c,d);
  unsigned* ans = cache_oblivious_mat_mul<unsigned>(n,o,q,axb,cxd);
  t.e();

  t.p("N ZC MM");

  unsigned* rns;

  StartMatMul<unsigned> ab = StartMatMul<unsigned>(n,m,o,a,b);
  StartMatMul<unsigned> cd = StartMatMul<unsigned>(o,p,q,c,d);
  EndMatMul<unsigned> abcd = EndMatMul<unsigned>(&rns);


  raft::map map;
  map += ab["axb"] >> abcd["a"];
  map += cd["axb"] >> abcd["b"];
  map += ab["n"] >> abcd["n"];
  map += ab["o"] >> abcd["m0"];
  map += cd["n"] >> abcd["m1"];
  map += cd["o"] >> abcd["o"];

  Timing<true> r;
  r.s();
  map.exe();
  r.e();

  for_mat_inner(n,q,[ans,rns,q](unsigned i, unsigned j){
    assert(ans[q * i + j] == rns[q * i + j]);
  });

  r.p("RL MC MM");

#if 0
  unsigned* zns;
  StartMatMul_1Copy<unsigned> zab = StartMatMul_1Copy<unsigned>(n,m,o,a,b);
  StartMatMul_1Copy<unsigned> zcd = StartMatMul_1Copy<unsigned>(o,p,q,c,d);
  EndMatMul_1Copy<unsigned> zabcd = EndMatMul_1Copy<unsigned>(&zns);

  raft::map zap;
  zap += zab["axb"] >> zabcd["a"];
  zap += zcd["axb"] >> zabcd["b"];
  zap += zab["n"] >> zabcd["n"];
  zap += zab["o"] >> zabcd["m0"];
  zap += zcd["n"] >> zabcd["m1"];
  zap += zcd["o"] >> zabcd["o"];

  Timing<true> z;
  z.s();
  zap.exe();
  z.e();


  for_mat_inner(n,q,[ans,zns,q](unsigned i, unsigned j){
    assert(ans[q * i + j] == zns[q * i + j]);
  });

  z.p("RaftLib One Copy Mat Mult");
#endif

  return 0;
}
