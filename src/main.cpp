#include <iostream>
#include <sstream>
#include "mat_mul.h"
#include "timing.h"
#include <raft>

unsigned x = 1, y = 4, z = 7, w = 13;

unsigned rand() {
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
  }

  unsigned* a = (unsigned *) malloc(n * m * sizeof(unsigned));
  unsigned* b = (unsigned *) malloc(m * o * sizeof(unsigned));
  unsigned* c = (unsigned *) malloc(o * p * sizeof(unsigned));
  unsigned* d = (unsigned *) malloc(p * q * sizeof(unsigned));

  fill<unsigned>(n,m,a, []{return rand() % 1000;});
  fill<unsigned>(m,o,b, []{return rand() % 1000;});
  fill<unsigned>(o,p,c, []{return rand() % 1000;});
  fill<unsigned>(p,q,d, []{return rand() % 1000;});
  Timing<true> t;

  t.s();
  unsigned* axb = cache_oblivious_mat_mul<unsigned>(n,m,o,a,b);
  unsigned* cxd = cache_oblivious_mat_mul<unsigned>(o,p,q,c,d);
  unsigned* ans = cache_oblivious_mat_mul<unsigned>(n,o,q,axb,cxd);
  t.e();

  unsigned* rns;

  COMatMul<unsigned> ab = COMatMul<unsigned>(n,m,o,a,b);
  COMatMul<unsigned> cd = COMatMul<unsigned>(o,p,q,c,d);
  COMatMul<unsigned> abcd = COMatMul<unsigned>(&rns);


  raft::map map;
  map += ab["axb"] >> abcd["a"];
  map += cd["axb"] >> abcd["b"];
  map += ab["n"] >> abcd["n"];
  map += ab["o"] >> abcd["m"];
  map += cd["o"] >> abcd["o"];

  Timing<true> r;
  r.s();
  map.exe();
  r.e();

  print_mat(n,q,ans);
  print_mat(n,q,rns);
  for_mat<unsigned>(n,q,[ans,rns,q](unsigned i, unsigned j){
    assert(ans[q * i + j] == rns[q * i + j]);
  });
  t.p("Normal Mat Mult");
  r.p("RaftLib Mat Mult");
  return 0;
}
