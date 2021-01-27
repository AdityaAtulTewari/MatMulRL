#include <iostream>
#include <sstream>
#include "mat_mul.h"
#include "timing.h"
#include <raft>
#include <getopt.h>

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

void usage(char* arg0)
{
  std::cerr << "Usage:\t" << arg0 << " [-c -v -d] n m o p q" << std::endl;
}

enum ALLOC_TYPE
{
 STD_ALLOC,
 DYN_ALLOC,
 VTL_ALLOC
};

void parse_args(int argc, char** argv, unsigned* n, unsigned* m, unsigned* o, unsigned* p, unsigned* q, bool* check, ALLOC_TYPE* at, bool* sched)
{
  int opt;
  char* arg0 = argv[0];
  auto us = [arg0] () {usage(arg0);};
  while((opt = getopt(argc, argv, "cvdq")) != -1)
  {
    switch(opt)
    {
      case 'v' :
        *at = VTL_ALLOC;
        break;
      case 'd' :
        *at = DYN_ALLOC;
        break;
      case 'q' :
        *sched = true;
        break;
      case 'c' :
        *check = true;
        break;
    }
  }
  std::istringstream sarr[5];
  unsigned darr[5];
  unsigned i;
  for(i = 0; i < 5; i++)
  {
    if(optind + i == argc)
    {
      std::cerr << "You have too few unsigned int arguments: " << i << " out of 5" << std::endl;
      us();
      exit(-3);
    }
    sarr[i] = std::istringstream(argv[optind + i]);
    if(!(sarr[i] >> darr[i]))
    {
      std::cerr << "Your argument at " << optind + i << " was malformed." << std::endl;
      std::cerr << "It should have been an unsigned int" << std::endl;
      us();
      exit(-2);
    }
  }
  if(i + optind != argc)
  {
    std::cerr << "You have too many arguments." << std::endl;
    us();
    exit(-1);
  }

  *n = darr[0];
  *m = darr[1];
  *o = darr[2];
  *p = darr[3];
  *q = darr[4];
}

int main(int argc, char** argv)
{
  unsigned n;
  unsigned m;
  unsigned o;
  unsigned p;
  unsigned q;
  bool check = false;
  bool sched = false;
  ALLOC_TYPE at = STD_ALLOC;
  parse_args(argc, argv, &n, &m, &o, &p, &q, &check, &at, &sched);

  unsigned* a = (unsigned *) malloc(n * m * sizeof(unsigned));
  unsigned* b = (unsigned *) malloc(m * o * sizeof(unsigned));
  unsigned* c = (unsigned *) malloc(o * p * sizeof(unsigned));
  unsigned* d = (unsigned *) malloc(p * q * sizeof(unsigned));

  fill<unsigned>(n,m,a, []{return rando() % 1000;});
  fill<unsigned>(m,o,b, []{return rando() % 1000;});
  fill<unsigned>(o,p,c, []{return rando() % 1000;});
  fill<unsigned>(p,q,d, []{return rando() % 1000;});


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
#ifdef VL
  if(at == VTL_ALLOC)
  {
#ifdef QT
    if(sched) map.exe<partition_dummy,simple_schedule,vlalloc,no_parallel>();
    else
#endif
    map.exe<partition_dummy,pool_schedule,vlalloc,no_parallel>();
  }
  else
#endif

  if(at == DYN_ALLOC)
  {
#ifdef QT
    if(sched) map.exe<partition_dummy, pool_schedule, dynalloc, no_parallel>();
    else
#endif
      map.exe<partition_dummy, simple_schedule, dynalloc, no_parallel>();
  }
  else
#ifdef QT
    if(sched) map.exe<partition_dummy, pool_schedule, stdalloc, no_parallel>();
    else
#endif
    map.exe<partition_dummy, simple_schedule, stdalloc, no_parallel>();

  r.e();

  r.p("RL MC MM");

  if(check)
  {
    Timing<true> t;

    t.s();
    unsigned* axb = cache_oblivious_mat_mul<unsigned>(n,m,o,a,b);
    unsigned* cxd = cache_oblivious_mat_mul<unsigned>(o,p,q,c,d);
    unsigned* ans = cache_oblivious_mat_mul<unsigned>(n,o,q,axb,cxd);
    t.e();

    t.p("SE ZC MM");
    if(check) for_mat_inner(n,q,[ans,rns,q](unsigned i, unsigned j){
      assert(ans[q * i + j] == rns[q * i + j]);
    });
  }


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

