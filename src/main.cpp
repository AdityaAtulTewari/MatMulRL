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
  std::cerr << "Usage:\t" << arg0 << " [-c -v -d -q[1-3]] n m o p q s" << std::endl;
}

enum ALLOC_TYPE
{
 STD_ALLOC,
 DYN_ALLOC,
 VTL_ALLOC
};

void parse_args(int argc, char** argv,
    unsigned* n, unsigned* m, unsigned* o, unsigned* p, unsigned* q, unsigned* s, bool* check, ALLOC_TYPE* at, bool* sched)
{
  int opt;
  char* arg0 = argv[0];
  auto us = [arg0] () {usage(arg0);};
  int helper;
  while((opt = getopt(argc, argv, "scvdq:")) != -1)
  {
    std::ostringstream num_hwpar;
    switch(opt)
    {
      case 'v' :
        *at = VTL_ALLOC;
        break;
      case 'd' :
        *at = DYN_ALLOC;
        break;
      case 's' :
        *at = STD_ALLOC;
        break;
      case 'q' :
        helper = atoi(optarg);
        if(1 > helper || 3 < helper)
        {
          std::cerr << "You failed to properly specify the number of qthreads" << std::endl;
          us();
          exit(-4);
        }
        helper++;
        if(setenv("QT_NUM_SHEPHERDS", "1", 1) ||
            !(num_hwpar << helper) ||
            setenv("QT_HWPAR", num_hwpar.str().c_str(), 1) ||
            setenv("QT_NUM_WORKERS_PER_SHEPHERD", num_hwpar.str().c_str(), 1))
        {
          std::cerr << "Setting environment variables failed" << std::endl;
          us();
          exit(-5);
        }
        *sched = true;
        break;

      case 'c' :
        *check = true;
        break;
    }
  }
  std::istringstream sarr[6];
  unsigned darr[6];
  unsigned i;
  for(i = 0; i < 6; i++)
  {
    if(optind + i == argc)
    {
      std::cerr << "You have too few unsigned int arguments: " << i << " out of 6" << std::endl;
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
  *s = darr[5];
}

int main(int argc, char** argv)
{
  unsigned n;
  unsigned m;
  unsigned o;
  unsigned p;
  unsigned q;
  unsigned s;
  bool check = false;
  bool sched = false;
  ALLOC_TYPE at = STD_ALLOC;
  parse_args(argc, argv, &n, &m, &o, &p, &q, &s, &check, &at, &sched);

  unsigned** a = (unsigned **) malloc(s * sizeof(unsigned*));
  unsigned** b = (unsigned **) malloc(s * sizeof(unsigned*));
  unsigned** c = (unsigned **) malloc(s * sizeof(unsigned*));
  unsigned** d = (unsigned **) malloc(s * sizeof(unsigned*));

  for(unsigned i = 0; i < s; i++)
  {
    a[i] = (unsigned*) malloc(n * m * sizeof(unsigned));
    b[i] = (unsigned*) malloc(m * o * sizeof(unsigned));
    c[i] = (unsigned*) malloc(o * p * sizeof(unsigned));
    d[i] = (unsigned*) malloc(p * q * sizeof(unsigned));
    fill<unsigned>(n,m,a[i], []{return rando() % 1000;});
    fill<unsigned>(m,o,b[i], []{return rando() % 1000;});
    fill<unsigned>(o,p,c[i], []{return rando() % 1000;});
    fill<unsigned>(p,q,d[i], []{return rando() % 1000;});
  }


  unsigned** rns = (unsigned**) malloc(s * sizeof(unsigned*));

  StartMatMul<unsigned> ab = StartMatMul<unsigned>(n,m,o,a,b,rns,s);
  StartMatMul<unsigned> cd = StartMatMul<unsigned>(o,p,q,c,d,rns,s);
  EndMatMul<unsigned> abcd = EndMatMul<unsigned>();


  raft::map map;
  map += ab["axb"] >> abcd["a"];
  map += cd["axb"] >> abcd["b"];
  map += ab["n"] >> abcd["n"];
  map += ab["o"] >> abcd["m0"];
  map += cd["n"] >> abcd["m1"];
  map += cd["o"] >> abcd["o"];
  map += ab["rns"] >> abcd["rns0"];
  map += cd["rns"] >> abcd["rns1"];

  Timing<true> r;
  r.s();
#ifdef VL
  if(at == VTL_ALLOC)
  {
#if USEQTHREADS
    if(sched) map.exe<partition_dummy,simple_schedule,vlalloc,no_parallel>();
    else
#endif /* USEQTHREADS */
    map.exe<partition_dummy,pool_schedule,vlalloc,no_parallel>();
  }
  else
#endif /* VL */

  if(at == DYN_ALLOC)
  {
#ifdef USEQTHREADS
    if(sched) map.exe<partition_dummy, pool_schedule, dynalloc, no_parallel>();
    else
#endif /* USEQTHREADS */
      map.exe<partition_dummy, simple_schedule, dynalloc, no_parallel>();
  }
  else
#ifdef USEQTHREADS
    if(sched) map.exe<partition_dummy, pool_schedule, stdalloc, no_parallel>();
    else
#endif /* USEQTHREADS */
    map.exe<partition_dummy, simple_schedule, stdalloc, no_parallel>();

  r.e();

  r.p("RL MC MM");

  if(check)
  {
    Timing<true> t;
    unsigned** ans = (unsigned**) malloc(s * sizeof(unsigned*));

    t.s();
    for(unsigned l = 0; l < s; l++)
    {
      unsigned* axb = cache_oblivious_mat_mul<unsigned>(n,m,o,a[l],b[l]);
      unsigned* cxd = cache_oblivious_mat_mul<unsigned>(o,p,q,c[l],d[l]);
      ans[l] = cache_oblivious_mat_mul<unsigned>(n,o,q,axb,cxd);
      free(axb);
      free(cxd);
    }
    t.e();

    t.p("SE ZC MM");
    for(unsigned l = 0; l < s; l++)
    {
      for_mat_inner(n,q,[ans,rns,q,l](unsigned i, unsigned j){
        assert(ans[l][q * i + j] == rns[l][q * i + j]);
      });
    }
  }

  return 0;
}
