//
// Created by Aditya Tewari on 1/20/21.
//

#ifndef MATMUL_MAT_MUL_H
#define MATMUL_MAT_MUL_H


template <typename P, typename Q, typename F>
void for_mat(unsigned n, unsigned m, F f, Q post, P pre)
{
  for(unsigned i = 0; i < n; i++)
  {
    pre(i);
    for(unsigned j = 0; j < m; j++) f(i,j);
    post(i);
  }
}

template <typename F>
void for_mat_inner(unsigned n, unsigned m, F f)
{
  for_mat(n,m,f,[](unsigned i){(void) i;},[](unsigned i){(void) i;});
}

template <typename T, typename F>
void fill(unsigned n, unsigned m, T* mat, F f)
{
  for_mat_inner(n,m,[mat,m,f] (unsigned i, unsigned j)
  {
    mat[i * m + j] = f();
  });
}

template <typename F, typename Q>
void for_mat_post(unsigned n, unsigned m, F f, Q post)
{
  for_mat(n,m,f,post,[](unsigned i){(void) i;});
}

template <typename T>
void print_mat(unsigned n, unsigned q, T* ans)
{
  auto in = [ans, q](unsigned i, unsigned j)
  {
    std::cerr << ans[i * q + j];
  };
  auto post = [](unsigned i)
  {
    (void) i;
    std::cerr << std::endl;
  };
  for_mat_post<T>(n,q,in,post);
}

template<typename T>
int cache_oblivious_mul(T* ans, unsigned n, unsigned m, unsigned o, T* nxm, T* mxo)
{
  for_mat_inner(n,o,[ans,nxm,mxo,m,o](unsigned i, unsigned j)
  {
    ans[i*o + j] = 0;
    for(int k = 0; k < m; k++)
    {
      ans[i*o + j] += nxm[i*m + k] * mxo[k*o + j];
    }
  });
  return 0;
}

template<typename T>
T* cache_oblivious_mat_mul(unsigned n, unsigned m, unsigned o, T* nxm, T* mxo)
{
  T* ans = (T*) malloc(n * o * sizeof(T));
  cache_oblivious_mul<T>(ans,n,m,o,nxm,mxo);
  return ans;
}

#include <raft>

template<typename T>
class StartMatMul : public raft::kernel
{
private:
  unsigned n,m,o;
  T** a;
  T** b;
  T** rns;
  unsigned s;
public:
  StartMatMul(unsigned n, unsigned m, unsigned o, T** a, T** b, T** rns, unsigned s) : raft::kernel(),n(n),m(m),o(o),a(a),b(b),rns(rns),s(s)
  {
    output.addPort<unsigned>("n");
    output.addPort<unsigned>("o");
    output.addPort<T>("axb");
    output.addPort<uintptr_t>("rns");
  }
  raft::kstatus run() override
  {
    for(unsigned l = 0; l < s; l++)
    {
      output["n"].push(n);
      output["o"].push(o);
      output["rns"].push((uintptr_t)&rns[l]);


      T ans;
      for_mat_inner(n,o,[&](unsigned i, unsigned j)
      {
        ans = 0;
        for(int k = 0; k < m; k++)
        {
          ans += a[l][i*m + k] * b[l][k*o + j];
        }
        output["axb"].push(ans);
      });
    }
    return raft::stop;
  }
};

template<typename T>
class EndMatMul : public raft::kernel
{
  typedef T** foo;
public:
  EndMatMul() : raft::kernel()
  {
    input.addPort<unsigned>("n");
    input.addPort<unsigned>("m0");
    input.addPort<unsigned>("m1");
    input.addPort<unsigned>("o");
    input.addPort<T>("a");
    input.addPort<T>("b");
    input.addPort<uintptr_t>("rns0");
    input.addPort<uintptr_t>("rns1");
  }
  raft::kstatus run() override
  {
    unsigned n,m,m_b,o;
    uintptr_t rns0;
    uintptr_t rns1;
    input["n"].pop(n);
    input["m0"].pop(m);
    input["m1"].pop(m_b);
    input["o"].pop(o);

    //Check to make sure this multiplication is valid
    assert(m == m_b);

    auto nxm = (T*) malloc(n * m * sizeof(T));
    auto mxo = (T*) malloc(m * o * sizeof(T));
    input["rns0"].pop(rns0);
    input["rns1"].pop(rns1);
    assert(rns0 == rns1);
    auto rns = (T**) rns0;
    *rns = (T*) malloc(n * o * sizeof(T));

    for_mat_inner(n,m,[nxm,this,m](unsigned i, unsigned j)
    {
      input["a"].pop(nxm[i*m + j]);
    });
    for_mat_inner(m,o,[mxo,this,o](unsigned i, unsigned j)
    {
      input["b"].pop(mxo[i*o + j]);
    });

    T ans;
    for_mat_inner(n,o,[&](unsigned i, unsigned j)
    {
      ans = 0;
      for(int k = 0; k < m; k++)
      {
        ans += nxm[i*m + k] * mxo[k*o + j];
      }
      (*rns)[i*o + j] = ans;
    });
    free(nxm);
    free(mxo);
    return raft::proceed;
  }
};

#endif //MATMUL_MAT_MUL_H
