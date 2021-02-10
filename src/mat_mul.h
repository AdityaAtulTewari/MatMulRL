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
  T* ans = (T*) calloc(n * o, sizeof(T));
  cache_oblivious_mul<T>(ans,n,m,o,nxm,mxo);
  return ans;
}


#include <raft>

template<typename T>
class StartMatMul : public raft::kernel
{
private:
  unsigned n,m,o;
  T* a;
  T* b;
public:
  StartMatMul(unsigned n, unsigned m, unsigned o, T* a, T* b) : raft::kernel(),n(n),m(m),o(o),a(a),b(b)
  {
    output.addPort<unsigned>("n");
    output.addPort<unsigned>("o");
    output.addPort<T>("axb");
  }
  raft::kstatus run() override
  {
    output["n"].push(n);
    output["o"].push(o);

    T ans;
    for_mat_inner(n,o,[&](unsigned i, unsigned j)
    {
      ans = 0;
      for(int k = 0; k < m; k++)
      {
        ans += a[i*m + k] * b[k*o + j];
      }
      output["axb"].push(ans);
    });

    return raft::stop;
  }
};

template<typename T>
class EndMatMul : public raft::kernel
{
private:
  T** rns;
public:
  EndMatMul(T** rns) : raft::kernel(),rns(rns)
  {
    input.addPort<unsigned>("n");
    input.addPort<unsigned>("m0");
    input.addPort<unsigned>("m1");
    input.addPort<unsigned>("o");
    input.addPort<unsigned>("a");
    input.addPort<unsigned>("b");
  }
  raft::kstatus run() override
  {
    unsigned n,m,m_b,o;
    input["n"].pop(n);
    input["m0"].pop(m);
    input["m1"].pop(m_b);
    input["o"].pop(o);

    //Check to make sure this multiplication is valid
    assert(m == m_b);

    auto nxm = (T*) malloc(n * m * sizeof(T));
    auto mxo = (T*) malloc(m * o * sizeof(T));
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
    return raft::stop;
  }
};

template<typename T>
class StartMatMul_1Copy : public raft::kernel
{
private:
  unsigned n,m,o;
  T* a;
  T* b;
public:
  StartMatMul_1Copy(unsigned n, unsigned m, unsigned o, T* a, T* b) : raft::kernel(),n(n),m(m),o(o),a(a),b(b)
  {
    output.addPort<unsigned>("n");
    output.addPort<unsigned>("o");
    output.addPort<T>("axb");
  }
  raft::kstatus run() override
  {

    auto nxo = output["axb"].allocate_range<T>(n * o);

    for_mat_inner(n,o,[&](unsigned i, unsigned j)
    {
      nxo[i*o + j].get() = 0u;
      for(int k = 0; k < m; k++)
      {
        nxo[i*o + j].get() += a[i*m + k] * b[k*o + j];
      }
    });

    output["axb"].send_range();

    output["n"].push(n);
    output["o"].push(o);

    return raft::stop;
  }
};

template<typename T>
class EndMatMul_1Copy : public raft::kernel
{
private:
  T** rns;
public:
  EndMatMul_1Copy(T** rns) : raft::kernel(),rns(rns)
  {
    input.addPort<unsigned>("n");
    input.addPort<unsigned>("m0");
    input.addPort<unsigned>("m1");
    input.addPort<unsigned>("o");
    input.addPort<unsigned>("a");
    input.addPort<unsigned>("b");
  }
  raft::kstatus run() override
  {
    unsigned n,m,m_b,o;
    input["n"].pop(n);
    input["m0"].pop(m);
    input["m1"].pop(m_b);
    input["o"].pop(o);

    //Check to make sure this multiplication is valid
    assert(m == m_b);

    *rns = (T*) malloc(n * o * sizeof(T));
    std::vector<std::pair<unsigned,raft::signal>> nxm = std::vector<std::pair<unsigned,raft::signal>>(n * m);
    std::vector<std::pair<unsigned,raft::signal>> mxo = std::vector<std::pair<unsigned,raft::signal>>(m * o);
    input["a"].pop_range<T>(nxm,n * m);
    input["b"].pop_range<T>(mxo,m * o);

    T ans;
    for_mat_inner(n,o,[&](unsigned i, unsigned j)
    {
      ans = 0;
      for(int k = 0; k < m; k++)
      {
        ans += nxm[i*m + k].first * mxo[k*o + j].first;
      }
      (*rns)[i*o + j] = ans;
    });
    input["a"].unpeek();
    input["b"].unpeek();
    input["a"].recycle(n * m);
    input["b"].recycle(m * o);
    return raft::stop;
  }
};

#endif //MATMUL_MAT_MUL_H
