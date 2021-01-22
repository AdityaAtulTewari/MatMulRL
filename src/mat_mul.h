//
// Created by Aditya Tewari on 1/20/21.
//

#ifndef MATMUL_MAT_MUL_H
#define MATMUL_MAT_MUL_H

template <typename T, typename F>
void fill(unsigned n, unsigned m, T* mat, F f)
{
  for_mat(n,m,[mat,m, f] (unsigned i, unsigned j)
  {
    mat[i * m + j] = f();
  });
}

template <typename T, typename P, typename Q, typename F>
void for_mat(unsigned n, unsigned m, F f, Q post = [](unsigned i){(void) i;}, P pre = [](unsigned i) {(void) i;})
{
  for(unsigned i = 0; i < n; i++)
  {
    pre(i);
    for(unsigned j = 0; j < m; j++) f(i,j);
    post(i);
  }
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
  for_mat(n,q,in,post);
}

template<typename T>
T* cache_oblivious_mat_mul(unsigned n, unsigned m, unsigned o, T* nxm, T* mxo)
{
  T* ans = (T*) calloc(n * o, sizeof(T));
  cache_oblivious_mat_mul(n,m,o,nxm,mxo,ans);
  return ans;
}

template<typename T>
int cache_oblivious_mat_mul(unsigned n, unsigned m, unsigned o, T* nxm, T* mxo, T* ans)
{
  for_mat(n,o,[ans,nxm,mxo,m,o](unsigned i, unsigned j)
  {
    ans[i*o + j] = 0;
    for(int k = 0; k < m; k++)
    {
      ans[i*o + j] += nxm[i*m + k] * mxo[k*o + j];
    }
  });
  return 0;
}

#include <raft>
template<typename T>
class COMatMul : public raft::kernel
{
private:
  bool end;
  T** rns;
public:
  COMatMul(T** rns) : raft::kernel(), rns(rns)
  {
    input.addPort<unsigned>("n");
    input.addPort<unsigned>("m");
    input.addPort<unsigned>("o");
    input.addPort<T>("a");
    input.addPort<T>("b");
    output.addPort<unsigned>("n");
    output.addPort<unsigned>("o");
    output.addPort<T>("axb");
    end = true;
  }
  COMatMul(unsigned n, unsigned m, unsigned o, T* a, T* b) : raft::kernel()
  {
    input.addPort<unsigned>("n");
    input.addPort<unsigned>("m");
    input.addPort<unsigned>("o");
    input.addPort<T>("a");
    input.addPort<T>("b");
    output.addPort<unsigned>("n");
    output.addPort<unsigned>("o");
    output.addPort<T>("axb");
    input["n"].push(n);
    input["m"].push(m);
    input["o"].push(o);
    for_mat<T>(n,m,[&](unsigned i, unsigned j){ input["a"].push(a[i*m + j]);});
    for_mat<T>(m,o,[&](unsigned i, unsigned j){ input["b"].push(b[i*o + j]);});
    end = false;
  }
  raft::kstatus run() override
  {
    unsigned n;
    unsigned m;
    unsigned o;
    input["n"].pop(n);
    input["m"].pop(m);
    input["o"].pop(o);
    output["n"].push(n);
    output["o"].push(o);
    if(end) *rns = malloc(n * o * sizeof(T));
    auto nxm = input["a"].peek_range(n * m);
    auto mxo = input["b"].peek_range(m * o);

    T ans;
    for_mat(n,o,[&](unsigned i, unsigned j)
    {
      ans = 0;
      for(int k = 0; k < m; k++)
      {
        ans += nxm[i*m + k] * mxo[k*o + j];
      }
      if(!end) output["axb"].push(ans);
      else (*rns)[i * o + j] = ans;
    });

    input["a"].unpeek();
    input["b"].unpeek();
    input["a"].recycle(n * m);
    input["b"].recycle(m * o);
    return raft::stop;
  }
};

#endif //MATMUL_MAT_MUL_H
