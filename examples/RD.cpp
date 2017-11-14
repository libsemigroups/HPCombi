#include "perm16.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>  // less<>
#include <iostream>
#include <set>
#include <sparsehash/dense_hash_set>
#include <sparsehash/sparse_hash_set>
#include <unordered_set>
#include <vector>
#include <x86intrin.h>

using namespace std;
using namespace HPCombi;

constexpr PTransf16 id =
    epu8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

constexpr PTransf16 s0 =
    epu8{0, 1, 2, 3, 4, 5, 6, 8, 7, 9, 10, 11, 12, 13, 14, 15};

constexpr PTransf16 s1e =
    epu8{0, 1, 2, 3, 4, 5, 7, 6, 9, 8, 10, 11, 12, 13, 14, 15};
constexpr PTransf16 s1f =
    epu8{0, 1, 2, 3, 4, 5, 8, 9, 6, 7, 10, 11, 12, 13, 14, 15};

constexpr PTransf16 s2 =
    epu8{0, 1, 2, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 13, 14, 15};
constexpr PTransf16 s3 =
    epu8{0, 1, 2, 3, 5, 4, 6, 7, 8, 9, 11, 10, 12, 13, 14, 15};
constexpr PTransf16 s4 =
    epu8{0, 1, 2, 4, 3, 5, 6, 7, 8, 9, 10, 12, 11, 13, 14, 15};
constexpr PTransf16 s5 =
    epu8{0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 12, 14, 15};
constexpr PTransf16 s6 =
    epu8{0, 2, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 13, 15};
constexpr PTransf16 s7 =
    epu8{1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14};

constexpr uint8_t FF = 0xff;
constexpr uint8_t FE = 0xfe;

constexpr PTransf16 gene =
    epu8{FF, FF, FF, FF, FF, FF, FF, FF, 8, 9, 10, 11, 12, 13, 14, 15};
constexpr PTransf16 genf =
    epu8{FF, FF, FF, FF, FF, FF, FF, 7, FF, 9, 10, 11, 12, 13, 14, 15};

inline PTransf16 act1(PTransf16 x, PTransf16 y) {
  return x * y;
}

inline PTransf16 act0(PTransf16 x, PTransf16 y) {
  PTransf16 minab, maxab, mask, b = x.permuted(y);
  mask = _mm_cmplt_epi8(y, Perm16::one());
  minab = _mm_min_epi8(x, b);
  maxab = _mm_max_epi8(x, b);
  return static_cast<epu8>(_mm_blendv_epi8(maxab, minab, mask)) | (y.v == FF);
}

int main() {
  // vector<PTransf16> gens {gene, genf, s1e, s1f, s2, s3, s4, s5};
  // vector<PTransf16> gens {gene, genf, s1e, s1f};
  vector<PTransf16> gens{gene, genf, s1e, s1f, s2, s3, s4, s5, s6};
  // vector<PTransf16> gens {gene, s1e, s2, s3, s4, s5, s6};
  // const PTransf16 toFind =
  //  {FF,FF,FF,FF,FF,FF,FF,FF,  FF, FF, FF, FF, FF, 13, 14, 15};
  // cout << act0(s2,genf) << endl;
  int lg = 0;

  using google::dense_hash_set;
  using google::sparse_hash_set;

  // sparse_hash_set<PTransf16, hash<PTransf16>, equal_to<PTransf16>> res;
  dense_hash_set<PTransf16, hash<PTransf16>, equal_to<PTransf16>> res;
  res.set_empty_key(
      {FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE});
  res.resize(250000000);

  // unordered_set<PTransf16> res;
  // res.reserve(250000000);

  res.insert(id);

  vector<PTransf16> todo, newtodo;
  todo.push_back(id);
  while (todo.size()) {
    newtodo.clear();
    lg++;
    for (auto v : todo) {
      for (auto g : gens) {
        auto el = act0(v, g);
        if (res.insert(el).second)
          newtodo.push_back(el);
        //        if (el == toFind) cout << v << endl;
      }
    }
    std::swap(todo, newtodo);
    cout << lg << ", todo = " << todo.size() << ", res = " << res.size()
         << ", #Bucks = " << res.bucket_count() << endl;
    // cout << "Trouve " << (res.find(toFind) != res.end()) << endl;
    // if (res.find(toFind) != res.end()) break;
  }
  cout << "res =  " << res.size() << endl;
  // for (auto v : res)  cout << v << endl;
  exit(0);
}
