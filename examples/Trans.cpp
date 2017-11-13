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

constexpr Vect16 id =
    epu8{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

constexpr Vect16 s = epu8{1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

constexpr Vect16 cy =
    epu8{1, 2, 3, 4, 5, 6, 7, 8, 0, 9, 10, 11, 12, 13, 14, 15};
constexpr Vect16 pi =
    epu8{0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

constexpr uint8_t FF = 0xff;
constexpr uint8_t FE = 0xfe;

inline Vect16 act1(Vect16 x, Vect16 y) {
  return static_cast<epu8>(_mm_shuffle_epi8(x, y)) | (y.v == FF);
}

inline Vect16 act0(Vect16 x, Vect16 y) {
  Vect16 minab, maxab, mask, b = x.permuted(y);
  mask = _mm_cmplt_epi8(y, Perm16::one());
  minab = _mm_min_epi8(x, b);
  maxab = _mm_max_epi8(x, b);
  return static_cast<epu8>(_mm_blendv_epi8(maxab, minab, mask)) | (y.v == FF);
}

struct eqVect16 {
  bool operator()(const Vect16 &s1, const Vect16 &s2) const { return s1 == s2; }
};

int main() {
  const vector<Vect16> gens{s, cy, pi};
  int lg = 0;

  using google::dense_hash_set;
  using google::sparse_hash_set;

  // sparse_hash_set<Vect16, hash<Vect16>, eqVect16> res;
  dense_hash_set<Vect16, hash<Vect16>, eqVect16> res;
  res.set_empty_key(
      {FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE});
  res.resize(500000000);

  // unordered_set<Vect16> res;
  // res.reserve(500000000);

  res.insert(id);

  vector<Vect16> todo, newtodo;
  todo.push_back(id);
  while (todo.size()) {
    newtodo.clear();
    lg++;
    for (auto v : todo) {
      for (auto g : gens) {
        auto el = act1(v, g);
        if (res.insert(el).second)
          newtodo.push_back(el);
      }
    }
    std::swap(todo, newtodo);
    cout << lg << ", todo = " << todo.size() << ", res = " << res.size()
         << ", #Bucks = " << res.bucket_count() << endl;
  }
  cout << "res =  " << res.size() << endl;
  exit(0);
}
