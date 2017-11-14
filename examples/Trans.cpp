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

constexpr Transf16 s =
    epu8{1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
constexpr Transf16 cy =
    epu8{1, 2, 3, 4, 5, 6, 7, 8, 0, 9, 10, 11, 12, 13, 14, 15};
constexpr Transf16 pi =
    epu8{0, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};

constexpr uint8_t FE = 0xfe;

int main() {
  const vector<Transf16> gens{s, cy, pi};
  int lg = 0;

  using google::dense_hash_set;
  using google::sparse_hash_set;

  // sparse_hash_set<Transf16, hash<Transf16>, eqTransf16> res;
  dense_hash_set<Transf16, std::hash<Transf16>, std::equal_to<Transf16>> res;
  res.set_empty_key({FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE});
  res.resize(500000000);

  // unordered_set<Transf16> res;
  // res.reserve(500000000);

  res.insert(Transf16::one());

  vector<Transf16> todo, newtodo;
  todo.push_back(Transf16::one());
  while (todo.size()) {
    newtodo.clear();
    lg++;
    for (auto v : todo) {
      for (auto g : gens) {
        auto el = v * g;
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
