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

const Transf16 s  {1, 0, 2, 3, 4, 5, 6, 7, 8};
const Transf16 cy {1, 2, 3, 4, 5, 6, 7, 8, 0};
const Transf16 pi {0, 0, 2, 3, 4, 5, 6, 7, 8};
const vector<Transf16> gens{s, cy, pi};

/* James favourite 
const Transf16 a1 {1, 7, 2, 6, 0, 4, 1, 5};
const Transf16 a2 {2, 4, 6, 1, 4, 5, 2, 7};
const Transf16 a3 {3, 0, 7, 2, 4, 6, 2, 4};
const Transf16 a4 {3, 2, 3, 4, 5, 3, 0, 1};
const Transf16 a5 {4, 3, 7, 7, 4, 5, 0, 4};
const Transf16 a6 {5, 6, 3, 0, 3, 0, 5, 1};
const Transf16 a7 {6, 0, 1, 1, 1, 6, 3, 4};
const Transf16 a8 {7, 7, 4, 0, 6, 4, 1, 7};
const vector<Transf16> gens{a1,a2,a3,a4,a5,a6,a7,a8};
*/

const uint8_t FE = 0xfe;

int main() {
  int lg = 0;

  using google::dense_hash_set;
  using google::sparse_hash_set;

  // sparse_hash_set<Transf16, hash<Transf16>, eqTransf16> res;
  dense_hash_set<Transf16, std::hash<Transf16>, std::equal_to<Transf16>> res;
  res.set_empty_key({FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE, FE});
  // res.resize(500000000);

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
