#include <x86intrin.h>
#include <cassert>
#include <cstdint>
#include <array>
#include <vector>
#include <unordered_set>
#include <set>
#include <iostream>
#include <functional>  // less<>
#include "perm16.hpp"

using namespace std;
using namespace IVMPG;

using epu8 = uint8_t __attribute__ ((vector_size (16)));

constexpr Vect16 id =
  epu8 {0, 1, 2, 3, 4, 5, 6, 7  ,  8, 9, 10, 11, 12, 13, 14, 15};

constexpr Vect16 s0 =
  epu8 {0, 1, 2, 3, 4, 5, 6, 8  ,  7, 9, 10, 11, 12, 13, 14, 15};

constexpr Vect16 s1e =
  epu8 {0, 1, 2, 3, 4, 5, 7, 6  ,  9, 8, 10, 11, 12, 13, 14, 15};
constexpr Vect16 s1f =
  epu8 {0, 1, 2, 3, 4, 5, 8, 9  ,  6, 7, 10, 11, 12, 13, 14, 15};

constexpr Vect16 s2 =
  epu8 {0, 1, 2, 3, 4, 6, 5, 7  ,  8, 10, 9, 11, 12, 13, 14, 15};
constexpr Vect16 s3 =
  epu8 {0, 1, 2, 3, 5, 4, 6, 7  ,  8, 9, 11, 10, 12, 13, 14, 15};
constexpr Vect16 s4 =
  epu8 {0, 1, 2, 4, 3, 5, 6, 7  ,  8, 9, 10, 12, 11, 13, 14, 15};
constexpr Vect16 s5 =
  epu8 {0, 1, 3, 2, 4, 5, 6, 7  ,  8, 9, 10, 11, 13, 12, 14, 15};
constexpr Vect16 s6 =
  epu8 {0, 2, 1, 3, 4, 5, 6, 7  ,  8, 9, 10, 11, 12, 14, 13, 15};
constexpr Vect16 s7 =
  epu8 {1, 0, 2, 3, 4, 5, 6, 7  ,  8, 9, 10, 11, 12, 13, 15, 14};

constexpr uint8_t FF = 0xff;

constexpr Vect16 gene =
  epu8 {FF,FF,FF,FF,FF,FF,FF,FF,  8, 9, 10, 11, 12, 13, 14, 15};
constexpr Vect16 genf =
  epu8 {FF,FF,FF,FF,FF,FF,FF, 7, FF, 9, 10, 11, 12, 13, 14, 15};


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

int main() {
  //vector<Vect16> gens {gene, genf, s1e, s1f, s2, s3, s4, s5};
  // vector<Vect16> gens {gene, genf, s1e, s1f};
  vector<Vect16> gens {gene, genf, s1e, s1f, s2, s3, s4, s5};

  // cout << act0(s2,genf) << endl;

  int lg = 0;
  unordered_set<Vect16> res;
  res.insert(id);
  res.reserve(250000000);

  vector<Vect16> todo;
  todo.push_back(id);
  while (todo.size()) {
    vector<Vect16> newtodo {};
    lg ++;
    for (auto v : todo) {
      for (auto g : gens) {
        auto el = act1(v, g);
        if (res.find(el) == res.end()) {
          res.insert(el);
          newtodo.push_back(el);
        }
      }
    }
    todo = std::move(newtodo);
    cout << lg << ", todo = " << todo.size() << ", res = " << res.size() <<
      ", #Bucks = " << res.bucket_count() << endl;
  }
  cout << "res =  " << res.size() << endl;
  // for (auto v : res)  cout << v << endl;
  exit(0);
}


