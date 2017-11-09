#include <x86intrin.h>
#include <cassert>
#include <cstdint>
#include <array>
#include <vector>
#include <string>
#include <unordered_set>
#include <set>
#include <iostream>
#include <iomanip>
#include <functional>  // less<>
#include <sparsehash/dense_hash_map>
#include "perm16.hpp"

template <typename T>
std::ostream& operator<< (std::ostream& out, const std::vector<T>& v) {
  out << '[';
  if ( !v.empty() ) {
    auto i = v.begin();
    for (; i != --v.end(); ++i)
      out << std::setw(2) << *i << ",";
    out << std::setw(2) << *i;
  }
  out << "]";
  return out;
}

using namespace std;
using namespace IVMPG;

struct eqVect16 {
  bool operator()(const Vect16 &s1, const Vect16 &s2) const {
    return s1 == s2;
  }
};


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
constexpr uint8_t FE = 0xfe;

constexpr Vect16 gene =
  epu8 {FF,FF,FF,FF,FF,FF,FF,FF,  8, 9, 10, 11, 12, 13, 14, 15};
constexpr Vect16 genf =
  epu8 {FF,FF,FF,FF,FF,FF,FF, 7, FF, 9, 10, 11, 12, 13, 14, 15};

// const vector<Vect16> gens {gene, genf, s1e, s1f};
const vector<Vect16> gens {gene, s0, s1e, s2, s3};
const int nprint = 4;
google::dense_hash_map<Vect16, std::vector<int>, hash<Vect16>, eqVect16> elems;

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

inline Vect16 mult0(Vect16 x, Vect16 y) {
  auto res = x;
  for (uint8_t i : elems[y]) res = act0(res, gens[i]);
  return res;
}


std::vector<int> sym_renner(Vect16 v, int n) {
  std::vector<int> res;
  for (int i=8-n; i < 8+n; i++) {
    if (v[i] == 0xff) res.push_back(0);
    else if (v[i] < 8) res.push_back(v[i]-8);
    else res.push_back(v[i]-7);
  }
  return res;
}

int main() {
  int lg = 0;

  cout << sizeof(std::vector<int>) << endl;
  
  elems.set_empty_key({FE,FE,FE,FE,FE,FE,FE,FE,FE,FE,FE,FE,FE,FE,FE,FE});
  elems[id] = {};

  int nidemp = 1;
  cout << "Idemp : " << setw(3) << nidemp << " "
       << sym_renner(id, nprint) << endl;


  vector<Vect16> todo, newtodo;
  todo.push_back(id);
  while (todo.size()) {
    newtodo.clear();
    lg ++;
    for (auto v : todo) {
      for (uint8_t i = 0; i < gens.size(); i++) {
        auto g = gens[i];
        Vect16 el = act0(v, g);
        if (elems.find(el) == elems.end()) {
          newtodo.push_back(el);
          auto newword = elems[v];
          newword.push_back(i);
          elems[el] = newword;
          if (mult0(el, el) == el) {
            nidemp++;
            cout << "Idemp : " << setw(3) << nidemp << " "
                 << sym_renner(el, nprint) << endl;
          }
        }
      }
    }
    std::swap(todo, newtodo);
    // cout << lg << ", todo = " << todo.size() << ", elems = " << elems.size() <<
    //  ", #Bucks = " << elems.bucket_count() << endl;
    // cout << "Trouve " << (elems.find(toFind) != elems.end()) << endl;
    // if (elems.find(toFind) != elems.end()) break;
  }
  Vect16 bla {FF,FF,FF,FF,FF,FF,FF,FF,FF,FF,10,11,12,13,14,15};
  cout << elems[bla] << endl;
  cout << "elems =  " << elems.size() << endl;
  exit(0);
}


