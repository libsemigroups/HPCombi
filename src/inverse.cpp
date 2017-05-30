#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <vector>
#include <algorithm>
#include <x86intrin.h>

#include "perm16.hpp"
#include "testtools.hpp"

using namespace std;
using namespace std::chrono;
using namespace IVMPG;

const std::array<Vect16, 3> inverting_rounds =
//     0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
  {{ { 0,  1,  2,  3,  8,  9, 10, 11,  4,  5,  6,  7, 12, 13, 14, 15},
     { 0,  1,  4,  5,  8,  9, 12, 13,  2,  3,  6,  7, 10, 11, 14, 15},
     { 0,  2,  4,  6,  8, 10, 12, 14,  1,  3,  5,  7,  9, 11, 13, 15} }};

const char FIND_IN_PERM = (_SIDD_UBYTE_OPS | _SIDD_CMP_EQUAL_ANY |
			   _SIDD_UNIT_MASK | _SIDD_NEGATIVE_POLARITY);

Perm16 invPerm(Perm16 s) {
  Vect16 res;
  res.v8 = -epi8(_mm_cmpestrm(s.v, 8, idv, 16, FIND_IN_PERM));
  for (Vect16 round : inverting_rounds) {
    s = s * round;
    res.v8 <<= 1;
    res.v8 -= epi8(_mm_cmpestrm(s.v, 8, idv, 16, FIND_IN_PERM));
  }
  return res;
}

int main() {
  std::srand(std::time(0));

  Perm16 p = { 5, 4,12,15,10, 8, 9, 2, 3,13,14, 0, 1, 7,11, 6};

  p = Perm16::random();

  cout << p << endl << p.inverse() << endl;
  cout << invPerm(p) << endl;
  assert(p.inverse() == p.inverse_sort());
  assert(p * p.inverse() == Perm16::one);
  assert(p.inverse() * p == Perm16::one);

  cout << endl << endl;
  uint_fast64_t sz = 10000000;
  auto sample = rand_perms(sz);
  auto inv = sample;
  auto inv2 = sample;
  auto inv3 = sample;

  high_resolution_clock::time_point tstart, tfin;

  tstart = high_resolution_clock::now();
  for (uint_fast64_t i=0; i < sz; i++) inv[i] = sample[i].inverse();
  tfin = high_resolution_clock::now();
  auto tmref = duration_cast<duration<double>>(tfin - tstart);
  cout << "timeref  = " << tmref.count() << "s" << endl;

  tstart = high_resolution_clock::now();
  for (uint_fast64_t i=0; i < sz; i++) inv2[i] = sample[i].inverse_sort();
  tfin = high_resolution_clock::now();
  auto tmsort = duration_cast<duration<double>>(tfin - tstart);
  cout << "timesort = " << tmsort.count() << "s";

  for (uint_fast64_t i=0; i < sz; i++) assert(inv[i] == inv2[i]);

  cout << ", speedup = " << tmref.count()/tmsort.count();
  cout << endl;

  tstart = high_resolution_clock::now();
  for (uint_fast64_t i=0; i < sz; i++) inv3[i] = invPerm(sample[i]);
  tfin = high_resolution_clock::now();
  auto tmnew = duration_cast<duration<double>>(tfin - tstart);
  cout << "timenew  = " << tmnew.count() << "s";

  for (uint_fast64_t i=0; i < sz; i++) assert(inv[i] == inv3[i]);

  cout << ", speedup = " << tmref.count()/tmnew.count();
  cout << endl;

  return EXIT_SUCCESS;
}
