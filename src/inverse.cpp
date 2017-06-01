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

int main() {
  std::srand(std::time(0));

  Perm16 p = { 5, 4,12,15,10, 8, 9, 2, 3,13,14, 0, 1, 7,11, 6};

  p = Perm16::random();

  cout << p << endl << p.inverse_ref() << endl;
  cout << p.inverse() << endl;
  assert(p.inverse_ref() == p.inverse_sort());
  assert(p.inverse_ref() == p.inverse());
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
  for (uint_fast64_t i=0; i < sz; i++) inv[i] = sample[i].inverse_ref();
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
  for (uint_fast64_t i=0; i < sz; i++) inv3[i] = sample[i].inverse();
  tfin = high_resolution_clock::now();
  auto tmnew = duration_cast<duration<double>>(tfin - tstart);
  cout << "timenew  = " << tmnew.count() << "s";

  for (uint_fast64_t i=0; i < sz; i++) assert(inv[i] == inv3[i]);

  cout << ", speedup = " << tmref.count()/tmnew.count();
  cout << endl;

  return EXIT_SUCCESS;
}
