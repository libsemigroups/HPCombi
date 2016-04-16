#include <iostream>
#include <iomanip>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <array>
#include <vector>
#include <algorithm>
#include <x86intrin.h>

#include"perm16.hpp"

using namespace std;
using namespace std::chrono;
using namespace IVMPG;

Perm16 randperm() {
  Perm16 res = Perm16::one;
  random_shuffle ( res.p.begin(), res.p.end() );
  return res;
}

/* From Ruskey : Combinatoriaj Generation page 138 */
Perm16 unrankSJT(int n, int r) {
  int j, k, rem, c;
  array<int, 16> dir;
  Perm16 res = Perm16::one;
  for (j=0; j<n; j++) res[j] = 0xFF;
  for (j=n-1; j >= 0; j--) {
    rem = r % (j + 1);
    r = r / (j + 1);
    if ((r & 1) != 0) {
      k = -1; dir[j] = +1;
    } else {
      k = n; dir[j] = -1;
    }
    c = -1;
    do {
      k = k + dir[j];
      if (res[k] == 0xFF) c++;
    } while (c < rem);
    res[k] = j;
  }
  return res;
}

uint8_t nb_cycles_def(Perm16 p) {
  Vect16 v {};
  int i, j, c = 0;
  for (i = 0; i < 16; i++) {
    if (v[i] == 0) {
      for (j=i; v[j] == 0; j = p[j]) v[j] = 1;
      c++;
    }
  }
  return c;
}


uint8_t nb_cycles(Perm16 p) {
  Vect16 x0, x1 = Perm16::one;
  Perm16 pp = p;
  do {
    x0 = x1;
    x1 = _mm_min_epi8(x0, x0.permuted(pp));
    pp = pp*pp;
  } while (x0 != x1);
  x0.v8 = (Perm16::one.v8 == x1.v8);
  return _mm_popcnt_u32(_mm_movemask_epi8(x0));
}

uint8_t nb_cycles2(Perm16 p) {
  Vect16 x0, x1 = Perm16::one;
  Perm16 pp = p;
  do {
    x0 = _mm_min_epi8(x1, x1.permuted(pp));
    pp = pp*pp;
    x1 = _mm_min_epi8(x0, x0.permuted(pp));
    pp = pp*pp;
  } while (x0 != x1);
  x0.v8 = (Perm16::one.v8 == x1.v8);
  return _mm_popcnt_u32(_mm_movemask_epi8(x0));
}

/** This is by far the fastest implementation *38 the default implem up there **/
uint8_t nb_cycles_unroll(Perm16 p) {
  Vect16 x0, x1 = Perm16::one;
  x0 = _mm_min_epi8(x1, x1.permuted(p));
  p = p*p;
  x1 = _mm_min_epi8(x0, x0.permuted(p));
  p = p*p;
  x0 = _mm_min_epi8(x1, x1.permuted(p));
  p = p*p;
  x1 = _mm_min_epi8(x0, x0.permuted(p));
  x0.v8 = (Perm16::one.v8 == x1.v8);
  return _mm_popcnt_u32(_mm_movemask_epi8(x0));
}

auto func = {nb_cycles_def, nb_cycles, nb_cycles2, nb_cycles_unroll};

using Statistic = array<uint64_t, 17>;

std::ostream & operator<<(std::ostream & stream, Statistic const &term) {
  stream << "[" << unsigned(term[0]);
  for (unsigned i=1; i < 17; i++) stream << "," << unsigned(term[i]);
  stream << "]";
  return stream;
}


constexpr unsigned int factorial(unsigned int n) {
  return n > 1 ? n * factorial(n-1) : 1;
}


vector<Perm16> rand_perms(int sz) {
  vector<Perm16> res(sz);
  std::srand(std::time(0));
  for (int i = 0; i < sz; i++) res[i] = randperm();
  return res;
}

vector<Perm16> all_perms(int n) {
  vector<Perm16> res(factorial(n));
  for (unsigned int i = 0; i < res.size(); i++) res[i] = unrankSJT(n, i);
  return res;
}


template <uint8_t ncycles(Perm16 p)> double timef(vector<Perm16> v,
						  double reftime) {
  high_resolution_clock::time_point tstart, tfin;
  Statistic stat = {};
  uint_fast64_t sz = v.size();

  tstart = high_resolution_clock::now();
  for (uint_fast64_t i=0; i < sz; i++)
    stat[ncycles(v[i])]++;
  tfin = high_resolution_clock::now();

  auto tm = duration_cast<duration<double>>(tfin - tstart);
  cout << stat << endl;
  cout << "time = " << tm.count() << "s";
  if (reftime != 0) cout << ", speedup = " << reftime/tm.count();
  cout << endl;
  return tm.count();
}

void timeit(vector<Perm16> v) {
  double sp_def;

  cout << "Reference: " << endl; sp_def = timef<nb_cycles_def>(v, 0.);
  cout << "Loop 1: " << endl; timef<nb_cycles>(v, sp_def);
  cout << "Loop 2: " << endl; timef<nb_cycles2>(v, sp_def);
  cout << "Unroll: " << endl; timef<nb_cycles_unroll>(v, sp_def);
}


int main() {
  Perm16 p = { 5, 4,12,15,10, 8, 9, 2, 3,13,14, 0, 1, 7,11, 6};

  for (auto f : func) cout << f(p) << " ";
  cout << endl;

  timeit(rand_perms(10000000));
  cout << endl;

  timeit(all_perms(11));

  return EXIT_SUCCESS;
}
