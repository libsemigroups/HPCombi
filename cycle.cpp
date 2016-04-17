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


// Sorting network Knuth AoCP3 Fig. 51 p 229.
static const array<Perm16, 9> rounds =
 // 	 0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15
    {{ { 1, 0, 3, 2, 5, 4, 7, 6, 9, 8,11,10,13,12,15,14},
       { 2, 3, 0, 1, 6, 7, 4, 5,10,11, 8, 9,14,15,12,13},
       { 4, 5, 6, 7, 0, 1, 2, 3,12,13,14,15, 8, 9,10,11},
       { 8, 9,10,11,12,13,14,15, 0, 1, 2, 3, 4, 5, 6, 7},
       { 0, 2, 1,12, 8,10, 9,11, 4, 6, 5, 7, 3,14,13,15},
       { 0, 4, 8,10, 1, 9,12,13, 2, 5, 3,14, 6, 7,11,15},
       { 0, 1, 4, 5, 2, 3, 8, 9, 6, 7,12,13,10,11,14,15},
       { 0, 1, 2, 6, 4, 8, 3,10, 5,12, 7,11, 9,13,14,15},
       { 0, 1, 2, 4, 3, 6, 5, 8, 7,10, 9,12,11,13,14,15}
       }};

Vect16 sort(Vect16 a) {
  for (Perm16 round : rounds) {
    Vect16 minab, maxab, blend, mask, b = a.permuted(round);

    mask = _mm_cmplt_epi8(round, Perm16::one);
    minab = _mm_min_epi8(a, b);
    maxab = _mm_max_epi8(a, b);

    a = _mm_blendv_epi8(minab, maxab, mask);
  }
  return a;
}

Vect16 revsort(Vect16 a) {
  for (Perm16 round : rounds) {
    Vect16 minab, maxab, blend, mask, b = a.permuted(round);

    mask = _mm_cmplt_epi8(round, Perm16::one);
    minab = _mm_min_epi8(a, b);
    maxab = _mm_max_epi8(a, b);

    a = _mm_blendv_epi8(maxab, minab, mask);
  }
  return a;
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
inline Vect16 cycles_mask_unroll(Perm16 p) {
  Vect16 x0, x1 = Perm16::one;
  x0 = _mm_min_epi8(x1, x1.permuted(p));
  p = p*p;
  x1 = _mm_min_epi8(x0, x0.permuted(p));
  p = p*p;
  x0 = _mm_min_epi8(x1, x1.permuted(p));
  p = p*p;
  x1 = _mm_min_epi8(x0, x0.permuted(p));
  return x1;
}

inline uint8_t nb_cycles_unroll(Perm16 p) {
  Perm16 res;
  res.v8 = (Perm16::one.v8 == cycles_mask_unroll(p).v8);
  return _mm_popcnt_u32(_mm_movemask_epi8(res));
}


Vect16 evalutation(Vect16 v) {
  Perm16 turn = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,0};
  Vect16 res;
  res.v8 = -(Perm16::one.v8 == v.v8);
  for (int i = 0; i<15; i++) {
    v = v.permuted(turn);
    res.v8 -= (Perm16::one.v8 == v.v8);
  }
  return res;
}

Vect16 cycle_type(Perm16 p) {
  return revsort(evalutation(cycles_mask_unroll(p)));
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
  for (int i = 0; i < sz; i++) res[i] = Perm16::random();
  return res;
}

vector<Perm16> all_perms(int n) {
  vector<Perm16> res(factorial(n));
  for (unsigned int i = 0; i < res.size(); i++) res[i] = Perm16::unrankSJT(n, i);
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
  std::srand(std::time(0));

  Perm16 p = { 5, 4,12,15,10, 8, 9, 2, 3,13,14, 0, 1, 7,11, 6};

  p = Perm16::random();
  cout << Perm16::one << endl
       << p << endl
       << cycles_mask_unroll(p) << endl
       << evalutation(cycles_mask_unroll(p)) << " #= "
       << unsigned(nb_cycles_unroll(p)) << endl
       << cycle_type(p) << endl;

  for (auto f : func) cout << f(p) << " ";
  cout << endl;

  timeit(rand_perms(10000000));
  cout << endl;

  timeit(all_perms(11));

  return EXIT_SUCCESS;
}
