#include <iostream>
#include <chrono>
#include <cstdlib>
#include <array>
#include <algorithm>
#include <x86intrin.h>

#include"perm16.hpp"

using namespace std;
using namespace std::chrono;
using namespace IVMPG;

Perm16 randperm() {
  Perm16 res = Perm16::one();
  random_shuffle ( res.p.begin(), res.p.end() );
  return res;
}

/* From Ruskey : Combinatoriaj Generation page 138 */
Perm16 unrankSJT(int r, int n) {
  int j, k, rem, c;
  array<int, 16> dir;
  Perm16 res = Perm16::one();
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

int nb_cycles_def(Perm16 p) {
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


int nb_cycles(Perm16 p) {
  Vect16 x0, x1 = Perm16::one();
  Perm16 pp = p;
  do {
    x0 = x1;
    x1.v = _mm_min_epi8(x0.v, x0.permuted(pp).v);
    pp = pp*pp;
  } while (x0 != x1);
  x0.v8 = (Perm16::one().v8 == x1.v8);
  return _mm_popcnt_u32(_mm_movemask_epi8(x0.v));
}

using Statistic = array<unsigned long int, 16>;

std::ostream & operator<<(std::ostream & stream, Statistic const &term) {
  stream << "[" << unsigned(term[0]);
  for (unsigned i=1; i < 16; i++) stream << "," << unsigned(term[i]);
  stream << "]";
  return stream;
}


const int sz = 10000;

constexpr unsigned int factorial(unsigned int n) {
  return n > 1 ? n * factorial(n-1) : 1;
}

int main() {
  high_resolution_clock::time_point tstart, tfin;

  Statistic stat = {}, stat_def = {};

  std::srand(std::time(0));

  // Perm16 v1, v = {5,0,2,7,3,6,1,4};
  // Perm16 v = {1,14,0,9,4,15,2,10,5,12,11,3,13,7,8,6};
  // array<Perm16, sz> vv;
  Perm16* vv = (Perm16*) malloc(sz*sizeof(Perm16));

  for (int i = 0; i<24; i++) {
    cout << unrankSJT(i, 4) << endl;
    assert(unrankSJT(i, 4).is_permutation(4));
  }
  for (int i = 0; i < sz; i++) vv[i] = randperm();
  for (int i = 0; i < sz; i++) assert(nb_cycles(vv[i]) == nb_cycles_def(vv[i]));

  cout << vv[0] << " " << nb_cycles(vv[0]) << " " << nb_cycles_def(vv[0]) << endl;

  tstart = high_resolution_clock::now();
  for (int i=0; i < sz; i++)
    stat[nb_cycles(vv[i])]++;
  tfin = high_resolution_clock::now();

  auto tm = duration_cast<duration<double>>(tfin - tstart);
  cout << stat << " (time = " << tm.count() << "s)." << std::endl;

  tstart = high_resolution_clock::now();
  for (int i=0; i < sz; i++)
    stat_def[nb_cycles_def(vv[i])]++;
  tfin = high_resolution_clock::now();

  auto tm_def = duration_cast<duration<double>>(tfin - tstart);
  cout << stat_def << " (time = " << tm_def.count() << "s)." << std::endl;
  cout << "Speedup = " << tm_def.count()/tm.count() << endl;

  return EXIT_SUCCESS;
}

