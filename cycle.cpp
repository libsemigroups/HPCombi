#include <iostream>
#include <chrono>
#include <cstdlib>
#include <array>
#include <vector>
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
Perm16 unrankSJT(int n, int r) {
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

using Statistic = array<unsigned long int, 17>;

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

void timeit(vector<Perm16> v) {
  high_resolution_clock::time_point tstart, tfin;
  Statistic stat = {}, stat_def = {};

  for (unsigned int i=0; i < v.size(); i++)
    assert(nb_cycles(v[i]) == nb_cycles_def(v[i]));

  tstart = high_resolution_clock::now();
  for (unsigned int i=0; i < v.size(); i++)
    stat[nb_cycles(v[i])]++;
  tfin = high_resolution_clock::now();

  auto tm = duration_cast<duration<double>>(tfin - tstart);
  cout << stat << " (time = " << tm.count() << "s)." << std::endl;

  tstart = high_resolution_clock::now();
  for (unsigned int i=0; i < v.size(); i++)
    stat_def[nb_cycles_def(v[i])]++;
  tfin = high_resolution_clock::now();

  auto tm_def = duration_cast<duration<double>>(tfin - tstart);
  cout << stat_def << " (time = " << tm_def.count() << "s)." << std::endl;
  cout << "Speedup = " << tm_def.count()/tm.count() << endl;
}

int main() {

  timeit(rand_perms(10000000));
  timeit(all_perms(11));

  return EXIT_SUCCESS;
}

