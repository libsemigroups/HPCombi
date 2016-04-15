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

Perm16 insertion_sort (Perm16 a){
  for (int i = 0; i < 16; i++)
    for (int j = i; j > 0 && a[j] < a[j-1]; j--)
      std::swap(a[j], a[j-1]);
  return a;
}

int main() {
  // Perm16 a = { 5, 4,12,15,10, 8, 9, 2, 3,13,14, 0, 1, 7,11, 6};
  high_resolution_clock::time_point tstart, tfin;

  for (Perm16 round : rounds) {
    assert(round.is_permutation());
    assert (round*round==Perm16::one);
  }

  auto vrand = rand_perms(10000000);

  tstart = high_resolution_clock::now();
  for (Perm16 v : vrand) assert(sort(v) == Perm16::one);
  tfin = high_resolution_clock::now();

  auto tm = duration_cast<duration<double>>(tfin - tstart);
  cout << "time = " << tm.count() << "s" << endl;

  tstart = high_resolution_clock::now();
  for (Perm16 v : vrand) assert(insertion_sort(v) == Perm16::one);
  tfin = high_resolution_clock::now();

  tm = duration_cast<duration<double>>(tfin - tstart);
  cout << "time = " << tm.count() << "s" << endl;

  tstart = high_resolution_clock::now();
  for (Perm16 v : vrand) {
    Perm16 vv = v;
    std::sort (vv.p.begin(), vv.p.end());
    assert(vv == Perm16::one);
  }
  tfin = high_resolution_clock::now();

  tm = duration_cast<duration<double>>(tfin - tstart);
  cout << "time = " << tm.count() << "s" << endl;

  return EXIT_SUCCESS;
}
