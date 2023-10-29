#include <array>
#include <cassert>
#include <cstdint>
#include <functional>  // less<>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#ifdef HPCOMBI_HAVE_DENSEHASHSET
#include <sparsehash/dense_hash_map>
#else
#include <unordered_map>
#endif
#include "simde/x86/sse4.1.h"  // for simde_mm_max_epu8, simde...

#include "hpcombi/perm16.hpp"

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &v) {
  out << '[';
  if (!v.empty()) {
    auto i = v.begin();
    for (; i != --v.end(); ++i)
        out << std::setw(2) << *i << ",";
    out << std::setw(2) << *i;
  }
  out << "]";
  return out;
}

using namespace std;
using namespace HPCombi;

std::vector<epu8> subsets;
std::vector<epu8> subperm;

epu8 tosubset(uint16_t n) {
    epu8 res {};
    for (int i = 0; i < 16; i++) {
        if (((n >> i) & 1) != 0) res[i] = 0xff;
    }
    if (simde_mm_movemask_epi8(res) != n) cout << n << "BUG" << res << endl;
    return res;
}

epu8 subset_to_perm(epu8 s) {
    epu8 res = Epu8({},0xff);
    int c = 0;
    for (int i = 0; i < 16; i++) {
        if (s[i] != 0) {
            res[c] = i;
            c++;
        }
    }
    return res;
}

void make_subsets_of_size(int n, int k) {
    int n2 = 1 << n;
    for (uint16_t i=0; i < n2; i++) {
        if (__builtin_popcountl(i) == k) {
            subsets.push_back(tosubset(i));
            subperm.push_back(subset_to_perm(tosubset(i)));
        }
    }
}

template <int Size>
epu8 extract_pattern(epu8 perm, epu8 permset) {
    epu8 cst = Epu8({}, Size);
    epu8 res = permuted(perm, permset) | (epu8id >= cst);
    res = sort_perm(res) & (epu8id < cst);
    return res;
}

template <int Size>
bool has_pattern(epu8 perm, epu8 patt) {
    for (size_t i = 0; i < subperm.size(); i++) {
        epu8 extr = extract_pattern<Size>(perm, subperm[i]);
        if (equal(extr, patt)) return true;
    }
    return false;
}

int main() {
    cout << hex;
    int n = 8, k = 4, n2 = 1 << n;
    make_subsets_of_size(n, k);
    cout << subsets.size() << endl;
    epu8 perm = {1,4,2,0,3,5,6,7};
    int i = 42;
    cout << Perm16::one() << endl;
    cout << perm << endl;
    cout << subsets[i] << endl;
    cout << simde_mm_movemask_epi8(subsets[i]) << endl;
    cout << extract_pattern<4>(perm, subperm[i]) << endl;
    cout << int(has_pattern<4>(perm, epu8 {2,1,0,3})) << endl;
    cout << int(has_pattern<4>(perm, epu8 {3,2,1,0})) << endl;
}
