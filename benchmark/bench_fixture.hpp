//****************************************************************************//
//       Copyright (C) 2016 Florent Hivert <Florent.Hivert@lri.fr>,           //
//                                                                            //
//  Distributed under the terms of the GNU General Public License (GPL)       //
//                                                                            //
//    This code is distributed in the hope that it will be useful,            //
//    but WITHOUT ANY WARRANTY; without even the implied warranty of          //
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU       //
//   General Public License for more details.                                 //
//                                                                            //
//  The full text of the GPL is available at:                                 //
//                                                                            //
//                  http://www.gnu.org/licenses/                              //
//****************************************************************************//
#ifndef BENCH_FIXTURE
#define BENCH_FIXTURE

#include "epu.hpp"

using HPCombi::epu8;

constexpr uint_fast64_t size = 1000;
// constexpr uint_fast64_t repeat = 100;

std::vector<epu8> rand_epu8(size_t sz) {
    std::vector<epu8> res;
    for (size_t i = 0; i < sz; i++)
        res.push_back(HPCombi::random_epu8(256));
    return res;
}

inline epu8 rand_perm() {
    epu8 res = HPCombi::epu8id;
    auto &ar = HPCombi::as_array(res);
    std::random_shuffle(ar.begin(), ar.end());
    return res;
}

std::vector<epu8> rand_perms(int sz) {
    std::vector<epu8> res(sz);
    std::srand(std::time(0));
    for (int i = 0; i < sz; i++)
        res[i] = rand_perm();
    return res;
}

class Fix_epu8 {
  public:
    Fix_epu8() : vects(rand_epu8(size)), perms(rand_perms(size)) {}
    ~Fix_epu8() {}
    const std::vector<epu8> vects;
    const std::vector<epu8> perms;
};

#endif  // BENCH_FIXTURE
