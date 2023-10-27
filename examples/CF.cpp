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

#include "perm16.hpp"
#include <array>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <x86intrin.h>

using namespace std;
using namespace HPCombi;

class EqEpu8 {

    const epu8 elem;
    const uint64_t size;
    class EqEpu8Iterator;

  public:
    EqEpu8(epu8 x, uint64_t sz = 16) : elem(x), size(sz){};

    EqEpu8Iterator begin() { return {*this, 0, 1}; }
    EqEpu8Iterator end() { return {*this, 0, size}; }

  private:
    class EqEpu8Iterator {

        const EqEpu8 &eqv;
        uint64_t i, j;

      public:
        EqEpu8Iterator(const EqEpu8 &_eqv, uint64_t _i, uint64_t _j)
            : eqv(_eqv), i(_i), j(_j) {
            find_eq();
        }

        void find_eq() {
            for (; j < eqv.size; j++) {
                for (; i < j; i++)
                    if (eqv.elem[i] == eqv.elem[j])
                        return;
                i = 0;
            }
        }
        EqEpu8Iterator &operator++() {
            assert(j < eqv.size);
            ++i;
            find_eq();
            return *this;
        }
        std::pair<uint64_t, uint64_t> operator*() const { return {i, j}; }
        bool operator!=(const EqEpu8Iterator &it) const {
            return i != it.i or j != it.j;
        }
    };
};

int main() {
    epu8 a{0, 2, 3, 4, 0, 6, 2, 2, 0};

    for (auto p : EqEpu8(a, 9)) {
        cout << "(" << p.first << ", " << p.second << ")" << endl;
    }
    cout << endl;
    exit(0);
}
