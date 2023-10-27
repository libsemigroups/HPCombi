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
/** @file
 * @brief Example of how to use #HPCombi::pow with
 * #HPCombi::power_helper::Monoid
 */

#include "power.hpp"
#include <cassert>
#include <string>

namespace HPCombi {
namespace power_helper {

// Algebraic monoid for string with concatenation
template <> struct Monoid<std::string> {

    // The one of the string monoid
    static std::string one() { return {}; };

    /* The product of two strings that is their concatenation
     * @param a the first string to be concatenated
     * @param b the second string to be concatenated
     * @return the concatenation of \a a and \a b
     */
    static std::string prod(std::string a, std::string b) { return a + b; }
};

}  // namespace power_helper
}  // namespace HPCombi

int main() {
    assert(HPCombi::pow<0>(std::string("ab")) == "");
    assert(HPCombi::pow<4>(std::string("ab")) == "abababab");
    assert(HPCombi::pow<5>(std::string("abc")) == "abcabcabcabcabc");
}
