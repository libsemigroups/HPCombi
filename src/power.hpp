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

#ifndef PERM16_POWER_HPP_INCLUDED
#define PERM16_POWER_HPP_INCLUDED

/** Forward declaration */
namespace power_helper {
  template <typename T> struct Monoid;
};


/** A generic compile time exponentiation function
 *
 *  @param exp    the power
 *  @param x      the number to exponentiate
 *  @return       x to the power exp
 *
 *  The function to compute a specific power is optimized at compile time.
 *
 *  To use for a specific type the user should pass a Monoid structure (see
 *  below) as third parameter to the template. Alternatively a default monoid
 *  structure can be defined for a given type by overloading the struct
 *  power_helper::Monoid<T>
 *
 *  A Monoid structure is required to define two static members
 *  - one : the unit of the monoid
 *  - T mult(T, T) : the product of two elements in the monoid
 *
 *  Note: unfortunately boost::math::power is not enought general to handle
 *  monoids whose unit is not constructed as T(1).
 */
template<typename T, typename M = power_helper::Monoid<T> >
constexpr T square(const T x) { return M::mult(x, x); }

template<unsigned exp, typename T, typename M = power_helper::Monoid<T> >
constexpr T pow(const T x) {
  return
    (exp == 0) ? M::one :
    (exp % 2 == 0) ? square<T, M>(pow<exp/2>(x)) :
    M::mult(x, square<T, M>(pow<(exp-1)/2>(x)));
}



namespace power_helper {

/** Default class for numeric multiplicative monoids
 */

template <typename T> struct Monoid {
  static constexpr T one = 1;
  static constexpr T mult(T a, T b) { return a * b; }
};

};  // namespace power_helper

#endif  // PERM16_POWER_HPP_INCLUDED
