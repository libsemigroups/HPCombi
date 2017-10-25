//****************************************************************************//
//       Copyright (C) 2014 Florent Hivert <Florent.Hivert@lri.fr>,           //
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

#ifndef PERM16_TESTTOOLS_HPP_INCLUDED
#define PERM16_TESTTOOLS_HPP_INCLUDED

#include <chrono>
#include <vector>
#include "perm16.hpp"

namespace IVMPG {

constexpr unsigned int factorial(unsigned int n) {
  return n > 1 ? n * factorial(n-1) : 1;
}

std::vector<Perm16> rand_perms(int sz);
std::vector<Perm16> all_perms(int n);

// using a template allows us to ignore the differences between functors,
// function pointers and lambda
template<typename Func>
double timethat(Func fun, double reftime = 0) {
  std::chrono::high_resolution_clock::time_point tstart, tfin;

  tstart = std::chrono::high_resolution_clock::now();
  fun();
  tfin = std::chrono::high_resolution_clock::now();

  auto tm = std::chrono::duration_cast<
    std::chrono::duration<double>>(tfin - tstart);
  std::cout << "time = " << std::setprecision(3) << tm.count() << "s";
  if (reftime != 0) std::cout << ", speedup = " << reftime/tm.count();
  std::cout << std::endl;
  return tm.count();
}

}  // namespace IVMPG
#endif  // PERM16_TESTTOOLS_HPP_INCLUDED
