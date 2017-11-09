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

#include <iostream>
#include <iomanip>
#include <vector>
#include "testtools.hpp"

namespace HPCombi {

std::vector<Perm16> rand_perms(int sz) {
  std::vector<Perm16> res(sz);
  std::srand(std::time(0));
  for (int i = 0; i < sz; i++) res[i] = Perm16::random();
  return res;
}

std::vector<Perm16> all_perms(int n) {
  std::vector<Perm16> res(factorial(n));
  for (unsigned int i = 0; i < res.size(); i++)
    res[i] = Perm16::unrankSJT(n, i);
  return res;
}

}  // namespace HPCombi
