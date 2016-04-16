/******************************************************************************/
/*       Copyright (C) 2014 Florent Hivert <Florent.Hivert@lri.fr>,           */
/*                                                                            */
/*  Distributed under the terms of the GNU General Public License (GPL)       */
/*                                                                            */
/*    This code is distributed in the hope that it will be useful,            */
/*    but WITHOUT ANY WARRANTY; without even the implied warranty of          */
/*    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU       */
/*   General Public License for more details.                                 */
/*                                                                            */
/*  The full text of the GPL is available at:                                 */
/*                                                                            */
/*                  http://www.gnu.org/licenses/                              */
/******************************************************************************/

#include "perm16.hpp"
#include <iostream>
#include <iomanip>

namespace IVMPG {

// Definition since previously *only* declared
const constexpr size_t Vect16::Size;

std::ostream & operator<<(std::ostream & stream, Vect16 const &term) {
  stream << "[" << unsigned(term[0]);
  for (unsigned i=1; i < Vect16::Size; i++)
    stream << "," << std::setw(2) << unsigned(term[i]);
  stream << "]";
  return stream;
}

} //  namespace IVMPG
