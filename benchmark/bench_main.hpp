////////////////////////////////////////////////////////////////////////////////
//       Copyright (C) 2023 James D. Mitchell <jdm3@st-andrews.ac.uk>         //
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
////////////////////////////////////////////////////////////////////////////////

#define BENCHMARK_MEM_FN(mem_fn, sample)                                       \
    BENCHMARK(#mem_fn) {                                                       \
        for (auto &elem : sample) {                                            \
            volatile auto dummy = elem.mem_fn();                               \
        }                                                                      \
        return true;                                                           \
    };

#define BENCHMARK_FREE_FN(msg, free_fn, sample)                                \
    BENCHMARK(#free_fn " " msg) {                                              \
        for (auto elem : sample) {                                             \
            volatile auto dummy = free_fn(elem);                               \
        }                                                                      \
        return true;                                                           \
    };

#define BENCHMARK_LAMBDA(msg, free_fn, sample)                                 \
    BENCHMARK(#free_fn " " msg) {                                              \
        auto lambda__xxx = [](auto val) { return free_fn(val); };              \
        for (auto elem : sample) {                                             \
            volatile auto dummy = lambda__xxx(elem);                           \
        }                                                                      \
        return true;                                                           \
    };

#define BENCHMARK_MEM_FN_PAIR_EQ(mem_fn, sample)                               \
    BENCHMARK(#mem_fn) {                                                       \
        for (auto &pair : sample) {                                            \
            auto val =                                                         \
                std::make_pair(pair.first.mem_fn(), pair.second.mem_fn());     \
            REQUIRE(val.first == val.second);                                  \
        }                                                                      \
        return true;                                                           \
    };

#define BENCHMARK_MEM_FN_PAIR(mem_fn, sample)                                  \
    BENCHMARK(#mem_fn) {                                                       \
        for (auto &pair : sample) {                                            \
            volatile auto val = pair.first.mem_fn(pair.second);                \
        }                                                                      \
        return true;                                                           \
    };
