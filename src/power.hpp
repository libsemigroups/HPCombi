#ifndef POWER_HPP_
#define POWER_HPP_


namespace power_helper {
    template <typename T> struct Monoid {
        static constexpr T one = 1;
        static constexpr T mult(T a, T b) { return a * b; };
    };
};

template<unsigned exp, typename T>
inline constexpr T pow(const T x) {
    using namespace power_helper;
    return
        (exp == 0) ? Monoid<T>::one :
        (exp % 2 == 0) ?
            Monoid<T>::mult(pow<exp/2>(x), pow<exp/2>(x)) :
            Monoid<T>::mult(x, Monoid<T>::mult(pow<(exp-1)/2>(x),
                                               pow<(exp-1)/2>(x)));
}

#endif // POWER_HPP_
