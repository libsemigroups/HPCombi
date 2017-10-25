CXXFLAGS    = -fcilkplus -std=c++14 -g -Wall -O3 -funroll-loops -mavx -mtune=native -flax-vector-conversions

PROGS = demovect perm32 perm64 permbig # demovect32_avx2
all: $(PROGS)

$(PROGS): % : %.cpp

clean:
	rm -rf $(PROGS)
