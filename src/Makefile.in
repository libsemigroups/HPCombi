CXX = g++-5
CXXFLAGS    = -std=c++11 -g -Wall -O3 -funroll-loops -mavx -mtune=native
# CXXFLAGS    = -std=c++11 -g -Wall -mavx -mtune=native
# -Winline -fcilkplus 
# -Wall -Wno-missing-braces -Wno-unused-variable  -Wno-attributes

TARGET = cycle sort inverse
all: $(TARGET)

cycle.o: cycle.cpp perm16.hpp testtools.hpp
cycle: cycle.o perm16.o testtools.o
	$(CXX) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@

sort.o: sort.cpp perm16.hpp testtools.hpp
sort: sort.o perm16.o testtools.o
	$(CXX) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@

inverse.o: inverse.cpp perm16.hpp testtools.hpp
inverse: inverse.o perm16.o testtools.o
	$(CXX) $(LDFLAGS) $^ $(LOADLIBES) $(LDLIBS) -o $@

clean:
	rm -rf $(TARGET) *.o
