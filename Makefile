# Created by Unium on 21.02.26

CXX      := g++
CXXFLAGS := -std=c++20 -Isrc -Wall -Wextra -pthread

ifeq ($(BUILD),debug)
	CXXFLAGS += -g
else
	CXXFLAGS += -O3 -ffast-math -march=native -flto -funroll-loops -DNDEBUG
endif

SRCS := $(shell find src -name "*.cpp" ! -name "main.cpp" ! -name "tests.cpp")
OBJS := $(SRCS:.cpp=.o)

all: llm tests

llm: $(OBJS) src/main.o
	$(CXX) $(CXXFLAGS) -o $@ $^

tests: $(OBJS) src/tests.o
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) src/main.o src/tests.o llm tests
