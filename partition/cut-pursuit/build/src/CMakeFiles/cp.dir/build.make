# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /var/lib/snapd/snap/cmake/252/bin/cmake

# The command to remove a file.
RM = /var/lib/snapd/snap/cmake/252/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /mnt/edisk/superpoint_graph/partition/cut-pursuit

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /mnt/edisk/superpoint_graph/partition/cut-pursuit/build

# Include any dependencies generated for this target.
include src/CMakeFiles/cp.dir/depend.make

# Include the progress variables for this target.
include src/CMakeFiles/cp.dir/progress.make

# Include the compile flags for this target's objects.
include src/CMakeFiles/cp.dir/flags.make

src/CMakeFiles/cp.dir/cutpursuit.cpp.o: src/CMakeFiles/cp.dir/flags.make
src/CMakeFiles/cp.dir/cutpursuit.cpp.o: ../src/cutpursuit.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/mnt/edisk/superpoint_graph/partition/cut-pursuit/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object src/CMakeFiles/cp.dir/cutpursuit.cpp.o"
	cd /mnt/edisk/superpoint_graph/partition/cut-pursuit/build/src && /bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cp.dir/cutpursuit.cpp.o -c /mnt/edisk/superpoint_graph/partition/cut-pursuit/src/cutpursuit.cpp

src/CMakeFiles/cp.dir/cutpursuit.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cp.dir/cutpursuit.cpp.i"
	cd /mnt/edisk/superpoint_graph/partition/cut-pursuit/build/src && /bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /mnt/edisk/superpoint_graph/partition/cut-pursuit/src/cutpursuit.cpp > CMakeFiles/cp.dir/cutpursuit.cpp.i

src/CMakeFiles/cp.dir/cutpursuit.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cp.dir/cutpursuit.cpp.s"
	cd /mnt/edisk/superpoint_graph/partition/cut-pursuit/build/src && /bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /mnt/edisk/superpoint_graph/partition/cut-pursuit/src/cutpursuit.cpp -o CMakeFiles/cp.dir/cutpursuit.cpp.s

# Object files for target cp
cp_OBJECTS = \
"CMakeFiles/cp.dir/cutpursuit.cpp.o"

# External object files for target cp
cp_EXTERNAL_OBJECTS =

src/libcp.so: src/CMakeFiles/cp.dir/cutpursuit.cpp.o
src/libcp.so: src/CMakeFiles/cp.dir/build.make
src/libcp.so: /mnt/edisk/anaconda3/lib/libboost_numpy37.so
src/libcp.so: /mnt/edisk/anaconda3/lib/libboost_python37.so
src/libcp.so: /mnt/edisk/anaconda3/lib/libpython3.7m.so
src/libcp.so: src/CMakeFiles/cp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/mnt/edisk/superpoint_graph/partition/cut-pursuit/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX shared library libcp.so"
	cd /mnt/edisk/superpoint_graph/partition/cut-pursuit/build/src && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
src/CMakeFiles/cp.dir/build: src/libcp.so

.PHONY : src/CMakeFiles/cp.dir/build

src/CMakeFiles/cp.dir/clean:
	cd /mnt/edisk/superpoint_graph/partition/cut-pursuit/build/src && $(CMAKE_COMMAND) -P CMakeFiles/cp.dir/cmake_clean.cmake
.PHONY : src/CMakeFiles/cp.dir/clean

src/CMakeFiles/cp.dir/depend:
	cd /mnt/edisk/superpoint_graph/partition/cut-pursuit/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /mnt/edisk/superpoint_graph/partition/cut-pursuit /mnt/edisk/superpoint_graph/partition/cut-pursuit/src /mnt/edisk/superpoint_graph/partition/cut-pursuit/build /mnt/edisk/superpoint_graph/partition/cut-pursuit/build/src /mnt/edisk/superpoint_graph/partition/cut-pursuit/build/src/CMakeFiles/cp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : src/CMakeFiles/cp.dir/depend

