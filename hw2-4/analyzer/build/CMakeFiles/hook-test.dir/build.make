# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/lib/python3.8/dist-packages/cmake/data/bin/cmake

# The command to remove a file.
RM = /usr/local/lib/python3.8/dist-packages/cmake/data/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/guofangyu/projects/hw02/hw2-4/analyzer

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/guofangyu/projects/hw02/hw2-4/analyzer/build

# Include any dependencies generated for this target.
include CMakeFiles/hook-test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/hook-test.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/hook-test.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/hook-test.dir/flags.make

CMakeFiles/hook-test.dir/hook-test.cpp.o: CMakeFiles/hook-test.dir/flags.make
CMakeFiles/hook-test.dir/hook-test.cpp.o: /home/guofangyu/projects/hw02/hw2-4/analyzer/hook-test.cpp
CMakeFiles/hook-test.dir/hook-test.cpp.o: CMakeFiles/hook-test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/guofangyu/projects/hw02/hw2-4/analyzer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/hook-test.dir/hook-test.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/hook-test.dir/hook-test.cpp.o -MF CMakeFiles/hook-test.dir/hook-test.cpp.o.d -o CMakeFiles/hook-test.dir/hook-test.cpp.o -c /home/guofangyu/projects/hw02/hw2-4/analyzer/hook-test.cpp

CMakeFiles/hook-test.dir/hook-test.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/hook-test.dir/hook-test.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/guofangyu/projects/hw02/hw2-4/analyzer/hook-test.cpp > CMakeFiles/hook-test.dir/hook-test.cpp.i

CMakeFiles/hook-test.dir/hook-test.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/hook-test.dir/hook-test.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/guofangyu/projects/hw02/hw2-4/analyzer/hook-test.cpp -o CMakeFiles/hook-test.dir/hook-test.cpp.s

# Object files for target hook-test
hook__test_OBJECTS = \
"CMakeFiles/hook-test.dir/hook-test.cpp.o"

# External object files for target hook-test
hook__test_EXTERNAL_OBJECTS =

hook-test: CMakeFiles/hook-test.dir/hook-test.cpp.o
hook-test: CMakeFiles/hook-test.dir/build.make
hook-test: /home/guofangyu/projects/hw02/hw2-4/analyzer/libtorch/lib/libtorch.so
hook-test: /home/guofangyu/projects/hw02/hw2-4/analyzer/libtorch/lib/libc10.so
hook-test: /home/guofangyu/projects/hw02/hw2-4/analyzer/libtorch/lib/libkineto.a
hook-test: /home/guofangyu/projects/hw02/hw2-4/analyzer/libtorch/lib/libc10.so
hook-test: CMakeFiles/hook-test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/guofangyu/projects/hw02/hw2-4/analyzer/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable hook-test"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/hook-test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/hook-test.dir/build: hook-test
.PHONY : CMakeFiles/hook-test.dir/build

CMakeFiles/hook-test.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/hook-test.dir/cmake_clean.cmake
.PHONY : CMakeFiles/hook-test.dir/clean

CMakeFiles/hook-test.dir/depend:
	cd /home/guofangyu/projects/hw02/hw2-4/analyzer/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/guofangyu/projects/hw02/hw2-4/analyzer /home/guofangyu/projects/hw02/hw2-4/analyzer /home/guofangyu/projects/hw02/hw2-4/analyzer/build /home/guofangyu/projects/hw02/hw2-4/analyzer/build /home/guofangyu/projects/hw02/hw2-4/analyzer/build/CMakeFiles/hook-test.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/hook-test.dir/depend

