execute_process(COMMAND "/home/hao/Documents/CNN_SLAM/data/ros/build/ros-laser-geometry-1.6.4/catkin_generated/python_distutils_install.sh" RESULT_VARIABLE res)

if(NOT res EQUAL 0)
  message(FATAL_ERROR "execute_process(/home/hao/Documents/CNN_SLAM/data/ros/build/ros-laser-geometry-1.6.4/catkin_generated/python_distutils_install.sh) returned error code ")
endif()
