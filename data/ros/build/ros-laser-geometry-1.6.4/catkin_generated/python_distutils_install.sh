#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
    DESTDIR_ARG="--root=$DESTDIR"
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/hao/Documents/CNN_SLAM/data/ros/src/ros-laser-geometry-1.6.4"

# snsure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/hao/Documents/CNN_SLAM/data/ros/install/lib/python2.7/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/hao/Documents/CNN_SLAM/data/ros/install/lib/python2.7/dist-packages:/home/hao/Documents/CNN_SLAM/data/ros/build/lib/python2.7/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/hao/Documents/CNN_SLAM/data/ros/build" \
    "/usr/bin/python" \
    "/home/hao/Documents/CNN_SLAM/data/ros/src/ros-laser-geometry-1.6.4/setup.py" \
    build --build-base "/home/hao/Documents/CNN_SLAM/data/ros/build/ros-laser-geometry-1.6.4" \
    install \
    $DESTDIR_ARG \
    --install-layout=deb --prefix="/home/hao/Documents/CNN_SLAM/data/ros/install" --install-scripts="/home/hao/Documents/CNN_SLAM/data/ros/install/bin"
