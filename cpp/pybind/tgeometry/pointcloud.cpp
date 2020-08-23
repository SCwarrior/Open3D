// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2020 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

#include "open3d/tgeometry/PointCloud.h"

#include "pybind/tgeometry/geometry.h"

namespace open3d {
namespace tgeometry {

void pybind_pointcloud(py::module& m) {
    py::class_<PointCloud, PyGeometry<PointCloud>, std::unique_ptr<PointCloud>,
               Geometry>
            pointcloud(m, "PointCloud",
                       "A pointcloud contains a set of 3D points.");

    // Constructors.
    pointcloud
            .def(py::init<core::Dtype, const core::Device&>(), "dtype"_a,
                 "device"_a)
            .def(py::init<const core::TensorList&>(), "points"_a)
            .def(py::init<const std::unordered_map<std::string,
                                                   core::TensorList>&>(),
                 "map_keys_to_tensorlists"_a);

    // Point's attributes: points, colors, normals.
    pointcloud.def_property_readonly("point", &PointCloud::GetPointAttrPybind);
}

}  // namespace tgeometry
}  // namespace open3d
