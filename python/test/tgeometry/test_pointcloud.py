# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2020 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

import open3d as o3d
import open3d.core as o3c
import numpy as np
import pytest


def test_pointcloud():
    device = o3c.Device("CPU:0")
    dtype = o3c.Dtype.Float32

    # Constructor.
    pcd = o3d.tgeometry.PointCloud(dtype, device)

    # Accesser.
    assert "points" in pcd.point
    assert "colors" not in pcd.point
    assert isinstance(pcd.point, o3d.tgeometry.TensorListMap)
    assert isinstance(pcd.point["points"], o3c.TensorList)

    # Assignment.
    pcd.point["points"] = o3c.TensorList(o3c.SizeVector([3]), dtype, device)
    pcd.point["colors"] = o3c.TensorList(o3c.SizeVector([3]), dtype, device)
    assert len(pcd.point["points"]) == 0
    assert len(pcd.point["colors"]) == 0

    one_point = o3c.Tensor.ones((3,), dtype, device)
    one_color = o3c.Tensor.ones((3,), dtype, device)
    pcd.point["points"].push_back(one_point)
    pcd.point["colors"].push_back(one_color)
    assert len(pcd.point["points"]) == 1
    assert len(pcd.point["colors"]) == 1
