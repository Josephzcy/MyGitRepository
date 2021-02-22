
from plyfile import PlyData
import pandas as pd
import numpy  as np
import pyqtgraph.opengl as gl

# from pyqtgraph.Qt import QtGui
from OpenGL.GL import glLineWidth
import pyqtgraph as pg

import collections


class plot3d(object):
    def __init__(self):
        self.app = pg.mkQApp()
        self.view = gl.GLViewWidget()
        coord = gl.GLAxisItem()
        glLineWidth(1)
        coord.setSize(10,10,10)
        self.view.addItem(coord)
    def add_points(self, points):
        points_item = gl.GLScatterPlotItem(pos=points, size=2)
        self.view.addItem(points_item)
    def add_line(self, p1, p2):
        lines = np.array([[p1[0], p1[1], p1[2]],
                          [p2[0], p2[1], p2[2]]])
        lines_item = gl.GLLinePlotItem(pos=lines, mode='lines',
                                       color=(1,0,0,1), width=3, antialias=True)
        self.view.addItem(lines_item)
    def show(self):
        self.view.show()
        self.app.exec()
    

def inte_to_rgb(pc_inte):
    minimum, maximum = np.min(pc_inte), np.max(pc_inte)
    ratio = 2 * (pc_inte-minimum) / (maximum - minimum)

    b = (np.maximum((1 - ratio), 0))
    r = (np.maximum((ratio - 1), 0))
    g = 1 - b - r
    return np.stack([r, g, b, np.ones_like(r)]).transpose()

def show_lidar(pc_velo):
    p3d = plot3d()
    points = pc_velo[:, 0:3]
    # pc_inte = pc_velo[:, 3]
    # pc_color = inte_to_rgb(pc_inte)
    # pc_color=set_color(pc_velo)
    p3d.add_points(points)
    p3d.show()


if __name__=="__main__":
    with open('point_cloud.ply', 'rb') as point_file_ply:
        plydata = PlyData.read(point_file_ply)
        data=plydata.elements[0].data
        
        data_pd = pd.DataFrame(data)  
       
        pointCloud = np.zeros(data_pd.shape, dtype=np.float)  
        property_names = data[0].dtype.names                # 读取property的名字
        for i, name in enumerate(property_names):           # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
            pointCloud[:, i] = data_pd[name]
        
        d = np.sqrt(pointCloud[:,0] * pointCloud[:,0] + pointCloud[:,1] * pointCloud[:,1] + pointCloud[:,2] * pointCloud[:,2])

        data_pd["d"]=np.rint(d)
        # print(data_pd["d"])
        # print(data_pd["d"].describe())
        class_type=collections.Counter(data_pd["d"])
        print("\n",class_type)
        print("\n","the count of type:",len(class_type))
        #数据可视化
        show_lidar(pointCloud)
    point_file_ply.close()