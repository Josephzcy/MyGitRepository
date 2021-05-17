import json
import os
import numpy as np
import pandas as pd
from functools import reduce


car_id=[];car_R=[];car_angle=[];car_O=[];car_score=[];car_point_density=[]
FilePath="./Data/three/result.json"
if __name__ == "__main__":
         
    if os.path.exists(FilePath):
        with open(FilePath) as fjson:
            labels= json.loads(fjson.read())   #list 10 帧点云
            gt =labels["tps"]
            for frame_id in gt :  #24270 dict  哪一帧print(frame_id), print("\n")
                for object_set in  frame_id.values():
                    # print("目标集合：")
                    # print(object_set)
                    for object in object_set:
                        if object["gt_type"]==0:  #car
                            car_id.append(object["id"])
                            R=pow(reduce(lambda x,y:x+y, (map(lambda x: x**2, object["gt_relativePos"]))),0.5)  #print(object["gt_relativePos"]), print(square_sum)
                            car_R.append(np.around(R,2))
                            car_angle.append(np.around(object["radial_angle"],2))
                           
                            # total_points=object["total_points"]
                            gt_size_V=reduce(lambda x,y:x*y,object["gt_size"])  #print(object["gt_relativePos"]), print(square_sum)
                            point_size_V=reduce(lambda x,y:x*y,object["point_size"]) 
                            car_O.append(np.around((1-point_size_V/gt_size_V),2))
                            # point_density=np.around(object["total_points"]/point_size_V,2)
                            # car_point_density.append(point_density)
                            car_score.append(np.around(object["result_prob"],2))                                                              
                        else:
                            print("other type object:{}".format(object["gt_type"]))
            gt = labels["fns"]   #list 10 帧点云
            for frame_id in gt :  #24270 dict  哪一帧print(frame_id), print("\n")
                for object_set in  frame_id.values():
                    # print("目标集合：")
                    # print(object_set)
                    for object in object_set:
                        if object["gt_type"]==0:  #car
                            car_id.append(object["id"])
                            R=pow(reduce(lambda x,y:x+y, (map(lambda x: x**2, object["gt_relativePos"]))),0.5)  #print(object["gt_relativePos"]), print(square_sum)
                            car_R.append(np.around(R,2))
                            car_angle.append(np.around(object["radial_angle"],2))
                           
                            # total_points=object["total_points"]
                            gt_size_V=reduce(lambda x,y:x*y,object["gt_size"])  #print(object["gt_relativePos"]), print(square_sum)
                            point_size_V=reduce(lambda x,y:x*y,object["point_size"]) 

                            # point_density=np.around(object["total_points"]/point_size_V,2)
                            # car_point_density.append(point_density)

                            car_O.append(np.around((1-point_size_V/gt_size_V),2))
                            car_score.append(0.00)                                                              
                        else:
                            print("other type object:{}".format(object["gt_type"]))            
        fjson.close()
    else:
        print("file no exist")

    car=pd.DataFrame()
    car['id']=car_id
    car['R']=car_R
    car['angle']=car_angle
    car['occusion']=car_O
    # car['density']=car_point_density
    car['score']=car_score
    car=car.sample(frac=1).reset_index(drop=True)
    car.to_csv("./Result/FactorCarSix.csv",index=False)


                
                           
                                           


        
   