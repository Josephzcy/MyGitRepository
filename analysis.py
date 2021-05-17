import os
import pandas as pd
import numpy as np
from scipy.optimize import leastsq
from numpy import linalg as la
from numpy import*  #include mat
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import copy
import seaborn as sns

path="./DataSource/CarDataAlloy/"  


def ReadData(path_data):  # 传入存储的list    
    list_path_data=[]
    print("是否开始执行")
    print(os.path.exists(path_data))
    if os.path.exists(path_data):
        for ｆdata in os.listdir(path_data):
            pathdata = os.path.join(path_data,fdata)    #当前文件路径
            list_path_data.append(pathdata) 
    return list_path_data

  
def PointToPlaneR(x,y,z,PlanePara):
    d=np.abs(PlanePara[0]*x+PlanePara[1]*y+PlanePara[2]*z+PlanePara[3])/np.linalg.norm(PlanePara[:3])
    d=np.around(d,4)
    return d


def SourceDataVisualization(data1,data2):
    fig=plt.figure(figsize=(20,20)) 
    ax_Source= fig.add_subplot(111, projection='3d')
    ax_Source.set_xlabel('X Label')
    ax_Source.set_ylabel('Y Label')
    ax_Source.set_zlabel('Z Label')

    x1=data1["Points:0"]
    y1=data1["Points:1"]
    z1=data1["Points:2"]
    
    x2=data2["Points:0"]
    y2=data2["Points:1"]
    z2=data2["Points:2"]

    # ax_Source.plot([0],[0],[0],c='r',marker='o')
    ax_Source.scatter(x1,y1,z1,c='r',marker='o',s=80)
    ax_Source.scatter(x2,y2,z2,c='y',marker='o',s=80) 

#outiers handing 四分位数法
def Quartile_Outiers_Removing(data):
    print("yuanshi:",len(data))
    # data[["intensity","distance_m"]].boxplot()
    percentile=np.percentile(data["distance_m"],[0,25,50,75,100])
    IQR=percentile[3]-percentile[1]
    UpLimit=percentile[3]+1*IQR
    DownLimit=percentile[1]-1*IQR

    for index in data.index.tolist():
        if data["distance_m"][index]>=UpLimit or data["distance_m"][index]<=DownLimit:
                data.drop(index=index,inplace=True)        
    data.reset_index(drop=True,inplace=True)
    print("distance:",len(data))
    print(data["intensity"].describe())
    I_percentile=np.percentile(data["intensity"],[0,25,50,75,100])
    I_IQR=I_percentile[3]-I_percentile[1]
    I_UpLimit=I_percentile[3]+1*I_IQR
    I_DownLimit=I_percentile[1]-1*I_IQR

    for index in data.index.tolist():
        if data["intensity"][index]>I_UpLimit or data["intensity"][index]<I_DownLimit:
                data.drop(index=index,inplace=True)        
    data.reset_index(drop=True,inplace=True)
    print("intensity:",len(data))

def PointToPointR(row,center):
    x=row["Points:0"]-center[0]
    y=row["Points:1"]-center[1]
    z=row["Points:2"]-center[2]
    d=sqrt(pow(x,2)+pow(y,2)+pow(z,2))
    d=np.around(d,3)
    return d

def SvdGetPlanePara(fitdata):
    ColumnsName=["Points:0","Points:1","Points:2"]
    x=fitdata["Points:0"]
    y=fitdata["Points:1"]
    z=fitdata["Points:2"]
    xmean=x.mean()
    ymean=y.mean()
    zmean=z.mean()

    ParaMatrix=mat([x-xmean,y-ymean,z-zmean]).T
    U,sigma,VT=la.svd(ParaMatrix)
    a,b,c=VT.T[:,2]
    d=-(a*xmean+b*ymean+c*zmean)
    print(a,b,c,d)
    
    a=a.tolist()[0][0]
    b=b.tolist()[0][0]
    c=c.tolist()[0][0]
    d=d.tolist()[0][0]

    return np.array([a,b,c,d])

def DrawFitPlane(fitdata,PlanePara):
    x=fitdata["Points:0"]
    y=fitdata["Points:1"]
    z=fitdata["Points:2"]
    
    print("Svd Fit Plane:")
    fig=plt.figure(figsize=(20,20)) 
    axsvd= fig.add_subplot(111, projection='3d')
    axsvd.set_xlabel('X Label')
    axsvd.set_ylabel('Y Label')
    axsvd.set_zlabel('Z Label')
    axsvd.axis([min(x),max(x),min(y),max(y)])
    
    # xy=np.meshgrid(x,y)
    # J=-(PlanePara[0]*xy[0]+PlanePara[1]*xy[1]+PlanePara[3])/PlanePara[2]
    # axsvd.plot_surface(xy[0],xy[1],J, rstride=1, cstride=1, cmap='spring')  

    # axsvd.plot([0],[0],[0],c='r',marker='o')
    axsvd.scatter(x,y,z,c='r',marker='o',s=80)
    
def SvdGetIncidentAngle(fitdata,SvdPlanePara):  
    '''
    平面的法向量数据：ax+by+c+d=0   单位方向向量n=(a,b,c),
    过原点的直线的方向向量:(x,y,z)
    l_mod=d
    '''
    x=fitdata["Points:0"]
    y=fitdata["Points:1"]
    z=fitdata["Points:2"]

    lmod=fitdata["distance_m"]
    n_mod=np.linalg.norm(SvdPlanePara[:3], ord=2) #n_mod=1
    l= [ np.array( [ x.iloc[i],y.iloc[i],z.iloc[i] ] ) for i in range(len(x)) ] 
   
    ln=np.around([abs(np.sum(SvdPlanePara[:3]*l[i]))  for i in range(len(l))],3)
   
    cosAngle=ln/lmod
    Angle=np.around(np.degrees(np.arccos(cosAngle)),3)
    return Angle

def DataMearge(FileList,DataSet):
    ColumnsName=["intensity","Points:0","Points:1","Points:2","distance_m"]
    for FilePath in FileList:
        
        fitdata=np.around(pd.read_csv(FilePath)[ColumnsName],3)
        print("FileName:",FilePath,len(fitdata))
       
        Quartile_Outiers_Removing(fitdata)
        print("filtering:",len(fitdata))

        data1=copy.deepcopy(fitdata)  #source data
        data2=copy .deepcopy(fitdata)
        if len(data1)<4:
            continue
        center=np.around([fitdata["Points:0"].mean(),fitdata["Points:1"].mean(),fitdata["Points:2"].mean()],3)
        print("AreaCenter:",center)
        d=fitdata.apply(lambda row: PointToPointR(row,center),axis=1)
        for index in d.index.tolist():  
            if d[index]>2*d.std():  
                    data1.drop(index=index,inplace=True)
            if d[index]<2*d.std():  
                    data2.drop(index=index,inplace=True)
        fitdata.reset_index(drop=True,inplace=True)
        
    
        print(len(data1),len(data2))
        SourceDataVisualization(data1,data2)


        # IMean=np.around(data1["intensity"].mean())
        # dMean=np.around(data1["distance_m"].mean(),3)

        I=np.around(data1["intensity"].mean())
        d=np.around(data1["distance_m"].mean(),3)


        SvdPlanePara=SvdGetPlanePara(data1)
        SvdAngle=SvdGetIncidentAngle(data1,SvdPlanePara)
        AngleMean=np.around(SvdAngle.mean())
        
        NewData=pd.DataFrame({'intensity':[I],'distance_m':[d],'Angle':[AngleMean]})
        DataSet= pd.concat([DataSet,NewData],ignore_index=True)      
    DataSet.to_csv("./DataSet/datasetthree.csv",index=False)
    # return  DataSet
    return  0

FilePath="./Data/FactocCar.csv"
      
if __name__=="__main__":

    pd.read_csv(FilePath)
    factor_data=pd.read_csv(FilePath)

    print(factor_data)
    # print("score:",factor_data['score'].value_counts());print("\n")
    print("angle:",factor_data['angle'].value_counts());print("\n")
    print("occusion:",factor_data['occusion'].value_counts());print("\n")
    print("R:",factor_data['R'].value_counts());print("\n")
    
    # plt.figure(1);plt.hist(factor_data['score'],orientation = 'vertical',histtype = 'bar', color ='red');plt.title('score')
    plt.figure(2);plt.hist(factor_data['angle'],orientation = 'vertical',histtype = 'bar', color ='red');plt.title('angle')
    plt.figure(3);plt.hist(factor_data['occusion'],orientation = 'vertical',histtype = 'bar', color ='red',bins=20);plt.title('occusion')
    plt.figure(4);plt.hist(factor_data['R'],orientation = 'vertical',histtype = 'bar', color ='red',bins=16);plt.title('R')

    # S=f(R) where angle ==3.08 and occusion==0.83
    print("S=f(R) where angle ==3.08 and occusion==0.17")
    d_condition=(factor_data['angle']==0.05 ) & (factor_data['occusion']==0.17)
    d_data=factor_data[d_condition]
    print(d_data)


    # S=f(angle) where R ==12.45 and occusion==0.83
    print("S=f(angle) where R ==12.45 and occusion==0.83")
    angle_condition=(factor_data['R']==12.45 ) & (factor_data['occusion']==0.17)
    angle_data=factor_data[angle_condition]
    print(angle_data)

    # S=f(O) where R ==12.45 and angle ==2.99
    print("S=f(angle) where R ==12.45 and angle ==2.99")
    angle_condition=(factor_data['R']==12.45 ) & (factor_data['angle']==2.99)
    angle_data=factor_data[angle_condition]
    print(angle_data)


    #correlation analysis
    correlation = factor_data.corr()
    plt.figure(5);plt.title('Correlation of Numeric Features with Score')
    sns.heatmap(correlation,square = True, vmax=0.8)
    sns.pairplot(factor_data,size = 2 ,kind ='scatter',diag_kind='kde')

    

    
    

    # print("R:",factor_data['R'].value_counts());print("\n")

    # ColumnsName=["intensity","distance_m","Angle"]
    # DataSet= pd.DataFrame(columns=ColumnsName)      #将不同的文件拼接起来,并更新索引

    # FileList=ReadData(path)
    # print(FileList[:2])
    # DataSet=DataMearge(FileList[:1],DataSet)
    plt.show()





   

   
    