import matplotlib.pyplot as plt
import numpy as np
import math
 
r1=1;r2=1;rd=0
 
angle1_circle = [i * math.pi / 180 for i in range(0, 360)]  
angle2_circle = [i * -6*math.pi / 180 for i in range(0, 360)]  

x =(r1+rd)*np.sin(angle1_circle)+r2*np.sin(angle2_circle) 
y =(r1+rd)*np.cos(angle1_circle)+r2*np.cos(angle2_circle)

plt.plot(x, y, color='red', linewidth=2)
plt.title("livox_prism_scan_lidar")
plt.axis('equal')
plt.axis('scaled')
plt.show()





