####### Importing Libraries
import numpy as np
import matplotlib.pyplot as plt

###### Curve Plot
def plot():
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    x_axes = np.arange(start=0,stop=12,step=2)

    ax.bar(x_axes,res3d[0],zs=0,color='navy',label='Multi-scale Res3D-ViViT')
    ax.bar(x_axes+0.8,msres3d[0],zs=0,color='maroon',label='TDSNet')

    ax.bar(x_axes,res3d[1],zs=-0.05,zdir='y',color='navy')
    ax.bar(x_axes+0.8,msres3d[1],zs=-0.05,zdir='y',color='maroon')

    ax.invert_yaxis()
    ax.set_zlim(0.0,np.max(res3d[1]+msres3d[1])+0.01)

    ax.set_zlabel('$d_{UNQ}$',fontsize=12)
    ax.set_ylabel('$Accuracy~(\%)$',fontsize=12)
    ax.set_xlabel('$Gesture$',fontsize=12)

    ax.set_xticks(x_axes,labels=g_list,fontsize=5.5,rotation=15)
    ax.legend(frameon=True,fontsize=8)

    plt.show()

###### Joint Plotting

##### Defining essentials
g_list = ['Pinch index', 'Palm tilt', 'Fast swipe', 'Push', 'Finger rub', 'Circle',]

res3d = [[100.0,84.80,45.60,91.20,88.80,76.00],
        [0.2011,0.2721,0.3652,0.1918,0.3380,0.2826],
        [0.3150,0.3638,0.1850,0.3816,0.3808,0.3512]]

msres3d = [[99.20,93.60,71.20,92.00,90.40,86.40],
        [0.1920,0.2116,0.3666,0.1084,0.3111,0.3501],
        [0.3676,0.3097,0.1860,0.2360,0.3566,0.4075]]

res3d_vivit = [[100.0,92.00,75.20,100.00,93.60,56.80],
               [0.1533,0.3220,0.4290,0.1689,0.3166,0.5631],
               [0.2901,0.4659,0.3067,0.2957,0.3566,0.4675]]

msres3d_vivit = [[100.00,96.00,76.80,92.80,99.20,71.20],
                 [0.1097,0.2510,0.3990,0.2043,0.1812,0.4613],
                 [0.2081,0.4068,0.2733,0.3687,0.2947,0.4656]]

tdsnet = [[100.00,98.40,85.60,96.80,99.20,68.00],
          [0.0148,0.1661,0.3920,0.1232,0.0813,0.4268],
          [0.0580,0.3484,0.3143,0.3713,0.2554,0.3284]]

fig, (ax11, ax12) = plt.subplots(nrows=2, ncols=1, figsize=(4,8), subplot_kw=dict(projection='3d'))

for ax in [ax11, ax12]:

    if(ax == ax11):

        x_axes = np.arange(start=0,stop=12,step=2)

        ax.bar(x_axes,msres3d_vivit[0],zs=0,color='navy',label='Multi-scale Res3D-ViViT')
        ax.bar(x_axes+0.8,tdsnet[0],zs=0,color='maroon',label='TDSNet')

        ax.bar(x_axes,msres3d_vivit[1],zs=-0.2,zdir='y',color='navy')
        ax.bar(x_axes+0.8,tdsnet[1],zs=-1.0,zdir='y',color='maroon')

        ax.invert_yaxis()
        ax.set_zlim(0.0,np.max(msres3d_vivit[1]+tdsnet[1])+0.01)

        ax.set_zlabel('$d_{UNQ}$',fontsize=12)
        ax.set_ylabel('$Accuracy~(\%)$',fontsize=12)
        ax.set_xlabel('$Gesture$',fontsize=12)

        ax.set_xticks(x_axes,labels=g_list,fontsize=5.5,rotation=15)
        ax.legend(frameon=True,fontsize=8)
        ax.view_init(elev=26, azim=-69) 

    if(ax == ax12):

        x_axes = np.arange(start=0,stop=12,step=2)

        ax.bar(x_axes,msres3d_vivit[0],zs=0,color='navy',label='Multi-scale Res3D-ViViT')
        ax.bar(x_axes+0.8,tdsnet[0],zs=0,color='maroon',label='TDSNet')

        ax.bar(x_axes,msres3d_vivit[2],zs=0,zdir='y',color='navy')
        ax.bar(x_axes+0.8,tdsnet[2],zs=-1.0,zdir='y',color='maroon')

        ax.invert_yaxis()
        ax.set_zlim(0.0,np.max(msres3d_vivit[2]+tdsnet[2])+0.01)

        ax.set_zlabel('$d_{VRB}$',fontsize=12)
        ax.set_ylabel('$Accuracy~(\%)$',fontsize=12)
        ax.set_xlabel('$Gesture$',fontsize=12)

        ax.set_xticks(x_axes,labels=g_list,fontsize=5.5,rotation=15)
        ax.legend(frameon=True,fontsize=8)
        ax.view_init(elev=26, azim=-69) 

plt.show()




