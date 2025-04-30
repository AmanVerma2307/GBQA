import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--choice',
                        type=str,
                        default='margin',
                        help="margin/id curve")
    args = parser.parse_args()

    margin_val = [0,0.01,0.1,0.25,0.5,0.75,1.0]
    lambda_id = [0,0.1,0.25,0.50,1.0,1.5,2.0]

    d_unq = [[0.0146,0.0268,0.0207,0.0359,0.0366,0.0126,0.1635],
            [0.1601,0.1375,0.1972,0.3429,0.0692,0.0216,0.0571],
            [0.3920,0.4371,0.4367,0.3854,0.5062,0.2593,0.5027],
            [0.1232,0.1904,0.0899,0.2176,0.1605,0.1325,0.0957],
            [0.0813,0.0811,0.0575,0.0530,0.0846,0.0578,0.0637],
            [0.4260,0.4315,0.4424,0.3160,0.3047,0.4120,0.3275]]
    d_unq = np.array(d_unq)

    d_vrb = [[0.0753,0.1290,0.0817,0.2239,0.1323,0.0416,0.9477],
            [0.4560,0.4815,0.4840,0.5578,0.2576,0.4322,0.2870],
            [0.3909,0.4182,0.3669,0.4110,0.3126,0.3422,0.3502],
            [0.4111,0.4197,0.3301,0.5620,0.3420,0.3562,0.3297],
            [0.3106,0.2763,0.1942,0.1340,0.3291,0.2256,0.2456],
            [0.3425,0.4990,0.4573,0.3737,0.4139,0.3955,0.5275]]
    d_vrb = np.array(d_vrb)

    d_unq_id = [[0.0146,0.0313,0.1285,0.2139,0.2878,0.2829,0.2613],
                [0.1609,0.1944,0.1506,0.1169,0.2116,0.2038,0.2027],
                [0.3920,0.4950,0.4226,0.4409,0.4592,0.5705,0.5596],
                [0.1232,0.1216,0.1683,0.2488,0.3265,0.2969,0.3381],
                [0.0811,0.0905,0.2469,0.2579,0.2959,0.2314,0.2466],
                [0.4260,0.3815,0.2295,0.4754,0.4023,0.4368,0.4552]]
    d_unq_id = np.array(d_unq_id)

    if(args.choice == 'margin'):

        fig, (ax11, ax12) = plt.subplots(nrows=1, ncols=2, figsize=(8,4))

        for ax in [ax11,ax12]:

            if(ax == ax11):
                ax.plot(margin_val,d_unq[0],label='Pinch index',linewidth=3,marker='o',markersize=8)
                ax.plot(margin_val,d_unq[1],label='Palm tilt',linewidth=3,marker='o',markersize=8)
                ax.plot(margin_val,d_unq[2],label='Fast swipe',linewidth=3,marker='o',markersize=8)
                ax.plot(margin_val,d_unq[3],label='Push',linewidth=3,marker='o',markersize=8)
                ax.plot(margin_val,d_unq[4],label='Finger rub',linewidth=3,marker='o',markersize=8)
                ax.plot(margin_val,d_unq[5],label='Circle',linewidth=3,marker='o',markersize=8)
                ax.legend(frameon=True,fontsize=8)
                ax.set_xlabel("margin ($m$)",fontsize=10)            
                ax.set_ylabel('$d_{UNQ}$',fontsize=12)

            if(ax == ax12):
                ax.plot(margin_val,d_vrb[0],label='Pinch index',linewidth=3,marker='o',markersize=8)
                ax.plot(margin_val,d_vrb[1],label='Palm tilt',linewidth=3,marker='o',markersize=8)
                ax.plot(margin_val,d_vrb[2],label='Fast swipe',linewidth=3,marker='o',markersize=8)
                ax.plot(margin_val,d_vrb[3],label='Push',linewidth=3,marker='o',markersize=8)
                ax.plot(margin_val,d_vrb[4],label='Finger rub',linewidth=3,marker='o',markersize=8)
                ax.plot(margin_val,d_vrb[5],label='Circle',linewidth=3,marker='o',markersize=8)
                ax.legend(frameon=True,fontsize=8)
                ax.set_xlabel("margin ($m$)",fontsize=10)            
                ax.set_ylabel('$d_{VRB}$',fontsize=12)

        plt.show()

    if(args.choice == 'id'):

        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4))

        ax.plot(lambda_id,d_unq_id[0],label='Pinch index',linewidth=3,marker='o',markersize=8)
        ax.plot(lambda_id,d_unq_id[1],label='Palm tilt',linewidth=3,marker='o',markersize=8)
        ax.plot(lambda_id,d_unq_id[2],label='Fast swipe',linewidth=3,marker='o',markersize=8)
        ax.plot(lambda_id,d_unq_id[3],label='Push',linewidth=3,marker='o',markersize=8)
        ax.plot(lambda_id,d_unq_id[4],label='Finger rub',linewidth=3,marker='o',markersize=8)
        ax.plot(lambda_id,d_unq_id[5],label='Circle',linewidth=3,marker='o',markersize=8)

        ax.legend(frameon=True,fontsize=8)
        ax.set_xlabel("$\lambda_{ID}$",fontsize=10)            
        ax.set_ylabel('$d_{UNQ}$',fontsize=12)

        plt.show()


if __name__ == "__main__":
    main()