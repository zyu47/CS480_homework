import numpy as np
import matplotlib.pyplot as plt

######################################################################
# Machine Learning Utilities. 
#
#  draw (a neural network)
######################################################################



######################################################################
# Associated with  neuralnetworks.py
# Draw a neural network with weights in each layer as a matrix
######################################################################

def draw(VsArg,WArg, inputNames = None, outputNames = None, gray = False):
    def isOdd(x):
        return x % 2 != 0

    W = VsArg + [WArg]
    nLayers = len(W)

    # calculate xlim and ylim for whole network plot
    #  Assume 4 characters fit between each wire
    #  -0.5 is to leave 0.5 spacing before first wire
    xlim = max(map(len,inputNames))/4.0 if inputNames else 1
    ylim = 0
    
    for li in range(nLayers):
        ni,no = W[li].shape  #no means number outputs this layer
        if not isOdd(li):
            ylim += ni + 0.5
        else:
            xlim += ni + 0.5

    ni,no = W[nLayers-1].shape  #no means number outputs this layer
    if isOdd(nLayers):
        xlim += no + 0.5
    else:
        ylim += no + 0.5

    # Add space for output names
    if outputNames:
        if isOdd(nLayers):
            ylim += 0.25
        else:
            xlim += round(max(map(len,outputNames))/4.0)

    ax = plt.gca()

    x0 = 1
    y0 = 0 # to allow for constant input to first layer
    # First Layer
    if inputNames:
        # addx = max(map(len,inputNames))*0.1
        y = 0.55
        for n in inputNames:
            y += 1
            ax.text(x0-len(n)*0.2, y, n)
            x0 = max([1,max(map(len,inputNames))/4.0])

    for li in range(nLayers):
        Wi = W[li]
        ni,no = Wi.shape
        if not isOdd(li):
            # Odd layer index. Vertical layer. Origin is upper left.
            # Constant input
            ax.text(x0-0.2, y0+0.5, '1')
            for li in range(ni):
                ax.plot((x0,x0+no-0.5), (y0+li+0.5, y0+li+0.5),color='gray')
            # output lines
            for li in range(no):
                ax.plot((x0+1+li-0.5, x0+1+li-0.5), (y0, y0+ni+1),color='gray')
            # cell "bodies"
            xs = x0 + np.arange(no) + 0.5
            ys = np.array([y0+ni+0.5]*no)
            ax.scatter(xs,ys,marker='v',s=1000,c='gray')
            # weights
            if gray:
                colors = np.array(["black","gray"])[(Wi.flat >= 0)+0]
            else:
                colors = np.array(["red","green"])[(Wi.flat >= 0)+0]
            xs = np.arange(no)+ x0+0.5
            ys = np.arange(ni)+ y0 + 0.5
            aWi = abs(Wi)
            aWi = aWi / np.max(aWi) * 20 #50
            coords = np.meshgrid(xs,ys)
            #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
            ax.scatter(coords[0],coords[1],marker='s',s=aWi**2,c=colors)
            y0 += ni + 1
            x0 += -1 ## shift for next layer's constant input
        else:
            # Even layer index. Horizontal layer. Origin is upper left.
            # Constant input
            ax.text(x0+0.5, y0-0.2, '1')
            # input lines
            for li in range(ni):
                ax.plot((x0+li+0.5,  x0+li+0.5), (y0,y0+no-0.5),color='gray')
            # output lines
            for li in range(no):
                ax.plot((x0, x0+ni+1), (y0+li+0.5, y0+li+0.5),color='gray')
            # cell "bodies"
            xs = np.array([x0 + ni + 0.5]*no)
            ys = y0 + 0.5 + np.arange(no)
            ax.scatter(xs,ys,marker='>',s=1000,c='gray')
            # weights
            Wiflat = Wi.T.flatten()
            if gray:
                colors = np.array(["black","gray"])[(Wiflat >= 0)+0]
            else:
                colors = np.array(["red","green"])[(Wiflat >= 0)+0]
            xs = np.arange(ni)+x0 + 0.5
            ys = np.arange(no)+y0 + 0.5
            coords = np.meshgrid(xs,ys)
            aWi = abs(Wiflat)
            aWi = aWi / np.max(aWi) * 20 # 50
            #ax.scatter(coords[0],coords[1],marker='o',s=2*np.pi*aWi**2,c=colors)
            ax.scatter(coords[0],coords[1],marker='s',s=aWi**2,c=colors)
            x0 += ni + 1
            y0 -= 1 ##shift to allow for next layer's constant input

    # Last layer output labels 
    if outputNames:
        if isOdd(nLayers):
            x = x0+1.5
            for n in outputNames:
                x += 1
                ax.text(x, y0+0.5, n)
        else:
            y = y0+0.6
            for n in outputNames:
                y += 1
                ax.text(x0+0.2, y, n)
    ax.axis([0,xlim, ylim,0])
    ax.axis('off')
