import numpy as np
import datetime
import seaborn as sns
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


def load_npy_data(file_path):
    data = np.load(file_path)
    return data

def extract_state(bufferINFO):
    #Extract (x,y)
    bufferINFO = np.array(bufferINFO)
    x = np.ones((len(bufferINFO), len(bufferINFO[0])))
    y = np.ones((len(bufferINFO), len(bufferINFO[0])))
    psi = np.ones((len(bufferINFO), len(bufferINFO[0])))
    v = np.ones((len(bufferINFO), len(bufferINFO[0])))
    for i in range(0, len(bufferINFO)):
        for j in range(0, len(bufferINFO[i])):
            x[i, j] = bufferINFO[i][j, 0]
            y[i, j] = bufferINFO[i][j, 1]
            psi[i,j] = bufferINFO[i][j, 2]
            v[i, j] = bufferINFO[i][j, 4]
    return x,y,psi,v

def distance(x1,x2,y1,y2):
    d = np.sqrt((x1-x2)**2+(y1-y2)**2)
    return d

def degree_distance(psi_1,psi_2):
    d = abs(np.degrees(psi_1-psi_2))
    if d >=180:
        d = 360-d
    return d

def Show_Performance(bufferINFO,file_name):
    '''
    Relative_distance:
    r_d = 1/N Σ |<x_i-x_j,y_i-y_j>|

    Relative heading:
    r_h = 1/N Σ |ψ_i-ψj|
    '''
    if len(bufferINFO) != 0:
        x,y,psi,v = extract_state(bufferINFO)
    else:
        return None
    # Draw the relative distance plot
    pd.options.display.notebook_repr_html = False  # 表格显示
    plt.rcParams['figure.dpi'] = 350  # 图形分辨率
    sns.set_theme(style='darkgrid')  # 图形主题
    Rd = []
    Rh = []
    Rd_l = []
    Rh_l = []
    nx = np.mat(x)
    for i in range(1, nx.shape[1]):
        Rd_i = []
        Rh_i = []
        Rd_l_i = []
        Rh_l_i = []
        for t in range(nx.shape[0]):
            r_d = [distance(x[t,i],x[t,j],y[t,i],y[t,j]) for j in range(1, x.shape[1])]
            r_h = [degree_distance(psi[t,i],psi[t,j]) for j in range(1, x.shape[1])]
            Rd_i.append(r_d)
            Rh_i.append(r_h)
            r_d = distance(x[t,i],x[t,0],y[t,i],y[t,0])
            r_h = degree_distance(psi[t,i],psi[t,0])
            Rd_l_i.append(r_d)
            Rh_l_i.append(r_h)

        Rd_i = np.array(Rd_i)
        Rh_i = np.array(Rh_i)
        Rd.append(Rd_i.T)
        Rh.append(Rh_i.T)

        Rd_l.append(Rd_l_i)
        Rh_l.append(Rh_l_i)
    path_Rd_f = './Performance/Relative Distance/Relative_Distance' + file_name + '_followers.svg'
    path_Rh_f = './Performance/Relative Heading/Relative_Heading' + file_name + '_followers.svg'
    print('开始绘制Followers系列图：')
    Draw_plot_f(Rh, path_Rh_f)
    Draw_plot_f(Rd, path_Rd_f)
    print('开始绘制Leader系列图：')
    path_Rd_l = './Performance/Relative Distance/Relative_Distance' + file_name + '_leader.svg'
    path_Rh_l = './Performance/Relative Heading/Relative_Heading' + file_name + '_leader.svg'
    Draw_plot_l(Rh_l, path_Rh_l)
    Draw_plot_l(Rd_l, path_Rd_l)

def Draw_plot_l(Rd,path):
    for i in range(len(Rd)):
        df = pd.DataFrame(dict(X=range(len(Rd[i])), Y=Rd[i]))
        #sns.scatterplot(x=df['X'], y=df['Y'])
        plt.plot(df['X'], df['Y'], linestyle='-.', label='Follower%d' %(i+1))
    plt.legend()
    # plt.show()
    plt.savefig(path, format='svg', bbox_inches='tight')
    plt.clf()


def Draw_plot_f(Rh,path):
    Rh = np.array(Rh)

    label = ['Follower%d'%(i+1) for i in range(Rh.shape[0])]
    df = []
    for i in range(Rh.shape[0]):
        df.append(pd.DataFrame(Rh[i]).melt(var_name='Timesteps', value_name='Relative'))
        df[i]['Follower'] = label[i]
    df = pd.concat(df)  # 合并
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="Timesteps", y="Relative", hue="Follower", style="Follower", data=df)
    #plt.show()
    plt.savefig(path, format='svg', bbox_inches='tight')
    plt.clf()


def get_cruve(n_agent):
    Data = []
    path = './Performance/Reward/' + str(n_agent)+'agents.svg'
    for i in range(1,4):
        data = np.load('./data/reward data/'+str(n_agent)+'agents_'+str(i)+'/'+'flocking999000'+'.npy')
        Data.append(data)
    df = []
    df.append(pd.DataFrame(Data).melt(var_name='Timestep', value_name='Reward'))
    df = pd.concat(df)
    sns.lineplot(data=df, x=r'Timestep', y="Reward")
    plt.tick_params(labelsize=12)
    plt.savefig(path, format='svg', bbox_inches='tight')
    plt.clf()

'''

    df[i]['Algorithms']= label[i]


df=pd.concat(df) # 合并

plt.show()
'''


if __name__ == '__main__':
    #file_name = 'flocking999000.npy'
    #bufferINFO = load_npy_data('./data/flocking data/2agents_5/'+file_name)
    #Show_Performance(bufferINFO,file_name)
    Data = get_cruve(5)