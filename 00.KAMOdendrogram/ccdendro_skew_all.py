import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.cluster import hierarchy
from scipy.stats import skewnorm

def set_sampledict(alpha,loc,scale,delta_cc,ndata):
    # CC(A-B) is set here
    cc_diff = loc - delta_cc

    sample_dict=[{"name":"A-A", "alpha":alpha,"loc":loc,"scale":scale},
             {"name":"A-B", "alpha":alpha,"loc": cc_diff,"scale":scale},
             {"name":"B-B","alpha":alpha,"loc":loc,"scale":scale}]

    # Figure name
    prefix="alpha_%.1f_%.3f_%.3f_dcc_%.2f_nd_%05d" % (
        sample_dict[1]['alpha'],sample_dict[1]['loc'], sample_dict[1]['scale'],delta_cc,ndata)

    files=glob.glob("%s*"%prefix)
    n_exist=len(files)
    if n_exist != 0:
        index=n_exist
        prefix="%s_%02d"%(prefix,index)

    # Dendrogram title
    title_s = "INPUT: (%s: alpha: %8.3f loc:%8.3f scale:%8.3f)(%s alpha:%8.3f loc:%8.3f scale:%8.3f)(%s: alpha:%8.3f loc:%8.3f scale:%8.3f)" \
        % (sample_dict[0]['name'], sample_dict[0]['alpha'],sample_dict[0]['loc'], sample_dict[0]['scale'], \
        sample_dict[1]['name'], sample_dict[1]['alpha'],sample_dict[1]['loc'], sample_dict[1]['scale'], \
        sample_dict[2]['name'], sample_dict[2]['alpha'],sample_dict[2]['loc'], sample_dict[2]['scale'])

    return sample_dict, prefix, title_s

def make_skew_random_cc(str_combination, sample_dict):
    for idx,s in enumerate(sample_dict):
        if s['name']==str_combination:
            break
    alpha=s['alpha']
    loc=s['loc']
    scale=s['scale']
    # print(str_combination,alpha,loc,scale)
    randcc=skewnorm.rvs(alpha, loc, scale)

    return randcc

def make_CC_table(n_sample, sample_dict):
    # half of n sample belongs to each type of molecule
    n_half = int(n_sample/2)
    sample_list=[]
    for i in np.arange(0,n_half):
        sample_list.append("A")
    for i in np.arange(0,n_half):
        sample_list.append("B")

    dist_list = []
    name_list=[]
    cc_list=[]

    aa=[]
    ab=[]
    bb=[]

    ofile=open("cc.dat","w")

    for idx1,s1 in enumerate(sample_list):
        for s2 in sample_list[idx1+1:]:
            if s1=="A" and s2=="A":
                name_list.append("A-A")
                cctmp = make_skew_random_cc("A-A", sample_dict)
                aa.append(cctmp)
            elif s1=="B" and s2=="B":
                name_list.append("B-B")
                cctmp = make_skew_random_cc("B-B", sample_dict)
                bb.append(cctmp)
            else:
                name_list.append("A-B")
                cctmp = make_skew_random_cc("A-B", sample_dict)
                ab.append(cctmp)

            if cctmp>1.0:
                cctmp=1.0
            dist = np.sqrt(1-cctmp*cctmp)
            ofile.write("%9.5f\n"%cctmp)
            dist_list.append(dist)
            cc_list.append(cctmp)

    ofile.close()
    # Making numpy array before returning
    aa_a=np.array(aa)
    bb_a=np.array(bb)
    ab_a=np.array(ab)

    return sample_list, dist_list, name_list, cc_list, aa_a, bb_a, ab_a

    # make_CC_table

def makeFigure(figure_prefix, input_title, dist_list, sample_list, aaa, bba, aba):
    # Histgram of CC
    fig = plt.figure(figsize=(25,10))
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95) 
    spec = gridspec.GridSpec(ncols=2, nrows=1, width_ratios=[1, 5])
    ax1=fig.add_subplot(spec[0])
    ax2=fig.add_subplot(spec[1])

    ax1.set_xlim(0.70,1.0)
    #ax1.hist([apo_apo,apo_ben,ben_ben],bins=20,label=["apo-apo","apo-ben", "ben-ben"],alpha=0.5)
    ax1.hist(aaa,bins=20,alpha=0.5,label="AA")
    ax1.hist(aba,bins=20,alpha=0.5,label="AB")
    ax1.hist(bba,bins=20,alpha=0.5,label="BB")
    ax1.legend(loc="upper left")

    outfile=open("results.dat","w")
    outfile.write("AA(mean,std,median)=%12.5f %12.5f %12.5f\n"% (aaa.mean(), aaa.std(), np.median(aaa)))
    outfile.write("AB(mean,std,median)=%12.5f %12.5f %12.5f\n"% (aba.mean(), aba.std(), np.median(aba)))
    outfile.write("BB(mean,std,median)=%12.5f %12.5f %12.5f\n"% (bba.mean(), bba.std(), np.median(bba)))
    outfile.close()

    Z = hierarchy.linkage(dist_list, 'ward')
    title_result="\nAA(mean:%5.3f std:%5.3f median:%5.3f) AB(mean:%5.3f std:%5.3f median:%5.3f) BB(mean:%5.3f std:%5.3f median:%5.3f)" % \
        (aaa.mean(), aaa.std(), np.median(aaa), \
        aba.mean(), aba.std(), np.median(aba), \
        bba.mean(), bba.std(), np.median(bba))

    plt.title(input_title+title_result)

    dn = hierarchy.dendrogram(Z,labels=sample_list, leaf_font_size=10)
    #ax2.annotate("Comment here", (2, 4), xycoords='data',
    #          xytext=(1, 1), textcoords='offset points',
    #          arrowprops=dict(arrowstyle="->",
    #                          connectionstyle="arc3,rad=-0.2"))

    plt.savefig("%s.jpg"%figure_prefix)
    #plt.show()

# Main routine
alpha=-10.0
loc=0.98
delta_cc = 0.01
scale=0.03

n_sample=1000

for delta_cc in [0.01,0.02,0.03]:
    for scale in [0.03, 0.04, 0.05, 0.06, 0.10]:
        for n_sample in [100,200,500,1000]:
            sample_dict, figure_prefix, input_title = set_sampledict(alpha,loc,scale,delta_cc,n_sample)
            sample_list, dist_list, name_list, cc_list, aaa, bba, aba = make_CC_table(n_sample, sample_dict)
            makeFigure(figure_prefix, input_title, dist_list, sample_list, aaa, bba, aba)
