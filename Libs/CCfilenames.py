import os,sys
import matplotlib.pyplot as plt

class CCfilenames:
    def __init__(self, ccfile_list, ccfile_nouse, ccvalue_file):
        self.ccfile_list = ccfile_list
        self.ccfile_nouse = ccfile_nouse
        self.ccvalue_file = ccvalue_file
        self.isInit = False
    
    def init(self):
        self.allfilees = {}
        lines = open(self.ccfile_list).readlines()
        for i,filename in enumerate(lines):
            # i, filenameを辞書に登録する
            self.allfilees[i] = filename
        # no-use
        self.nouse = {}
        lines = open(self.ccfile_nouse).readlines()
        for i,filename in enumerate(lines):
            # i, filename を self.nouseに登録する
            self.nouse[i] = filename

        # cctable.dat
        #  0   16  0.8344  147
        #  0   17  0.8825  203
        #  0   18  0.4317   46
        #  0   19  0.7400  203
        #  0   20  0.7030   85
        #  0   21  0.8013  119
        self.ccvalues = []
        lines = open(self.ccvalue_file).readlines()
        for i,ccvalue in enumerate(lines[1:]):
            # 左から順に、index1, index2, ccvalue, n_reflsの順に１行に入っている
            # これを、self.ccvaluesに登録する
            cols = ccvalue.split()
            index1 = int(cols[0])
            index2 = int(cols[1])
            ccvalue = float(cols[2])
            n_refls = int(cols[3])
            tmp_dic = {}
            tmp_dic["index1"] = index1
            tmp_dic["index2"] = index2
            tmp_dic["ccvalue"] = ccvalue
            tmp_dic["n_refls"] = n_refls
            self.ccvalues.append(tmp_dic)

        # self.ccvalues を pandads.DataFrameに変換する
        import pandas as pd
        self.ccdf = pd.DataFrame(self.ccvalues)
        self.isInit = True

    def plotCC(self):
        if self.isInit == False:
            self.init()

        # dataframeを利用して、index1, index2, ccvalueのマップをプロットする
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.ccdf["index1"], self.ccdf["index2"], c=self.ccdf["ccvalue"], cmap="jet")
        # plt.show()
        plt.savefig("cc.png")
    
    def findUnuseData(self):
        if self.isInit == False:
            self.init()

        # self.nouseに含まれている情報は
        # (i, filename) と、ファイルインデックスとファイル名
        # self.allfiles にも同様に登録されているので、
        # self.nouseにあるファイル名がself.allfilesにあるかを確認し、
        # あった場合には、self.allfilesにあるインデックスを登録する
        self.no_use_index = []
        for i in self.nouse:
            filename = self.nouse[i]
            for j in self.allfilees:
                if filename == self.allfilees[j]:
                    self.no_use_index.append(j)

    def groupPlot(self):
        if self.isInit == False:
            self.init()
        self.findUnuseData()
        # self.ccdfの中で index1 もしくは index2　に
        # self.no_use_indexに含まれるインデックスがある場合には、
        # その行は"悪いデータ”として、self.ccdfに、bad_flag = True
        # を設定する。
        self.ccdf["bad_flag"] = False
        for i in self.ccdf.index:
            index1 = self.ccdf["index1"][i]
            index2 = self.ccdf["index2"][i]
            if index1 in self.no_use_index or index2 in self.no_use_index:
                self.ccdf["bad_flag"][i] = True
        # bad_flag が True のものと Falseのもので色をかえて、index1, index2, ccvalueをプロットする
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.ccdf["index1"][self.ccdf["bad_flag"] == False], self.ccdf["index2"][self.ccdf["bad_flag"] == False], c=self.ccdf["ccvalue"][self.ccdf["bad_flag"] == False], cmap="jet")
        ax.scatter(self.ccdf["index1"][self.ccdf["bad_flag"] == True], self.ccdf["index2"][self.ccdf["bad_flag"] == True], c=self.ccdf["ccvalue"][self.ccdf["bad_flag"] == True], cmap="jet", marker="x")
        # 特に数値がマイナスの場合には 黒色を設定する.
        # ただしdotを小さく表示する
        ax.scatter(self.ccdf["index1"][self.ccdf["ccvalue"] < 0], self.ccdf["index2"][self.ccdf["ccvalue"] < 0], c="black", cmap="jet", s=5)

        # plt.show()
        plt.savefig("cc.png")

    def checkCCtalbe(self):
        if self.isInit == False:
            self.init()
        
        ccindices = []
        # self.ccdfに含まれているindex1, index2に含まれる整数を
        # 網羅して配列にする
        for i in self.ccdf["index1"]:
            ccindices.append(i)
        for j in self.ccdf["index2"]:
            ccindices.append(j)
        # ccindices の重複を削除
        ccindices = list(set(ccindices))
        print(len(ccindices))
        
if __name__ == "__main__":
    ccfile = CCfilenames(sys.argv[1],sys.argv[2],sys.argv[3])
    # ccfile.init()
    ccfile.groupPlot()