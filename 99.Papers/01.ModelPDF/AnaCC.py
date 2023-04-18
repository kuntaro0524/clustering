import logging
# matplotlib pyplotのインポート
import matplotlib.pyplot as plt
import pandas as pd

# 入力される様々な方法に対して、CC分布の統計値を計算・プロットするクラス
class AnaCC():
    def __init__(self):
        print("initialization!")

        # loggerを設定する
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        # ファイルに出力する
        fh = logging.FileHandler('AnaCC.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)
        # コンソールに出力する
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        self.isDebug = False

    # cluster_number は文字列で指定する
    def read_clusters(self, cluster_number, file_path="CLUSTERS.txt"):
        # CLUSTERS.txtを読み取り、クラスター番号を指定して、そのクラスターに含まれるIDのリストを返す
        id_list = []

        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                cols = line.split()
                if cols[0] == cluster_number:
                    id_list = [int(ID) - 1 for ID in cols[3:]]
                    break
        return id_list

    def read_cctable(self, cluster_number, clusters_file="CLUSTERS.txt", cctable_file="cctable.dat"):
        # cctable.datを読み取り、クラスター番号を指定して、そのクラスターに含まれるCCの値のリストを返す
        id_list = self.read_clusters(cluster_number, clusters_file)

        cc_values = []

        with open(cctable_file, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                cols = line.split()
                i, j = int(cols[0]), int(cols[1])

                if i in id_list and j in id_list:
                    cc_values.append(float(cols[2]))

        # cc_values が格納されたDataFrameを返す
        ret = pd.DataFrame(cc_values, columns=["cc"])
    
        return ret

    def get_id_list_from_clusters(self, cluster_number, file_path="CLUSTERS.txt"):
        id_list = []

        with open(file_path, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                cols = line.split()
                #print(cols[0])
                if cols[0] == cluster_number:
                    id_list = [int(ID) - 1 for ID in cols[3:]]
                    break

        return id_list

    def get_cc_various_values_from_cctable(self, cluster_number1, cluster_number2, clusters_file="CLUSTERS.txt", cctable_file="cctable.dat", listname="filenames.lst"):
        id_list1 = self.get_id_list_from_clusters(cluster_number1, clusters_file)
        id_list2 = self.get_id_list_from_clusters(cluster_number2, clusters_file)

        # isDebugがTrueの場合は、id_list1とid_list2を表示する
        if self.isDebug:
            print(id_list1)
            print(id_list2)

        cc_values = []
        cctype_list=[]
        i_list=[]
        j_list=[]

        # 同じクラスタIDに含まれるインデックスどうし、または異なるクラスタIDに含まれるインデックスどうしのCC値を取得する
        # iとjの両方がid_list1に含まれている場合には対応するCC値を取得→type: "AA"をキーとしてccをdictに格納→配列として保存する
        # iがid_list1に含まれているか、jがid_list2に含まれている場合 またはiがid_list2に含まれているか、jがid_list1に含まれている場合には "AB" をキーとしてccをdictに格納→配列として保存する
        # iとjの両方がid_list2に含まれている場合には "BB" をキーとしてccをdictに格納→配列として保存する

        with open(cctable_file, "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                cols = line.split()
                i, j = int(cols[0]), int(cols[1])
                # iとjの両方がid_list1に含まれている場合には対応するCC値を取得→type: "AA"をキーとしてccをdictに格納→配列として保存する
                if i in id_list1 and j in id_list1:
                    cctype_list.append("AA")
                    cc_values.append(float(cols[2]))
                    i_list.append(i)
                    j_list.append(j)
                # iとjの両方がid_list2に含まれている場合には "BB" をキーとしてccをdictに格納→配列として保存する
                elif i in id_list2 and j in id_list2:
                    cctype_list.append("BB")
                    cc_values.append(float(cols[2]))
                    i_list.append(i)
                    j_list.append(j)
                # i, j がどちらにも含まれていない場合には、何もしない
                elif (i not in id_list1 and i not in id_list2) or (j not in id_list1 and j not in id_list2):
                    continue
                # 残りはAB
                else:
                    cctype_list.append("AB")
                    cc_values.append(float(cols[2])) 
                    i_list.append(i)
                    j_list.append(j)

        # cc_values が格納されたDataFrameを返す
        ret = pd.DataFrame(cc_values, columns=["cc"])
        # retにcctype_listを追加する
        ret["cctype"] = cctype_list
        # retにi_listを追加する
        ret["i"] = i_list
        # retにj_listを追加する
        ret["j"] = j_list
    
        return ret

    def crossCheck(self, cluster1_name, cluster2_name, cc_thresh=0.9165, clusters_file="CLUSTERS.txt", cctable_file="cctable.dat", listname="filenames.lst"):
        # cluster1_nameとcluster2_nameのクラスタIDを取得する
        id_list1 = self.get_id_list_from_clusters(cluster1_name, clusters_file)
        # cluster1_nameとcluster2_nameのクラスタIDを取得する
        id_list2 = self.get_id_list_from_clusters(cluster2_name, clusters_file)

        # self.get_cc_various_values_from_cctable()を用いてdataframeを取得する
        df_all = self.get_cc_various_values_from_cctable(cluster1_name, cluster2_name, clusters_file, cctable_file=cctable_file)

        # CC値がcc_thresh以上のもののみを取り出す
        df_all = df_all[df_all["cc"] >= cc_thresh]

        # "AB"のi, jの値を取り出す
        df_AB = df_all[df_all["cctype"] == "AB"]
        # error flag
        error_flag = False
        # df_ABに含まれる行を１行ずつ処理する
        for index, row in df_AB.iterrows():
            i, j = row["i"], row["j"]
            # i がid_list1に含まれている場合には cluster1_name とする
            # i がid_list2に含まれている場合には cluster2_name とする
            # j がid_list1に含まれている場合には cluster1_name とする
            # j がid_list2に含まれている場合には cluster2_name とする
            if i in id_list1:
                cluster_i = cluster1_name
            elif i in id_list2:
                cluster_i = cluster2_name
            if j in id_list1:
                cluster_j = cluster1_name
            elif j in id_list2:
                cluster_j = cluster2_name
            
            # 正しくラベルがついているか確認する
            # 期待しているのは cluster_i と cluster_j が異なる場合のみ
            if cluster_i != cluster_j:
                self.logger.info(f"i: {i}, j: {j}, cluster_i: {cluster_i}, cluster_j: {cluster_j}")
                print("i, j, cluster_i, cluster_j", i, j, cluster_i, cluster_j)
            else:
                self.logger.info(f"ERROR i: {i}, j: {j}, cluster_i: {cluster_i}, cluster_j: {cluster_j}")
                print("i, j, cluster_i, cluster_j", i, j, cluster_i, cluster_j)
                print("Error: cluster_i == cluster_j")
                error_flag = True

        return error_flag
    
    # CCの統計値を計算して表示する
    def calcAndShowCC(self, cluster_number1, cluster_number2, threshold=0.8, clusters_file="CLUSTERS.txt", cctable_file="cctable.dat", listname="filenames.lst"):
        df_all = self.get_cc_various_values_from_cctable(cluster_number1, cluster_number2, clusters_file, cctable_file=cctable_file)
        
        # CC値がthreshold以上のもののみを取り出す
        df_all = df_all[df_all["cc"] >= threshold]

        # AAのCC値のみを取り出す
        df_AA = df_all[df_all["cctype"] == "AA"]
        # BBのCC値のみを取り出す
        df_BB = df_all[df_all["cctype"] == "BB"]
        # ABのCC値のみを取り出す
        df_AB = df_all[df_all["cctype"] == "AB"]
        
        # mean, std, median,データ数をそれぞれのDataframeから計算して１行１データフレームで表示
        # AA
        df = pd.DataFrame([df_AA["cc"].mean(), df_AA["cc"].std(), df_AA["cc"].median(), len(df_AA)], index=["mean", "std", "median", "count"], columns=["AA"])
        # BB
        df["BB"] = [df_BB["cc"].mean(), df_BB["cc"].std(), df_BB["cc"].median(), len(df_BB)]
        # AB
        df["AB"] = [df_AB["cc"].mean(), df_AB["cc"].std(), df_AB["cc"].median(), len(df_AB)]

        # データフレームをcsvファイルに出力する
        # AA, BB, AB の順に結果を表示する
        # 1行目はAA それぞれのCC値の平均値、標準偏差、中央値、データ数を表示する
        # 2行目はBB それぞれのCC値の平均値、標準偏差、中央値、データ数を表示する
        # 3行目はAB それぞれのCC値の平均値、標準偏差、中央値、データ数を表示する
        # 桁数は小数点以下4桁まで
        df.to_csv("CC.csv", float_format="%.4f")
        
    # filesname.lstには、ファイル名が1行1つずつ記載されている
    # そのファイルとcctaable.datからCCの組み合わせを取得する
    def get_cc_from_filenamelst(self,threshold=0.8):
        # filesname.lstを読み込む
        # １行につき１サンプル名が入っている
        # 読み込んだらサンプル名をリストに格納する
        with open("filenames.lst", "r") as f:
            lines = f.readlines()
            filenames = [line.strip() for line in lines]

        # cctaable.datを読み込む
        # i,j,cc, nref が１行に空白区切りで入っている
        # 1行ずつ処理する
        # i, j はそれぞれfilenameのインデックスに対応する
        # filesname[i]とfilesname[j]のファイル名を取得して表示する
        self.apo_apo = []
        self.apo_ben = []
        self.ben_ben = []
        with open("cctable.dat", "r") as f:
            lines = f.readlines()
            for line in lines[1:]:
                cols = line.split()
                i, j = int(cols[0]), int(cols[1])

                # もしファイル名が "apo" "apo"であれば
                # apo_apoのリストにCCの数値を追加する
                if filenames[i] == "apo" and filenames[j] == "apo":
                    self.apo_apo.append(float(cols[2]))
                # もしファイル名が "benz" "benz"であれば
                # ben_benのリストにCCの数値を追加する
                elif filenames[i] == "benz" and filenames[j] == "benz":
                    self.ben_ben.append(float(cols[2]))
                else:
                    self.apo_ben.append(float(cols[2]))

        # それぞれをデータフレームに変換して表示する
        df_apo_apo = pd.DataFrame(self.apo_apo, columns=["cc"])
        df_apo_ben = pd.DataFrame(self.apo_ben, columns=["cc"])
        df_ben_ben = pd.DataFrame(self.ben_ben, columns=["cc"])

        # CC値がthreshold以上のもののみを取り出す
        df_apo_apo = df_apo_apo[df_apo_apo["cc"] >= threshold]
        df_apo_ben = df_apo_ben[df_apo_ben["cc"] >= threshold]
        df_ben_ben = df_ben_ben[df_ben_ben["cc"] >= threshold]

        print(df_apo_ben)

        # mean, std, median, データ数 をそれぞれのDataframeから計算して１行１データフレームで表示
        df_mean = pd.DataFrame([df_apo_apo["cc"].mean(), df_apo_ben["cc"].mean(), df_ben_ben["cc"].mean()], columns=["mean"], index=["apo_apo", "apo_ben", "ben_ben"])
        df_std = pd.DataFrame([df_apo_apo["cc"].std(), df_apo_ben["cc"].std(), df_ben_ben["cc"].std()], columns=["std"], index=["apo_apo", "apo_ben", "ben_ben"])
        df_median = pd.DataFrame([df_apo_apo["cc"].median(), df_apo_ben["cc"].median(), df_ben_ben["cc"].median()], columns=["median"], index=["apo_apo", "apo_ben", "ben_ben"])
        df_count = pd.DataFrame([df_apo_apo["cc"].count(), df_apo_ben["cc"].count(), df_ben_ben["cc"].count()], columns=["count"], index=["apo_apo", "apo_ben", "ben_ben"])
        df = pd.concat([df_mean, df_std, df_median, df_count], axis=1)
        print(df)

        # データフレームをcsvファイルに出力する
        # AA, BB, AB の順に結果を表示する
        # 1行目はAA それぞれのCC値の平均値、標準偏差、中央値、データ数を表示する
        # 2行目はBB それぞれのCC値の平均値、標準偏差、中央値、データ数を表示する
        # 3行目はAB それぞれのCC値の平均値、標準偏差、中央値、データ数を表示する
        # 桁数は小数点以下4桁まで
        df.to_csv("CC.csv", float_format="%.4f")
        

# mainが定義されていなかったら
if __name__ == "__main__":
    import sys
    ana = AnaCC()
    #print(ana.read_cctable("0072"))
    print("From cluster number")
    cluster1=sys.argv[1]
    cluster2=sys.argv[2]
    # しきい値を引数から得る
    # しきい値が指定されていなければ0.8を使う
    if len(sys.argv) > 3:
        threshold = float(sys.argv[3])
    else:
        threshold = 0.9165

    # check if this function works
    error_flag = ana.crossCheck(cluster1, cluster2, cc_thresh=0.9165, clusters_file="CLUSTERS.txt", cctable_file="cctable.dat", listname="filenames.lst")
    # if error_flag is True, then there is an error
    if error_flag:
        print("Error: cross check failed")
        sys.exit(1)
    else:
        print("######################")
        ana.calcAndShowCC(cluster1, cluster2, threshold=threshold)
        #ana.get_cc_from_filenamelst(threshold=threshold)