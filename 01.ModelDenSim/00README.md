Finally: 
Run all_simulate.sh
to use 'main_control.py'

After generation of "cc.dat" use
dendro.py
to draw dendro gram.

other programs are generated during the developement.
calc.sh
calc_y.sh
cc_calc.py
hist.py
make_cctable.py
rot_y_control.py

# 経緯
きんにくんの卒論発表の準備中に以下のことに気づいた
１）論文のリバイスで投稿したシミュレーションのσの数値に間違いがあった。分散の数値をσと思っていた。
２）ランダムにCCを抽出した場合、３者のCCの比較をする場合、２者どうしのCCの比較に「本当は関連がある」はず
→しかし、ランダムにCCを抽出するということはその関係を破棄している

ということで、本当の本当のシミュレーションをするためには
「構造空間にある、ある構造をベースにして、１対多のCC計算について矛盾の内容なCCになるように仕向ける必要がある」

# やっていること
+ PDBモデルをCCP4-PDBSETを利用して全分子を傾ける
+ 変形したPDBモデルからphenix.fmodelでMTZを作成
+ MTZどうしでCC計算ができる

# 試していること　2023/02/08
+ 全分子を傾けるときに、AとBという２つの構造を「X軸について＃＃度回転した構造」「Y軸について??度回転した構造」とした
+ ##や??が0degに近づくと、それらの構造は似通ってくるはずである
+ この回転の角度にガウス分布を想定し、mu, sigmaを変化させてCCまで計算している
