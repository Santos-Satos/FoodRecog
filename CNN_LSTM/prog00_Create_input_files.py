import glob
import os

# 入力に使用するWAVファイルを取得するプログラム
def main():
	# ユーザ変数
	find_dir = "../2019_Noguchi/"
	ch = "6"
	mfcc_dir = "./mfcc/"
	food_list = ["BEF", "BFW", "CBG", "COH", "DRY", "GMI", "GUM", "JL3", "ORO", "RTZ", "W03", "W20"]

	# ここから処理開始
	
	# base_dir以下のch.wavをすべて取得
	fs = glob.glob(find_dir+"**/*_"+ch+".wav", recursive=True)
	
	lines = ""
	cnt = 0
	for i in range(len(fs)):
		for food in food_list:
			if food not in fs[i].split("/")[-1]:continue
			lines += os.path.abspath(fs[i]) + " " + os.path.abspath(mfcc_dir+str(cnt)+".bin") + " " + food + "\n"
			cnt += 1
			break
	with open("input_info.txt", mode="w") as f:
		f.write(lines)
	

if __name__ == "__main__":
	main()
