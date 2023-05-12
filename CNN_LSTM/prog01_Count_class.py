
# foodクラスのデータ数をカウントするプログラム
def main():

	# ここから処理
	with open("input_info.txt", mode="r") as f:
		src_txt = f.read()
	
	food_class = []
	data_cnt = {}
	i = 0
	for line in src_txt.split("\n"):
		if line == "": break
		
		food = line.split(" ")[-1] 
		if food not in food_class:
			food_class.append(food)
			data_cnt[food] = 0
		data_cnt[food] += 1
		i+=1
	
	print(data_cnt)
	
	with open("Food_class.txt", mode="w") as f:
		f.write(" ".join(food_class))

if __name__ == "__main__":
	main()
