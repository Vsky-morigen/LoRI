file_path = r"C:/Users/Vsky/Desktop/ccl/personality/code/LoRI/dataset/empatheticdialogues/train.csv"

with open(file_path, "rb") as f:  # 二进制模式
    for i, line in enumerate(f, 1):
        try:
            line.decode("utf-8")
        except UnicodeDecodeError as e:
            print(f"Line {i} has invalid utf-8: {e}")
            # 打印出出错位置前后20个字节，方便定位坏字符
            bad_bytes = line[e.start-10:e.end+10]
            print("Nearby bytes:", bad_bytes)
            break  # 先只看第一个错误
