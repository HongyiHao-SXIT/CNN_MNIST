import os

# 指定要处理的目录
directory = 'pupil_dataset/train'

# 遍历指定目录下的所有文件
for filename in os.listdir(directory):
    if filename.endswith('.jpg'):
        # 构建新的文件名
        new_filename = 'e9b4fd28-' + filename
        # 构建旧文件的完整路径
        old_file_path = os.path.join(directory, filename)
        # 构建新文件的完整路径
        new_file_path = os.path.join(directory, new_filename)
        # 重命名文件
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {filename} to {new_filename}")