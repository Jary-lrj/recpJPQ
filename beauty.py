# 初始化一个字典来存储用户和他们的 item 数量
user_item_count = {}

# 读取文件
with open("data.txt", "r") as file:
    for line in file:
        user, item = line.strip().split()  # 假设用户和 item 用空格分隔
        if user in user_item_count:
            user_item_count[user] += 1
        else:
            user_item_count[user] = 1

# 找到拥有最多 item 的用户
max_items_user = max(user_item_count, key=user_item_count.get)
max_items_count = user_item_count[max_items_user]

# 输出结果
print(f"拥有最多 item 的用户: {max_items_user}, 数量: {max_items_count}")
