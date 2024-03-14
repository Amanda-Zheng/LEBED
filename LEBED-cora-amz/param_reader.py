import os
default_folder = [
'/Users/xinzheng/Documents/Workspace/git/ICLR24-LeBed/0921-backup/logs-cora/infer',
'/Users/xinzheng/Documents/Workspace/git/ICLR24-LeBed/0921-backup/logs-amazon/infer',
'/Users/xinzheng/Documents/Workspace/git/ICLR24-LeBed/0921-backup/logs-domain/infer',
'/Users/xinzheng/Documents/Workspace/git/ICLR24-LeBed/0921-backup/logs-domain/infer',
'/Users/xinzheng/Documents/Workspace/git/ICLR24-LeBed/0921-backup/logs-domain/infer',
'/Users/xinzheng/Documents/Workspace/git/ICLR24-LeBed/0921-backup/logs-arxiv/infer',
]
file_path = [
[
'PNorm-cora-GCN-0-20230831-170837-473891',
'PNorm-cora-GAT-0-20230831-171218-502672',
'PNorm-cora-GIN-0-20230901-103256-913366',
'PNorm-cora-SAGE-0-20230831-171611-755431',
'PNorm-cora-MLP-0-20230831-172458-603507'
],
[
'PNorm-amazon-photo-GCN-0-20230831-215726-602878',
'PNorm-amazon-photo-SAGE-0-20230831-232723-260254',
'PNorm-amazon-photo-GAT-0-20230831-222402-118603',
'PNorm-amazon-photo-GIN-0-20230901-073723-844890',
'PNorm-amazon-photo-MLP-0-20230901-121126-199959'
],
[
'PNorm-acm-to-dblp-network-GCN-0-MODE-1_2_3_4-DOMAIN-both-20230910-080337-478528',
'PNorm-acm-to-dblp-network-GAT-0-MODE-1_2_3_4-DOMAIN-both-20230909-162918-209506',
'PNorm-acm-to-dblp-network-SAGE-0-MODE-1_2_3_4-DOMAIN-both-20230910-140427-882638',
'PNorm-acm-to-dblp-network-GIN-0-MODE-1_2_3_4-DOMAIN-both-20230909-160033-500087',
'PNorm-acm-to-dblp-network-MLP-0-MODE-1_2_3_4-DOMAIN-both-20230910-114831-111647',
],
[
'PNorm-dblp-to-acm-network-GCN-0-MODE-1_2_3_4-DOMAIN-both-20230908-220503-342140',
'PNorm-dblp-to-acm-network-GAT-0-MODE-1_2_3_4-DOMAIN-both-20230909-004809-188364',
'PNorm-dblp-to-acm-network-SAGE-0-MODE-1_2_3_4-DOMAIN-both-20230908-235644-412717',
'PNorm-dblp-to-acm-network-GIN-0-MODE-1_2_3_4-DOMAIN-both-20230908-235853-262532',
'PNorm-dblp-to-acm-network-MLP-0-MODE-1_2_3_4-DOMAIN-both-20230911-221659-747784',
],
[
'PNorm-network-to-acm-dblp-GCN-0-MODE-1_2_3_4-DOMAIN-both-20230909-004910-491004',
'PNorm-network-to-acm-dblp-GAT-0-MODE-1_2_3_4-DOMAIN-both-20230909-000510-605725',
'PNorm-network-to-acm-dblp-SAGE-0-MODE-1_2_3_4-DOMAIN-both-20230912-054929-142681',
'PNorm-network-to-acm-dblp-GIN-0-MODE-1_2_3_4-DOMAIN-both-20230913-215932-389667',
'PNorm-network-to-acm-dblp-MLP-0-MODE-1_2_3_4-DOMAIN-both-20230909-004550-953675',
],
[
'PNorm-ogb-arxiv-GCN-0-20230912-051829-109861',
'PNorm-ogb-arxiv-GAT-0-20230913-123017-268016',
'PNorm-ogb-arxiv-SAGE-0-20230915-075013-682937',
'PNorm-ogb-arxiv-GIN-0-20230920-004822-916940',
'PNorm-ogb-arxiv-MLP-0-20230917-035232-723246',
]
]
output_file = '/Users/xinzheng/Documents/Workspace/git/ICLR24-LeBed/0921-backup/param_space.txt'
with open(output_file, 'w') as output:
    # 嵌套循环遍历 default_folder 和 file_path
    for i in range(len(default_folder)):
        for j in range(len(file_path[i])):
            filename = os.path.join(default_folder[i], file_path[i][j], 'infer.log')
            with open(filename, 'r') as file:
                lines = file.readlines()

                # 提取最后一行
                last_line = lines[-2]  # 使用 [-2] 来获取倒数第二行

                # 打印或使用最后一行的内容
                print(last_line)

                # 写入最后一行到输出文件
                output.write(last_line)

print(f"All last lines saved to {output_file}")