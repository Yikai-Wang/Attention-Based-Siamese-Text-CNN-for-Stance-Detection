import csv
train_percent = 0.9

csv_file_body = list(csv.reader(open('train_bodies_origin.csv','r',encoding='utf-8')))
header_body = csv_file_body[0]
csv_file_body.remove(header_body)

csv_file_stance = list(csv.reader(open('train_stances_origin.csv','r',encoding='utf-8')))
header_stance = csv_file_stance[0]
csv_file_stance.remove(header_stance)

bodies = {}
bodies_test = {}
bodies_train_list = []
bodies_test_list = []

stances_train_list = []
stances_test_list = []

body_length = len(csv_file_body)

index = 0
for item in csv_file_body:
	bodies[item[0]] = item[1]
	if index < body_length*train_percent:
		bodies_train_list.append(item)
	else:
		bodies_test[ item[0] ] = item[1]
		bodies_test_list.append(item)
	index += 1

for item in csv_file_stance:
	if item[1] in bodies_test:
		stances_test_list.append(item)
	else:
		stances_train_list.append(item)

with open("train_bodies.csv","w",encoding="utf-8") as train_bodies_new:
	writer1 = csv.writer(train_bodies_new)
	writer1.writerow(header_body)
	writer1.writerows(bodies_train_list)

with open("train_stances.csv","w",encoding="utf-8") as train_stances_new:
	writer2 = csv.writer(train_stances_new)
	writer2.writerow(header_stance)
	writer2.writerows(stances_train_list)

with open("test_bodies.csv","w",encoding="utf-8") as test_bodies_new:
	writer3 = csv.writer(test_bodies_new)
	writer3.writerow(header_body)
	writer3.writerows(bodies_test_list)

with open("test_stances_labeled.csv","w",encoding="utf-8") as test_stances_new:
	writer4 = csv.writer(test_stances_new)
	writer4.writerow(header_stance)
	writer4.writerows(stances_test_list)
# index = 0
# for item in csv_file_stance:
# 	stances[(item[0], item[1])] = item[2]
# 	if index < sample_length*train_percent:
# 		stances_train[ (item[0] ,item[1] )] = item[2]
# 	else:
# 		stances_test[ (item[0] ,item[1] )] = item[2]
# 	index += 1