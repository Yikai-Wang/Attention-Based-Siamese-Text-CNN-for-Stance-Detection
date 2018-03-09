import csv
result = list(csv.reader(open("predictions_test.csv", "r", encoding="utf-8")))
result.remove(result[0])
real_result = [i[0] for i in result]

oringin_answer = list(csv.reader(
	open("test_stances_unlabeled.csv", "r", encoding="utf-8")))
oringin_answer.remove(oringin_answer[0])

real_answer = []

for item in oringin_answer:
	if item:
		real_answer.append(item[2])


unrelated_samples = 0
for item in real_answer:
	if item == 'unrelated':
		unrelated_samples += 1
related_samples = len(real_answer) - unrelated_samples

unrelated_result = 0
related_result = 0

samples_num = len(real_result)
for i in range(samples_num):
	if real_result[i] == real_answer[i]:
		if real_result[i] == 'unrelated':
			unrelated_result += 1
		else:
			related_result += 1

score1 = unrelated_result/unrelated_samples
score2 = related_result/related_samples
relative_score = score1*0.25+score2*0.75

print(relative_score)


	




