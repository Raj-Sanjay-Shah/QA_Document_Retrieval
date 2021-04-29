from bert import QA

def compare(item1):
    return item1[1]
top_3_docs = 'app.json'
model = QA('model')
import json
with open(top_3_docs) as f:
  data = json.load(f)

for i in data['documents']:
    print("New Document")
    ans = []
    list_para = i.split('\n\n')
    for j in list_para:
        answer = model.predict(j,data['question'])
        ans.append([answer[0]['answer'],answer[0]['confidence']])
    sorted(ans, key=compare,reverse = True)
    print(*ans[0:5],sep='\n')
