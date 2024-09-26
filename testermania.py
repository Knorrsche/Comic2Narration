import pandas as pd

def pk_score(your_data, evaluator_data, k):
    N = len(your_data)
    mismatches = 0
    total_windows = N - k

    for i in range(total_windows):
        if (your_data[i] == your_data[i + k]) != (evaluator_data[i] == evaluator_data[i + k]):
            mismatches += 1

    pk = mismatches / total_windows
    return pk

data = []
data.append(pd.read_json(r'C:\Users\derra\Desktop\Paper\Alley_Oop.json'))
data.append(pd.read_json(r'C:\Users\derra\Desktop\Paper\Champ.json'))
data.append(pd.read_json(r'C:\Users\derra\Desktop\Paper\Treasure_Comics.json'))
data.append(pd.read_json(r'C:\Users\derra\Desktop\Paper\Western_Love.json'))
data.append(pd.read_json(r'C:\Users\derra\Desktop\Paper\Person_1.json'))
data.append(pd.read_json(r'C:\Users\derra\Desktop\Paper\Person_2.json'))
data.append(pd.read_json(r'C:\Users\derra\Desktop\Paper\Person_3.json'))
data.append(pd.read_json(r'C:\Users\derra\Desktop\Paper\Person_4.json'))
data.append(pd.read_json(r'C:\Users\derra\Desktop\Paper\Person_5.json'))
data.append(pd.read_json(r'C:\Users\derra\Desktop\Paper\Person_6.json'))

original_data = data[:4]
person_data = data[4:]

pk_scores_list = []

for i, orig_df in enumerate(original_data):
    orig_chapters = pd.json_normalize(orig_df['comic']['chapters'])
    for j, person_df in enumerate(person_data):
        person_chapters = pd.json_normalize(person_df['comic']['chapters'])

        common_titles = set(orig_chapters['title']).intersection(set(person_chapters['title']))

        for title in common_titles:
            orig_chapter = orig_chapters[orig_chapters['title'] == title].iloc[0]
            person_chapter = person_chapters[person_chapters['title'] == title].iloc[0]

            orig_pages = [panel for page in orig_chapter['pages'] for panel in page]
            person_pages = [panel for page in person_chapter['pages'] for panel in page]

            k = 5
            score = pk_score(person_pages, orig_pages, k)

            pk_scores_list.append({
                'title': title,
                'pk_score': score
            })

pk_scores_df = pd.DataFrame(pk_scores_list)

average_pk = pk_scores_df['pk_score'].mean()

print(pk_scores_df)
print(f"Average Pk Score: {average_pk}")
