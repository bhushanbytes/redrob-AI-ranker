import json

with open("data/candidates.jsonl", "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        # candidate = json.loads(line)

        # print(f"\nCandidate {i+1}")
        # print(candidate["candidate_id"])
        # print(candidate["profile"]["current_title"])
        # print(candidate["profile"]["years_of_experience"])
        # print(candidate["profile"]["location"])
        # # print(candidate["profile"]["headline"])
        # # print(candidate["profile"]["summary"])
        # print(candidate["skills"])
        # print(candidate["career_history"])

        candidate = json.loads(line)

        print(candidate["career_history"][0]["description"])
        


        if i ==  0 :
            break