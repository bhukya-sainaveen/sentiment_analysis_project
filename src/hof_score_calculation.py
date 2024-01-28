# hof_score_calculation.py

def count(text, word_list):
    # Function to calculate each count of HATE, OFFENSE and PROFANE words
    cnt = 0
    for val in text:
        if val in word_list:
            cnt += 1
    return cnt
