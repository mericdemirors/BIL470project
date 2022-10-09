def get_votes(link, vote_threshold=10000):
    import requests, re
    request_text = requests.get(link).text
    if "No Ratings Available" in request_text or "Well, what if there is no webpage?" in request_text or int(((request_text[request_text.index("IMDb users have given a ")-15:request_text.index("IMDb users have given a ")-1].strip()).replace(",",""))) < vote_threshold:
        return [0,0,0,0,0,0,0,0,0,0], 0.0

    star_votes = re.findall("<div class=\"leftAligned\">(.*?)</div>", request_text)
    rating = re.findall("span class=\"ipl-rating-star__rating\">(.*?)</span>", request_text)
    stars = [int(vote.replace(",","")) for vote in star_votes[1:11]]
    
    return stars, float(rating[0])

def fill_dataset(dataset, end):
    import os
    cwd = os.getcwd()
    for i in range(6000, end):
        try:
            stars, rating = get_votes(dataset.iloc[i]["rating link"])
            dataset.loc[i,["1","2","3","4","5","6","7","8","9","10"]] = stars
            dataset.loc[i,["Rating","Votes"]] = rating, sum(stars)
        except:
            pass

        if i%1000==0 and i>0:
            print(str(i)+". movie done")
            dataset.to_csv(cwd+"/movie_data_till_"+str(i)+".csv", index=False)

# MAIN
import pandas as pd
dataset = pd.read_csv("movie_data_at_5000_6000.csv")
fill_dataset(dataset, 6001)