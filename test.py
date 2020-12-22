def all_teams(k):
    teams=[]
    for i in range(k+1):
        for j in range(k-i+1):
            team=[0]*i+[1]*j+[2]*(k-j-i)
            teams.append(teams)
            #print(team)
    return teams

for i in range(8):
    print(i,len(all_teams(i)))