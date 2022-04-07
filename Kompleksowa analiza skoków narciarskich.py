#!/usr/bin/env python
# coding: utf-8

# <h1>Kompleksowa analiza skoków narciarskich<h1>

# ![image.png](attachment:image.png)

# Skoki  narciarskie to sport, któty charakteryzuje się rozbieżnością w wynikach zawodników z różnych krajów. Zawodnicy z niektórych krajów (głównie europejskich i Japonii) stale osiągają lepsze wyniki niż zawodnicy z innych krajów. Ostatecznym celem tego projektu jest zbudowanie modelu, mającego na ocelu określenie co tak naprawde wpływa na długość skoku skoczka narciarskiego. Dodatkowo,przeprowadzona zostanie analiza danych, która może byc ciekawa dla osób, interesujacych się skokami narciarskimi.

# <h2>Pobieranie danych <h2>

# 
# 
# Wszystkie dane do tego projektu zostały zebrane ze strony internetowej FIS. Większość danych, informacje zawierające wyniki zawodów i specyfikacje skoczni, zostały pobrane z Kaggle.
# 
# https://www.kaggle.com/code/dominicbolton/skijump-exploratory-analysis/notebook

# <h2>Import pakietów<h2>

# In[364]:


import math
import seaborn as sns
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns; sns.set_theme(color_codes=True)
import os
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from fuzzywuzzy import process
from fuzzywuzzy import fuzz
import time


# <h2>Ratingi<h2>

# In[175]:


ratings=pd.read_csv(r"all_ratings.csv")
ratings


# <h2>Imiona zawodnikow<h2>

# In[176]:


names = pd.read_csv(r"all_names.csv")
names


# <h2>Wszystkie konkursy<h2>

# In[177]:


comps = pd.read_csv(r"all_comps_r.csv")
comps


# <h2>Rezultaty konkursów<h2>

# In[178]:


results = pd.read_csv(r"all_results.csv")
results


# <h2>Inne statystyki np wiatr <h2>

# In[179]:


stats = pd.read_csv(r"all_stats_r.csv")
stats


# In[180]:


stats=stats.rename(columns={'fis_code': 'id'})
stats


# <h2>Dane personalne o zawodnikach<h2>

# In[181]:


athletes = pd.read_csv(r"all_athlete_data.csv")
athletes.drop(columns=['Unnamed: 0'], inplace=True)
athletes


# 
# <h2>Łaczenie danych w jeden zbior <h2>

# Łączymy kolumny ze zbiorów danych dotyczących zawodów, wyników i sportowców.Stosujemy tu funkcję MERGE.

# In[182]:


athletes_results = pd.merge(left = results, right = athletes, left_on='codex', right_on='xcode')

athletes_results


# In[183]:


athletes_comp_results = pd.merge(left=comps, right=athletes_results, left_on = 'id', right_on = 'id')
athletes_comp_results.tail()


# In[184]:


final = pd.merge(left=stats, right=athletes_comp_results, left_on = 'id', right_on = 'id')
final.tail()


# <h2>Wyświetlanie wszystkich kolumn<h2>

# W tym miejscu mozna zastosowac funkcje, za ktora pomoca moge wyswtelac wszystkie wiersze w dataframie.Jednak nie polecam dlugotrwałego jej stosowania,
# poniewaz bardzo spowalnia działanie notebooka.

# In[185]:



#show all columns when visalizing 
pd.set_option('max_columns', None)


# <h1>Czyszczenie danych<h1>

# <h2>Usuniecie powtarzajacych sie kolumn<h2>

# Niektóre kolumny powtarzały się w wielu zbiorach danych i po polaczeniu maja etykietki z koncowka x i y.Trzeba te duplikaty usunac.

# In[186]:



#usuniecie powtarzajacych sie kolumn
final.rename(columns={'gate_x': 'gate','codex_x': 'comp_codex', 'hill_size_x': 'hill_size', 'xcode': 'athlete_codex', 'gender_y': 'gender','points':'total_points', 'loc': 'ranking'}, inplace=True)
final.drop(columns=['hill_size_y','codex_y','gender_x', 'hill_size_y', 'codex_y','gate_y'], inplace=True)
final.head()


# <h2>Usuwanie niepotrzebnych wartosci<h2>

# Usune sobie note poszczegolnych sedziow,poniewaz wystarczy zwykla nota koncowa, te informacje nie przydadzą się w dalszej analizie.

# In[187]:



final.drop(columns=['note_1', 'note_2', 'note_3', 'note_4', 'note_5','round_type'], inplace=True)



# <h2>Obsługa brakujących wartości<h2>

# <h3>Sprawdzenie ile jest brakujacych wartosci <h3>

# In[188]:


final.isnull().sum()


# <h2>Usuniecie wartosci brakujacych i zastepowanie mediana <h2>

# Nie chce stracic tych wszystkich wierszow ,wiec zdecydowalam:
#     
# *zastapic puste wartosci kolumn wartoscia mediana=>gate.factor, wind.factor, dist_points, note_points, total_points, bib, gate, birthdate, wind, wind_comp, gate_points, ranking, training
# 
# 
# *natomiast usnac wartosci:club, round, weather_type

# In[189]:


df = pd.DataFrame(final)


# In[190]:


# uzupełnijmy
# to remove
df['humid'] = df['humid'].fillna(df['humid'].median())
df['snow'] = df['snow'].fillna(df['snow'].median())
df['air'] = df['air'].fillna(df['air'].median())
df['max_wind'] = df['max_wind'].fillna(df['max_wind'].median())
df['avg_wind'] = df['avg_wind'].fillna(df['avg_wind'].median())
df['min_wind'] = df['min_wind'].fillna(df['min_wind'].median())

df['gate'] = df['gate'].fillna(df['gate'].median())
df['counted_jumpers'] = df['counted_jumpers'].fillna(df['counted_jumpers'].median())
df['all_jumpers'] = df['all_jumpers'].fillna(df['all_jumpers'].median())
df['all_countries'] = df['all_countries'].fillna(df['all_countries'].median())
df['gate.factor'] = df['gate.factor'].fillna(df['gate.factor'].median())
df['wind.factor'] = df['wind.factor'].fillna(df['wind.factor'].median())

df['dist_points'] = df['dist_points'].fillna(df['dist_points'].median())
df['note_points'] = df['note_points'].fillna(df['note_points'].median())
df['total_points'] = df['total_points'].fillna(df['total_points'].median())
df['ranking'] = df['ranking'].fillna(df['ranking'].median())
df['bib'] = df['bib'].fillna(df['bib'].median())
df['training'] = df['training'].fillna(df['training'].median())
df['wind'] = df['wind'].fillna(df['wind'].median())
df['wind_comp'] = df['wind_comp'].fillna(df['wind_comp'].median())
df['birthdate'] = df['birthdate'].fillna(df['birthdate'].median())
df['gate_points'] = df['gate_points'].fillna(df['gate_points'].median())


# In[191]:



df.isnull().sum()


# In[192]:



df = df.dropna()
df.head()


# <h2>Ile wierszy stracilismy?<h2>

# In[193]:


len(final)-len(df)


# <h2>Ustawienie daty konkurus jako miesiac tylko i wylacznie i zapisanie jej w zmiennej month<h2>

# Wartości czasu są przetwarzane tak, aby data  komp zawierała tylko miesiąc .

# In[194]:



df['month'] = df['date'].str[5:7]
df.head()


# <h2>Fuzzy matching<h2>

# Czy kiedykolwiek chciałeś porównać ciągi znaków, które odnoszą się do tej samej rzeczy, ale zostały napisane nieco inaczej, zawierały literówki lub błędy ortograficzne? Jest to problem, z którym można się spotkać w różnych sytuacjach.
# 
# W naszym przypadku dopasowanie rozmyte między nazwami klubów jest minimalizowane przez utworzenie kolumny club-id, która nadaje klubom o wystarczająco podobnych nazwach ten sam identyfikator. Dopasowanie jest prawdopodobnie niedoskonałe i w pewnym stopniu zależy od interpretacji,lecz może pomóc w interpretacji tej zmiennej jeśli użyjemy jej do swoich analiz.
# 
# 

# In[195]:


#fix fuzzydla nazw klubów
#utworzyć kolumnę "identyfikator klubu" dla wystarczająco podobnych nazw klubów

clubs = df[['club', 'nationality']]
club_names = np.sort(clubs.club.unique())

def matchscore(s1, s2, limit=1):
    return fuzz.partial_ratio(s1,s2)

def make_id(names, minscore):
    ids = [1]
    idnum = 1
    for i in range(len(names)-1):
        if matchscore(names[i+1], names[i]) > minscore:
            ids.append(idnum)
        else:
            idnum += 1
            ids.append(idnum)
    return ids

club_ids = make_id(club_names, 70)

club_id_df = pd.DataFrame({'club': club_names, 'club_id': club_ids})

df = pd.merge(left = df, right = club_id_df, left_on='club', right_on='club')


#zmień kolumny na wartości liczbowe tam, gdzie to konieczne
df = df.apply(pd.to_numeric, errors='ignore')


# In[196]:


df.head()


# <h1>Exploratory Data Analysis <h1>

# <h2>Kraje z sukcesami<h2>

# Ski narciarskie to dziwna dyscyplina, w której sukcesy odnoszą głównie zawodnicy z sześciu krajów. Warto byłoby wiec zbadac, czy fakt, iz zawodnik przynalezy do kraju z sukcesami ma wpływ na jego skoki.
# 
# 
# To, czy zawodnik pochodzi z jednego z wymienionych krajów osiągających najlepsze wyniki, jest zapisywane w kolumnie meaningful_nationality.
# 
# 
# 
# 
# 
# W tej kolumnie:
# 
# *jesli zawodnik pochodzi z kraju z sukcesami-wiersz przymuje wartosc jeden
# *bez sukcesow-zero

# In[197]:


conditions = [
    (df['nationality'] == 'GER'),
    (df['nationality'] =='NOR'),
    (df['nationality'] =='AUT'),
    (df['nationality'] =='POL'),
    (df['nationality'] =='SLO'),
    (df['nationality'] =='JPN')
    
    ]

# create a list of the values we want to assign for each condition
values = ['1','1','1','1','1','1']

# create a new column and use np.select to assign values to it using our lists as arguments
df['meaningful_nationality'] = np.select(conditions, values)

# display updated DataFrame
df.head()


# <h2>Czy konkurs odbywal sie w kraju z sukcesami?<h2>

# In[198]:


conditions = [
    (df['country'] == 'GER'),
    (df['country'] =='NOR'),
    (df['country'] =='AUT'),
    (df['country'] =='POL'),
    (df['country'] =='SLO'),
    (df['country'] =='JPN')
    
    ]

# create a list of the values we want to assign for each condition
values = ['1','1','1','1','1','1']

# create a new column and use np.select to assign values to it using our lists as arguments
df['meaningful_country'] = np.select(conditions, values)

# display updated DataFrame
df


# <h2>Ilu zawodników należy do nacji które sie liczą w skokach?<h2>

# <h3>Grupowanie ilu zawodników w naszej bazie nalezy do posczególnych nacji<h3>

# In[199]:


df1=df.groupby('nationality').name.nunique()
df1 = pd.DataFrame(df1).reset_index() 
df1


# <h3>Dodanie kolumny pokazującej czy kraj jest z nacji zwycieskiej czy nie<h3>

# In[200]:


conditions = [
    (df1['nationality'] == 'GER'),
    (df1['nationality'] =='NOR'),
    (df1['nationality'] =='AUT'),
    (df1['nationality'] =='POL'),
    (df1['nationality'] =='SLO'),
    (df1['nationality'] =='JPN')
    
    ]

# utworzenie listy wartości, które chcemy przypisać do każdego warunku
values = ['1','1','1','1','1','1']

#utwórz nową kolumnę i użyj np.select, aby przypisać do niej wartości, używając naszych list jako argumentów
df1['meaningful_nationality'] = np.select(conditions, values)


# wyświetlanie zaktualizowanej ramki danych DataFrame
df1

df1[(df1.meaningful_nationality == '1')]
df1


# <h3>Wyfiltrowanie nacji z sukcesami<h3>

# In[201]:


df2=df1[(df1.meaningful_nationality == '1')]
df2


# <h3>Przeliczenie ilu było w sumie zawodników z nacji z sukcesami <h3>

# In[202]:


Sukces = df2['name'].sum()

print(Sukces)


# <h3>A ile bylo innych zawodnikow przez te lata?<h3>

# In[203]:


df3=df1[(df1.meaningful_nationality == '0')]
df3
Bezsukcesu = df3['name'].sum()

print(Bezsukcesu)


# <h3>A jak to wygláda procentowo?<h3>

# In[204]:


data = [809,714]
labels = ['Znaczace panstwa', 'Nieznaczace pastwa']
explode = (0, 0.1) 
plt.figure(figsize=(7,7))
plt.pie(data,labels=labels,autopct='%1.1f%%',explode=explode,shadow=True, startangle=90)
plt.show()


# A wiec jak widzimy, jest wiecej zawodnikow z nacji z sukcesami, jednak roznica nie jest az tak kolosalna.

# <h2>Zobaczmy tez jak ogolnie przedstawiala sie liczba skoczkow w tych latach w tych panstwach<h2>

# In[205]:


import seaborn as sns
sns.barplot(data=df1.nlargest(15,'name'), x="nationality", y="name")


# W ciagu badanego okresu(11 sezonów),najwiecej zawodnikow pochodzilo z Norwegii, Niemiec, Austrii i Polski a wiec zdecydowanie z najlepszych
# nacji z sukcesami.

# <h2>Ile miejscowek jest w kazdym panstwie do skokow?<h2>

# In[206]:


df1=df.groupby('country').place.nunique()
df1 = pd.DataFrame(df1).reset_index() 
df1


# In[207]:


sns.barplot(data=df1.nlargest(10,'place'), x="country", y="place")


# A wiec nie ma niespodzianki, tak jak widzimy w TV ze najczesnciej konkursy odbywaja sie u nacji z sukcesami, poniewaz na skoki wykladane sa te pieniadze.A w jakich krajach najczesciej?Zobaczmy.
# 
# 
# Miejscowki w skokach zapewne swiadcza ze jest duzo przetsrzeni do szkolenia juniorow w tych krajach. Zobaczmy wiec jak wyglada rozklad wieku w poszczegolnych krajach i podzielimy nieco nasz zbiór danych na kategorie, które nietety nie zostały opisane w oryginalnym zbiorze danych i doszłam do tego sama, przygladając się zbiorowi i znając się na skokach narciarckich.

# <h2>Rozklad wieku,srednia wieku skoczkow w poszczegolnych panstwach<h2>

# Dodajemy kolumnę wiek do naszego zbioru danych- w tym celu odejmujemy rok urodzenia od roku sezonu.

# In[208]:


# wiek
df['age'] = df.season - df.birthdate
df


# <h2>Type-jak wygląda podział na poziomy rywalizacji?<h2>

# Na podstawie kolumny type po dokładnym przeanalizowaniu danych(oczywiście znając i śledząc każdy z poziomó rywalizacji ) wydedukowałam, że poszczególne typy oznaczają różne poziomy rywalizacji:
# 
# *0-PŚ,igrzyska itp
# 
# *1-Puchar Kontynentalny
# 
# *2-LGP
# 
# *3-Fis Cup
# 
# *4-Mamuty
# 
# *5-MŚ
# 
# *6-MŚ Juniorow
# 

# <h2>Zbadam sredni wiek zawodników męskiego PS którzy PUNKTOWALI w danym konkursie<h2>

# Najpierw musimy ustalić kilka rzeczy:
#    
#   1.Wyfiltrujemy sobie wszystkich zawodników którzy PUNKTOWALI w konkursach-czyli przeszli do rundy 2.W tym celu najpierw musze sprawic ze zmienna mowiaca o rundzie zawodow bedzie rozplaszczona na kilka zmiennych, kazda mowiaca o okreslonej rundzie stanie sie zmienna binarna.
#     
#   2.Z tych zawodnikow filtruje sobie mezczyzn
#     
#   3.Filtruje tylko PS
#     
#   4.Obliczam sredni wiek dopiero.

# In[209]:


df_copy = df.copy()

X = df
y = df['round']

nominal = ['round'] 

# enkodowanie one-hot 
temp = X.drop(columns = nominal)#wywalamy kolumne nominal -to sa same zmienne ktorych nie bediemy rozpalszcac na duumies
dummies = pd.get_dummies(X[nominal])#przeksztalcamy te zmienne  na dummies czyli kazda zmienna ma swoje wartosci a tu rozplaszczamy ta zmenna
dfdum = pd.concat([temp,dummies], axis = 1)#laczymy do siebie zmienne z rozplaszczone i te ktorych nie chcielismy zamienic na dummmiesy
dfdum.head()#chcemy zeby to laczenie bylo po kolumnach a nie po wierszach!


# In[210]:


dfdum.columns = dfdum.columns.str.replace(' ', '_')
dfdum
dfdum['round_2nd_round_']=pd.to_numeric(dfdum['round_2nd_round_'])
dfdum


tylko2runda=dfdum[dfdum['round_2nd_round_']==1]
tylko2runda.head()


# In[211]:


#filtruje tylko mezczyzn
mezczyzni=tylko2runda[tylko2runda['gender']=='M']
mezczyzni

#filtruje tylko PS
PS=mezczyzni[mezczyzni['type']==0]
PS.head()


# In[212]:


#grupuje po narodowosci i sezonie oraz wyliczam dla nich srednia wieku
PS1=PS.groupby(['nationality', 'season']).age.mean()
PS2 = pd.DataFrame(PS1).reset_index() 
PS2


# <h3>Wykres,pokazujący średnią wieku dla danego kraju dla zawodników którzy punktowali w PS mezczyzn<h3>

# In[213]:


POL1 = PS2[PS2.nationality == 'POL']
AUT1 = PS2[PS2.nationality == 'AUT']
GER1 = PS2[PS2.nationality == 'GER']
JPN1 = PS2[PS2.nationality == 'JPN']
NOR1 = PS2[PS2.nationality == 'NOR']
SLO1 = PS2[PS2.nationality == 'SLO']
POL1





plt.figure(figsize=(15, 10), dpi=80)


plt.plot(POL1.season, POL1.age, color='darkgreen',label='POL') 
plt.plot(AUT1.season, AUT1.age, color='red',label='AUT') 
plt.plot(GER1.season, GER1.age, color='pink',label='GER') 
plt.plot(JPN1.season, JPN1.age, color='orange',label='JPN') 
plt.plot(NOR1.season, NOR1.age, color='violet',label='NOR')
plt.plot(SLO1.season, SLO1.age, color='yellow',label='SLO')
plt.title('Wiek w poszczegolnych sezonach') 
plt.legend()


# Jest to sredni wiek zawodnika w poszczegolnych sezonach w PS mezczyzn ktorzy punktowali.Jedyne zastrzezenie moze budzic fakt ze np jesli byl zawodnik starszy i wystepowal wiecej razy niz mlodszy i wiecej sie kwalifikowal  to ta srednia jest zawyzona, bo liza sie wszystkie skoki w sezonie.
# 
# 
# W Polsce np obnizenie sredniego wieku w 2012 roku bylo spowodowane odejsciem Malysza a od tego czasu  nasza kadra stopniowo sie starzeje(szczegolnie widocznie po roku 2017) i niestety nie dolaczaja nowi skoczkowie.W 2019 kariere w japonii zawiesil tez np.Noriaki kasai,co dokladnie widac jak bardzo jego wiek powodowal ze srednia wieku w Japonii byla taka duza.

# <h3>Rozkład gęstości wieku<h3>

# In[214]:


sns.kdeplot(data=PS, x='age', shade=True)


# <h2>Ilu zawodnikow punktowalo w poszczegolnych sezonach?<h2>

# In[215]:


df1=PS.groupby('season').name.nunique()
df1 = pd.DataFrame(df1).reset_index() 
df1


# Tyle bylo skoczkow w poszczegolnych sezonach

# In[216]:


df1["name"].iloc[3]


# <h2>A ile średnio sezonów zawodnicy występują w PS?<h2>

# In[217]:


# num_years_seen
df['years_seen'] = df.groupby('athlete_codex').season.transform('nunique')


# In[218]:


PS['years_seen'] = PS.groupby('athlete_codex').season.transform('nunique')


# Kolumna years_seen pokazuje nam, ile sezonow z dostepnych w danych 11 zawodnik wystepowal w PS.Zdecydowałam sie zsumowac lata w ktorej dany zawodnik wystepuje
# poniewaz chce miec oglada na wplyw"calkowitej dlugosci kariery" w tym dziesiecioleciu na wyniki danego zawodnika.

# <h3>Jesli chcemy zobaczyc ile lat w PS wystepowal dany zawodnik,mozemy tu sobie wyfitrowac<h3>

# In[219]:


lata=PS.groupby(['name']).years_seen.mean()
lata2 = pd.DataFrame(lata).reset_index() 
lata2


# In[220]:



ZYLA=lata2[lata2.name=='ZYLA Piotr']
ZYLA


# Mozemy w ten sam sposob wyfiltrowac sobie kogo tylko chcemy :)

# <h3>A teraz pokazmy ile zawodnikow wystepowalo w PS rok,dwa,trzy itd...<h3>

# In[221]:


lata3=lata2.groupby('years_seen').size()
lata3 = pd.DataFrame(lata3).reset_index() 
lata4 = lata3.rename(columns = {0: 'number'}, inplace = False)
lata4.head()


# <h3>Wykres gęstości występowania zawodnika w PS <h3>

# In[222]:


sns.kdeplot(data=lata2, x='years_seen', shade=True)


# <h3>Pokazanie tego samego na bar plocie<h3>

# In[223]:


sns.barplot(data=lata4.nlargest(15,'years_seen'), x="years_seen", y="number")


# <h3>A w jakich miesiącach odbywał się PŚ?<h3>

# In[224]:


miesiac1=PS.groupby(['season','month'])['id'].count()
miesiac1=pd.DataFrame(miesiac1).reset_index()
miesiac1.head()
rok1=miesiac1[(miesiac1['season'] ==2020)]
miesiac1.tail()


sns.barplot(data=rok1, x="month", y="id")


# <H3>Dla porownania zobaczmy w jakich miesiącach odbywał się LGP mezczyzn<h3>

# In[225]:


LGP=mezczyzni[mezczyzni['type']==2]
PS.head()

miesiac1=LGP.groupby(['season','month'])['id'].count()
miesiac1=pd.DataFrame(miesiac1).reset_index()
miesiac1.head()
rok1=miesiac1[(miesiac1['season'] ==2020)]
miesiac1.tail()


sns.barplot(data=rok1, x="month", y="id")


# <h2>Jaka nacja miala najwiecej zwyciezcow?Ktory zawodnik byl tz."zawodnikiem 10-lecia?"<h2>

# In[226]:


PS.tail()


# In[227]:


zwyciezcy=PS[PS['ranking']==1]
zwyciezcy.tail()


# In[228]:


zwyciezcy1=zwyciezcy.id.nunique()
#zwyciezcy1=zwyciezcy.groupby('name').id.nunique()

#zwyciezcy2=pd.DataFrame(zwyciezcy1).reset_index() 
zwyciezcy1


# Mamy dane ze 162 konkursow PS rozgrywanych mniej wiecej przez ostatnie 10-11 lat.Kto najczesciej wygrywal w tych konkursach?

# In[229]:


zwyciezcy2=zwyciezcy.groupby('name').id.nunique()
zwyciezcy2=pd.DataFrame(zwyciezcy2).reset_index()
zwyciezcy2


# In[230]:


sns.barplot(data=zwyciezcy2.nlargest(15,'id'), x="id", y="name")


# Mimo ze zbior danych nie zawiera z pewnoscia wszystkich konkursow w karierach tych zawodnikow(niektorzy zaczeli np dopiero w 2015,a niektorzy jak Malysz zakonczyli w 1 sezonie zbierania tych danych.Jednak widzimy nazwiska z sukcesami. W tym okresie dla tych danych to wlasnie ci zawodnicy byli w topie.)

# <h2>Skocznie do lotów<h2>

# In[231]:


# seperate flying hills as k-point > 160
df.loc[:,'flying_hill'] = df['k.point'] > 160
df.loc[:,'percent_k'] = df.dist/df['k.point']
df.head()


# In[232]:


# seperate flying hills as k-point > 160
PS.loc[:,'flying_hill'] = PS['k.point'] > 160
PS.loc[:,'percent_k'] = PS.dist/PS['k.point']
PS.head()


# <h3>A ile bylo zawodow rozgrywanych na skoczniach mamucich?<h3>

# In[233]:


PS1=PS[(PS['k.point'] >160)]
PS1
Mamuty = PS1['id'].nunique()

print(Mamuty)




# <h2>Ile % zawodow ylo rozgrywane na skoczniach do lotów?<h2>

# In[234]:


data = [22,162]
labels = ['Mamut', 'Zwykła skocznia']
explode = (0, 0.1) 
plt.figure(figsize=(7,7))
plt.pie(data,labels=labels,autopct='%1.1f%%',explode=explode,shadow=True, startangle=90)
plt.show()


# <h2>Gdzie najczesciej odbywaly sie zawody><h2>

# In[235]:


PS['place']=PS['place'].replace('Garmisch-Partenkirchen', 'Garmisch Partenkirchen')
PS['place']=PS['place'].replace('Tauplitz/Bad Mitterndorf', 'Tauplitz/ Bad Mitterndorf')


# In[236]:


miejsca=PS.groupby('place').id.nunique()
miejsca=pd.DataFrame(miejsca).reset_index()
miejsca


# In[237]:


sns.barplot(data=miejsca.nlargest(12,'id'), x="id", y="place")


# <h2>Ile punktów najczęsciej przyznajemy za jeden metr na skoczni?<h2>

# In[238]:


punkty=PS.groupby('meter.value').id.nunique()
punkty=pd.DataFrame(punkty).reset_index()
punkty


# In[239]:


sns.barplot(data=punkty.nlargest(3,'id'), x="meter.value", y="id")


# To niec dziwnego, bo 1.8 punktu to rekompensata przyznawana na skoczniach duzych,1.2 na mamucich które są żadsze i 2.0 na skoczniach małych które są bardzo rzadko.

# <h2>Czy mozna to zrobić łatwiej?Profiling<h2>

# ![image.png](attachment:image.png)

# Mamy tu wydruk z profilingu.Profiling to specjalna funkcja, która wypluwa nam informacje o naszym zbiorze danych.
# Profiling wykoprzystamy sobie, żeby zobaczyć statystyki nie tylko dla PS ale dla całego zbioru danych, żebyśmy nie musieli powtarzac czynnosci, które zrobilismy dla PS. Oczywiscie mozemy przefiltrowac zbior danych po PS i wrzucic do profilingu.
# 
# Jako ze ja zdecydowalam sie wykonac model na danych calkowitych by nie tracic mozliwosci danych oraz przyjrzec sie takze wplywowi plci oraz rangi zawodow na dlugoisc skoku, zostawie akurat profiling dla podstawowego zbioru danych.

# <h2>Z ilu sezonów mamy dane?Z ilu konkursów w sezonie?<h2>

# In[240]:


sezony=PS.groupby('season').id.nunique()
sezony=pd.DataFrame(sezony).reset_index()
sezony


# In[241]:


sns.barplot(data=sezony, x="season", y="id")


# Zazwyczaj w sezonie jest 20-23 konkursy wiec dane sa nieco wybrakowane.Aczkolwiek zawieraja wiekszosc konkursow na podstawie czego mozemy znac ogolna
# sytuacje, jak wygladal sezon .

# <h2>Ile skoków zostalo oddanych w danym roku w PS?<h2>

# In[242]:


liczbaskokow=PS.groupby(['season'])['name'].count()
liczbaskokow=pd.DataFrame(liczbaskokow).reset_index()
liczbaskokow

#sns.barplot(data=sezony, x="season", y="name")


# In[243]:


sns.barplot(data=liczbaskokow, x="season", y="name")


# ![image.png](attachment:image.png)

# A tu mamy wydruk z profilingu ktory pokazuje liczbe wszystkich oddanych skokow w sezonie we wszystkich kategoriach(PS,PK,MS itd.)

# <h1>Rozkłady zmiennych,zależności, histogramy,profiling<h1>

# <h2>Profiling<h2>

# A co jesli nie musimy sie wysilac i niektore pobiezne(oczywiscie algorytm nie rozpozna nam po numerze ps,pk itd) statystyki mozemy wyswietlic
# za pomoca dwoch linijek komendy?

# In[244]:


from pandas_profiling import ProfileReport
profile = ProfileReport(df, title='Skoki narciarskie Zbiór Danych')
profile


# <h2>Spostrzeżenia<h2>

# 
# *mozemy zobaczyc co jest skorelowane  zczym, na orzyklad jest wiele oczywistych rzeczy jak:punkt k jest silnie skorelowany z wielkością
#     skoczni lub gate.factor (punkty za wiatr) sa skorelowane z rozmiarem skoczni(hill.size).Warto do modelu wybrać zmienne ktore nie sa skorelowane same ze soba,
#     poniewaz inaczej mozemy napotkac problem współliniowości, którego nalezy unikać. Np bez sensu bedzie wybranie jednoczesnie tak skorelowanych zmiennych jak ate.factor i hill.size
#     
# *dla ułatwienia mozemy w profilingu wyswietlic tez macierze korelacji
# 
# *w zakladce variables mozemy obejrzec sobie rozklady kazdej zmiennej ktora zamierzamy uzyc do modelu.
# 
# 
# 

# <h2>SNS PAIRPLOT<h2>

# Daje dla próbki zbioru danych.Oczywiście lepiej dać dla całości, jednak trzeba nastawić się na bardzo długie oczekiwanie na wyniki.

# In[198]:


PSs=(PS.sample(100))
PSs.head()


# In[ ]:


sns.pairplot(PSs)


# In[260]:


df.head()


# <h2>Histogram odległości i innych zmiennych które zawre w modelu<h2>

# W modelu nie bede rozrozniac PS,PK itd, wiec biore statystyki z df.

# In[263]:


df1=df[['humid','snow','weather_type','hill_size','k.point','meter.value','type','speed','dist','dist_points','ranking','wind','gate','birthdate','all_jumpers','month','gender','meaningful_nationality','age','years_seen','team']]
df1.head()


# <h3>Histogramy<h3>

# In[247]:


plt.hist(df1['dist'])


# In[248]:


plt.hist(df1['wind'])


# In[249]:


plt.hist(df1['gate'])


# <h2>Scatter ploty<h2>

# <h2>Odleglosc/wiatr<h2>

# In[250]:


x = df1['dist']
y = df1['wind']
plt.scatter(x, y)
plt.rcParams.update({'figure.figsize':(50,15), 'figure.dpi':100})
plt.title('Simple Scatter plot')
plt.xlabel('dist')
plt.ylabel('wind')
plt.show()


# <h2>odleglosc/predkosc na progu<h2>

# In[251]:


x = df1['dist']
y = df1['speed']
plt.scatter(x, y)
plt.rcParams.update({'figure.figsize':(50,15), 'figure.dpi':100})
plt.title('Simple Scatter plot')
plt.xlabel('dist')
plt.ylabel('speed')
plt.show()


# Widzimy ze wyraznie na mamucich skoczniach sa wieksze predkosci na progu.

# <h2>Jak mozna to przedstawic ladniej?-przykład z wiatrem i odległością<h2>

# In[252]:


# Import Data
#df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/mpg_ggplot2.csv")

# Create Fig and gridspec
fig = plt.figure(figsize=(26, 15), dpi= 100)
grid = plt.GridSpec(4, 4, hspace=1.0, wspace=0.2)

# Define the axes
ax_main = fig.add_subplot(grid[:-1, :-1])
ax_right = fig.add_subplot(grid[:-1, -1], xticklabels=[], yticklabels=[])
ax_bottom = fig.add_subplot(grid[-1, 0:-1], xticklabels=[], yticklabels=[])

# Scatterplot on main ax
ax_main.scatter('dist', 'wind', s=df1.dist, c=df1.wind, alpha=.9, data=df1, cmap="tab10", edgecolors='gray', linewidths=.5)

# histogram on the right
ax_bottom.hist(df1.dist, 40, histtype='stepfilled', orientation='vertical', color='deeppink')
ax_bottom.invert_yaxis()

# histogram in the bottom
ax_right.hist(df1.wind, 40, histtype='stepfilled', orientation='horizontal', color='deeppink')

# Decorations
ax_main.set(title='Scatterplot with Histograms \n dist vs wind', xlabel='dist', ylabel='wind')
ax_main.title.set_fontsize(20)
for item in ([ax_main.xaxis.label, ax_main.yaxis.label] + ax_main.get_xticklabels() + ax_main.get_yticklabels()):
    item.set_fontsize(14)

xlabels = ax_main.get_xticks().tolist()
ax_main.set_xticklabels(xlabels)
plt.show()


# <h2>Typy danych<h2>

# Zanim przejdziemy do dalszej analizy mozemy wywietlic typy danych,moze bedzie to pomocne. Zasadniczo powinno sie zawsze przed rozpoczeciem 
# analizy wykonac ten krok.

# <h2>Statystyki<h2>

# Podstawowe statystyki tylko wartosci liczbowych

# In[254]:


df1.describe(exclude = 'object')


# <h2>Korelacja- w skokach jak wiadomo wiele rzeczy ma wplyw na inne rzeczy<h2>

# In[264]:


import seaborn as sn
df.corr()
sns.set(font_scale=3.0)
corrMatrix = df1.corr()
sn.heatmap(corrMatrix, annot=True)

plt.figure(figsize=(27,17))
plt.show()


# Widzimy macierz korelacji. Zmienna dist jest najbardziej skorelowana z :k.pont,hill size(to oczywiste ze dlugosc skoku zalezy od rozmiaru skoczni),w dalszej kolejnosci z :speed,type,yearss_seen,age,team,snow.Postaram sie uzyc tych zmiennych by zbudowac model.

# <h1>Model ekonometryczny-od czego zalezy dlugosc skoku skoczka narciarskiego? <h1>

# <h2>Budowa modelu<h2>

# <h2>Zmienna objaśniana:ranking<h2>

# In[265]:


df['ranking'].head()


# Mowi ktora pozycje w rankingu uzyskal dany zawodnik.Teraz zbadamy od czego najbardziej zalezy ta pozycja w rankingu, budujac model ekonometryczny
# 

# <h2>Metoda Najmniejszych Kwadratów<h2>

# Metoda najmniejszych kwadratów – najczęściej stosowana metoda dopasowania modelu liniowego do danych, np. w analizie korelacji czy analizie regresji. Polega ona na dopasowaniu linii prostej, która będzie leżała jak najbliżej wszystkich wyników – tak aby suma odległości wszystkich punktów od linii była minimalna.Jest to jedna z najprostrzych i najczesciej stosowanych metod w ekonometrii.

# In[286]:


import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.api as sm
import statsmodels.stats.api as sms


# In[427]:


df.columns


# In[426]:


df['home_comp'] = df['country'] == df['nationality']


# <h2>Zmienne do modelu<h2>

# In[433]:


cols = ['home_comp','month','team','type','humid','snow','weather_type','counted_jumpers','dist','wind','ranking','meter.value', 'country', 'hill_size', 'k.point',  'gate.factor', 'wind.factor', 'wind_comp','round','gate','gate_points','nationality','gender','age','meaningful_nationality','meaningful_country','flying_hill','years_seen','speed']
df_v1 = df[cols]
df_v1.columns = df_v1.columns.str.replace('.', '_')

df_v1.head()


# In[434]:


df_v1.columns


# <h2>Dummy Variables<h2>

# In[435]:


data_copy = df_v1.copy()

X = df_v1.drop(['ranking'], axis = 1)
y = df_v1['ranking']

df_v1['gender'].mask(df_v1['gender'] == 'female', 0, inplace=True)

df_v1.loc[df_v1["month"] == 1, "month"] = 'january'
df_v1.loc[df_v1["month"] == 2, "month"] = 'february'
df_v1.loc[df_v1["month"] == 3, "month"] = 'march'
df_v1.loc[df_v1["month"] == 4, "month"] = 'april'
df_v1.loc[df_v1["month"] == 5, "month"] = 'may'
df_v1.loc[df_v1["month"] == 6, "month"] = 'june'
df_v1.loc[df_v1["month"] == 7, "month"] = 'july'
df_v1.loc[df_v1["month"] == 8, "month"] = 'august'
df_v1.loc[df_v1["month"] == 9, "month"] = 'september'
df_v1.loc[df_v1["month"] == 10, "month"] = 'october'
df_v1.loc[df_v1["month"] == 11, "month"] = 'november'
df_v1.loc[df_v1["month"] == 12, "month"] = 'december'
df_v1.loc[df_v1["type"] == 0, "type"] = 'PS'
df_v1.loc[df_v1["type"] == 1, "type"] = 'PK'
df_v1.loc[df_v1["type"] == 2, "type"] = 'LGP'
df_v1.loc[df_v1["type"] == 3, "type"] = 'FC'
df_v1.loc[df_v1["type"] == 4, "type"] = 'MAMUT'
df_v1.loc[df_v1["type"] == 5, "type"] = 'MS'
df_v1.loc[df_v1["type"] == 6, "type"] = 'MSJ'




df_v1
X=df_v1


# In[436]:


nominal = ['month','type','gender','round','weather_type','nationality','country'] 

# enkodowanie one-hot 
temp = X.drop(columns = nominal)#wywalamy kolumne nominal -to sa same zmienne ktorych nie bediemy rozpalszcac na duumies
dummies = pd.get_dummies(X[nominal])#przeksztalcamy te zmienne  na dummies czyli kazda zmienna ma swoje wartosci a tu rozplaszczamy ta zmenna
X = pd.concat([temp,dummies], axis = 1)#laczymy do siebie zmienne z rozplaszczone i te ktorych nie chcielismy zamienic na dummmiesy

X = pd.DataFrame(X)
X.columns = X.columns.str.replace(',', '_')
X.columns = X.columns.str.replace('/', '_')
X.columns = X.columns.str.replace(' ', '_')

X#chcemy zeby to laczenie bylo po kolumnach a nie po wierszach!


# In[438]:


Base_model=smf.ols('dist~hill_size+wind+age+meaningful_nationality+years_seen+gender_M+speed+team+snow+humid+weather_type_sunny+round_1st_round_+round_2nd_round_+round_trial_round_+type_FC+type_LGP+type_MAMUT+type_MSJ+type_PK',data = X).fit()


# In[439]:


print(Base_model.summary())


# <h2>Podsumowanie naszego modelu podstawowego, na pierwszy rzut oka:<h2>

# 
#     
# *R^2 wyszlo dosc duze zatysfakconujace. R-squared: Model wyjaśnia około 86% zmienności -bardzo dobrze
# 
# *Prob (F-statystyka): Współczynniki są istotne zbiorczo na poziomie istotności 5%.
# 
# *prawie wszystkie zmienne oprocz humid sa istotne statystycznie,
# 
# *jest duza wspollniniowosc-co wyrzuca nam nawet komunikat
# 
# *test DW-nie ma autokorelacji skladnika losowego-to dobrze.Zasadą jest, że wartości statystyki testu DW w zakresie od 1,5 do 2,5 są względnie normalne. Wartości spoza tego przedziału mogą jednak być powodem do niepokoju.
# 
# 

# <h2>Poprawa modelu<h2>

# <h3>Wyrzucam humid<h3>

# In[349]:


Base_model2=smf.ols('dist~hill_size+wind+age+meaningful_nationality+years_seen+gender_M+speed+team+snow+weather_type_sunny+round_1st_round_+round_2nd_round_+round_trial_round_+type_FC+type_LGP+type_MAMUT+type_MSJ+type_PK',data = X).fit()


# In[350]:


print(Base_model2.summary())


# <h2>Problem wspolliniowosci<h2>

# Jednym z ważnych założeń regresji liniowej jest to, że powinna istnieć liniowa zależność między każdym z predyktorów (x₁, x₂ itd.) a wynikiem y. Jeśli jednak między predyktorami występuje korelacja (np. x₁ i x₂ są silnie skorelowane), nie można już określić wpływu jednego z nich przy zachowaniu stałej wartości drugiego, ponieważ oba predyktory zmieniają się razem. W rezultacie współczynniki (w₁ i w₂) są teraz mniej dokładne, a zatem mniej interpretowalne.
# 
# Usuwanie wieloliniowości
# Podczas trenowania modelu uczenia maszynowego ważne jest, aby na etapie wstępnego przetwarzania danych usunąć ze zbioru cech cechy, które wykazują wieloliniowość. Można to zrobić za pomocą metody znanej jako VIF - współczynnik inflacji wariancji.
# 
# 
# Współczynnik VIF pozwala określić siłę korelacji między różnymi zmiennymi niezależnymi. Oblicza się go, biorąc zmienną i poddając ją regresji względem wszystkich innych zmiennych.
# VIF oblicza, jak bardzo wariancja współczynnika jest zawyżona ze względu na jego liniową zależność z innymi predyktorami. Stąd jego nazwa.

# <h2>VIF<h2>

# In[351]:


import pandas as pd
from sklearn.linear_model import LinearRegression
def calculate_vif(df, features):    
    vif, tolerance = {}, {}
    # all the features that you want to examine
    for feature in features:
        # extract all the other features you will regress against
        X = [f for f in features if f != feature]        
        X, y = df[X], df[feature]
        # extract r-squared from the fit
        r2 = LinearRegression().fit(X, y).score(X, y)                
        
        # calculate tolerance
        tolerance[feature] = 1 - r2
        # calculate VIF
        vif[feature] = 1/(tolerance[feature])
    # return VIF DataFrame
    return pd.DataFrame({'VIF': vif, 'Tolerance': tolerance})


# In[352]:


X.corr()


# Widzimy ze na warosc dlugosci skoku najnatdziej ejst skorelowana z hill_size,k_point,wind_factor,gate_factor,flying_hill

# In[353]:


calculate_vif(df=X, features=['hill_size','wind','age','meaningful_nationality','years_seen','gender_M','speed','team','snow','weather_type_sunny','round_1st_round_','round_2nd_round_','round_trial_round_','type_FC','type_LGP','type_MAMUT','type_MSJ','type_PK'])


# Wiac ze istnieje najwiekza  korelacja jesli chodzi o hill size .Spróbowałam usunąć tą zmienną,lecz (zobacz ponizej), znacznie osłabia to R2 w naszym modelu. Jako ze korelacja zadnej ze mniennych (w tym hill_size) nie przekracza 5 , to zostawie tak jak jest jak na razie.

# In[354]:


Base_model3=smf.ols('dist~wind+age+meaningful_nationality+years_seen+gender_M+speed+team+snow+weather_type_sunny+round_1st_round_+round_2nd_round_+round_trial_round_+type_FC+type_LGP+type_MAMUT+type_MSJ+type_PK',data = X).fit()


# In[355]:



print(Base_model2.summary())


# <h2>Test Jarque-Bera-na normalnosc skladnika losoego<h2>

# ![image-2.png](attachment:image-2.png)

# 
# Normalność rozkładu składnika losowego ułatwia konstrukcję testów statystycznych przydatnych do weryfikacji modelu 
# ekonometrycznego. Przetestuję ją za pomocą testu Jarque-Bera. Opiera się on na weryfikacji podobieństwa trzeciego i 
# czwartego momentu rozkładu składnika losowego modelu do znanych wartości tych momentów w rozkładzie normalnym. 
# 
# 𝐻0:𝑆𝑘ł𝑎𝑑𝑛𝑖𝑘⁡ 𝑙𝑜𝑠𝑜𝑤𝑦⁡ 𝑚𝑎 ⁡𝑟𝑜𝑧𝑘ł𝑎𝑑⁡ 𝑛𝑜𝑟𝑚𝑎𝑙𝑛𝑦.
# 
# 𝐻1:𝑆𝑘ł𝑎𝑑𝑛𝑖𝑘⁡ 𝑙𝑜𝑠𝑜𝑤𝑦 ⁡𝑛𝑖𝑒⁡ 𝑚𝑎 ⁡𝑟𝑜𝑧𝑘ł𝑎𝑑𝑢⁡ 𝑛𝑜𝑟𝑚𝑎𝑙𝑛𝑒𝑔𝑜. 
#  
# Statystyka testu chi-kwadrat wynosi 1167577 Zatem:
# Na poziomie istotności ∝ = 0,05 nie ma podstaw do odrzucenia hipotezy zerowej, mówiącej o normalności składnika 
# losowego.

# <h3>Rozklad skladnika losowego<h3>

#  Test Jarque'a-Bery jest testem dobroci dopasowania, który mierzy, czy dane z próby mają skośność i kurtozę zbliżone do rozkładu normalnego.
# 
# Statystyka testu Jarque'a-Bery jest zawsze dodatnia, ALE! jeśli nie jest bliska zeru, oznacza to, że dane z próby nie mają rozkładu normalnego.

# In[357]:


import statsmodels


# In[361]:


statsmodels.stats.stattools.jarque_bera(Base_model2.resid)


# In[362]:


name = ['Jarque-Bera', 'Chi^2 two-tail prob.', 'Skew', 'Kurtosis']
test = sms.jarque_bera(Base_model3.resid)
print(name, test)


# Dane mają bardziej skośny rozkład normalny. Popularnym sposobem poprawienia tego stanu rzeczy jest przekształcenie zmiennych w logarytm i wykorzystanie  w modelu. Biorąc pod uwagę rozkład, przydatny może być również inny model regresji, taki jak regresja kwantylowa lub nieparametryczna regresja liniowa.
# 
# Przyjrzyjmy się, jak przekształcenie modelu w logarytm zmienia rozkład zmiennej.

# <h3>Skosnosc i kurtoza-co to jest?<h3>

# Skośność
# 
# Rozkład normalny jest symetryczny i ma skośność równą 0. Rozkład z dużą skośnością dodatnią ma długi ogon prawostronny. Gdy zaś współczynnik skośności jest ujemny, rozkład ma długi kraniec z lewej strony.
#  
#  
# Kurtoza
# 
# Kurtoza jest to względna miara koncentracji i spłaszczenia rozkładu (termin stosowany w statystyce i rachunku prawdopodobieństwa). Określa rozmieszczenie i koncentrację wartości (zbiorowości) w pobliżu średniej.
# 
#  W przypadku rozkładu normalnego wartość statystyki kurtozy wynosi trzy. Kurtoza dodatnia wskazuje, że w danych istnieje więcej dodatnich wartości odstających niż w przypadku rozkładu normalnego. Kurtoza ujemna wskazuje, że w danych istnieje mniej dodatnich wartości odstających niż w przypadku rozkładu normalnego.

# In[363]:


plt.suptitle('Rozklad reszt skladnika losowego', fontsize=20)


ax = plt.hist(Base_model2.resid)
plt.xlim(-100,100)
plt.xlabel('Residuals')
plt.show


# A wiec patrzac na rozklad jest lekko skosny w lewo lecz zasadniczo zbiega do normalnego.Jesli chcemy uzyskac rozklad calkowicie normalny  nalezy pokombinowac z innymi metodami niz regresja liniowa :)

# <h2>Heteroskedastycznosc-test Breuscha-Pagana<h2>

# 
# Drugim sposobem na sprawdzenie, czy w modelu występuje problem heteroskedastyczności jest test Breuscha-Pagana, który 
# polega na oszacowaniu równania regresji, w którym zmienną objaśniającą jest kwadrat reszt podzielonych przez odchylenie 
# standardowe.
# 
# 𝐻0:𝑆𝑘ł𝑎𝑑𝑛𝑖𝑘⁡𝑙𝑜𝑠𝑜𝑤𝑦⁡𝑗𝑒𝑠𝑡⁡ℎ𝑜𝑚𝑜𝑠𝑘𝑒𝑑𝑎𝑠𝑡𝑦𝑐𝑧𝑛𝑦. 
# 
# 𝐻1:𝑆𝑘ł𝑎𝑑𝑛𝑖𝑘⁡𝑙𝑜𝑠𝑜𝑤𝑦⁡𝑗𝑒𝑠𝑡⁡ℎ𝑒𝑡𝑒𝑟𝑜𝑠𝑘𝑒𝑑𝑎𝑠𝑡𝑦𝑐𝑧𝑛𝑦. 
# 
# 
# Statystyka testu chi-kwadrat wynosi duzo powyzej zera , a krytyczny poziom istotności 0,0.
# Przy poziomie istotności ∝ = 0,05 nie ma więc powodu do odrzucenia hipotezy zerowej, mówiącej o homoskedastyczności 
# składnika losowego. Nie występuje więc problem heteroskedastyczności

# In[375]:


name = ['Lagrange multiplier statistic', 'p-value', 
        'f-value', 'f p-value']
test = sms.het_breuschpagan(Base_model2.resid, Base_model2.model.exog)
print(name, test)


# Nie ma problemu heteroskedastycznosci na szczsscie.

# <h2>Autokorelacja<h2>

# In[420]:


from statsmodels.stats.stattools import durbin_watson

#perform Durbin-Watson test
durbin_watson(Base_model2.resid)


# In[380]:


X.head()


# Statystyka testowa wynosi 0.957. Ponieważ nie mieści się ona w przedziale od 1,5 do 2,5, można uznać, że autokorelacja stanowi problem w tym modelu regresji.
# 
# Statystyka Durbina Watsona (DW) jest testem autokorelacji w resztach z modelu statystycznego lub analizy regresji. Statystyka Durbina-Watsona ma zawsze wartość z przedziału od 0 do 4. Wartość 2,0 oznacza, że w próbie nie wykryto autokorelacji. Wartości od 0 do mniej niż 2 wskazują na autokorelację dodatnią, a wartości od 2 do 4 oznaczają autokorelację ujemną.

# In[419]:


Base_model5=smf.ols('dist.shift(1)~hill_size+wind+age+years_seen+gender_M+speed+team+snow+weather_type_sunny+weather_type_sunny_to_partly_cloudy+weather_type_snowy__windy+weather_type_snowy__foggy+weather_type_rain_+round_1st_round_+round_2nd_round_+round_trial_round_+type_FC+type_LGP+type_MAMUT+type_MSJ+type_PK+country_NOR+country_POL+country_GER+country_JPN+country_FRA+country_ITA+country_CZE+country_AUT+country_FIN+country_CAN+country_USA+country_TUR+nationality_POL+nationality_RUS+nationality_SLO+nationality_NOR+nationality_NED+nationality_GER+nationality_SVK+nationality_USA+nationality_SRB+nationality_FIN+nationality_FRA+nationality_EST+nationality_CZE+nationality_CHN+nationality_BLR+nationality_BUL+month_august+month_november+month_december+month_february+month_march+month_january+month_september+month_october',data = X).fit()

print(Base_model5.summary())


# In[ ]:


from from statsmodels.stats.stattools import durbin_watson

#perform Durbin-Watson test
durbin_watson(Base_model5.resid)statsmodels.stats.stattools import durbin_watson

#perform Durbin-Watson test
durbin_watson(Base_model5.resid)


# Zamienienie zmiennej meaningful nationality na dummies z nationality zwiekszylo normalnosc modelu i zmniejszylo jego autokorelacje. Dodanie opoznienia do zmiennej objasnianej sprawilo ze autokorelacja zostala znacznie zmniejszona.

# <h1>Podsumowanie<h1>

# Musimy być swiadomi, ze w naszym modelu moze wystapic problem endogenicznosci, który zostanie omówiony poniżej. Problem ten poruszamtakze w innym moim projekcie-oczekiwana dlugosc zycia.
# 
# W obecnym projekcie postaram się poszukać idealnego instrumentu i niebawem dołączyć go do modelu by jescze trochę go poprawić. Na tą chwilę jednak mozna przedstawic juz pare wnioskow.

# <h2>Wnioski-co tak napraede wplywa na długosc skoku narciarskiego?<h2>

# Poprzeanalizowaniu dokłądnie danych oraz przygotowaniu prostego modelu ekonometrycznego można stwierdzić, że na długośc skoku narciarskiego ma wpływ wiele rzeczy,miedzy innymi:
#     
#     
# *pogoda:snieg i mgla wplywa negatywnie
#         
# *wiatr-kiedy wieje pod narty zawodnik leci dluzej
#     
# *ranga zawodow-na mamutach odleglosc jest znacznie dluzsza, dodatkowo mniejsze odleglosci w porownaniu do PS notujemy w PK i LGP,
#     
# *nacje z sukcesami skacza srednio o kilka metrow dalej niz te bez sukcesow
#     
# *wiek i doswiadczenie zawodnika wplywaja pozytywnie na jego skoki.
#     
# *wyzsza predkosc na progu wydluza skok
#     
# *zawodnicy uzyskuja dluzsze odleglosci w konkursach druzynowych niz indywidualnych

# <h2>Endogenicznosc<h2>

# <h3>Przyczyny<h3>

# Endogeniczność jest zjawiskiem występującym powszechnie w ekonometrii.Endogeniczność może być wynikiem błędu 
# pomiarowego, autoregresji, pominiętych zmiennych lub błędów w wyborze próby statystycznej. Więc warto zastanowić się 
# ,czy mamy z nią do czynienia. Szczególnie, że podczas przygotowywania danych do mojego modelu, napotkałam miedzy 
# innymi problem braków danych, które musiałam usunąć , by kontynuować dalszą analizę. W 
# wyniku tego działania straciłam dane. Dodatkowo na badane przeze mnie 
# zjawisko wpływa całe spektrum zmiennych. Łatwo wiec jest pominąć jakąś ważną i kluczową dla istnienia modelu zmienną. 
# Co znaczy właściwie endogeniczność? O parametrze mówi się, że jest endogeniczny, jeśli jest on skorelowany ze 
# składnikiem losowym. W moim modelu na składnik losowy może składać się wiele rzeczy. Zmienna z modelu, która może być silnie skorelowana ze składnikiem losowym to np. counted_jumpers	
# ..

# <h3>Jak poradzic sobie z tym zjawiskiem?Metoda zmiennych instrumentalnych <h3>

# 
# Aby wyeliminować prawdopodobną endogeniczność, korzystam z metody zmiennych instrumentalnych .Najpierw należy
# zdecydować się na instrument, który będzie spełniał odpowiednie kryteria. Dobry, mocny instrument powinien być silnie 
# skorelowany ze zmienną endogeniczną  i nieskorelowany ze składnikiem losowym, a więc nie może 
# on wpływać bezpośrednio na dystans skoku. Wybieram instrument counted_jumpers.
# 

# <h3>Wybor dobrego instrumentu<h3>

# Instrument immun może być właśnie takim dobrym kandydatem na mocną zmienną instrumentalną, gdyż:
# 1.Jest on silnie skorelowany ze zmienną endogeniczną 
# 
# 2.Równocześnie :nie jest skorelowana bezpośrednio ze składnikiem losowym,

# <h3>idealny instrument<h3>

# Idealnym instrumentem moze byc counted_jumpers+>idealny instrumenty poniewaz nie wplywa za bardzo na odleglosc skoczba a np jest skorelowany z nacjami w skoach,poniewaz są wypracowywane limity skoczkow-np.lepszy kraj czsto ma 7 skoczkow w PS a gorszy tylko 2-3.,poniewaz kraj nie spelnia odpowiednich kryteriow.

# In[ ]:




