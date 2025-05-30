from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from math import sqrt
import os
import json
import glob

list_bestanden = []             #Regels 9 t/m 26 zorgen ervoor dat alle namen van de teksten woorden toegevoegd aan een lijst.

for root, dirs, files in os.walk('.'):
    for name in files:
        if name.endswith((".txt")):
            if name == "tekst1.txt":
                list_bestanden.append(name)
            if name.startswith(("tekst")):
                if name.startswith(("tekst1")):
                    lol = 0
                else:
                    list_bestanden.append(name)

for root, dirs, files in os.walk('.'):
    for name in files:
        if name.endswith((".txt")):
            if name.startswith(("tekst1")):
                    list_bestanden.append(name)

                            #Regels 29 t/m 42 zorgen ervoor dat elke tekst leesbaar is.
for i in range(0,(len(list_bestanden)-1)):
    bestand = open(list_bestanden[i])
    text = bestand.read()
    text = text.replace("\n","")
    text = text.replace("\t","")
    text = text.replace("\\","")
    text = text.replace("/","")
    text = text.replace("Â","")
    text = text.replace("Ã","e ")
    text = text.replace("«","")
    text = text.replace(" het ","")
    text = text.replace(" een ","")
    text = text.replace(" is ","")
    text = text.replace(" de ","")
    text = text.replace(" op ","")
    text = text.replace(" onder ","")
    for j in text:
        dutchcheck = glob.glob("dutch.txt")
        dutchcheck.read()
        if j in dutchcheck:
            text.replace(j,"")
        dutchcheck.close()
        englishcheck = glob.glob("english.txt")
        englishcheck.read()
        if j in englishcheck:
            text.replace(j,"")
        englishcheck.close()
    number = i + 1
    name = "text"+str(number)
    exec(name + " = text")
    bestand.close()
                            #Regels 44 t/m 54 laten een berekening uitvoeren waardoor we weten hoeveel een bepaald woord voorkomt.
def wordcount(path):
    wordcounts = {}
    with open(path) as file:
        for line in file.readlines():
            for word in line.split():
                if (word in wordcounts.keys()): 
                    wordcounts[word] += 1
                else: 
                    wordcounts[word] = 1
                    
    return wordcounts
                            #Regels 56 t/m 59 zorgen ervoor dat elke tekst de berekening uitgevoerd krijgt.
for j in range(0,(len(list_bestanden)-1)):
    number = j + 1
    name = "textfreq"+str(number)
    exec(name + " = wordcount('tekst" + str(number) + ".txt')")


app = Flask(__name__)


@app.route('/tekst', methods=['GET', 'POST'])
                            #Regels 67 t/m 81 zorgen ervoor dat de tekst die je aanmaakt in een nieuw document komt en u krijgt een melding welke tekst en waar dit staat.
def voeg_toe_pagina():
    if request.method == 'POST':
        if request.form['submit_button'] == 'Sla de tekst op':
            new_text = request.form["new_text"]
            
            number_new_bestand = len(list_bestanden)
            new_name = "tekst" + str(number_new_bestand) + ".txt"
            
            new_bestand = open(new_name, "w")
            new_bestand.write(new_text)
            new_bestand.close()
                        
            melding = "This text you have submited to " + new_name + ": " + new_text
                
            return melding
            
            
    elif request.method == 'GET':
        return render_template('Teksttoevoegen.html')



@app.route('/', methods=['GET', 'POST'])
                                #Regels 91 t/m 157 zorgen ervoor de query wordt opgehaald en daar een berekening over wordt gedaan om de score te bepalen.
def search_engine_pagina():
    if request.method == 'POST':
        if request.form['submit_button'] == 'Run Query':
            query = request.form["query"]
                        
            c1 = 0                      #de variable c1 t/m c10 staan voor een nulpunt, de frequentie van de woorden in de query worden hierbij opgeteld.
            for k, v in textfreq1.items(): #k staat voor de query en v voor de frequentie
                if k in query:
                    c1 = c1 + (v*v)
            vector_1 = sqrt(c1)         #vector_1 t/m vector_10 zijn de vectoren en dienden hierbij als score.
            
            c2 = 0
            for k, v in textfreq2.items():
                if k in query:
                    c2 = c2 + (v*v)
            vector_2 = sqrt(c2)
            
            
            c3 = 0
            for k, v in textfreq3.items():
                if k in query:
                    c3 = c3 + (v*v)
            vector_3 = sqrt(c3)
            
            
            c4 = 0
            for k, v in textfreq4.items():
                if k in query:
                    c4 = c4 + (v*v)
            vector_4 = sqrt(c4)
            
            
            c5 = 0
            for k, v in textfreq5.items():
                if k in query:
                    c5 = c5 + (v*v)
            vector_5 = sqrt(c5)
            
            c6 = 0
            for k, v in textfreq6.items():
                if k in query:
                    c6 = c6 + (v*v)
            vector_6 = sqrt(c6)
            
            c7 = 0
            for k, v in textfreq7.items():
                if k in query:
                    c7 = c7 + (v*v)
            vector_7 = sqrt(c7)
            
            c8 = 0
            for k, v in textfreq8.items():
                if k in query:
                    c8 = c8 + (v*v)
            vector_8 = sqrt(c8)
            
            c9 = 0
            for k, v in textfreq9.items():
                if k in query:
                    c9 = c9 + (v*v)
            vector_9 = sqrt(c9)
            
            c10 = 0
            for k, v in textfreq10.items():
                if k in query:
                    c10 = c10 + (v*v)
            vector_10 = sqrt(c10)
            
            #Regels 161 t/m 175 halen alle scores bij elkaar en rangschikt de documenten met de score die ze hierboven hebben gekregen.
            
            data_text = dict()
            data_text = {text1 : vector_1, text2 : vector_2, text3 : vector_3, text4 : vector_4, text5 : vector_5, text6 : vector_6, text7 : vector_7, text8 : vector_8, text9 : vector_9, text10 : vector_10}
            list_b = []
            list_text = []

            for k in sorted(data_text, key=data_text.get, reverse=True):
                list_b.append(data_text[k])
                list_text.append(k)
                
            data_bestand = dict()
            data_bestand = {1 : vector_1, 2 : vector_2, 3 : vector_3, 4 : vector_4, 5 : vector_5, 6 : vector_6, 7 : vector_7, 8 : vector_8, 9 : vector_9, 10 : vector_10}
            list_bestand = []

            for k in sorted(data_bestand, key=data_bestand.get, reverse=True):
                list_bestand.append(k)
        
            #Regels 179 t/m 228 kijkt als eerste of de checkbox is aangevinkt en daarna zet het de scores in json query om het vervolgens te laten zien op het scherm.
        
            json_query1 = ""
            json_query2 = ""
            json_query3 = ""
            json_query4 = ""
            json_query5 = ""
            
            try:
                if request.form['show_scores'] == "1":
                    queryreturn1 = dict()
                    queryreturn1 = {"Rank" : 1,"Score" : list_b[0],"Bestand" : list_bestand[0],"Inhoud" : list_text[0][0:350]}
                    json_query1 = json.dumps(queryreturn1)
            
                    queryreturn2 = dict()
                    queryreturn2 = {"Rank" : 2,"Score" : list_b[1],"Bestand" : list_bestand[1],"Inhoud" : list_text[1][0:350]}
                    json_query2 = json.dumps(queryreturn2)
            
                    queryreturn3 = dict()
                    queryreturn3 = {"Rank" : 3,"Score" : list_b[2],"Bestand" : list_bestand[2],"Inhoud" : list_text[2][0:350]}
                    json_query3 = json.dumps(queryreturn3)
            
                    queryreturn4 = dict()
                    queryreturn4 = {"Rank" : 4,"Score" : list_b[3],"Bestand" : list_bestand[3],"Inhoud" : list_text[3][0:350]}
                    json_query4 = json.dumps(queryreturn4)
            
                    queryreturn5 = dict()
                    queryreturn5 = {"Rank" : 5,"Score" : list_b[4],"Bestand" : list_bestand[4],"Inhoud" : list_text[4][0:350]}
                    json_query5 = json.dumps(queryreturn5)
            except:
                queryreturn1 = dict()
                queryreturn1 = {"Rank" : 1,"Score" : "-","Bestand" : list_bestand[0],"Inhoud" : list_text[0][0:350]}
                json_query1 = json.dumps(queryreturn1)
            
                queryreturn2 = dict()
                queryreturn2 = {"Rank" : 2,"Score" : "-","Bestand" : list_bestand[1],"Inhoud" : list_text[1][0:350]}
                json_query2 = json.dumps(queryreturn2)
            
                queryreturn3 = dict()
                queryreturn3 = {"Rank" : 3,"Score" : "-","Bestand" : list_bestand[2],"Inhoud" : list_text[2][0:350]}
                json_query3 = json.dumps(queryreturn3)
            
                queryreturn4 = dict()
                queryreturn4 = {"Rank" : 4,"Score" : "-","Bestand" : list_bestand[3],"Inhoud" : list_text[3][0:350]}
                json_query4 = json.dumps(queryreturn4)
            
                queryreturn5 = dict()
                queryreturn5 = {"Rank" : 5,"Score" : "-","Bestand" : list_bestand[4],"Inhoud" : list_text[4][0:350]}
                json_query5 = json.dumps(queryreturn5)            
            
            
            return jsonify(json_query1, json_query2, json_query3, json_query4, json_query5) #(request.form["query"])
        else:
            pass 
    elif request.method == 'GET':
        return render_template('searchengine.html')
    
  

if __name__ == "__main__":
    app.config['TEMPLATES_AUTO_RELOAD']=True
    app.config['DEBUG'] = True
    app.config['SERVER_NAME'] = "127.0.0.1:5000"
    app.run()