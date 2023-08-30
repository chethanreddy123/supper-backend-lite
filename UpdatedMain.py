from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import pandas as pd
import random
from datetime import datetime, timedelta
import google.generativeai as palm
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
import os
import prompts

load_dotenv()

palm.configure(api_key=os.environ.get("PALM_API_KEY"))

def generateTextWithPalm(prompt):
    models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
    model = models[0].name

    completion = palm.generate_text(
        model=model,
        prompt=prompt,
        temperature=0.1,
        max_output_tokens=5000,
    )
    return completion.result

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

C1 = MongoClient('mongodb+srv://chethanreddy2002:1234@cluster0.xihwp.mongodb.net/?retryWrites=true&w=majority')
myData1 = C1['Test']['NewCompanies']
List_Of_Clusters = [myData1]

OldMax, OldMin, OldRange = (10401986.77325441, -10327433.682112753, 20729420.455367163)
NewMax = 1000
NewMin = 1
NewRange = NewMax - NewMin

def NewValue(x):
    return (((x - OldMin) * NewRange) / OldRange) + NewMin

@app.post("/domainSearch")
async def domainSearch(info: Request):
    req_info = await info.json()
    CurrString = dict(req_info)["SearchedString"]
    Results = []

    def get_data(n):
        for i in List_Of_Clusters[n].find({"Function": CurrString}):
            yield i

    for i in range(1):
        if len(Results) >= 3:
            break
        cuList = get_data(i)
        while True:
            try:
                item = next(cuList)
                del item['_id']
                Results.append(item)
                if len(Results) == 3:
                    break
            except StopIteration:
                break

    Final_Data = {"List": Results}
    return Final_Data

@app.post("/companySearch")
async def companySearch(info: Request):
    req_info = await info.json()
    CurrString = dict(req_info)["SearchedString"]
    Results = []

    def get_data(n):
        for i in List_Of_Clusters[n].find({"SupplierName": CurrString}):
            yield i

    check = False

    for i in range(1):
        cuList = get_data(i)
        if check == True:
            break
        while True:
            try:
                item = next(cuList)
                del item['_id']
                Results.append(item)
                check = True
                break
            except StopIteration:
                break

    return Results[0]

@app.post("/predictionSearch")
async def predictionSearch(info: Request):
    req_info = await info.json()
    CurrString = dict(req_info)["SearchedString"]
    Results = []

    def get_data(n):
        for i in List_Of_Clusters[n].find({"SupplierName": CurrString}):
            yield i

    for i in range(1):
        cuList = get_data(i)
        while True:
            try:
                item = next(cuList)
                del item['_id']
                Results.append(item)
            except StopIteration:
                break

    CostList = []

    for i in Results:
        if i['Cost'] > 1000 or i['Cost'] < 0:
            CostList.append((NewValue(i['Cost']), i['Year']))
        else:
            CostList.append((i['Cost'], i['Year']))

    CostList = sorted(CostList, key=lambda x: x[1])

    X, y = [], []

    for i, j in CostList:
        X.append(int(str(j)[2:]))
        y.append(i)

    Df = pd.DataFrame({"x": X, "y": y})

    poly = PolynomialFeatures(degree=5)
    X_poly = poly.fit_transform(Df[['x']])

    poly.fit(X_poly, Df['y'])
    lin2 = LinearRegression()
    lin2.fit(X_poly, Df['y'])

    X = Df[['x']]
    y = Df['y']
    x_prec = pd.DataFrame({"x": [23 + i for i in range(2)]})[['x']]

    CurrListx = list(X.x)
    CurrListy = list(y)

    PrecListx = list(x_prec.x)
    PrecListy = list(lin2.predict(poly.fit_transform(x_prec)))

    PrecListy = [round(i, 2) for i in PrecListy]
    CurrListy = [round(i, 2) for i in CurrListy]

    FinalX = CurrListx + PrecListx
    FinalY = CurrListy + PrecListy

    oldMax = max(FinalY)
    oldMin = min(FinalY)

    oldRange = oldMax - oldMin

    NewMax = 100
    NewMin = 1

    NewRange = NewMax - NewMin

    changer = lambda OldValue: (((OldValue - oldMin) * NewRange) / oldRange) + NewMin

    FinalY = CurrListy + PrecListy

    FinalY = [round(changer(i), 2) for i in FinalY]

    FinalX = [int("20" + str(i)) for i in FinalX]

    PlotData = {
        "Years": FinalX,
        "Performance": FinalY,
    }

    return PlotData

@app.post("/numberSearch")
async def numberSearch(info: Request):
    req_info = await info.json()
    CurrString = int(dict(req_info)["No_of_Companies"])
    Results = []

    query = list(List_Of_Clusters[0].find({}).limit(CurrString))
    for i in query:
        del i['_id']
    finalDict = {
        "List_of_Companies": query
    }

    data = finalDict
    sorted_companies = sorted(data["List_of_Companies"], key=lambda x: x["Cost"], reverse=True)

    data["List_of_Companies"] = sorted_companies

    return data

@app.post("/palmChat")
async def palmChat(info: Request):
    req_info = await info.json()
    ListOfCompanies = dict(req_info)["ListOfCompanies"]
    query = dict(req_info)["Query"]
    result = generateTextWithPalm(prompts.getAnalysisPrompt(query, ListOfCompanies))
    return result
