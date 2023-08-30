def getAnalysisPrompt(query, listOfCompanies):
    prompt = f''' Give below is query from the user on list of companies given below in list.
    Query : {query}
    List of companies : {listOfCompanies}
    '''

    return prompt