def getAnalysisPrompt(query, listOfCompanies):
    prompt = f''' Give below is query from the user on list of companies with detail of each company mentioned.
    Analysis the details and answer the query with a small descriptive explanation of the analysis.
    Query : {query}
    List of companies : {listOfCompanies}
    '''
    return prompt