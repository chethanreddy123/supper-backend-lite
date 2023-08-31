def getAnalysisPrompt(query, listOfCompanies):
    prompt = f''' Analyze the following details for each company in the list. 
    Provide a concise and descriptive explanation of your analysis in response to the query.
    
    Query: {query}
    
    List of Companies: {listOfCompanies}
    
    Note: Please keep the explanations brief and focused, avoiding excessive length and code outputs. 
    There will be 10 entities in JSON format for each company, and the query pertains to these 10 entities. 
    Ensure you have read and understood all the entities before conducting your analysis.
    Don't use any bold letters in the answer.
    '''
    return prompt
