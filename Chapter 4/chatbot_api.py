#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: rahulkumar

Usage : 

Execute:
    python -m rasa_nlu.server --path ./rasa_version/projects    
    python chatbot_api.py    

Call bellow api to execute TFIDF version    
http://localhost:8080/version1?query=Can I get an Americano

Call bellow api to execute RASA version
http://localhost:8080/version2?query=where is Indian cafe
"""

from tfidf_version import tfidf_bot
import requests
import web

urls = (
    '/version1', 'TFIDF',
    '/version2', 'RASA'
)


class TFIDF:
    def GET(self):
        
        userdata = web.input()
        
        query=userdata.query
        
        data = tfidf_bot.bot_engine(query=query)
        
        return data


class RASA:
    def GET(self):
        
        userdata = web.input()
        
        query=userdata.query
        
        data = requests.get('http://localhost:5000/parse?q='+query).json()
        
        return data

if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()