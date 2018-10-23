#!/usr/bin/env python2
# -*- coding: utf-8 -*-

"""
Usage : Explained in the book. Refer chapter 4 -- > Serving chatbots section
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
