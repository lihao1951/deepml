#!/usr/bin/python
# -*- coding:utf-8 -*-
"""
Author LiHao
Time 2018/11/30 18:01
"""
import os
import sys
import uuid
import time

class Event(object):
    """
    定义事件类
    """
    def __init__(self,title,cont,event_time,keywords):
        self._title = title #事件标题
        self._cont = cont #事件内容
        self._event_time = event_time #事件时间
        self._keywords = keywords #关键字
        self._id = self._make_random_id() #事件id 随机生成
        self._news_docs = [] #新闻
        self._news_comments = [] #新闻评论
        self._weibo_primary_docs = [] #微博原发
        self._weibo_transmit_docs = [] #微博转发
        self._weibo_comments = [] #微博评论
        self._app_docs = [] #app新闻
        self._app_comments = [] #app评论
        self._weixin_docs = [] #微信文章
        self._weixin_comments = [] #微信评论
        self._blog_docs = [] #博客数据
        self._twitter_docs = [] #推特数据
        self._oversea_news_docs = [] #海外新闻
        self._forum_docs = [] #论坛数据
        self._bbs_docs = [] #bbs数据
        self._related_locations = [] #相关地域
        self._related_persons = [] #相关人物
        self._related_organizations = [] #相关组织
    #事件id
    @property
    def id(self):
        return self._id
    #事件标题
    @property
    def title(self):
        return self._title
    @title.setter
    def title(self,title):
        if isinstance(title,str):
            self._title = title
    #事件内容
    @property
    def cont(self):
        return self._cont
    @cont.setter
    def cont(self,c):
        if isinstance(c,str):
            self._cont = c
    #生成随机id
    def _make_random_id(self):
        return str(uuid.uuid1())
    #新闻通道
    @property
    def newsdocs(self):
        return self._news_docs
    @newsdocs.setter
    def newsdocs(self,news):
        if isinstance(news,str):
            self._news_docs.append(news)
        if isinstance(news,list):
            self._news_docs.extend(news)
    #原发微博
    @property
    def primaryweibodocs(self):
        return self._weibo_primary_docs
    @primaryweibodocs.setter
    def primaryweibodocs(self,weibos):
        if isinstance(weibos,str):
            self._weibo_primary_docs.append(weibos)
        if isinstance(weibos,list):
            self._weibo_primary_docs.extend(weibos)
    #转发微博
    @property
    def transmitweibodocs(self):
        return self._weibo_transmit_docs
    @transmitweibodocs.setter
    def transmitweibodocs(self,weibos):
        if isinstance(weibos,str):
            self._weibo_transmit_docs.append(weibos)
        if isinstance(weibos,list):
            self._weibo_transmit_docs.extend(weibos)
    #微博评论
    @property
    def commentsweibo(self):
        return self._weibo_comments
    @commentsweibo.setter
    def commentsweibo(self,weibo):
        if isinstance(weibo,str):
            self._weibo_comments.append(weibo)
        if isinstance(weibo,list):
            self._weibo_comments.extend(weibo)
    #app新闻
    @property
    def appdocs(self):
        return self._app_docs
    @appdocs.setter
    def appdocs(self,app):
        if isinstance(app,str):
            self._app_docs.append(app)
        if isinstance(app,list):
            self._app_docs.extend(app)
    #app评论
    @property
    def commentsapp(self):
        return self._app_comments
    @commentsapp.setter
    def commentsapp(self,app):
        if isinstance(app,str):
            self._app_comments.append(app)
        if isinstance(app,list):
            self._app_comments.extend(app)
    #weixin
    @property
    def weixindocs(self):
        return self._weixin_docs
    @weixindocs.setter
    def weixindocs(self,weixin):
        if isinstance(weixin,str):
            self._weixin_docs.append(weixin)
        if isinstance(weixin,list):
            self._weixin_docs.extend(weixin)
    #weixin comments
    @property
    def commentsweixin(self):
        return self._weixin_comments
    @commentsweixin.setter
    def commentsweixin(self,weixin):
        if isinstance(weixin,str):
            self._weixin_comments.append(weixin)
        if isinstance(weixin,list):
            self._weixin_comments.extend(weixin)
    #blog
    @property
    def blogdocs(self):
        return self._blog_docs
    @blogdocs.setter
    def blogdocs(self,blog):
        if isinstance(blog,str):
            self._blog_docs.append(blog)
        if isinstance(blog,list):
            self._blog_docs.extend(blog)
    # twitter
    @property
    def twitterdocs(self):
        return self._twitter_docs
    @twitterdocs.setter
    def twitterdocs(self,twitter):
        if isinstance(twitter,str):
            self._twitter_docs.append(twitter)
        if isinstance(twitter,list):
            self._twitter_docs.extend(twitter)
    # oversea news docs
    @property
    def overseadocs(self):
        return self._oversea_news_docs
    @overseadocs.setter
    def overseadocs(self,oversea):
        if isinstance(oversea,str):
            self._oversea_news_docs.append(oversea)
        if isinstance(oversea,list):
            self._oversea_news_docs.extend(oversea)
    # TODO 其他字段的获取及存储