#!/usr/bin/python
# -*- coding: utf-8 -*-
#import a
import gol as gl
 
gl._init()
gl.set_value('name', 'cc')
gl.set_value('score', 90)
name = gl.get_value('name')
score = gl.get_value('score')
 
print("%s: %s" % (name, score))

