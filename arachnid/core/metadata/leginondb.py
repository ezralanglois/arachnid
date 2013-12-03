'''
.. Created on Dec 3, 2013
.. codeauthor:: robertlanglois
'''

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker #scoped_session
from sqlalchemy.orm import relationship
from sqlalchemy import Table, Column, Integer, String, ForeignKey, DateTime
import sqlalchemy
import sys

Base = declarative_base()

project_user_table = Table('projectowners', Base.metadata,
    Column('REF|leginondata|UserData|user', Integer, ForeignKey('UserData.DEF_id')),
    Column('REF|projects|project', Integer, ForeignKey('projects.DEF_id'))
)
project_user_table.__bind_key__='projectdb'


project_session_table = Table('projectexperiments', Base.metadata,
    Column('REF|projects|project', Integer, ForeignKey('projects.DEF_id')),
    Column('REF|leginondata|SessionData|session', Integer, ForeignKey('SessionData.DEF_id'))
)
project_session_table.__bind_key__='projectdb'

class User(Base):
    '''
    '''
    __bind_key__ = 'leginondb'
    __tablename__='UserData'
    
    username = Column('username', String)
    id = Column('DEF_id', Integer, primary_key=True)
    projects = relationship("Projects", secondary=project_user_table)
    sessions = relationship("Session")

class Projects(Base):
    '''
    '''
    __bind_key__ = 'projectdb'
    __tablename__='projects'
    
    id = Column('DEF_id', Integer, primary_key=True)
    name = Column('name', String)
    timestamp = Column('DEF_timestamp', DateTime)
    sessions = relationship("Projects", secondary=project_session_table)

class Session(Base):
    __bind_key__ = 'leginondb'
    __tablename__='SessionData'
    
    name = Column('name', String)
    id = Column('DEF_id', Integer, primary_key=True)
    imagedata = relationship("ImageData")
    user = Column('REF|UserData|user', Integer, ForeignKey('UserData.DEF_id')) #

class ImageData(Base):
    __bind_key__ = 'leginondb'
    __tablename__='AcquisitionImageData'
    
    filename = Column('filename', String)
    id = Column('DEF_id', Integer, primary_key=True)
    session = Column('REF|SessionData|session', Integer, ForeignKey('SessionData.DEF_id'))
    
def projects_for_user(username, password, hostname, leginondb='leginondb', projectdb='projectdb'):
    '''
    '''
    
    leginondb;
    projectdb;
    hostname;
    password;
    leginondb = sqlalchemy.create_engine('mysql://{username}:{password}@{hostname}/{leginondb}'.format(**locals()))
    projectdb = sqlalchemy.create_engine('mysql://{username}:{password}@{hostname}/{projectdb}'.format(**locals()))
    local_vars = locals()
    binds = dict([(v, local_vars[getattr(v, '__bind_key__')])for v in sys.modules[__name__].__dict__.values() if hasattr(v, '__bind_key__')])
    SessionDB = sessionmaker(autocommit=False,autoflush=False)
    db_session = SessionDB(binds=binds)
    rs = db_session.query(User).filter(User.username==username)
    return rs
