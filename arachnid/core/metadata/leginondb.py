''' Interface to access data in the Leginon Database

This interface relies on `SQLAlchemy<http://www.sqlalchemy.org/>`_ to build
a relational mapper between Python Objects and SQL. This is not a complete
mapping of the Leginon database, but sufficent to get all information
necessary to process the data.

.. Created on Dec 3, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker #scoped_session
from sqlalchemy.orm import relationship
from sqlalchemy import Table, Column, Integer, String, ForeignKey, DateTime, desc, Float
import sqlalchemy
import logging
import sys

Base = declarative_base()

project_user_table = Table('projectowners', Base.metadata,
    Column('REF|leginondata|UserData|user', Integer, ForeignKey('UserData.DEF_id')),
    Column('REF|projects|project', Integer, ForeignKey('projects.DEF_id'))
)
project_user_table.__bind_key__='projectdb'

if 1 == 1:
    project_session_table = Table('projectexperiments', Base.metadata,
        Column('REF|projects|project', Integer, ForeignKey('projects.DEF_id')),
        Column('REF|leginondata|SessionData|session', Integer, ForeignKey('SessionData.DEF_id'))
    )
    project_session_table.__bind_key__='projectdb'

else:
    
    class ProjectExperiments(Base):
        '''
        '''
        __bind_key__ = 'projectdb'
        __tablename__='projectexperiments'
        
        id = Column('DEF_id', Integer, primary_key=True)
        project_id = Column('REF|projects|project', Integer, ForeignKey('projects.DEF_id'))
        session_id = Column('REF|leginondata|SessionData|session', Integer)

class User(Base):
    '''
    '''
    __bind_key__ = 'leginondb'
    __tablename__='UserData'
    
    username = Column('username', String)
    id = Column('DEF_id', Integer, primary_key=True)
    projects = relationship("Projects", secondary=project_user_table)
    sessions = relationship("Session", order_by=lambda: desc(Session.timestamp))

class Projects(Base):
    '''
    '''
    __bind_key__ = 'projectdb'
    __tablename__='projects'
    
    id = Column('DEF_id', Integer, primary_key=True)
    name = Column('name', String)
    timestamp = Column('DEF_timestamp', DateTime)
    #sessions = relationship("ProjectExperiments")  # - Workaround
    #sessions = relationship("Session", secondary=project_session_table) - Bug? assumes secondary is on other database
    
class Session(Base):
    __bind_key__ = 'leginondb'
    __tablename__='SessionData'
    
    name = Column('name', String)
    id = Column('DEF_id', Integer, primary_key=True)
    imagedata = relationship("ImageData")
    imagefilter = relationship("ImageData", lazy="dynamic")
    timestamp = Column('DEF_timestamp', DateTime)
    user = Column('REF|UserData|user', Integer, ForeignKey('UserData.DEF_id'))
    projects = relationship("Projects", secondary=project_session_table)
    instrument_id = Column('REF|InstrumentData|instrument', Integer, ForeignKey('InstrumentData.DEF_id'))
    instrument = relationship("Instrument", uselist=False)
    calibration = relationship("PixelSizeCalibration", uselist=False)
    scope = relationship("ScopeEM", uselist=False)
    camera = relationship("CameraEM", uselist=False)

class ImageData(Base):
    __bind_key__ = 'leginondb'
    __tablename__='AcquisitionImageData'
    
    filename = Column('filename', String)
    id = Column('DEF_id', Integer, primary_key=True)
    session = Column('REF|SessionData|session', Integer, ForeignKey('SessionData.DEF_id'))
    timestamp = Column('DEF_timestamp', DateTime)
    scope_id = Column('REF|ScopeEMData|scope', Integer, ForeignKey('ScopeEMData.DEF_id'))
    camera_id = Column('REF|CameraEMData|camera', Integer, ForeignKey('CameraEMData.DEF_id'))
    filename = Column('filename', String)
    label = Column('label', String)
    scope = relationship("ScopeEM", uselist=False)
    camera = relationship("CameraEM", uselist=False)
    #pixeltype = Column('pixeltype', Float) - numpy.dtype

class ScopeEM(Base):
    __bind_key__ = 'leginondb'
    __tablename__='ScopeEMData'
    
    id = Column('DEF_id', Integer, primary_key=True)
    voltage = Column('high tension', Float)
    magnification = Column('magnification', Float)
    session_id = Column('REF|SessionData|session', Integer, ForeignKey('SessionData.DEF_id'))
    instrument_id = Column('REF|InstrumentData|tem', Integer, ForeignKey('InstrumentData.DEF_id'))
    instrument = relationship("Instrument", uselist=False)

class CameraEM(Base):
    __bind_key__ = 'leginondb'
    __tablename__='CameraEMData'
    
    id = Column('DEF_id', Integer, primary_key=True)
    pixel_size = Column('SUBD|pixel size|x', Float)
    pixel_sizey = Column('SUBD|pixel size|y', Float)
    session_id = Column('REF|SessionData|session', Integer, ForeignKey('SessionData.DEF_id'))
    
class PixelSizeCalibration(Base):
    __bind_key__ = 'leginondb'
    __tablename__='PixelSizeCalibrationData'
    id = Column('DEF_id', Integer, primary_key=True)
    pixelsize = Column('pixelsize', Float)
    session_id=Column('REF|SessionData|session', Integer, ForeignKey('SessionData.DEF_id'))
    #session = relationship("PixelSizeCalibration", uselist=False, backref='calibration')
    
class Instrument(Base):
    __bind_key__ = 'leginondb'
    __tablename__='InstrumentData'
    
    id = Column('DEF_id', Integer, primary_key=True)
    cs = Column('cs', Float)
    pixel_size = Column('camera pixel size', Float)

def find_exposures(session):
    '''
    '''
    
    return session.imagefilter.filter(ImageData.label=='Exposure') #filter(ImageData.filename=='13jul06a_40S-DHX-GMPPNP-eIF3_00013gr_00217sq_00002hl_00003en')

def projects_for_user(username, password, leginondb='leginondb', projectdb='projectdb', alternate_user=None):
    '''
    '''
    
    leginondb;projectdb;password; # pyflakes hack
    leginondb = sqlalchemy.create_engine('mysql://{username}:{password}@{leginondb}'.format(**locals()))
    projectdb = sqlalchemy.create_engine('mysql://{username}:{password}@{projectdb}'.format(**locals()))
    local_vars = locals()
    binds = dict([(v, local_vars[getattr(v, '__bind_key__')])for v in sys.modules[__name__].__dict__.values() if hasattr(v, '__bind_key__')])
    SessionDB = sessionmaker(autocommit=False,autoflush=False)
    db_session = SessionDB(binds=binds)
    if not alternate_user: alternate_user = username
    rs = db_session.query(User).filter(User.username==alternate_user).all()
    if len(rs) > 0: return rs[0].sessions
    return []


