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
from sqlalchemy.orm import relationship, column_property
from sqlalchemy import Table, Column, Integer, String, ForeignKey, DateTime, desc, Float
from sqlalchemy import select, and_
import sqlalchemy
import logging
import sys

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

_sa_logger=logging.getLogger('sqlalchemy')
while len(_sa_logger.handlers) > 0: _sa_logger.removeHandler(_sa_logger.handlers[0])
_sa_logger.setLevel(logging.ERROR)

Base = declarative_base()



if 1 == 0:
    project_session_table = Table('projectexperiments', Base.metadata,
        Column('REF|projects|project', Integer, ForeignKey('projects.DEF_id')),
        Column('REF|leginondata|SessionData|session', Integer, ForeignKey('SessionData.DEF_id'))
    )
    project_session_table.__bind_key__='projectdb'
    
    project_user_table = Table('projectowners', Base.metadata,
        Column('REF|leginondata|UserData|user', Integer, ForeignKey('UserData.DEF_id')),
        Column('REF|projects|project', Integer, ForeignKey('projects.DEF_id'))
    )
    project_user_table.__bind_key__='projectdb'
else:
    
    class ProjectOwners(Base):
        '''
        '''
        __bind_key__ = 'projectdb'
        __tablename__='projectowners'
        
        id = Column('DEF_id', Integer, primary_key=True)
        user_id = Column('REF|leginondata|UserData|user', Integer, ForeignKey('UserData.DEF_id'))
        project_id = Column('REF|projects|project', Integer, ForeignKey('projects.DEF_id'))
    
    class ProjectExperiments(Base):
        '''
        '''
        __bind_key__ = 'projectdb'
        __tablename__='projectexperiments'
        
        id = Column('DEF_id', Integer, primary_key=True)
        project_id = Column('REF|projects|project', Integer, ForeignKey('projects.DEF_id'))
        session_id = Column('REF|leginondata|SessionData|session', Integer, ForeignKey('SessionData.DEF_id'))

class User(Base):
    '''
    '''
    __bind_key__ = 'leginondb'
    __tablename__='UserData'
    
    username = Column('username', String)
    id = Column('DEF_id', Integer, primary_key=True)
    projects = relationship("Projects", secondary=ProjectOwners.__table__)#project_user_table)
    sessions = relationship("Session", order_by=lambda: desc(Session.timestamp))
    #allsessions = relationship("Session", primaryjoin="and_(User.id==ProjectOwners.user_id, ProjectOwners.project_id==ProjectExperiments.project_id, Session.id==ProjectExperiments.session_id)", secondary=ProjectOwners.__table__, order_by=lambda: desc(Session.timestamp))

class Projects(Base):
    '''
    '''
    __bind_key__ = 'projectdb'
    __tablename__='projects'
    
    id = Column('DEF_id', Integer, primary_key=True)
    name = Column('name', String)
    timestamp = Column('DEF_timestamp', DateTime)
    #sessions = relationship("ProjectExperiments")  # - Workaround
    sessions = relationship("Session", secondary=ProjectExperiments.__table__) # - Bug? assumes secondary is on other database
    
class Session(Base):
    __bind_key__ = 'leginondb'
    __tablename__='SessionData'
    
    name = Column('name', String)
    id = Column('DEF_id', Integer, primary_key=True)
    imagedata = relationship("ImageData")
    exposures = relationship("ImageData", primaryjoin="and_(Session.id==ImageData.session, ImageData.label.startswith('Exposure'))")
    imagefilter = relationship("ImageData", lazy="dynamic")
    timestamp = Column('DEF_timestamp', DateTime)
    user = Column('REF|UserData|user', Integer, ForeignKey('UserData.DEF_id'))
    projects = relationship("Projects", secondary=ProjectExperiments.__table__)#project_session_table)
    instrument_id = Column('REF|InstrumentData|instrument', Integer, ForeignKey('InstrumentData.DEF_id'))
    instrument = relationship("Instrument", uselist=False)
    scope = relationship("ScopeEM", uselist=False)
    camera = relationship("CameraEM", uselist=False)

class ScopeEM(Base):
    __bind_key__ = 'leginondb'
    __tablename__='ScopeEMData'
    
    id = Column('DEF_id', Integer, primary_key=True)
    voltage = Column('high tension', Float)
    magnification = Column('magnification', Integer, ForeignKey('PixelSizeCalibrationData.magnification'))
    session_id = Column('REF|SessionData|session', Integer, ForeignKey('SessionData.DEF_id'))
    instrument_id = Column('REF|InstrumentData|tem', Integer, ForeignKey('InstrumentData.DEF_id'))
    instrument = relationship("Instrument", uselist=False)

class PixelSizeCalibration(Base):
    __bind_key__ = 'leginondb'
    __tablename__='PixelSizeCalibrationData'
    
    id = Column('DEF_id', Integer, primary_key=True)
    pixelsize = Column('pixelsize', Float)
    magnification = Column('magnification', Integer)
    session_id=Column('REF|SessionData|session', Integer, ForeignKey('SessionData.DEF_id'))
    camera_id=Column('REF|InstrumentData|ccdcamera', Integer, ForeignKey('CameraEMData.REF|InstrumentData|ccdcamera'))
    instrument_id=Column('REF|InstrumentData|tem', Integer)#, ForeignKey('ImageData.REF|ScopeEMData|scope'))
    #session = relationship("PixelSizeCalibration", uselist=False, backref='calibration')


class CameraEM(Base):
    __bind_key__ = 'leginondb'
    __tablename__='CameraEMData'
    
    id = Column('DEF_id', Integer, primary_key=True)
    pixel_size = Column('SUBD|pixel size|x', Float)
    pixel_sizey = Column('SUBD|pixel size|y', Float)
    session_id = Column('REF|SessionData|session', Integer, ForeignKey('SessionData.DEF_id'))
    instrument_id = Column('REF|InstrumentData|ccdcamera', Integer, ForeignKey('PixelSizeCalibrationData.REF|InstrumentData|ccdcamera'))
    #calibration = relationship("PixelSizeCalibration", uselist=False)

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
    magnification = column_property(select([ScopeEM.magnification]).where(scope_id==ScopeEM.id))
    pixelsize = column_property(select([PixelSizeCalibration.pixelsize]).where(
                                                                               and_(ScopeEM.magnification==PixelSizeCalibration.magnification, 
                                                                                    scope_id==ScopeEM.id,
                                                                                    camera_id==CameraEM.id,
                                                                                    CameraEM.instrument_id==PixelSizeCalibration.camera_id,  #All
                                                                                    ScopeEM.instrument_id==PixelSizeCalibration.instrument_id)).limit(1))
    #calibration = relationship('PixelSizeCalibration', secondary=ScopeEM.__table__, uselist=True)

    
class Instrument(Base):
    __bind_key__ = 'leginondb'
    __tablename__='InstrumentData'
    
    id = Column('DEF_id', Integer, primary_key=True)
    cs = Column('cs', Float)
    pixel_size = Column('camera pixel size', Float)
    #calibration = relationship("PixelSizeCalibration", uselist=False)

def find_exposures(session):
    '''
    '''
    
    return session.imagefilter.filter(ImageData.label=='Exposure') #filter(ImageData.filename=='13jul06a_40S-DHX-GMPPNP-eIF3_00013gr_00217sq_00002hl_00003en')

def projects_for_user(username, password, leginondb='leginondb', projectdb='projectdb', alternate_user=None):
    '''
    '''
    
    leginondb;projectdb;password; # pyflakes hack
    leginondb = sqlalchemy.create_engine('mysql://{username}:{password}@{leginondb}'.format(**locals()), echo=False, echo_pool=False)
    projectdb = sqlalchemy.create_engine('mysql://{username}:{password}@{projectdb}'.format(**locals()), echo=False, echo_pool=False)
    local_vars = locals()
    binds = dict([(v, local_vars[getattr(v, '__bind_key__')])for v in sys.modules[__name__].__dict__.values() if hasattr(v, '__bind_key__')])
    SessionDB = sessionmaker(autocommit=False,autoflush=False)
    db_session = SessionDB(binds=binds)
    if not alternate_user: alternate_user = username
    rs = db_session.query(User).filter(User.username==alternate_user).all()
    if len(rs) > 0: return rs[0]
    return []


