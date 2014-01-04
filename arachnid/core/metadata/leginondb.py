''' Interface to access data in the Leginon Database

This interface relies on `SQLAlchemy<http://www.sqlalchemy.org/>`_ to build
a relational mapper between Python Objects and SQL. This is not a complete
mapping of the Leginon database, but sufficent to get all information
necessary to process the data.

.. todo::

    needs to be replaced - multiple scopes
    scope = relationship("ScopeEM", uselist=False)

.. Created on Dec 3, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker #scoped_session
from sqlalchemy.orm import relationship, column_property
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, desc, Float
from sqlalchemy import select, and_
import sqlalchemy
import logging
import sys

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.DEBUG)

_sa_logger=logging.getLogger('sqlalchemy')
#while len(_sa_logger.handlers) > 0: _sa_logger.removeHandler(_sa_logger.handlers[0])
_sa_logger.setLevel(logging.ERROR)

Base = declarative_base()
    
class ProjectOwners(Base):
    __bind_key__ = 'projectdb'
    __tablename__='projectowners'
    
    id = Column('DEF_id', Integer, primary_key=True)
    user_id = Column('REF|leginondata|UserData|user', Integer, ForeignKey('UserData.DEF_id'))
    project_id = Column('REF|projects|project', Integer, ForeignKey('projects.DEF_id'))

class ProjectExperiments(Base):
    __bind_key__ = 'projectdb'
    __tablename__='projectexperiments'
    
    id = Column('DEF_id', Integer, primary_key=True)
    project_id = Column('REF|projects|project', Integer, ForeignKey('projects.DEF_id'))
    session_id = Column('REF|leginondata|SessionData|session', Integer, ForeignKey('SessionData.DEF_id'))
    
    session = relationship('Session', backref='experiments', primaryjoin='ProjectExperiments.session_id==Session.id', order_by=lambda: desc(Session.timestamp))
    project = relationship('Projects', backref="experiments", primaryjoin='ProjectExperiments.project_id==Projects.id', order_by=lambda: desc(Projects.timestamp))

class Projects(Base):
    __bind_key__ = 'projectdb'
    __tablename__='projects'
    
    id = Column('DEF_id', Integer, primary_key=True)
    name = Column('name', String)
    timestamp = Column('DEF_timestamp', DateTime)
    #sessions = relationship("ProjectExperiments")  # - Workaround
    #sessions = relationship("Session", secondary=ProjectExperiments.__table__) # - Bug? assumes secondary is on other database

class User(Base):
    __bind_key__ = 'leginondb'
    __tablename__='UserData'
    
    username = Column('username', String)
    id = Column('DEF_id', Integer, primary_key=True)
    projects = relationship("Projects", secondary=ProjectOwners.__table__, order_by=lambda: desc(Projects.timestamp))#project_user_table)
    sessions = relationship("Session", order_by=lambda: desc(Session.timestamp))
    firstname = Column('firstname', String)
    lastname = Column('lastname', String)
    fullname = column_property(firstname + " " + lastname)
    #allsessions = relationship("Session", primaryjoin="and_(User.id==ProjectOwners.user_id, ProjectOwners.project_id==ProjectExperiments.project_id, Session.id==ProjectExperiments.session_id)", secondary=ProjectOwners.__table__, order_by=lambda: desc(Session.timestamp))


class Session(Base):
    __bind_key__ = 'leginondb'
    __tablename__='SessionData'
    
    name = Column('name', String)
    id = Column('DEF_id', Integer, primary_key=True)
    exposures = relationship("ImageData", primaryjoin="and_(Session.id==ImageData.session, ImageData.label.startswith('Exposure'))")
    timestamp = Column('DEF_timestamp', DateTime)
    image_path = Column('image path', String)
    frame_path = Column('frame path', String)
    user = Column('REF|UserData|user', Integer, ForeignKey('UserData.DEF_id'))
    projects = relationship("Projects", secondary=ProjectExperiments.__table__)#, backref='sessions')#project_session_table)
    instrument_id = Column('REF|InstrumentData|instrument', Integer, ForeignKey('InstrumentData.DEF_id'))
    instrument = relationship("Instrument", uselist=False)
    scope = relationship("ScopeEM", uselist=False) # .. todo:: needs to be replaced - multiple scopes
    camera = relationship("CameraEM", uselist=False)
    # Unused
    #imagefilter = relationship("ImageData", lazy="dynamic")
    #imagedata = relationship("ImageData")
    

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
    type = Column('exposure type', String)
    
class NormImage(Base):
    __bind_key__ = 'leginondb'
    __tablename__='NormImageData'
    
    id = Column('DEF_id', Integer, primary_key=True)
    session_id = Column('REF|SessionData|session', Integer)
    filename = Column('filename', String)
    mrcimage = Column('MRC|image', String)
    norm_path = column_property(select([Session.image_path]).where(session_id==Session.id))
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
    norm_id = Column('REF|NormImageData|norm', Integer, ForeignKey('NormImageData.DEF_id'))
    filename = Column('filename', String)
    mrcimage = Column('MRC|image', String)
    label = Column('label', String)
    frame_list = Column('SEQ|use frames', Integer)
    scope = relationship("ScopeEM", uselist=False)
    camera = relationship("CameraEM", uselist=False)
    #pixeltype = Column('pixeltype', Float) - numpy.dtype
    norm_path = column_property(select([NormImage.norm_path]).where(norm_id==NormImage.id))
    norm_filename = column_property(select([NormImage.filename]).where(norm_id==NormImage.id))
    norm_mrcimage = column_property(select([NormImage.mrcimage]).where(norm_id==NormImage.id))
    magnification = column_property(select([ScopeEM.magnification]).where(scope_id==ScopeEM.id))
    pixelsize = column_property(select([PixelSizeCalibration.pixelsize]).where(
                                                                               and_(ScopeEM.magnification==PixelSizeCalibration.magnification, 
                                                                                    scope_id==ScopeEM.id,
                                                                                    camera_id==CameraEM.id,
                                                                                    CameraEM.instrument_id==PixelSizeCalibration.camera_id,  #All
                                                                                    ScopeEM.instrument_id==PixelSizeCalibration.instrument_id)).limit(1))
class Instrument(Base):
    __bind_key__ = 'leginondb'
    __tablename__='InstrumentData'
    
    id = Column('DEF_id', Integer, primary_key=True)
    cs = Column('cs', Float)

def query_session_info(username, password, leginondb, projectdb, session):
    ''' Get the user relational object to access the Leginon
    
    :Parameters:
    
    username : str
               Username 
    password : str
               Password
    leginondb : str
                Host/path to Leginon database, e.g. 111.222.32.143/leginondb
    projectdb  : str
                Host/path to Leginon Project database, e.g. 111.222.32.143/projectdb
    session : str
              Name of session to query
    
    :Returns:
    
    session : Session
              Relational user object that accesses 
              information from the database
    '''
    
    leginondb;projectdb;password; # pyflakes hack
    leginondb = sqlalchemy.create_engine('mysql://{username}:{password}@{leginondb}'.format(**locals()), echo=False, echo_pool=False)
    projectdb = sqlalchemy.create_engine('mysql://{username}:{password}@{projectdb}'.format(**locals()), echo=False, echo_pool=False)
    local_vars = locals()
    binds = dict([(v, local_vars[getattr(v, '__bind_key__')])for v in sys.modules[__name__].__dict__.values() if hasattr(v, '__bind_key__')])
    SessionDB = sessionmaker(autocommit=False,autoflush=False)
    db_session = SessionDB(binds=binds)
    rs = db_session.query(Session).filter(Session.name==session).all()
    if len(rs) > 0: return rs[0]
    return None

def query_user_info(username, password, leginondb, projectdb, targetuser=None):
    ''' Get the user relational object to access the Leginon
    
    :Parameters:
    
    username : str
               Username 
    password : str
               Password
    leginondb : str
                Host/path to Leginon database, e.g. 111.222.32.143/leginondb
    projectdb  : str
                Host/path to Leginon Project database, e.g. 111.222.32.143/projectdb
    targetuser : str, optional
                 Target user to query
    
    :Returns:
    
    user : User
           Relational user object that accesses 
           information from the database
    '''
    
    leginondb;projectdb;password; # pyflakes hack
    leginondb = sqlalchemy.create_engine('mysql://{username}:{password}@{leginondb}'.format(**locals()), echo=False, echo_pool=False)
    projectdb = sqlalchemy.create_engine('mysql://{username}:{password}@{projectdb}'.format(**locals()), echo=False, echo_pool=False)
    local_vars = locals()
    binds = dict([(v, local_vars[getattr(v, '__bind_key__')])for v in sys.modules[__name__].__dict__.values() if hasattr(v, '__bind_key__')])
    SessionDB = sessionmaker(autocommit=False,autoflush=False)
    db_session = SessionDB(binds=binds)
    if targetuser is None: targetuser = username
    rs = db_session.query(User).filter(User.username==targetuser).all()
    #if len(rs) > 0: return rs[0]
    if len(rs) > 0: 
        user = rs[0]
        experiments = []#user.projects[0].experiments
        projects = user.projects
        for i in xrange(len(projects)):
            experiments.extend(projects[i].experiments)
        return user, experiments
    return [], []


