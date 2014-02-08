''' Estimate the amount of time left in a task

@todo - weight task, total by job size

.. Created on Jan 11, 2013
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''
import time, numpy

class progress(object):
    ''' Progress monitor
    '''
    
    def __init__(self, total):
        '''Create a progress monitor with the expected
        number of tasks
        '''
        
        self.history = numpy.zeros((total+1, 2))
        self.history[0, 1] = time.time()
        self.completed = 0
        self.p_total = 0
        self.p_squared = 0
    
    def update(self, work=None):
        ''' Update the monitor
        
        :Parameters:
        
        work : int, optional
               Current work
        '''
        
        if work is None: work = self.history[self.completed, 0]+1
        epoch = time.time()
        if (epoch-self.history[self.completed, 1]) < 1 or self.completed == 0:
            if self.completed == 0: self.completed += 1
            self.history[self.completed, 0] = work
        else:
            rate = (self.history[self.completed, 1]-self.history[self.completed-1, 1]) / \
                   (self.history[self.completed, 0]-self.history[self.completed-1, 0])
            self.p_squared += rate * rate
            self.p_total += rate
            
            self.completed += 1
            self.history[self.completed, :] = (work, epoch)
        
    def predicted_rate(self):
        ''' Predict the work rate for remaining
        
        :Returns:
        
        rate : float
               Work rate
        '''
        
        if self.completed < 2: return None
        
        if 1 == 1:
            return numpy.mean(numpy.diff(self.history[:self.completed+1, 0])/numpy.diff(self.history[:self.completed+1, 1]))
        
        if 1 == 0:
            pessimistic_rate = (self.history[self.completed, 0]-self.history[0, 0]) / \
                               (self.history[self.completed, 1]-self.history[0, 1])
            return pessimistic_rate
            
        
        optimistic_rate = (self.history[self.completed, 0]-self.history[self.completed-1, 0]) / \
                          (self.history[self.completed, 1]-self.history[self.completed-1, 1])
        if self.completed < 3:        
            pessimistic_rate = (self.history[self.completed, 0]-self.history[0, 0]) / \
                               (self.history[self.completed, 1]-self.history[0, 1])
        else:
            avg = self.p_total/self.completed
            pessimistic_rate = 1.0 / ( avg + numpy.sqrt(self.p_squared / self.completed - avg* avg ) *\
                              (self.history[self.completed, 0]/self.history.shape[0]) )
        return (optimistic_rate + pessimistic_rate) / 2
    
    def time_remaining(self, format=False):
        ''' Predict the remaining time to complete work
        
        :Returns:
        
        remaining : float
                    Seconds remaining
        '''
        
        rate = self.predicted_rate()
        if rate is None: return "--"
        work_time_remaining = (self.history.shape[0] - self.history[self.completed, 0]) / rate
        work_time_elapsed = time.time() - self.history[self.completed, 1]
        if format: return elapsed_str(work_time_remaining - work_time_elapsed)
        return work_time_remaining - work_time_elapsed

def elapsed_str(secs):
    ''' Format elasped time in seconds into a string
    
    :Parameters:
    
    secs : float
           Number of seconds
    
    :Returns:
    
    elapse : str
             Human readable elapsed time
    '''
    
    if secs < 0: return "--"
    units = [(60*60*24*7, "w"), (60*60*24, "d"), (60*60, "h"), (60, "m"), (1, "s")]
    for i in xrange(len(units)):
        if secs > units[i][0]: break
    
    n = int(secs/units[i][0])
    parts = ['%d%s'%(n, units[i][1])]
    if i < len(units)-1:
        secs -= n*units[i][0]
        n = int(float(secs)/units[i+1][0])
        parts.append('%d%s'%(n, units[i+1][1]))
        
    return "".join(parts)


