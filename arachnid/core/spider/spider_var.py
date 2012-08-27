''' Defines a pool of SPIDER varibles

This module defines a pool of SPIDER variables that can be reused. It automatically
releases any variable when all references have been lost.


.. Created on Aug 24, 2012
.. codeauthor:: Robert Langlois <rl2528@columbia.edu>
'''


class spider_var_pool(list):
    ''' Pool of SPIDER variables
    '''
    __slots__=('var_curr')
    def __init__(self):
        self.var_curr = 0
    
    def get(self, force_new, hook=None, is_stack=False):
        ''' Get a SPIDER variable from the pool or create
        a new one if the pool is empty
        
        :Parameters:
        
        force_new : bool
                    Ensure a new variable is created
        hook : function
               Called when the variable is deleted
        is_stack : bool
                   Flag denoting whether the variable is a stack
        
        :Returns:
        
        var : spider_var
              A SPIDER variable
        '''
        
        return spider_var(self, force_new, hook, is_stack)
    
    def _poolget(self, force_new):
        if len(self) == 0 or force_new:
            self.var_curr = self.var_curr + 1
            return self.var_curr
        else: return self.pop()
        
    def _repool(self, val):
        self.append(int(val))
    
class spider_var(int):
    ''' SPIDER variable
    
    Release SPIDER incore variable back to pool when no longer
    in use.
    '''
    __slots__=('parent', 'hook', 'is_stack')
    def __new__(cls, parent, force_new, hook, is_stack):
        inst = super(spider_var, cls).__new__(cls, parent._poolget(force_new))
        inst.parent = parent
        inst.hook = hook
        inst.is_stack = is_stack
        return inst
    def __del__(self):
        if self.parent is not None: 
            if self.hook is not None:
                self.hook(self)
            self.parent._repool(self)
            self.parent = None
    def __exit__(self, type, value, traceback):
        if self.parent is not None: 
            if self.hook is not None:
                self.hook(self)
            self.parent._repool(self)
            self.parent = None
    def __enter__(self):
        return self
    