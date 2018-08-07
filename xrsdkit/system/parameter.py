class Parameter(object):
    
    def __init__(self,value,fixed=False,bounds=[None,None],constraint_expr=None):
        self.value = value
        self.fixed = fixed
        self.bounds = bounds
        self.constraint_expr = constraint_expr

    def to_dict(self):
        return dict(
            value=self.value,
            fixed=self.fixed,
            bounds=self.bounds,
            constraint_expr=self.constraint_expr
            ) 
    
    def update_from_dict(self,paramd):
        if 'value' in paramd:
            self.value = paramd['value']
        if 'fixed' in paramd:
            self.fixed = paramd['fixed']
        if 'bounds' in paramd:
            self.bounds = paramd['bounds']
        if 'constraint_expr' in paramd:
            self.constraint_expr = paramd['constraint_expr']


