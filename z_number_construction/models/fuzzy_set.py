class FS:
    '''
    Represents a trapezoidal fuzzy set FS(a,b,c,d). A particular case is a triangular fuzzy set FS(l,c,c,r).
    '''
    def __init__(self, a: float, b: float, c: float, d: float):
        assert a <= b <= c <= d, 'Must be that a <= b <= c <= d.'
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def membership_function(self):
        '''Returns the membership function of the fuzzy set.'''
        def mf(x):
            if x <= self.a or x >= self.d:
                return 0
            if self.b <= x <= self.c:
                return 1
            if self.a <= x <= self.b:
                return (x - self.a) / (self.b - self.a) if self.a != self.b else 0
            if self.c <= x <= self.d:
                return (self.d - x) / (self.d - self.c) if self.d != self.c else 0
        return mf
    
    def plot(self, axis, limits=None, **kwargs):
        '''Plots the membership function of the fuzzy set on the given axis.'''
        if limits is None:
            limits = (self.a, self.d)
        axis.plot([limits[0], self.a, self.b, self.c, self.d, limits[1]], [0, 0, 1, 1, 0, 0], **kwargs)

    def specificity(self, u: float) -> float:
        '''Returns the specificity of the fuzzy set for the given universal set of size `u`.'''
        assert u > 0, 'u must be greater than 0.'
        return 1 - (self.c + self.d - self.a - self.b) / (2 * u)
    
    def defuzzify(self, method: str = 'centroid') -> float:
        '''Represents the fuzzy set with a single value, either maximum of its membership function or x-coordinate of the centroid of the trapezoid therebeneath.'''
        if method == 'maximum':
            return (self.b + self.c) / 2
        if method == 'centroid':
            return (self.c ** 2 + self.d ** 2 + self.c * self.d - (self.a ** 2 + self.b ** 2 + self.a * self.b)) / (3 * (self.c + self.d - self.a - self.b))
        raise ValueError(f'Unknown defuzzification method: {method}, must be "maximum" or "centroid".')