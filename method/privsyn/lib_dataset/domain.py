from functools import reduce


class Domain:
    def __init__(self, attrs, shape):
        """ Construct a Domain object

        :param attrs: a list or tuple of attribute names
        :param shape: a list or tuple of domain sizes for each attribute
        """
        assert len(attrs) == len(shape), 'dimensions must be equal'

        self.attrs = tuple(attrs)
        self.shape = tuple(shape)
        self.config = dict(zip(attrs, shape))
        self.attr_index_mapping = dict(zip(attrs, range(len(attrs))))

    @staticmethod
    def fromdict(config):
        """ Construct a Domain object from a dictionary of { attr : size } values """
        return Domain(config.keys(), config.values())

    def change_shape(self, attr_name, new_shape):
        attr_index = self.attr_index_mapping[attr_name]
        shape = list(self.shape)
        shape[attr_index] = new_shape
        self.shape = tuple(shape)

        self.config[attr_name] = new_shape

    def project(self, attrs): #返回attrs子集的domain
        """ project the domain onto a subset of attributes

        :param attrs: the attributes to project onto
        :return: the projected Domain object
        """
        # return the projected domain
        if type(attrs) is str:
            attrs = [attrs]
            
        shape = list(self.config[a] for a in attrs)

        return Domain(attrs, shape)

    def marginalize(self, attrs): #返回剔除给定attrs的集合的domain
        """ marginalize out some attributes from the domain (opposite of project)

        :param attrs: the attributes to marginalize out
        :return: the marginalized Domain object
        """
        proj = [a for a in self.attrs if not a in attrs]

        return self.project(proj)

    def axes(self, attrs): #返回给定attrs的index
        """ return the axes tuple for the given attributes

        :param attrs: the attributes
        :return: a tuple with the corresponding axes
        """
        return tuple(self.attrs.index(a) for a in attrs)

    def transpose(self, attrs):
        """ reorder the attributes in the domain object """
        return self.project(attrs)

    def invert(self, attrs):
        """ returns the attributes in the domain not in the list """
        return [a for a in self.attrs if a not in attrs]

    def merge(self, other):
        """ merge this domain object with another

        :param other: another Domain object
        :return: a new domain object covering the full domain

        Example:
        >>> D1 = Domain(['a','b'], [10,20])
        >>> D2 = Domain(['b','c'], [20,30])
        >>> D1.merge(D2)
        Domain(['a','b','c'], [10,20,30])
        """
        extra = other.marginalize(self.attrs) #剔除重复元素，返回一个domain

        return Domain(self.attrs + extra.attrs, self.shape + extra.shape)

    def contains(self, other):
        """ determine if this domain contains another

        """
        return set(other.attrs) <= set(self.attrs)

    def size(self, attrs=None):
        """ return the total size of the domain """
        if attrs is None:
            return reduce(lambda x, y: x * y, self.shape, 1)

        return self.project(attrs).size()

    def sort(self, how='size'):
        """ return a new domain object, sorted by attribute size or attribute name """
        if how == 'size':
            attrs = sorted(self.attrs, key=self.size)
        elif how == 'name':
            attrs = sorted(self.attrs)

        return self.project(attrs)

    def canonical(self, attrs):
        """ return the canonical ordering of the attributes """
        return tuple(a for a in self.attrs if a in attrs)

    def __contains__(self, attr):
        return attr in self.attrs

    def __getitem__(self, a):
        """ return the size of an individual attribute
        :param a: the attribute
        """
        return self.config[a]

    def __iter__(self):
        """ iterator for the attributes in the domain """
        return self.attrs.__iter__()

    def __len__(self):
        return len(self.attrs)

    def __eq__(self, other):
        return self.attrs == other.attrs and self.shape == other.shape

    def __repr__(self):
        inner = ', '.join(['%s: %d' % x for x in zip(self.attrs, self.shape)])
        return 'Domain(%s)' % inner

    def __str__(self):
        return self.__repr__()
