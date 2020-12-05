"""graph.py implements classes that represent graphs.
"""

import collections
import functools


class GraphError(Exception):
    """Base exception class of the graph module."""
    pass

class NodeError(GraphError):
    """Error associated with nodes."""
    pass

class EdgeError(GraphError):
    """Error associated with edges."""
    pass


class Node(collections.UserDict):
    """Represents a graph node in an adjacency list.

    `Node` is a dict-like class that stores the adjacent nodes of a simple
    directed graph, the distance to those adjacent nodes, and other
    associated data.

    `Node` uses other `Node` objects as its keys, with the values representing
    the distance to a given node.

    Parameters
    ----------
    key : hashable
        Sets the attribute `k`.
    value : optional
        Sets the attribute `v`.
    blob : optional
        Sets the attribute `blob`.

    Attributes
    ----------
    k : hashable
        A unique, hashable key that identifies the node; read-only.
    v : optional
        A primary value associated with the node.
    blob : optional
        An arbitrary data blob associated with the node.
    """

    def __init__(self, key, value=None, blob=None):
        self.data = dict()
        self._k = key
        self.v = value
        self.blob = blob

    def __hash__(self):
        return hash(self._k)

    def __str__(self):
        if self.v is None:
            return f"Node({self.k!r})"
        return f"Node({self.k!r}, {self.v!r})"

    def __repr__(self):
        return f"Node({self.k!r}, {self.v!r}, {self.blob!r})"

    @property
    def k(self):
        return self._k

    def adde(self, node, dist):
        """Add an edge to `node` with distance `dist`."""
        self[node] = dist

    # def contract(self, other):
    #     self_other = self[other] if other in self else None
    #     other_self = other[self] if self in other else None
    #     for k, v in other.items():
    #         # Update edge (self, k)
    #         if self_other is None:
    #             self[k] = v
    #         elif k in self:
    #             self[k] = min(self_other + v, self[k])
    #         else:
    #             self[k] = self_other + v
    #         # Update edge (k, self), if exists
    #         if self in k:
    #             if other in k:  # Otherwise, no alternate path
    #                 k_other = k[other]
    #                 if other_self is None:
    #                     k[self] = min(k_other, k[self])
    #                 else:
    #                     k[self] = min(k_other + other_self, k[self])

    def cleave(self, new_key, dist, value=None, blob=None):
        """Cleave this node and create an edge to the new node with distance `dist`.
        """
        if new_key == self._k:
            raise KeyError(f"Duplicate node key: {new_key}")
        new_node = Node(new_key, value, blob)
        new_node.update(self)
        self.data = {new_node: dist}
        return new_node

    def basic_search(self, queue_class, select_func, append_func, dest=None, max_depth=None):
        """Generic implemenation of a basic graph search algorithm."""
        distances = {self: 0}
        parents = {self: None}
        queue = queue_class([self])
        i = 1
        found = False
        while queue:
            visit_node = select_func(queue)
            for neighbor in visit_node:
                if neighbor in distances:
                    continue
                distances[neighbor] = distances[visit_node] + 1
                if neighbor is dest:
                    found = True
                    break
                parents[neighbor] = visit_node
                append_func(queue, neighbor)
            if found:
                break
            if max_depth is not None and i > max_depth:
                break
        return distances, parents

    bfs = functools.partialmethod(
        basic_search,
        collections.deque,
        lambda q: q.popleft(),
        lambda q, x: q.append(x),
    )
    dfs = functools.partialmethod(
        basic_search,
        collections.deque,
        lambda q: q.popleft(),
        lambda q, x: q.appendleft(x),
    )


class _Decorators(object):
    """A helper class with useful decorators

    These decorators are collected in a class so that they can be used on
    bound methods.
    """

    @staticmethod
    def check_key(func):
        """Checks whether the graph object has a node with `key`."""
        @functools.wraps(func)
        def checked_func(self, key, *args, **kwargs):
            if not self.has(key):
                raise KeyError(f"Node {key} does not exist")
            return func(self, key, *args, **kwargs)
        if checked_func.__doc__ is not None:  # pylint: disable=no-member
            # pylint: disable=no-member
            checked_func.__doc__ += f"""
        Raises
        ------
        KeyError
            If `key` is not found.
            """
        return checked_func

    @staticmethod
    def check_key_pair(func):
        """Checks whether the graph object has nodes with `key1` and `key2`."""
        @functools.wraps(func)
        def checked_func(self, key1, key2, *args, **kwargs):
            if not self.has(key1):
                raise KeyError(f"Node {key1} does not exist")
            if not self.has(key2):
                raise KeyError(f"Node {key2} does not exist")
            return func(self, key1, key2, *args, **kwargs)
        if checked_func.__doc__ is not None:  # pylint: disable=no-member
            # pylint: disable=no-member
            checked_func.__doc__ += f"""
        Raises
        ------
        KeyError
            If `key1` or `key2` is not found.
        """
        return checked_func

    @staticmethod
    class VertexOrEdge(object):
        """Dispatch methods based on whether the key represents a node or an edge.

        `VertexOrEdge` creates a decorator parameterized by the methods it
        dispatches to. It is implemented as a full class in order to support
        bound methods,
        """
        def __init__(self, v_func, e_func):
            self.v_func = v_func
            self.e_func = e_func
        def __call__(self, func):
            v_func = self.v_func
            e_func = self.e_func
            @functools.wraps(func)
            def vertex_edge_func(this, key, *args, **kwargs):
                if isinstance(key, tuple):
                    if len(key) != 2:
                        raise TypeError(f"An edge must consist of exactly two vertices: {key}")
                    return e_func(this, *key, *args, **kwargs)
                return v_func(this, key, *args, **kwargs)
            vertex_edge_func.__doc__ = f"""
            Dispatch `self.{v_func.__name__}()` or `self.{e_func.__name__}()` depending on `key` type.

            Parameters
            ----------
            key : hashable or (hashable, hashable)

            Returns
            -------
            bool
                If `key` is a tuple of length two, return ``self.{e_func.__name__}(*key)``;
                otherwise, return ``self.{v_func.__name__}(key)``.
            """
            return vertex_edge_func


class AbstractGraph(object):
    """An abstract base class for all graphs"""
    pass

class BaseGraph(AbstractGraph):
    """The base class for all graph implementations.

    It defines several useful formatting and representation methods common to
    all graph classes.

    Methods
    -------
    __str__()
    __repr__()
    """

    data = None

    def __str__(self):
        adj_list = dict()
        for k, n in self.data.items():
            adj_list[k] = {n.k: d for n, d in n.items()}
        return str(adj_list)

    def __repr__(self):
        class_name = self.__class__.__name__
        dict_class = self.data.__class__.__name__
        edges = []
        for key, node in self.data.items():
            for neighbor, dist in node.items():
                edges.append((key, neighbor.k, dist))
        return f"{class_name}({edges=!r}, {dict_class=!s})"


class Digraph(BaseGraph, collections.UserDict):
    """A class representing simple directed graphs (digraphs)

    `Digraph` is a dict-like object that stores a mapping of unique, hashable
    keys to their corresponding `Node` objects. `Digraph` supports one edge
    per ordered pair of nodes as well as a weight or distance associated with
    each edge. For undirected graphs, use the `Graph` subclass instead.

    `Digraph` has methods to check for the existence, add, get or set the
    value of, and delete nodes or edges. In addition, `Digraph` supports
    standard syntax sugars of Python container classes for convenience.

    Attributes
    ----------
    data : dict_like
        The underlying dict-like object that maps keys to `Node` objects.
    directed : bool
        True for directed graphs.

    Methods
    -------
    __contains__(key)
    __delitem__(key)
    __getitem__(key)
    __setitem__(key)
    add(key, value=None)
    adde(key1, key2, dist=1)
    dele(key1, key2)
    deln(key)
    get(key)
    gete(key1, key2)
    has(key)
    isadj(key1, key2)
    neigh(key)
    nodes()
    set(key, value)
    sete(key1, key2, dist=1)

    See Also
    --------
    tulun.Graph : A subclass that implements simple undirected graphs.

    Notes
    -----
    `Digraph` raises an error if you attempt to add a loop with its methods,
    but you can get around this restriction by calling the `adde()` method on
    the underlying `Node` object. Do so at your own peril.

    Examples
    --------
    Common operations of of a `Digraph` object.

    >>> g = Digraph()
    >>> g.add('a')
    Node('a', None, None)
    >>> g['b'] = 4
    >>> g['b']
    Node('b', 4, None)
    >>> g.adde('a', 'b')
    >>> g
    Digraph(edges=[('a', 'b', 1)], dict_class=dict)
    >>> g['a', 'c'] = 2
    >>> g['b', 'c'] = 5
    >>> g['b', 'd'] = 3
    Digraph(edges=[('a', 'b', 1), ('a', 'c', 2), ('b', 'c', 5'), ('b', 'd', 3)], dict_class=dict)
    >>> del g['b', 'd']
    >>> g
    Digraph(edges=[('a', 'b', 1), ('a', 'c', 2), ('b', 'c', 5')], dict_class=dict)
    >>> del g['a']
    >>> g
    Digraph(edges=[('b', 'c', 5')], dict_class=dict)
    """

    directed = True

    def __init__(self, edges=None, dict_class=dict):
        self.data = dict_class()
        if edges is not None:
            self._init_edges(edges)

    def _init_edges(self, edges):
        for e in edges:
            if len(e) == 2:
                key1, key2 = e
                dist = 1
            elif len(e) == 3:
                key1, key2, dist = e
            else:
                raise ValueError(f"Edge description must be (key1, key2) or "
                                  "(key1, key2, dist): {e}")
            self.sete(key1, key2, dist)

    # Existence checks.

    def has(self, key):
        """Check whether key is of a valid node or is a valid node.

        Parameters
        ----------
        key : hashable or Node

        Returns
        -------
        bool
        """
        if isinstance(key, Node):
            return key.k in self.data
        return key in self.data

    @_Decorators.check_key
    def get(self, key):
        """-> Node associated with `key`.

        Parameters
        ----------
        key : hashable

        Returns
        -------
        Node
        """
        return self.data[key]
    _get = get.__wrapped__

    @_Decorators.check_key_pair
    def isadj(self, key1, key2):
        """Check whether edge (`key1`, `key2`) exists.

        Parameters
        ----------
        key1 : hashable
        key2 : hashable

        Returns
        -------
        bool
        """
        return self._get(key2) in self._get(key1)
    _isadj = isadj.__wrapped__

    @_Decorators.VertexOrEdge(has, isadj)
    def __contains__(self, key):
        pass

    # Getters.

    def nodes(self):
        """-> A dictionary view of the `Node` objects in `self`.
        """
        return self.data.values()

    @_Decorators.check_key
    def neigh(self, key):
        """-> A dictionary view of the neighbors of the node associated with `key`.
        """
        return self._get(key).keys()
    _neigh = neigh.__wrapped__

    @_Decorators.check_key_pair
    def gete(self, key1, key2):
        """-> Distance value associated with edge (`key1`, `key2`).
        """
        if self._isadj(key1, key2):
            return self._get(key1)[self._get(key2)]
        raise KeyError(f"Edge ({key1}, {key2}) does not exist")
    _gete = gete.__wrapped__

    @_Decorators.VertexOrEdge(get, gete)
    def __getitem__(self, key):
        pass

    # Adders.

    def _add(self, key, value):
        """Unconditionally overrides `key` with a new `Node` object. UNSAFE.
        """
        new_node = Node(key, value)
        self.data[key] = new_node
        return new_node

    def _set(self, key, value):
        """Sets value of node associated with `key` without checking for existence. UNSAFE.
        """
        self._get(key).v = value

    def add(self, key, value=None):
        """Add a new node with `key` or update an existing one with `value`.

        Parameters
        ----------
        key : hashable
        value : optional

        Returns
        -------
        Node
            The newly created `Node` object or the updated old object.

        Raises
        ------
        KeyError
            Raised if `value` is None but a node with `key` already exists.
            This a safeguard to prevent accidentally overriding an existing
            node.

        See Also
        --------
        set(key) : `value` is mandatory and no error is raised if a node
            with `key` exists.
        """
        if self.has(key):
            if value is None:
                raise NodeError(f"Node {key} already exists")
            self._set(key, value)
            return self._get(key)
        self._add(key, value)

    def set(self, key, value):
        """Add a new or update an existing node with `key` and `value`.

        Parameters
        ----------
        key : hashable
        value

        Returns
        -------
        Node
            The newly created `Node` object or the updated old object.

        See Also
        --------
        add(key) : `value` is optional. Raises an error if a node with
            `key` exists but `value` is not provided.
        """
        if self.has(key):
            self._set(key, value)
        self._add(key, value)

    def _adde(self, key1, key2, dist):
        self._get(key1).adde(self._get(key2), dist)

    def _check_adde(self, key1, key2):
        if key1 == key2:
            raise EdgeError(f"Digraph does not support loops: ({key1}, {key2})")
        if self._isadj(key1, key2):
            raise EdgeError(f"Edge ({key1}, {key2} already exists")

    def adde(self, key1, key2, dist=1):
        """Add a new edge (`key1`, `key2`) or update the existing one with `dist`.

        Parameters
        ----------
        key1 : hashable
        key2 : hashable
        dist : optional

        Raises
        ------
        EdgeError
            Raised if the edge (`key1`, `key2`) already exists, or would
            result in a loop. This a safeguard to prevent accidentally
            overriding an existing edge.

        See Also
        --------
        set(key) : `dist` is mandatory and no error is raised if the edge
            (`key1`, `key2`) exists.
        """
        if not self.has(key1):
            self._add(key1, None)
        if not self.has(key2):
            self._add(key2, None)
        self._check_adde(key1, key2)
        self._adde(key1, key2, dist)

    def sete(self, key1, key2, dist):
        """Add a new or update an existing edge (`key1`, `key2`) with `dist`.

        Parameters
        ----------
        key1 : hashable
        key2 : hashable
        dist : optional

        See Also
        --------
        add(key) : `dist` is optional. Raises an error if the edge (`key1`,
            `key2`) exists but `dist` is not provided.
        """
        if not self.has(key1):
            self._add(key1, None)
        if not self.has(key2):
            self._add(key2, None)
        self._adde(key1, key2, dist)

    @_Decorators.VertexOrEdge(set, sete)
    def __setitem__(self, key, item):
        pass

    # Deleters.

    @_Decorators.check_key
    def deln(self, key):
        """Delete the node with `key` from `self`.

        `deln()` walks through all nodes to delete edges pointing to the
        deleted node.
        """
        del_node = self._get(key)
        del self.data[key]
        for node in self.data.items():
            if del_node in node:
                del node[del_node]
    _deln = deln.__wrapped__

    @_Decorators.check_key_pair
    def dele(self, key1, key2):
        """Delete the edge (`key1`, `key2`).

        Unlike `Graph.dele()`, this method does **not** delete the edge
        (`key2`, `key1`).
        """
        if not self._isadj(key1, key2):
            raise KeyError(f"Edge ({key1}, {key2}) does not exist")
        del self._get(key1)[self._get(key2)]
    _dele = dele.__wrapped__

    @_Decorators.VertexOrEdge(deln, dele)
    def __delitem__(self, key):
        pass


class Graph(Digraph):
    """A class representing simple directed graphs (digraphs)

    `Graph` is a dict-like object that stores a mapping of unique, hashable
    keys to their corresponding `Node` objects. `Graph` supports one edge
    per pair of nodes as well as a weight or distance associated with
    each edge. For directed graphs, use the `Digraph` class instead.

    `Graph` is implemented as a subclass of `Digraph` that overrides the
    `adde()`, `sete()`, and `dele()` methods to make sure two nodes that are
    adjacent share the same edge.

    `Graph` has methods to check for the existence, add, get or set the
    value of, and delete nodes or edges. In addition, `Graph` supports
    standard syntax sugars of Python container classes for convenience.

    Attributes
    ----------
    data : dict_like
        The underlying dict-like object that maps keys to `Node` objects.
    directed : bool
        True for directed graphs

    Methods
    -------
    __contains__(key)
    __delitem__(key)
    __getitem__(key)
    __setitem__(key)
    add(key, value=None)
    adde(key1, key2, dist=1)
    dele(key1, key2)
    deln(key)
    get(key)
    gete(key1, key2)
    has(key)
    isadj(key1, key2)
    neigh(key)
    nodes()
    set(key, value)
    sete(key1, key2, dist=1)

    See Also
    --------
    tulun.Digraph: The superclass that implements simple directed graphs.
    """

    directed = False

    def adde(self, key1, key2, dist=1):
        """Add a new edge {`key1`, `key2`} or update the existing one with `dist`.

        Parameters
        ----------
        key1 : hashable
        key2 : hashable
        dist : optional

        Raises
        ------
        EdgeError
            Raised if the edge {`key1`, `key2`} already exists, or would
            result in a loop. This a safeguard to prevent accidentally
            overriding an existing edge.

        See Also
        --------
        set(key) : `dist` is mandatory and no error is raised if the edge
            {`key1`, `key2`} exists.
        """
        if not self.has(key1):
            self._add(key1, None)
        if not self.has(key2):
            self._add(key2, None)
        self._check_adde(key1, key2)
        self._adde(key1, key2, dist)
        self._adde(key2, key1, dist)

    def sete(self, key1, key2, dist):
        """Add a new or update an existing edge {`key1`, `key2`} with `dist`.

        Parameters
        ----------
        key1 : hashable
        key2 : hashable
        dist : optional

        See Also
        --------
        add(key) : `dist` is optional. Raises an error if the edge {`key1`,
            `key2`} exists but `dist` is not provided.
        """
        if not self.has(key1):
            self._add(key1, None)
        if not self.has(key2):
            self._add(key2, None)
        self._adde(key1, key2, dist)
        self._adde(key2, key1, dist)

    @_Decorators.check_key_pair
    def dele(self, key1, key2):
        """Delete the edge {`key1`, `key2`}.
        """
        if not self._isadj(key1, key2):
            raise KeyError(f"Edge ({key1}, {key2}) does not exist")
        node1 = self._get(key1)
        node2 = self._get(key2)
        del node1[node2]
        del node2[node1]
    _dele = dele.__wrapped__
